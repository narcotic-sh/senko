#!/usr/bin/env python3
"""Export and verify the static ONNX models used by the browser runtime.

The generated files are deployment artifacts. They belong in
``web/public/models`` locally (or in an artifact store in CI), not in git.
Every export is checked with ONNX Runtime CPU against the PyTorch source before
the manifest is written.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
import onnxconverter_common
import torch
from onnx import TensorProto
from onnxconverter_common import float16 as onnx_float16
from torch import nn
from torch.nn import functional as F

from pyannote_frontend_webgpu.pack import build_package as build_frontend_webgpu_package
from pyannote_tail_webgpu.pack import build_package as build_tail_webgpu_package


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "web" / "public" / "models"

OPSET_VERSION = 18
EXPORT_SEED = 0x5E_4B_0
PYANNOTE_BATCHES = (1, 8, 16, 32)
CAMPPLUS_BATCHES = (32, 64, 128)
CAMPPLUS_FP16_MAX_ABS_ERROR = 0.03
CAMPPLUS_FP16_MIN_COSINE_SIMILARITY = 0.9999
CAMPPLUS_FP16_MIN_POSITIVE_VALUE = 0.0
CAMPPLUS_FP16_MAX_FINITE_VALUE = 65_504.0
PYANNOTE_SAMPLES = 160_000
PYANNOTE_FRAMES = 589
PYANNOTE_FRONTEND_FEATURES = 60
PYANNOTE_LSTM_HIDDEN = 128
PYANNOTE_LSTM_OUTPUT_FEATURES = 2 * PYANNOTE_LSTM_HIDDEN
PYANNOTE_CLASSES = 7
PYANNOTE_LSTM_LAYERS = 4
PYANNOTE_LSTM_DIRECTIONS = ("forward", "reverse")
PYANNOTE_LSTM_GATE_ORDER = ("input", "forget", "cell", "output")
PYANNOTE_LSTM_WEIGHTS_FILE = "pyannote-segmentation-3.0-lstm-f32.bin"
PYANNOTE_LSTM_METADATA_FILE = "pyannote-segmentation-3.0-lstm.json"
PYANNOTE_LSTM_FP16_WEIGHTS_FILE = "pyannote-segmentation-3.0-lstm-f16.bin"
PYANNOTE_LSTM_FP16_METADATA_FILE = "pyannote-segmentation-3.0-lstm-f16.json"

PYANNOTE_WEIGHTS = (
    REPO_ROOT / "models" / "pyannote_segmentation_3.0" / "pytorch_model.bin"
)
PYANNOTE_CONFIG = (
    REPO_ROOT / "models" / "pyannote_segmentation_3.0" / "config.yaml"
)
CAMPPLUS_WEIGHTS = (
    REPO_ROOT
    / "models"
    / "speech_campplus_sv_zh_en_16k-common_advanced"
    / "campplus_cn_en_common.pt"
)
CAMPPLUS_SOURCE = REPO_ROOT / "tracing" / "coreml" / "camplusplus_coreml.py"
CAMPPLUS_REFERENCE_SOURCE = REPO_ROOT / "senko" / "camplusplus.py"

# Updating either weight file must be an explicit, reviewed operation. Besides
# making exports reproducible, these pins prevent accidentally publishing a
# local or partially downloaded checkpoint.
EXPECTED_SOURCE_SHA256 = {
    PYANNOTE_WEIGHTS: "da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298",
    CAMPPLUS_WEIGHTS: "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199",
}


@dataclass(frozen=True)
class ModelSpec:
    family: str
    batch: int
    file_name: str
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    precision: str = "float32"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def check_pinned_sources() -> None:
    for path, expected in EXPECTED_SOURCE_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required source model is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Unexpected SHA-256 for {relative(path)}: {actual}\n"
                f"Expected: {expected}\n"
                "Review the new checkpoint, then update EXPECTED_SOURCE_SHA256."
            )


def _load_python_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_campplus() -> nn.Module:
    module = _load_python_module("senko_browser_campplus", CAMPPLUS_SOURCE)
    model = module.CAMPPlusCoreML(feat_dim=80, embedding_size=192)
    state_dict = torch.load(CAMPPLUS_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_campplus_reference() -> nn.Module:
    module = _load_python_module("senko_reference_campplus", CAMPPLUS_REFERENCE_SOURCE)
    model = module.CAMPPlus(feat_dim=80, embedding_size=192)
    state_dict = torch.load(CAMPPLUS_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_pyannote_raw_logits() -> nn.Module:
    # These imports are intentionally local. CAM++ export does not need the
    # large pyannote dependency tree to be imported.
    from pyannote.audio import Model
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from pyannote.audio.models.segmentation import PyanNet

    safe_types = [
        torch.torch_version.TorchVersion,
        Specifications,
        Problem,
        Resolution,
        PyanNet,
    ]
    with torch.serialization.safe_globals(safe_types):
        model = Model.from_pretrained(PYANNOTE_WEIGHTS, map_location="cpu")

    if not isinstance(model.activation, nn.LogSoftmax):
        raise TypeError(
            "Expected pyannote segmentation-3.0 to end in LogSoftmax, got "
            f"{type(model.activation).__name__}."
        )

    # VAD uses argmax class decoding, so exporting classifier logits is exactly
    # equivalent while avoiding an unsupported WebGPU LogSoftmax node.
    model.activation = nn.Identity()
    freeze_pyannote_sincnet_filterbank(model)
    model.eval()
    return model


def freeze_pyannote_sincnet_filterbank(model: nn.Module) -> None:
    """Replace the inference-constant Sinc filterbank with a static Conv1d.

    ``ParamSincFB`` constructs its 251-tap kernels on every forward pass.  The
    legacy ONNX export expresses the mirrored half-kernel with a negative-step
    ``Slice``.  ONNX Runtime Web/JSEP currently drops one element from that
    slice, producing 250 values that cannot be reshaped to ``[40, 1, 251]``.

    The learned low/band parameters are fixed for inference, so materializing
    ``Encoder.get_filters()`` once is mathematically identical.  Besides
    avoiding the browser incompatibility, this removes trigonometry, slicing,
    concatenation, and shape work from every WebGPU inference.
    """

    try:
        encoder = model.sincnet.conv1d[0]  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError) as error:
        raise TypeError("Expected pyannote PyanNet SincNet encoder.") from error

    if not callable(getattr(encoder, "get_filters", None)):
        raise TypeError("Expected the first SincNet encoder to expose get_filters().")
    if not bool(getattr(encoder, "as_conv1d", False)):
        raise ValueError("Only the channels-first SincNet encoder can be frozen.")
    if bool(getattr(encoder, "is_pinv", False)):
        raise ValueError("A pseudo-inverse SincNet encoder cannot be frozen for analysis.")

    with torch.no_grad():
        filters = encoder.get_filters().detach().clone()
    if tuple(filters.shape) != (80, 1, 251):
        raise ValueError(
            "Unexpected pyannote SincNet filter shape: "
            f"{tuple(filters.shape)}; expected [80, 1, 251]."
        )

    frozen = nn.Conv1d(
        in_channels=filters.shape[1],
        out_channels=filters.shape[0],
        kernel_size=filters.shape[2],
        stride=encoder.stride,
        padding=encoder.padding,
        bias=False,
        device=filters.device,
        dtype=filters.dtype,
    )
    with torch.no_grad():
        frozen.weight.copy_(filters)
    frozen.weight.requires_grad_(False)
    model.sincnet.conv1d[0] = frozen  # type: ignore[attr-defined]


class PyannoteFrontend(nn.Module):
    """Static SincNet frontend with a batch-major GPU boundary."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.sincnet = model.sincnet  # type: ignore[attr-defined]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # SincNet emits [B, F, T]. The persistent LSTM consumes [B, T, F],
        # keeping each independent sequence contiguous in a storage buffer.
        return self.sincnet(waveform).transpose(1, 2)


class PyannoteTail(nn.Module):
    """Post-LSTM affine/activation classifier consuming ``[B, T, 256]``."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.linear = model.linear  # type: ignore[attr-defined]
        self.classifier = model.classifier  # type: ignore[attr-defined]

    def forward(self, recurrent: torch.Tensor) -> torch.Tensor:
        output = recurrent
        for linear in self.linear:
            output = F.leaky_relu(linear(output))
        return self.classifier(output)


def specs_for(family: str, batches: Iterable[int]) -> list[ModelSpec]:
    if family == "pyannote":
        return [
            ModelSpec(
                family=family,
                batch=batch,
                file_name=f"pyannote-segmentation-3.0-logits-b{batch}.onnx",
                input_name="waveform",
                output_name="logits",
                input_shape=(batch, 1, PYANNOTE_SAMPLES),
                output_shape=(batch, PYANNOTE_FRAMES, PYANNOTE_CLASSES),
            )
            for batch in batches
        ]
    if family == "pyannote_frontend":
        return [
            ModelSpec(
                family=family,
                batch=batch,
                file_name=f"pyannote-segmentation-3.0-frontend-b{batch}.onnx",
                input_name="waveform",
                output_name="features",
                input_shape=(batch, 1, PYANNOTE_SAMPLES),
                output_shape=(batch, PYANNOTE_FRAMES, PYANNOTE_FRONTEND_FEATURES),
            )
            for batch in batches
        ]
    if family == "pyannote_tail":
        return [
            ModelSpec(
                family=family,
                batch=batch,
                file_name=f"pyannote-segmentation-3.0-tail-b{batch}.onnx",
                input_name="recurrent",
                output_name="logits",
                input_shape=(batch, PYANNOTE_FRAMES, PYANNOTE_LSTM_OUTPUT_FEATURES),
                output_shape=(batch, PYANNOTE_FRAMES, PYANNOTE_CLASSES),
            )
            for batch in batches
        ]
    if family in {"campplus", "campplus_fp16"}:
        suffix = "-fp16" if family == "campplus_fp16" else ""
        return [
            ModelSpec(
                family=family,
                batch=batch,
                file_name=f"campplus-t150-b{batch}{suffix}.onnx",
                input_name="features",
                output_name="embeddings",
                input_shape=(batch, 150, 80),
                output_shape=(batch, 192),
                precision="float16" if family == "campplus_fp16" else "float32",
            )
            for batch in batches
        ]
    raise ValueError(f"Unknown model family: {family}")


def make_input(spec: ModelSpec) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    family_offsets = {
        "pyannote": 0,
        "pyannote_frontend": 200_000,
        "pyannote_tail": 300_000,
        "campplus": 100_000,
        "campplus_fp16": 100_000,
    }
    family_offset = family_offsets[spec.family]
    generator.manual_seed(EXPORT_SEED + family_offset + spec.batch)
    value = torch.randn(spec.input_shape, generator=generator, dtype=torch.float32)
    if spec.family in {"pyannote", "pyannote_frontend"}:
        # A realistic PCM scale also keeps the SincNet numerical test focused.
        value = (value * 0.1).clamp_(-1.0, 1.0)
    elif spec.family == "pyannote_tail":
        # Trained LSTM hidden states occupy a narrow subset of [-1, 1].
        value = value * 0.05
    return value


def export_onnx(model: nn.Module, sample: torch.Tensor, spec: ModelSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".onnx.incomplete")
    temporary_path.unlink(missing_ok=True)
    try:
        with torch.inference_mode():
            # The legacy exporter intentionally emits the four recurrent layers
            # as ONNX LSTM operators. The dynamo exporter currently decomposes
            # or rejects parts of this pyannote model.
            torch.onnx.export(
                model,
                (sample,),
                temporary_path,
                export_params=True,
                opset_version=OPSET_VERSION,
                do_constant_folding=True,
                input_names=[spec.input_name],
                output_names=[spec.output_name],
                dynamic_axes=None,
                dynamo=False,
            )

        graph = onnx.load(temporary_path, load_external_data=True)
        graph.producer_name = "senko-browser-model-tools"
        graph.producer_version = "0.1.0"
        metadata = {
            "senko.family": spec.family,
            "senko.batch_size": str(spec.batch),
            "senko.input_shape": ",".join(map(str, spec.input_shape)),
            "senko.output_shape": ",".join(map(str, spec.output_shape)),
            "senko.opset": str(OPSET_VERSION),
            "senko.precision": spec.precision,
        }
        if spec.family in {"pyannote", "pyannote_frontend"}:
            metadata["senko.frontend"] = "static-sinc-convolution"
        if spec.family in {"pyannote_frontend", "pyannote_tail"}:
            metadata["senko.boundary_layout"] = "batch,frame,feature"
        del graph.metadata_props[:]
        for key, value in sorted(metadata.items()):
            entry = graph.metadata_props.add()
            entry.key = key
            entry.value = value
        onnx.checker.check_model(graph, full_check=True)
        onnx.save_model(graph, temporary_path, save_as_external_data=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def export_campplus_fp16(
    fp32_path: Path,
    spec: ModelSpec,
    path: Path,
) -> None:
    """Convert a checked FP32 CAM++ graph to FP16 with FP32 API boundaries."""
    if spec.family != "campplus_fp16" or spec.precision != "float16":
        raise ValueError("FP16 conversion requires a campplus_fp16 model spec.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".onnx.incomplete")
    temporary_path.unlink(missing_ok=True)
    try:
        source = onnx.load(fp32_path, load_external_data=True)
        converted = onnx_float16.convert_float_to_float16(
            source,
            min_positive_val=CAMPPLUS_FP16_MIN_POSITIVE_VALUE,
            max_finite_val=CAMPPLUS_FP16_MAX_FINITE_VALUE,
            keep_io_types=True,
            disable_shape_infer=False,
        )
        converted.producer_name = "senko-browser-model-tools"
        converted.producer_version = "0.1.0"
        metadata = {entry.key: entry.value for entry in converted.metadata_props}
        metadata.update(
            {
                "senko.family": spec.family,
                "senko.precision": "float16-internal-float32-io",
                "senko.fp32_source_sha256": sha256(fp32_path),
                "senko.fp16_converter": (
                    f"onnxconverter-common-{onnxconverter_common.__version__}"
                ),
            }
        )
        del converted.metadata_props[:]
        for key, value in sorted(metadata.items()):
            entry = converted.metadata_props.add()
            entry.key = key
            entry.value = value
        onnx.checker.check_model(converted, full_check=True)
        onnx.save_model(converted, temporary_path, save_as_external_data=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def inspect_onnx(path: Path, spec: ModelSpec) -> dict[str, Any]:
    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    operators = Counter(node.op_type for node in graph.graph.node)

    if spec.family == "pyannote":
        if operators["LSTM"] != 4:
            raise RuntimeError(
                f"{path.name} must contain four LSTM operators, found {operators['LSTM']}."
            )
        forbidden = sorted({"LogSoftmax", "Softmax"}.intersection(operators))
        if forbidden:
            raise RuntimeError(f"{path.name} contains forbidden operators: {forbidden}")

    if spec.family in {"pyannote_frontend", "pyannote_tail"}:
        if operators["LSTM"] != 0:
            raise RuntimeError(f"{path.name} must not contain an ONNX LSTM node.")
        forbidden = sorted({"LogSoftmax", "Softmax"}.intersection(operators))
        if forbidden:
            raise RuntimeError(f"{path.name} contains forbidden operators: {forbidden}")

    if spec.family in {"pyannote", "pyannote_frontend"}:
        assert_static_sinc_convolution(graph, operators, path)

    initializer_types = Counter(
        TensorProto.DataType.Name(initializer.data_type).lower()
        for initializer in graph.graph.initializer
    )
    if spec.family == "campplus_fp16":
        assert_campplus_fp16_contract(graph, operators, initializer_types, path)

    return {
        "ir_version": graph.ir_version,
        "node_count": len(graph.graph.node),
        "operators": dict(sorted(operators.items())),
        "initializer_dtypes": dict(sorted(initializer_types.items())),
    }


def assert_campplus_fp16_contract(
    graph: onnx.ModelProto,
    operators: Counter[str],
    initializer_types: Counter[str],
    path: Path,
) -> None:
    input_types = [value.type.tensor_type.elem_type for value in graph.graph.input]
    output_types = [value.type.tensor_type.elem_type for value in graph.graph.output]
    if input_types != [TensorProto.FLOAT] or output_types != [TensorProto.FLOAT]:
        raise RuntimeError(
            f"{path.name} must retain float32 input/output boundaries; "
            f"got inputs {input_types}, outputs {output_types}."
        )
    if initializer_types["float16"] == 0 or initializer_types["float"] != 0:
        raise RuntimeError(
            f"{path.name} must store every floating initializer as float16; "
            f"found {dict(initializer_types)}."
        )
    if operators["Cast"] != 2:
        raise RuntimeError(
            f"{path.name} must contain exactly two FP32 boundary Cast nodes; "
            f"found {operators['Cast']}."
        )

    cast_targets = Counter()
    for node in graph.graph.node:
        if node.op_type != "Cast":
            continue
        to_attribute = next(
            (attribute for attribute in node.attribute if attribute.name == "to"),
            None,
        )
        if to_attribute is None:
            raise RuntimeError(f"{path.name} has a Cast node without a target dtype.")
        cast_targets[TensorProto.DataType.Name(to_attribute.i).lower()] += 1
    if cast_targets != Counter({"float": 1, "float16": 1}):
        raise RuntimeError(
            f"{path.name} must cast FP32->FP16 at input and FP16->FP32 at output; "
            f"found {dict(cast_targets)}."
        )


def assert_static_sinc_convolution(
    graph: onnx.ModelProto,
    operators: Counter[str],
    path: Path,
) -> None:
    dynamic_sinc_operators = sorted(
        {"Clip", "Cos", "Neg", "Sin", "Slice"}.intersection(operators)
    )
    if dynamic_sinc_operators:
        raise RuntimeError(
            f"{path.name} still constructs Sinc filters at runtime: "
            f"{dynamic_sinc_operators}"
        )

    initializer_names = {initializer.name for initializer in graph.graph.initializer}
    first_convolution = next(
        (node for node in graph.graph.node if node.op_type == "Conv"), None
    )
    if (
        first_convolution is None
        or len(first_convolution.input) < 2
        or first_convolution.input[1] not in initializer_names
    ):
        raise RuntimeError(
            f"{path.name} must use an initializer for its first convolution."
        )


def compare_outputs(
    source: np.ndarray, actual: np.ndarray, spec: ModelSpec
) -> dict[str, Any]:
    if tuple(source.shape) != spec.output_shape:
        raise RuntimeError(
            f"PyTorch output shape is {source.shape}; expected {spec.output_shape}."
        )
    if tuple(actual.shape) != spec.output_shape:
        raise RuntimeError(
            f"ONNX Runtime output shape is {actual.shape}; expected {spec.output_shape}."
        )

    difference = np.abs(source - actual)
    denominator = np.maximum(np.abs(source), 1e-5)
    metrics: dict[str, Any] = {
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "max_relative_error": float((difference / denominator).max(initial=0.0)),
    }

    if spec.family == "pyannote":
        source_classes = source.argmax(axis=-1)
        actual_classes = actual.argmax(axis=-1)
        agreement = float(np.mean(source_classes == actual_classes))
        metrics["argmax_agreement"] = agreement
        if metrics["max_abs_error"] > 5e-4 or agreement < 0.9999:
            raise RuntimeError(f"Pyannote verification failed: {metrics}")
    elif spec.family in {"campplus", "campplus_fp16"}:
        source_norm = source / np.maximum(
            np.linalg.norm(source, axis=-1, keepdims=True), 1e-12
        )
        actual_norm = actual / np.maximum(
            np.linalg.norm(actual, axis=-1, keepdims=True), 1e-12
        )
        similarities = np.sum(source_norm * actual_norm, axis=-1)
        metrics["minimum_cosine_similarity"] = float(similarities.min())
        metrics["mean_cosine_similarity"] = float(similarities.mean())
        max_abs_error = (
            CAMPPLUS_FP16_MAX_ABS_ERROR
            if spec.precision == "float16"
            else 5e-4
        )
        minimum_cosine = (
            CAMPPLUS_FP16_MIN_COSINE_SIMILARITY
            if spec.precision == "float16"
            else 0.99999
        )
        if (
            metrics["max_abs_error"] > max_abs_error
            or metrics["minimum_cosine_similarity"] < minimum_cosine
        ):
            raise RuntimeError(f"CAM++ verification failed: {metrics}")
    else:
        if metrics["max_abs_error"] > 5e-4:
            raise RuntimeError(f"Pyannote {spec.family} verification failed: {metrics}")

    return metrics


def verify_onnx(
    model: nn.Module, sample: torch.Tensor, spec: ModelSpec, path: Path
) -> dict[str, Any]:
    with torch.inference_mode():
        source = model(sample).detach().cpu().numpy()

    optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        if spec.precision == "float16"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    actual = run_onnx(sample, spec, path, optimization_level)
    return compare_outputs(source, actual, spec)


def run_onnx(
    sample: torch.Tensor,
    spec: ModelSpec,
    path: Path,
    optimization_level: ort.GraphOptimizationLevel,
) -> np.ndarray:
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization_level
    options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
    session = ort.InferenceSession(
        path.as_posix(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        [spec.output_name], {spec.input_name: sample.detach().cpu().numpy()}
    )[0]
    return actual


def verify_campplus_fp16(
    model: nn.Module,
    sample: torch.Tensor,
    spec: ModelSpec,
    path: Path,
    fp32_path: Path,
) -> dict[str, Any]:
    with torch.inference_mode():
        source = model(sample).detach().cpu().numpy()
    fp16_actual = run_onnx(
        sample,
        spec,
        path,
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    )
    verification = compare_outputs(source, fp16_actual, spec)
    fp32_spec = specs_for("campplus", (spec.batch,))[0]
    fp32_actual = run_onnx(
        sample,
        fp32_spec,
        fp32_path,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    )
    verification["comparison_to_float32_onnx"] = compare_outputs(
        fp32_actual,
        fp16_actual,
        spec,
    )
    return verification


def compare_campplus_reference(
    model: nn.Module, reference: nn.Module, sample: torch.Tensor
) -> dict[str, float]:
    """Measure the static export model against Senko's reference CAMPPlus.

    For T=150, every CAM layer sees fewer than 100 frames after TDNN
    downsampling, making ``2 * global_mean`` equivalent to the reference
    segmented-pooling context. The remaining intentional difference is biased
    variance with epsilon in the final statistics pool, matching the shipping
    Core ML model and avoiding the reference graph's invalid ONNX trace.
    """
    with torch.inference_mode():
        actual = model(sample).detach().cpu().numpy()
        expected = reference(sample).detach().cpu().numpy()
    difference = np.abs(expected - actual)
    expected_norm = expected / np.maximum(
        np.linalg.norm(expected, axis=-1, keepdims=True), 1e-12
    )
    actual_norm = actual / np.maximum(
        np.linalg.norm(actual, axis=-1, keepdims=True), 1e-12
    )
    similarities = np.sum(expected_norm * actual_norm, axis=-1)
    metrics = {
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
        "minimum_cosine_similarity": float(similarities.min()),
        "mean_cosine_similarity": float(similarities.mean()),
    }
    if metrics["minimum_cosine_similarity"] < 0.9999:
        raise RuntimeError(f"CAM++ reference-semantic verification failed: {metrics}")
    return metrics


def artifact_record(
    path: Path,
    spec: ModelSpec,
    graph_info: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "family": spec.family,
        "batch_size": spec.batch,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "precision": spec.precision,
        "opset": OPSET_VERSION,
        "input": {"name": spec.input_name, "shape": list(spec.input_shape)},
        "output": {"name": spec.output_name, "shape": list(spec.output_shape)},
        "graph": graph_info,
        "verification": verification,
    }
    if spec.family == "campplus_fp16":
        record["internal_dtype"] = "float16"
        record["io_dtype"] = "float32"
        record["conversion"] = {
            "tool": "onnxconverter-common",
            "version": onnxconverter_common.__version__,
            "keep_io_types": True,
            "min_positive_val": CAMPPLUS_FP16_MIN_POSITIVE_VALUE,
            "max_finite_val": CAMPPLUS_FP16_MAX_FINITE_VALUE,
        }
    return record


def pack_pyannote_lstm(
    model: nn.Module,
    precision: str = "float32",
) -> tuple[bytes, dict[str, Any]]:
    """Pack PyanNet's LSTM into a deterministic, WGSL-friendly byte buffer.

    PyTorch stores each gate matrix as ``[4H, input]``/``[4H, H]`` in IFGO
    order. Each matrix first concatenates the input and recurrent columns, then
    changes physical layout to ``[gate, column/4, hidden, input_lane]``. For a
    fixed input vec4, adjacent GPU lanes therefore fetch adjacent packed weight
    vectors instead of striding by a complete source row. The two PyTorch biases
    stay separate; the FP32 package preserves them exactly and the optional FP16
    package applies deterministic IEEE binary16 rounding.
    """

    if precision == "float32":
        storage_dtype = "<f4"
        tensor_dtype = "float32-le"
        metadata_version = 2
        package_format = "senko-persistent-lstm-f32-gc4h"
        weights_file = PYANNOTE_LSTM_WEIGHTS_FILE
    elif precision == "float16":
        storage_dtype = "<f2"
        tensor_dtype = "float16-le"
        metadata_version = 3
        package_format = "senko-persistent-lstm-f16-gc4h"
        weights_file = PYANNOTE_LSTM_FP16_WEIGHTS_FILE
    else:
        raise ValueError(
            f"Unsupported pyannote LSTM package precision: {precision!r}."
        )

    lstm = model.lstm  # type: ignore[attr-defined]
    expected = {
        "input_size": PYANNOTE_FRONTEND_FEATURES,
        "hidden_size": PYANNOTE_LSTM_HIDDEN,
        "num_layers": PYANNOTE_LSTM_LAYERS,
        "bidirectional": True,
        "batch_first": True,
    }
    for attribute, value in expected.items():
        if getattr(lstm, attribute) != value:
            raise ValueError(
                f"Unexpected pyannote LSTM {attribute}: {getattr(lstm, attribute)!r}; "
                f"expected {value!r}."
            )

    state = lstm.state_dict()
    payload = bytearray()
    layers: list[dict[str, Any]] = []
    for layer in range(PYANNOTE_LSTM_LAYERS):
        input_size = (
            PYANNOTE_FRONTEND_FEATURES
            if layer == 0
            else PYANNOTE_LSTM_OUTPUT_FEATURES
        )
        directions: list[dict[str, Any]] = []
        for direction in PYANNOTE_LSTM_DIRECTIONS:
            suffix = "" if direction == "forward" else "_reverse"
            source_names = {
                "weight_ih": f"weight_ih_l{layer}{suffix}",
                "weight_hh": f"weight_hh_l{layer}{suffix}",
                "bias_ih": f"bias_ih_l{layer}{suffix}",
                "bias_hh": f"bias_hh_l{layer}{suffix}",
            }
            arrays = {
                name: state[source_name].detach().cpu().numpy().astype("<f4", copy=False)
                for name, source_name in source_names.items()
            }
            expected_gate_rows = 4 * PYANNOTE_LSTM_HIDDEN
            if arrays["weight_ih"].shape != (expected_gate_rows, input_size):
                raise ValueError(
                    f"Unexpected {source_names['weight_ih']} shape: "
                    f"{arrays['weight_ih'].shape}."
                )
            if arrays["weight_hh"].shape != (
                expected_gate_rows,
                PYANNOTE_LSTM_HIDDEN,
            ):
                raise ValueError(
                    f"Unexpected {source_names['weight_hh']} shape: "
                    f"{arrays['weight_hh'].shape}."
                )

            fused = np.ascontiguousarray(
                np.concatenate((arrays["weight_ih"], arrays["weight_hh"]), axis=1),
                dtype="<f4",
            )
            columns = fused.shape[1]
            if columns % 4 != 0:
                raise ValueError(
                    f"Pyannote LSTM fused column count {columns} is not divisible by 4."
                )
            packed_matrix = np.ascontiguousarray(
                fused.reshape(4, PYANNOTE_LSTM_HIDDEN, columns // 4, 4)
                .transpose(0, 2, 1, 3),
                dtype="<f4",
            )
            tensors: dict[str, dict[str, Any]] = {}
            for name, value in (
                ("matrix", packed_matrix),
                ("bias_ih", arrays["bias_ih"]),
                ("bias_hh", arrays["bias_hh"]),
            ):
                if len(payload) % 256 != 0:
                    raise RuntimeError("LSTM package tensor is not 256-byte aligned.")
                offset = len(payload)
                raw = np.ascontiguousarray(value, dtype=storage_dtype).tobytes(
                    order="C"
                )
                payload.extend(raw)
                tensors[name] = {
                    "offset_bytes": offset,
                    "length_bytes": len(raw),
                    "shape": (
                        [4 * PYANNOTE_LSTM_HIDDEN, columns]
                        if name == "matrix"
                        else list(value.shape)
                    ),
                    "packed_shape": list(value.shape),
                    "dtype": tensor_dtype,
                    "layout": (
                        "gate-column4-hidden-input4"
                        if name == "matrix"
                        else "row-major"
                    ),
                }

            directions.append(
                {
                    "direction": direction,
                    "time_order": (
                        "0..588" if direction == "forward" else "588..0"
                    ),
                    "input_size": input_size,
                    "hidden_size": PYANNOTE_LSTM_HIDDEN,
                    "gate_order": list(PYANNOTE_LSTM_GATE_ORDER),
                    "matrix_columns": [
                        {"source": source_names["weight_ih"], "size": input_size},
                        {
                            "source": source_names["weight_hh"],
                            "size": PYANNOTE_LSTM_HIDDEN,
                        },
                    ],
                    "bias_sources": [
                        source_names["bias_ih"],
                        source_names["bias_hh"],
                    ],
                    "tensors": tensors,
                }
            )
        layers.append(
            {
                "layer": layer,
                "input_size": input_size,
                "output_size": PYANNOTE_LSTM_OUTPUT_FEATURES,
                "output_concatenation": ["forward", "reverse"],
                "directions": directions,
            }
        )

    weights = bytes(payload)
    if len(weights) % 256 != 0:
        raise RuntimeError("LSTM package size must be 256-byte aligned.")
    metadata: dict[str, Any] = {
        "version": metadata_version,
        "model": "pyannote-segmentation-3.0",
        "format": package_format,
        "byte_order": "little-endian",
        "alignment_bytes": 256,
        "boundary_layout": "batch,frame,feature",
        "frames": PYANNOTE_FRAMES,
        "num_layers": PYANNOTE_LSTM_LAYERS,
        "bidirectional": True,
        "hidden_size": PYANNOTE_LSTM_HIDDEN,
        "gate_order": list(PYANNOTE_LSTM_GATE_ORDER),
        "equations": {
            "affine": "z = W_fused @ concat(x_t, h_prev) + bias_ih + bias_hh",
            "gates": "i=sigmoid(z_i), f=sigmoid(z_f), g=tanh(z_g), o=sigmoid(z_o)",
            "cell": "c_t = f * c_prev + i * g",
            "hidden": "h_t = o * tanh(c_t)",
        },
        "weights": {
            "file": weights_file,
            "bytes": len(weights),
            "sha256": hashlib.sha256(weights).hexdigest(),
        },
        "layers": layers,
    }
    if precision == "float16":
        metadata["storage_dtype"] = "float16"
        metadata["accumulator_dtype"] = "float32"
        metadata["required_webgpu_features"] = ["shader-f16"]
    return weights, metadata


def write_or_verify_pyannote_lstm_package(
    model: nn.Module,
    output_dir: Path,
    verify_only: bool,
    precision: str = "float32",
) -> dict[str, Any]:
    weights, metadata = pack_pyannote_lstm(model, precision)
    if precision == "float32":
        weights_path = output_dir / PYANNOTE_LSTM_WEIGHTS_FILE
        metadata_path = output_dir / PYANNOTE_LSTM_METADATA_FILE
    elif precision == "float16":
        weights_path = output_dir / PYANNOTE_LSTM_FP16_WEIGHTS_FILE
        metadata_path = output_dir / PYANNOTE_LSTM_FP16_METADATA_FILE
    else:
        raise ValueError(
            f"Unsupported pyannote LSTM package precision: {precision!r}."
        )
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()

    if verify_only:
        if not weights_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Cannot verify missing pyannote LSTM package.")
        if weights_path.read_bytes() != weights:
            raise RuntimeError(f"{weights_path.name} does not match the checkpoint.")
        if metadata_path.read_bytes() != metadata_bytes:
            raise RuntimeError(f"{metadata_path.name} does not match the checkpoint.")
    else:
        temporary_weights = weights_path.with_suffix(".bin.incomplete")
        temporary_metadata = metadata_path.with_suffix(".json.incomplete")
        temporary_weights.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        try:
            temporary_weights.write_bytes(weights)
            temporary_metadata.write_bytes(metadata_bytes)
            temporary_weights.replace(weights_path)
            temporary_metadata.replace(metadata_path)
        finally:
            temporary_weights.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)

    return {
        "format": metadata["format"],
        "boundary_layout": metadata["boundary_layout"],
        "frames": PYANNOTE_FRAMES,
        "input_features": PYANNOTE_FRONTEND_FEATURES,
        "output_features": PYANNOTE_LSTM_OUTPUT_FEATURES,
        "weights": {
            "file": weights_path.name,
            "bytes": weights_path.stat().st_size,
            "sha256": sha256(weights_path),
        },
        "metadata": {
            "file": metadata_path.name,
            "bytes": metadata_path.stat().st_size,
            "sha256": sha256(metadata_path),
        },
    }


def split_variant(record: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in record.items()
        if key not in {"family", "batch_size", "input", "output"}
    }
    result["input_shape"] = record["input"]["shape"]
    result["output_shape"] = record["output"]["shape"]
    return result


def pyannote_boundary_buffers(batch: int) -> dict[str, int]:
    bytes_per_float = 4
    return {
        "waveform_bytes": batch * PYANNOTE_SAMPLES * bytes_per_float,
        "first_convolution_activation_bytes": batch * 80 * 15_975 * bytes_per_float,
        "frontend_output_bytes": (
            batch
            * PYANNOTE_FRAMES
            * PYANNOTE_FRONTEND_FEATURES
            * bytes_per_float
        ),
        "recurrent_output_bytes": (
            batch
            * PYANNOTE_FRAMES
            * PYANNOTE_LSTM_OUTPUT_FEATURES
            * bytes_per_float
        ),
        "tail_output_bytes": (
            batch * PYANNOTE_FRAMES * PYANNOTE_CLASSES * bytes_per_float
        ),
        "two_recurrent_ping_pong_buffers_bytes": (
            2
            * batch
            * PYANNOTE_FRAMES
            * PYANNOTE_LSTM_OUTPUT_FEATURES
            * bytes_per_float
        ),
        "input_affine_scratch_bytes": (
            batch
            * len(PYANNOTE_LSTM_DIRECTIONS)
            * PYANNOTE_FRAMES
            * len(PYANNOTE_LSTM_GATE_ORDER)
            * PYANNOTE_LSTM_HIDDEN
            * bytes_per_float
        ),
        "hidden_and_cell_state_bytes_per_layer": (
            2
            * len(PYANNOTE_LSTM_DIRECTIONS)
            * batch
            * PYANNOTE_LSTM_HIDDEN
            * bytes_per_float
        ),
    }


def _write_or_verify_bytes(path: Path, expected: bytes, verify_only: bool) -> None:
    if verify_only:
        if not path.is_file():
            raise FileNotFoundError(f"Cannot verify missing artifact: {path}")
        if path.read_bytes() != expected:
            raise RuntimeError(f"{path.name} does not match its source ONNX graph.")
        return
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(expected)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_or_verify_direct_webgpu_vad(
    output_dir: Path,
    verify_only: bool,
    lstm_package: dict[str, Any],
) -> dict[str, Any]:
    batch = 8
    frontend_source = output_dir / "pyannote-segmentation-3.0-frontend-b8.onnx"
    tail_source = output_dir / "pyannote-segmentation-3.0-tail-b8.onnx"
    frontend_binary_name = "pyannote-segmentation-3.0-frontend-webgpu-f16.bin"
    tail_binary_name = "pyannote-segmentation-3.0-tail-webgpu-f16.bin"
    frontend_binary, frontend_metadata = build_frontend_webgpu_package(
        frontend_source,
        frontend_binary_name,
        np.dtype(np.float16),
    )
    tail_binary, tail_metadata = build_tail_webgpu_package(
        tail_source,
        tail_binary_name,
    )
    artifacts = (
        (output_dir / frontend_binary_name, frontend_binary),
        (
            output_dir / "pyannote-segmentation-3.0-frontend-webgpu-f16.json",
            (json.dumps(frontend_metadata, indent=2) + "\n").encode(),
        ),
        (output_dir / tail_binary_name, tail_binary),
        (
            output_dir / "pyannote-segmentation-3.0-tail-webgpu-f16.json",
            (json.dumps(tail_metadata, indent=2) + "\n").encode(),
        ),
    )
    for path, payload in artifacts:
        _write_or_verify_bytes(path, payload, verify_only)

    frontend_metadata_path = artifacts[1][0]
    tail_metadata_path = artifacts[3][0]
    recurrent_bytes = batch * PYANNOTE_FRAMES * PYANNOTE_LSTM_OUTPUT_FEATURES * 4
    input_affine_scratch_bytes = 4 * recurrent_bytes
    lstm_gpu_bytes = (
        int(lstm_package["weights"]["bytes"])
        + 2 * recurrent_bytes
        + input_affine_scratch_bytes
        + PYANNOTE_LSTM_LAYERS * 32
    )
    frontend_gpu_bytes = (
        int(
            frontend_metadata["memory"]["planned_webgpu"]["aliased_arena"][
                "minimum_resident_gpu_bytes"
            ]
        )
        + 384
    )
    explicit_gpu_bytes = (
        frontend_gpu_bytes
        + lstm_gpu_bytes
        + int(tail_metadata["memory"]["explicit_gpu_bytes"])
    )
    return {
        "batches": {
            str(batch): {
                "format": "senko-pyannote-direct-webgpu-f16-v1",
                "frontend_metadata": {
                    "file": frontend_metadata_path.name,
                    "bytes": frontend_metadata_path.stat().st_size,
                    "sha256": sha256(frontend_metadata_path),
                },
                "tail_metadata": {
                    "file": tail_metadata_path.name,
                    "bytes": tail_metadata_path.stat().st_size,
                    "sha256": sha256(tail_metadata_path),
                },
                "explicit_gpu_bytes": explicit_gpu_bytes,
            }
        }
    }


def export_pyannote_split(
    model: nn.Module,
    batches: tuple[int, ...],
    output_dir: Path,
    verify_only: bool,
) -> dict[str, Any]:
    # Keep an exact FP32 package as the diagnostic baseline, but select the
    # browser-validated FP16 package in the production manifest.
    write_or_verify_pyannote_lstm_package(
        model, output_dir, verify_only, "float32"
    )
    package = write_or_verify_pyannote_lstm_package(
        model, output_dir, verify_only, "float16"
    )
    components = (
        ("frontend", "pyannote_frontend", PyannoteFrontend(model)),
        ("tail", "pyannote_tail", PyannoteTail(model)),
    )
    component_manifests: dict[str, Any] = {}
    for component, family, module in components:
        variants: dict[str, Any] = {}
        for spec in specs_for(family, batches):
            path = output_dir / spec.file_name
            sample = make_input(spec)
            if verify_only:
                if not path.is_file():
                    raise FileNotFoundError(f"Cannot verify missing artifact: {path}")
                print(f"Verifying {path.name} ...", flush=True)
            else:
                print(f"Exporting {path.name} ...", flush=True)
                export_onnx(module, sample, spec, path)
            graph_info = inspect_onnx(path, spec)
            verification = verify_onnx(module, sample, spec, path)
            record = artifact_record(path, spec, graph_info, verification)
            variants[str(spec.batch)] = split_variant(record)
            print(
                f"  {record['bytes'] / (1024 * 1024):.2f} MiB, "
                f"max abs error {verification['max_abs_error']:.3g}",
                flush=True,
            )

            del sample
            gc.collect()
        component_manifests[component] = {
            "id": f"pyannote-segmentation-3.0-{component}",
            "input": {
                "name": specs_for(family, (1,))[0].input_name,
                "dtype": "float32",
                "shape": (
                    ["batch", 1, PYANNOTE_SAMPLES]
                    if component == "frontend"
                    else ["batch", PYANNOTE_FRAMES, PYANNOTE_LSTM_OUTPUT_FEATURES]
                ),
            },
            "output": {
                "name": specs_for(family, (1,))[0].output_name,
                "dtype": "float32",
                "shape": (
                    ["batch", PYANNOTE_FRAMES, PYANNOTE_FRONTEND_FEATURES]
                    if component == "frontend"
                    else ["batch", PYANNOTE_FRAMES, PYANNOTE_CLASSES]
                ),
            },
            "batches": variants,
        }

    direct_webgpu = (
        write_or_verify_direct_webgpu_vad(output_dir, verify_only, package)
        if 8 in batches
        else {"batches": {}}
    )
    return {
        "version": 1,
        "boundary_layout": "batch,frame,feature",
        "frontend": component_manifests["frontend"],
        "lstm": package,
        "tail": component_manifests["tail"],
        "direct_webgpu": direct_webgpu,
        "buffer_bytes_by_batch": {
            str(batch): pyannote_boundary_buffers(batch) for batch in batches
        },
    }


def export_family(
    family: str,
    batches: tuple[int, ...],
    output_dir: Path,
    verify_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    print(f"Loading {family} source model ...", flush=True)
    model = load_pyannote_raw_logits() if family == "pyannote" else load_campplus()
    reference = load_campplus_reference() if family == "campplus" else None
    records: list[dict[str, Any]] = []
    segmentation_split: dict[str, Any] | None = None
    try:
        for spec in specs_for(family, batches):
            path = output_dir / spec.file_name
            sample = make_input(spec)
            if verify_only:
                if not path.is_file():
                    raise FileNotFoundError(f"Cannot verify missing artifact: {path}")
                print(f"Verifying {path.name} ...", flush=True)
            else:
                print(f"Exporting {path.name} ...", flush=True)
                export_onnx(model, sample, spec, path)

            graph_info = inspect_onnx(path, spec)
            verification = verify_onnx(model, sample, spec, path)
            if reference is not None:
                verification["reference_campplus"] = compare_campplus_reference(
                    model, reference, sample
                )
            record = artifact_record(path, spec, graph_info, verification)
            records.append(record)
            print(
                f"  {record['bytes'] / (1024 * 1024):.2f} MiB, "
                f"max abs error {verification['max_abs_error']:.3g}",
                flush=True,
            )
            if family == "campplus":
                fp16_spec = specs_for("campplus_fp16", (spec.batch,))[0]
                fp16_path = output_dir / fp16_spec.file_name
                if verify_only:
                    if not fp16_path.is_file():
                        raise FileNotFoundError(
                            f"Cannot verify missing artifact: {fp16_path}"
                        )
                    print(f"Verifying {fp16_path.name} ...", flush=True)
                else:
                    print(f"Converting {fp16_path.name} ...", flush=True)
                    export_campplus_fp16(path, fp16_spec, fp16_path)

                fp16_graph_info = inspect_onnx(fp16_path, fp16_spec)
                fp16_verification = verify_campplus_fp16(
                    model,
                    sample,
                    fp16_spec,
                    fp16_path,
                    path,
                )
                fp16_record = artifact_record(
                    fp16_path,
                    fp16_spec,
                    fp16_graph_info,
                    fp16_verification,
                )
                fp16_record["conversion"]["source"] = {
                    "file": path.name,
                    "sha256": sha256(path),
                }
                records.append(fp16_record)
                print(
                    f"  {fp16_record['bytes'] / (1024 * 1024):.2f} MiB, "
                    f"max abs error {fp16_verification['max_abs_error']:.3g}, "
                    "minimum cosine similarity "
                    f"{fp16_verification['minimum_cosine_similarity']:.7f}",
                    flush=True,
                )
            del sample
            gc.collect()
        if family == "pyannote":
            segmentation_split = export_pyannote_split(
                model,
                batches,
                output_dir,
                verify_only,
            )
    finally:
        del model
        del reference
        gc.collect()
    return records, segmentation_split


def parse_batches(value: str, allowed: tuple[int, ...]) -> tuple[int, ...]:
    try:
        batches = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Batch sizes must be comma-separated integers.") from error
    invalid = sorted(set(batches).difference(allowed))
    if not batches or invalid:
        raise argparse.ArgumentTypeError(
            f"Allowed batch sizes are {','.join(map(str, allowed))}; got {value!r}."
        )
    return batches


def manifest(
    records: list[dict[str, Any]],
    segmentation_split: dict[str, Any] | None = None,
    campplus_direct: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sorted_records = sorted(
        records,
        key=lambda record: (
            0 if record["family"] == "pyannote" else 1,
            record["batch_size"],
        ),
    )

    def batch_map(family: str) -> dict[str, Any]:
        batches: dict[str, Any] = {}
        for record in sorted_records:
            if record["family"] != family:
                continue
            batch_record = {
                key: value
                for key, value in record.items()
                if key not in {"family", "batch_size", "input", "output"}
            }
            batch_record["input_shape"] = record["input"]["shape"]
            batch_record["output_shape"] = record["output"]["shape"]
            batches[str(record["batch_size"])] = batch_record
        return batches

    result = {
        "version": 1,
        "generated_by": {
            "seed": EXPORT_SEED,
            "opset": OPSET_VERSION,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "onnxconverter_common": onnxconverter_common.__version__,
        },
        "models": {
            "segmentation": {
                "id": "pyannote-segmentation-3.0-logits",
                "input": {
                    "name": "waveform",
                    "dtype": "float32",
                    "shape": ["batch", 1, 160_000],
                },
                "output": {
                    "name": "logits",
                    "dtype": "float32",
                    "shape": ["batch", 589, 7],
                },
                "batches": batch_map("pyannote"),
            },
            "campplus": {
                "id": "campplus-t150",
                "input": {
                    "name": "features",
                    "dtype": "float32",
                    "shape": ["batch", 150, 80],
                },
                "output": {
                    "name": "embeddings",
                    "dtype": "float32",
                    "shape": ["batch", 192],
                },
                "batches": batch_map("campplus"),
            },
        },
        "sources": {
            "pyannote_segmentation_3_0": {
                "weights": relative(PYANNOTE_WEIGHTS),
                "weights_sha256": sha256(PYANNOTE_WEIGHTS),
                "config": relative(PYANNOTE_CONFIG),
                "config_sha256": sha256(PYANNOTE_CONFIG),
                "output_semantics": (
                    "classifier logits; final LogSoftmax removed; trained Sinc "
                    "filters materialized as a static convolution"
                ),
            },
            "campplus": {
                "weights": relative(CAMPPLUS_WEIGHTS),
                "weights_sha256": sha256(CAMPPLUS_WEIGHTS),
                "implementation": relative(CAMPPLUS_SOURCE),
                "implementation_sha256": sha256(CAMPPLUS_SOURCE),
                "reference_implementation": relative(CAMPPLUS_REFERENCE_SOURCE),
                "reference_implementation_sha256": sha256(CAMPPLUS_REFERENCE_SOURCE),
                "output_semantics": (
                    "Core ML-compatible CAMPPlus with fixed T=150; checked for "
                    "cosine parity against reference CAMPPlus"
                ),
            },
        },
    }
    fp16_batches = batch_map("campplus_fp16")
    if fp16_batches:
        result["models"]["campplus"]["precision_variants"] = {
            "float16": {
                "internal_dtype": "float16",
                "input_dtype": "float32",
                "output_dtype": "float32",
                "verification_thresholds": {
                    "maximum_absolute_error": CAMPPLUS_FP16_MAX_ABS_ERROR,
                    "minimum_cosine_similarity": (
                        CAMPPLUS_FP16_MIN_COSINE_SIMILARITY
                    ),
                },
                "ort_web": {
                    "required_graph_optimization_level": "basic",
                    "reason": (
                        "ORT/JSEP extended or all graph optimization produces "
                        "unsupported FusedConv(float16) nodes"
                    ),
                },
                "batches": fp16_batches,
            }
        }
    if segmentation_split is not None:
        result["models"]["segmentation"]["split"] = segmentation_split
    if campplus_direct is not None:
        result["models"]["campplus"]["direct_webgpu"] = campplus_direct
    return result


def merge_records(output_dir: Path, new_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_path = output_dir / "manifest.json"
    old_records: list[dict[str, Any]] = []
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("version") == 1:
            for runtime_key, family in (
                ("segmentation", "pyannote"),
                ("campplus", "campplus"),
            ):
                model = existing.get("models", {}).get(runtime_key, {})
                for batch, batch_record in model.get("batches", {}).items():
                    old_records.append(
                        {
                            "family": family,
                            "batch_size": int(batch),
                            "input": {
                                "name": model["input"]["name"],
                                "shape": batch_record["input_shape"],
                            },
                            "output": {
                                "name": model["output"]["name"],
                                "shape": batch_record["output_shape"],
                            },
                            **{
                                key: value
                                for key, value in batch_record.items()
                                if key not in {"input_shape", "output_shape"}
                            },
                        }
                    )

            campplus = existing.get("models", {}).get("campplus", {})
            fp16_variant = (
                campplus.get("precision_variants", {}).get("float16", {})
            )
            for batch, batch_record in fp16_variant.get("batches", {}).items():
                old_records.append(
                    {
                        "family": "campplus_fp16",
                        "batch_size": int(batch),
                        "input": {
                            "name": campplus["input"]["name"],
                            "shape": batch_record["input_shape"],
                        },
                        "output": {
                            "name": campplus["output"]["name"],
                            "shape": batch_record["output_shape"],
                        },
                        **{
                            key: value
                            for key, value in batch_record.items()
                            if key not in {"input_shape", "output_shape"}
                        },
                    }
                )

    replacements = {(record["family"], record["batch_size"]) for record in new_records}
    retained = [
        record
        for record in old_records
        if (record.get("family"), record.get("batch_size")) not in replacements
        and (output_dir / record.get("file", "")).is_file()
    ]
    return retained + new_records


def load_existing_segmentation_split(output_dir: Path) -> dict[str, Any] | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    existing = json.loads(manifest_path.read_text())
    split = existing.get("models", {}).get("segmentation", {}).get("split")
    return split if isinstance(split, dict) and split.get("version") == 1 else None


def load_existing_campplus_direct(output_dir: Path) -> dict[str, Any] | None:
    """Preserve the separately generated direct-WebGPU CAM++ package."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    existing = json.loads(manifest_path.read_text())
    direct = existing.get("models", {}).get("campplus", {}).get("direct_webgpu")
    if not isinstance(direct, dict):
        return None
    metadata = output_dir / str(direct.get("metadata", {}).get("file", ""))
    weights = output_dir / str(direct.get("weights", {}).get("file", ""))
    if (
        direct.get("format") != "senko-campplus-direct-webgpu-f16-v1"
        or not metadata.is_file()
        or not weights.is_file()
    ):
        return None
    return direct


def merge_segmentation_split(
    output_dir: Path,
    new_split: dict[str, Any] | None,
) -> dict[str, Any] | None:
    old_split = load_existing_segmentation_split(output_dir)
    if new_split is None:
        return old_split
    if old_split is None:
        return new_split

    merged = json.loads(json.dumps(new_split))
    for component in ("frontend", "tail"):
        merged_batches = merged[component]["batches"]
        old_batches = old_split.get(component, {}).get("batches", {})
        for batch, variant in old_batches.items():
            if batch in merged_batches or not isinstance(variant, dict):
                continue
            artifact = output_dir / str(variant.get("file", ""))
            if artifact.is_file():
                merged_batches[batch] = variant
        merged[component]["batches"] = dict(
            sorted(merged_batches.items(), key=lambda item: int(item[0]))
        )

    old_buffers = old_split.get("buffer_bytes_by_batch", {})
    merged_buffers = merged["buffer_bytes_by_batch"]
    for batch in merged["frontend"]["batches"]:
        if batch not in merged_buffers and batch in old_buffers:
            merged_buffers[batch] = old_buffers[batch]
    merged["buffer_bytes_by_batch"] = dict(
        sorted(merged_buffers.items(), key=lambda item: int(item[0]))
    )
    merged_direct = merged.setdefault("direct_webgpu", {"batches": {}})["batches"]
    old_direct = old_split.get("direct_webgpu", {}).get("batches", {})
    for batch, variant in old_direct.items():
        if batch in merged_direct or not isinstance(variant, dict):
            continue
        frontend = output_dir / str(
            variant.get("frontend_metadata", {}).get("file", "")
        )
        tail = output_dir / str(variant.get("tail_metadata", {}).get("file", ""))
        if frontend.is_file() and tail.is_file():
            merged_direct[batch] = variant
    merged["direct_webgpu"]["batches"] = dict(
        sorted(merged_direct.items(), key=lambda item: int(item[0]))
    )
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "family",
        choices=("all", "pyannote", "campplus"),
        nargs="?",
        default="all",
        help="Model family to export (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory (default: {relative(DEFAULT_OUTPUT_DIR)}).",
    )
    parser.add_argument(
        "--pyannote-batches",
        type=lambda value: parse_batches(value, PYANNOTE_BATCHES),
        default=PYANNOTE_BATCHES,
        metavar="BATCHES",
        help="Comma-separated subset of 1,8,16,32.",
    )
    parser.add_argument(
        "--campplus-batches",
        type=lambda value: parse_batches(value, CAMPPLUS_BATCHES),
        default=CAMPPLUS_BATCHES,
        metavar="BATCHES",
        help="Comma-separated subset of 32,64,128.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-run graph and numerical checks without exporting.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(EXPORT_SEED)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
    check_pinned_sources()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    families = ("pyannote", "campplus") if args.family == "all" else (args.family,)
    records: list[dict[str, Any]] = []
    segmentation_split: dict[str, Any] | None = None
    for family in families:
        batches = (
            args.pyannote_batches if family == "pyannote" else args.campplus_batches
        )
        family_records, family_split = export_family(
            family,
            batches,
            output_dir,
            args.verify_only,
        )
        records.extend(family_records)
        if family_split is not None:
            segmentation_split = family_split

    if args.verify_only:
        print(f"Verified {len(records)} model artifact(s).")
        return 0

    all_records = merge_records(output_dir, records)
    segmentation_split = merge_segmentation_split(output_dir, segmentation_split)
    campplus_direct = load_existing_campplus_direct(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest(all_records, segmentation_split, campplus_direct), indent=2
        )
        + "\n"
    )
    print(f"Wrote {relative(manifest_path)} with {len(all_records)} artifact(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
