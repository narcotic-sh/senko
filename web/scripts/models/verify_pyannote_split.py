#!/usr/bin/env python3
"""Compose frontend ONNX -> explicit LSTM -> tail ONNX and prove parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

import export_models


def load_package(metadata_path: Path) -> tuple[dict[str, Any], bytes]:
    metadata = json.loads(metadata_path.read_text())
    contracts = {
        2: ("senko-persistent-lstm-f32-gc4h", "float32-le"),
        3: ("senko-persistent-lstm-f16-gc4h", "float16-le"),
    }
    contract = contracts.get(metadata.get("version"))
    if contract is None or metadata.get("format") != contract[0]:
        raise RuntimeError(f"Unsupported LSTM metadata version in {metadata_path}.")
    weights_record = metadata["weights"]
    weights_path = metadata_path.with_name(weights_record["file"])
    weights = weights_path.read_bytes()
    if len(weights) != weights_record["bytes"]:
        raise RuntimeError(f"Unexpected byte count for {weights_path}.")
    if hashlib.sha256(weights).hexdigest() != weights_record["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {weights_path}.")
    for layer in metadata.get("layers", []):
        for direction in layer.get("directions", []):
            for record in direction.get("tensors", {}).values():
                if record.get("dtype") != contract[1]:
                    raise RuntimeError(
                        f"Mixed tensor dtype in LSTM package {metadata_path}."
                    )
    return metadata, weights


def package_tensor(
    weights: bytes,
    record: dict[str, Any],
) -> torch.Tensor:
    offset = int(record["offset_bytes"])
    length = int(record["length_bytes"])
    shape = tuple(int(value) for value in record["shape"])
    packed_shape = tuple(int(value) for value in record["packed_shape"])
    if offset % 256 != 0:
        raise RuntimeError(f"Tensor offset {offset} is not GPU-buffer aligned.")
    dtype_name = record.get("dtype")
    if dtype_name == "float32-le":
        dtype = np.dtype("<f4")
    elif dtype_name == "float16-le":
        dtype = np.dtype("<f2")
    else:
        raise RuntimeError(f"Unsupported packaged tensor dtype {dtype_name!r}.")
    expected = int(np.prod(packed_shape)) * dtype.itemsize
    if length != expected:
        raise RuntimeError(
            f"Tensor length {length} does not match packed shape {packed_shape}."
        )
    # Copy gives PyTorch writable, naturally aligned host storage.
    array = np.frombuffer(
        weights,
        dtype=dtype,
        count=expected // dtype.itemsize,
        offset=offset,
    ).copy().reshape(packed_shape)
    if record.get("layout") == "gate-column4-hidden-input4":
        array = array.transpose(0, 2, 1, 3).reshape(shape)
    elif record.get("layout") != "row-major" or packed_shape != shape:
        raise RuntimeError(f"Unsupported packaged tensor layout {record.get('layout')!r}.")
    # This matches the WebGPU path: storage is optionally f16, while all dot
    # products, biases, cell state, and hidden state are evaluated as f32.
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def explicit_lstm(
    features: np.ndarray | torch.Tensor,
    metadata: dict[str, Any],
    weights: bytes,
) -> torch.Tensor:
    """Reference the persistent WGSL equations using the packaged tensors."""

    output = (
        torch.from_numpy(features.copy())
        if isinstance(features, np.ndarray)
        else features.detach().clone()
    ).to(dtype=torch.float32, device="cpu")
    expected_input = (
        output.shape[0],
        export_models.PYANNOTE_FRAMES,
        export_models.PYANNOTE_FRONTEND_FEATURES,
    )
    if tuple(output.shape) != expected_input:
        raise ValueError(f"Expected frontend features {expected_input}, got {output.shape}.")

    with torch.inference_mode():
        for layer in metadata["layers"]:
            direction_outputs: list[torch.Tensor] = []
            for direction in layer["directions"]:
                tensors = direction["tensors"]
                matrix = package_tensor(weights, tensors["matrix"])
                bias_ih = package_tensor(weights, tensors["bias_ih"])
                bias_hh = package_tensor(weights, tensors["bias_hh"])
                batch, frames, _ = output.shape
                hidden = torch.zeros(
                    batch,
                    export_models.PYANNOTE_LSTM_HIDDEN,
                    dtype=torch.float32,
                )
                cell = torch.zeros_like(hidden)
                sequence = (
                    range(frames)
                    if direction["direction"] == "forward"
                    else range(frames - 1, -1, -1)
                )
                direction_output = torch.empty(
                    batch,
                    frames,
                    export_models.PYANNOTE_LSTM_HIDDEN,
                    dtype=torch.float32,
                )
                for frame in sequence:
                    joined = torch.cat((output[:, frame, :], hidden), dim=-1)
                    affine = joined @ matrix.T
                    affine.add_(bias_ih).add_(bias_hh)
                    input_gate, forget_gate, cell_gate, output_gate = affine.chunk(4, dim=-1)
                    input_gate.sigmoid_()
                    forget_gate.sigmoid_()
                    cell_gate.tanh_()
                    output_gate.sigmoid_()
                    cell = forget_gate * cell + input_gate * cell_gate
                    hidden = output_gate * torch.tanh(cell)
                    direction_output[:, frame, :] = hidden
                direction_outputs.append(direction_output)
            output = torch.cat(direction_outputs, dim=-1)
    return output


def cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
    return ort.InferenceSession(
        path.as_posix(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def error_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = np.abs(expected - actual)
    return {
        "max_abs_error": float(difference.max(initial=0.0)),
        "mean_abs_error": float(difference.mean()),
    }


def run(output_dir: Path, batch: int, lstm_precision: str = "float32") -> dict[str, Any]:
    export_models.check_pinned_sources()
    model = export_models.load_pyannote_raw_logits()
    frontend = export_models.PyannoteFrontend(model)
    tail = export_models.PyannoteTail(model)
    sample = export_models.make_input(
        export_models.specs_for("pyannote", (batch,))[0]
    )

    frontend_path = output_dir / f"pyannote-segmentation-3.0-frontend-b{batch}.onnx"
    tail_path = output_dir / f"pyannote-segmentation-3.0-tail-b{batch}.onnx"
    full_path = output_dir / f"pyannote-segmentation-3.0-logits-b{batch}.onnx"
    metadata_path = output_dir / (
        export_models.PYANNOTE_LSTM_METADATA_FILE
        if lstm_precision == "float32"
        else export_models.PYANNOTE_LSTM_FP16_METADATA_FILE
    )
    metadata, weights = load_package(metadata_path)

    with torch.inference_mode():
        source_frontend = frontend(sample)
        source_recurrent, _ = model.lstm(source_frontend)  # type: ignore[attr-defined]
        source_tail = tail(source_recurrent)
        source_full = model(sample)

    frontend_ort = cpu_session(frontend_path).run(
        ["features"],
        {"waveform": sample.numpy()},
    )[0]
    recurrent_explicit = explicit_lstm(frontend_ort, metadata, weights)
    tail_ort = cpu_session(tail_path).run(
        ["logits"],
        {"recurrent": recurrent_explicit.numpy()},
    )[0]
    full_ort = cpu_session(full_path).run(
        ["logits"],
        {"waveform": sample.numpy()},
    )[0]

    source_frontend_np = source_frontend.numpy()
    source_recurrent_np = source_recurrent.numpy()
    source_full_np = source_full.numpy()
    metrics: dict[str, Any] = {
        "batch": batch,
        "lstm_precision": lstm_precision,
        "shapes": {
            "frontend": list(frontend_ort.shape),
            "recurrent": list(recurrent_explicit.shape),
            "logits": list(tail_ort.shape),
        },
        "frontend_onnx_vs_torch": error_metrics(source_frontend_np, frontend_ort),
        "explicit_lstm_vs_torch": error_metrics(
            source_recurrent_np,
            recurrent_explicit.numpy(),
        ),
        "tail_torch_composition_vs_full_torch": error_metrics(
            source_full_np,
            source_tail.numpy(),
        ),
        "split_composition_vs_full_torch": error_metrics(source_full_np, tail_ort),
        "split_composition_vs_full_onnx": error_metrics(full_ort, tail_ort),
        "argmax_agreement_vs_full_torch": float(
            np.mean(tail_ort.argmax(axis=-1) == source_full_np.argmax(axis=-1))
        ),
    }
    if metrics["frontend_onnx_vs_torch"]["max_abs_error"] > 5e-4:
        raise RuntimeError(f"Frontend parity failed: {metrics}")
    recurrent_tolerance = 5e-4 if lstm_precision == "float32" else 2e-3
    composition_tolerance = 1e-3 if lstm_precision == "float32" else 5e-3
    if metrics["explicit_lstm_vs_torch"]["max_abs_error"] > recurrent_tolerance:
        raise RuntimeError(f"Explicit LSTM parity failed: {metrics}")
    if metrics["split_composition_vs_full_torch"]["max_abs_error"] > composition_tolerance:
        raise RuntimeError(f"Split composition parity failed: {metrics}")
    if metrics["argmax_agreement_vs_full_torch"] < 0.9999:
        raise RuntimeError(f"Split argmax parity failed: {metrics}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=export_models.DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=export_models.PYANNOTE_BATCHES,
        default=1,
    )
    parser.add_argument(
        "--lstm-precision",
        choices=("float32", "float16"),
        default="float32",
    )
    args = parser.parse_args()
    torch.manual_seed(export_models.EXPORT_SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result = run(args.output_dir.resolve(), args.batch, args.lstm_precision)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
