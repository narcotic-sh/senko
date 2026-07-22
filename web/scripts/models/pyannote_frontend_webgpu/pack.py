#!/usr/bin/env python3
"""Pack the static pyannote segmentation frontend for direct WGSL execution.

The ONNX frontend is only fifteen nodes, but its first convolution produces a
large ``[B, 80, 15975]`` tensor.  The package describes a seven-dispatch
lowering which fuses that convolution with Abs and MaxPool.  Convolution
weights are retiled so a WGSL invocation accumulates four output channels with
one ``vec4<f32>`` load.  Every binary section is 256-byte aligned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


FORMAT_MAGIC = b"SNKVADF1"
FORMAT_VERSION = 1
HEADER_BYTES = 256
SECTION_ALIGNMENT = 256
CHANNEL_TILE = 4
SAMPLES = 160_000
FRAMES = 589
FEATURES = 60

EXPECTED_OPERATORS = Counter(
    {
        "InstanceNormalization": 4,
        "Conv": 3,
        "MaxPool": 3,
        "LeakyRelu": 3,
        "Abs": 1,
        "Transpose": 1,
    }
)


def align_up(value: int, alignment: int = SECTION_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _little_endian_bytes(array: np.ndarray) -> bytes:
    if array.dtype.kind not in "fiu":
        raise TypeError(f"unsupported packed dtype {array.dtype}")
    return np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<")).tobytes()


def pack_conv_ki_o4(weight: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Retile ONNX ``[O,I,K]`` weights as ``[K,I,ceil(O/4),O-lane]``."""

    value = np.asarray(weight)
    if value.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError(f"pyannote frontend weights must be float16/float32, got {value.dtype}")
    if value.ndim != 3:
        raise ValueError(f"expected Conv1d [O,I,K], got {value.shape}")
    output_channels, input_channels, kernel = map(int, value.shape)
    padded_output = align_up(output_channels, CHANNEL_TILE)
    padded = np.zeros((padded_output, input_channels, kernel), dtype=value.dtype)
    padded[:output_channels] = value
    packed = (
        padded.reshape(padded_output // CHANNEL_TILE, CHANNEL_TILE, input_channels, kernel)
        .transpose(3, 2, 0, 1)
        .copy()
    )
    return packed, list(packed.shape)


def unpack_conv_ki_o4(packed: np.ndarray, logical_shape: Iterable[int]) -> np.ndarray:
    shape = tuple(int(value) for value in logical_shape)
    if len(shape) != 3:
        raise ValueError(f"expected logical Conv1d shape, got {shape}")
    output_channels, input_channels, kernel = shape
    padded_output = align_up(output_channels, CHANNEL_TILE)
    restored = packed.transpose(2, 3, 1, 0).reshape(
        padded_output, input_channels, kernel
    )
    return restored[:output_channels].copy()


def pack_o4(
    values: np.ndarray,
    output_channels: int,
    storage_dtype: np.dtype[Any] = np.dtype(np.float32),
) -> np.ndarray:
    flattened = np.asarray(values, dtype=storage_dtype).reshape(-1)
    if flattened.size != output_channels:
        raise ValueError(
            f"expected {output_channels} bias values, got {flattened.size}"
        )
    padded = np.zeros(align_up(output_channels, CHANNEL_TILE), dtype=storage_dtype)
    padded[:output_channels] = flattened
    return padded.reshape(-1, CHANNEL_TILE)


def pack_instance_norm_affine(gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Pack dynamic InstanceNorm affine as interleaved ``[gamma4,beta4]``."""

    gamma_value = np.asarray(gamma, dtype=np.float32).reshape(-1)
    beta_value = np.asarray(beta, dtype=np.float32).reshape(-1)
    if gamma_value.size != beta_value.size or gamma_value.size == 0:
        raise ValueError("InstanceNorm gamma/beta lengths differ or are empty")
    padded_channels = align_up(gamma_value.size, CHANNEL_TILE)
    packed = np.zeros((padded_channels // CHANNEL_TILE, 2, CHANNEL_TILE), dtype=np.float32)
    packed[:, 0, :] = np.pad(
        gamma_value, (0, padded_channels - gamma_value.size)
    ).reshape(-1, CHANNEL_TILE)
    packed[:, 1, :] = np.pad(
        beta_value, (0, padded_channels - beta_value.size)
    ).reshape(-1, CHANNEL_TILE)
    return packed


@dataclass
class BinaryBuilder:
    payload: bytearray
    sections: list[dict[str, Any]]

    @classmethod
    def create(cls) -> "BinaryBuilder":
        return cls(bytearray(HEADER_BYTES), [])

    def add(
        self,
        *,
        section_id: str,
        kind: str,
        array: np.ndarray,
        logical_shape: Iterable[int],
        packed_shape: Iterable[int],
        layout: str,
        source_tensors: Iterable[str],
    ) -> str:
        offset = align_up(len(self.payload))
        self.payload.extend(b"\0" * (offset - len(self.payload)))
        raw = _little_endian_bytes(array)
        self.payload.extend(raw)
        self.sections.append(
            {
                "id": section_id,
                "kind": kind,
                "byte_offset": offset,
                "byte_length": len(raw),
                "element_count": int(array.size),
                "dtype": "float16" if array.dtype == np.float16 else "float32",
                "logical_shape": [int(value) for value in logical_shape],
                "packed_shape": [int(value) for value in packed_shape],
                "layout": layout,
                "source_tensors": list(source_tensors),
            }
        )
        return section_id

    def finish(self, *, source_sha256: str, batch: int) -> bytes:
        final_size = align_up(len(self.payload))
        self.payload.extend(b"\0" * (final_size - len(self.payload)))
        payload_hash = hashlib.sha256(memoryview(self.payload)[HEADER_BYTES:]).digest()
        struct.pack_into(
            "<8sIIIIQ",
            self.payload,
            0,
            FORMAT_MAGIC,
            FORMAT_VERSION,
            HEADER_BYTES,
            SECTION_ALIGNMENT,
            len(self.sections),
            len(self.payload),
        )
        self.payload[32:64] = bytes.fromhex(source_sha256)
        self.payload[64:96] = payload_hash
        struct.pack_into("<IIII", self.payload, 96, batch, SAMPLES, FRAMES, FEATURES)
        return bytes(self.payload)


def parse_header(data: bytes) -> dict[str, Any]:
    if len(data) < HEADER_BYTES:
        raise ValueError("truncated pyannote frontend package")
    magic, version, header_bytes, alignment, section_count, total_bytes = struct.unpack_from(
        "<8sIIIIQ", data, 0
    )
    batch, samples, frames, features = struct.unpack_from("<IIII", data, 96)
    return {
        "magic": magic.decode("ascii"),
        "format_version": version,
        "header_bytes": header_bytes,
        "section_alignment": alignment,
        "section_count": section_count,
        "total_bytes": total_bytes,
        "source_sha256": data[32:64].hex(),
        "payload_sha256": data[64:96].hex(),
        "batch": batch,
        "samples": samples,
        "frames": frames,
        "features": features,
    }


def _attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            value = helper.get_attribute_value(attribute)
            return list(value) if isinstance(value, tuple) else value
    return default


def _tensor_shapes(model: onnx.ModelProto) -> tuple[dict[str, list[int]], dict[str, int]]:
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    shapes: dict[str, list[int]] = {}
    types: dict[str, int] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tensor_type = value.type.tensor_type
        shape: list[int] = []
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value"):
                raise ValueError(f"dynamic dimension remains for {value.name}")
            shape.append(int(dimension.dim_value))
        shapes[value.name] = shape
        types[value.name] = tensor_type.elem_type
    return shapes, types


def _tensor_bytes(shape: Iterable[int], elem_type: int) -> int:
    widths = {TensorProto.FLOAT: 4}
    if elem_type not in widths:
        raise ValueError(f"unsupported activation type {elem_type}")
    return math.prod(shape) * widths[elem_type]


def _graph_liveness(
    graph: onnx.GraphProto,
    shapes: dict[str, list[int]],
    types: dict[str, int],
) -> dict[str, Any]:
    static_names = {item.name for item in graph.initializer}
    last_use: dict[str, int] = {}
    for index, node in enumerate(graph.node):
        for name in node.input:
            if name:
                last_use[name] = index
    graph_outputs = {value.name for value in graph.output}
    live = {
        value.name: _tensor_bytes(shapes[value.name], types[value.name])
        for value in graph.input
        if value.name not in static_names
    }
    peak = sum(live.values())
    peak_node = "graph-input"
    peak_live = dict(live)
    for index, node in enumerate(graph.node):
        for name in node.output:
            if name in shapes:
                live[name] = _tensor_bytes(shapes[name], types[name])
        current = sum(live.values())
        if current > peak:
            peak = current
            peak_node = node.name or f"{node.op_type}:{index}"
            peak_live = dict(live)
        for name in set(node.input):
            if name in live and last_use.get(name) == index and name not in graph_outputs:
                del live[name]
    return {
        "conservative_topological_peak_bytes": peak,
        "peak_after_node": peak_node,
        "live_tensors_at_peak": [
            {"name": name, "shape": shapes[name], "bytes": size}
            for name, size in sorted(
                peak_live.items(), key=lambda item: item[1], reverse=True
            )
        ],
        "note": "Treats each ONNX value as materialized and does not assume fusion or aliasing.",
    }


def _require_frontend_contract(
    model: onnx.ModelProto,
    shapes: dict[str, list[int]],
    types: dict[str, int],
) -> int:
    graph = model.graph
    operators = Counter(node.op_type for node in graph.node)
    if operators != EXPECTED_OPERATORS:
        raise ValueError(f"unexpected frontend operators: {dict(operators)}")
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("pyannote frontend requires one input and one output")
    input_name = graph.input[0].name
    output_name = graph.output[0].name
    input_shape = shapes[input_name]
    batch = input_shape[0]
    if input_shape != [batch, 1, SAMPLES]:
        raise ValueError(f"expected [B,1,{SAMPLES}] waveform, got {input_shape}")
    if shapes[output_name] != [batch, FRAMES, FEATURES]:
        raise ValueError(f"expected [B,{FRAMES},{FEATURES}] features")
    if types[input_name] != TensorProto.FLOAT or types[output_name] != TensorProto.FLOAT:
        raise ValueError("pyannote frontend boundaries must be float32")
    if batch not in (1, 8, 16, 32):
        raise ValueError(f"unsupported static frontend batch {batch}")

    expected_sequence = [
        "InstanceNormalization",
        "Conv",
        "Abs",
        "MaxPool",
        "InstanceNormalization",
        "LeakyRelu",
        "Conv",
        "MaxPool",
        "InstanceNormalization",
        "LeakyRelu",
        "Conv",
        "MaxPool",
        "InstanceNormalization",
        "LeakyRelu",
        "Transpose",
    ]
    if [node.op_type for node in graph.node] != expected_sequence:
        raise ValueError("frontend node order no longer matches the fused lowering")
    return batch


def _conv_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        "kernel_shape": [int(value) for value in _attribute(node, "kernel_shape")],
        "strides": [int(value) for value in _attribute(node, "strides", [1])],
        "dilations": [int(value) for value in _attribute(node, "dilations", [1])],
        "pads": [int(value) for value in _attribute(node, "pads", [0, 0])],
        "group": int(_attribute(node, "group", 1)),
    }


def _pool_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        "kernel_shape": [int(value) for value in _attribute(node, "kernel_shape")],
        "strides": [int(value) for value in _attribute(node, "strides")],
        "dilations": [int(value) for value in _attribute(node, "dilations", [1])],
        "pads": [int(value) for value in _attribute(node, "pads", [0, 0])],
        "ceil_mode": int(_attribute(node, "ceil_mode", 0)),
    }


def _activation_plan(
    batch: int,
    binary_bytes: int,
    intermediate_element_bytes: int = 4,
) -> dict[str, Any]:
    waveform = batch * SAMPLES * 4
    pool0 = batch * 80 * 5_325 * intermediate_element_bytes
    pool1 = batch * 60 * 1_773 * intermediate_element_bytes
    output = batch * FEATURES * FRAMES * 4
    slot_a = max(pool0, output)
    slot_b = max(waveform, pool1)
    stats = align_up(batch * 80 * 2 * 4)
    return {
        "unfused_first_conv_activation_bytes": batch * 80 * 15_975 * 4,
        "fused_first_pool_activation_bytes": pool0,
        "saved_first_stage_activation_bytes": batch * 80 * 15_975 * 4 - pool0,
        "logical_tensors": {
            "waveform": waveform,
            "pool0_bct": pool0,
            "pool1_bct": pool1,
            "features_btf": output,
        },
        "aliased_arena": {
            "slot_a_bytes": slot_a,
            "slot_a_lifetimes": ["pool0_bct", "features_btf"],
            "slot_b_bytes": slot_b,
            "slot_b_lifetimes": ["waveform", "pool1_bct"],
            "statistics_bytes": stats,
            "activation_arena_bytes": slot_a + slot_b + stats,
            "weight_buffer_bytes": binary_bytes,
            "minimum_resident_gpu_bytes": slot_a + slot_b + stats + binary_bytes,
        },
        "assumptions": [
            "Waveform and pool1 alias because conv0 finishes before conv1 writes pool1.",
            "Pool0 and final BTF output alias because conv1 consumes pool0 before conv2 writes output.",
            "InstanceNorm statistics are retained only as per-(batch,channel) scale/shift pairs.",
            "Waveform/final boundaries and all reductions remain float32.",
            f"Pooled intermediates use {intermediate_element_bytes * 8}-bit storage.",
        ],
    }


def build_package(
    source: Path,
    binary_name: str,
    storage_dtype: np.dtype[Any] = np.dtype(np.float32),
) -> tuple[bytes, dict[str, Any]]:
    storage_dtype = np.dtype(storage_dtype)
    if storage_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError("frontend convolution storage must be float16 or float32")
    source_bytes = source.read_bytes()
    source_hash = sha256_bytes(source_bytes)
    model = onnx.load_model_from_string(source_bytes)
    onnx.checker.check_model(model, full_check=True)
    shapes, types = _tensor_shapes(model)
    batch = _require_frontend_contract(model, shapes, types)
    graph = model.graph
    static = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    builder = BinaryBuilder.create()
    convolutions: list[dict[str, Any]] = []
    normalizations: list[dict[str, Any]] = []
    pools: list[dict[str, Any]] = []

    for index, node in enumerate(graph.node):
        if node.op_type == "Conv":
            if len(node.input) < 2 or node.input[1] not in static:
                raise ValueError(f"dynamic convolution weight in {node.name}")
            source_weight = np.asarray(static[node.input[1]])
            if source_weight.dtype != np.float32:
                raise ValueError(f"non-float32 source convolution in {node.name}")
            weight = source_weight.astype(storage_dtype)
            packed_weight, packed_shape = pack_conv_ki_o4(weight)
            weight_id = builder.add(
                section_id=f"conv:{len(convolutions)}:weight",
                kind="conv_weight",
                array=packed_weight,
                logical_shape=weight.shape,
                packed_shape=packed_shape,
                layout="K_I_O4_O",
                source_tensors=[node.input[1]],
            )
            output_channels = int(weight.shape[0])
            bias_sources: list[str] = []
            if len(node.input) >= 3 and node.input[2]:
                bias = np.asarray(static[node.input[2]], dtype=storage_dtype)
                bias_sources.append(node.input[2])
            else:
                bias = np.zeros(output_channels, dtype=storage_dtype)
            packed_bias = pack_o4(bias, output_channels, storage_dtype)
            bias_id = builder.add(
                section_id=f"conv:{len(convolutions)}:bias",
                kind="conv_bias",
                array=packed_bias,
                logical_shape=[output_channels],
                packed_shape=packed_bias.shape,
                layout="O4",
                source_tensors=bias_sources,
            )
            attributes = _conv_attributes(node)
            if attributes["group"] != 1 or attributes["dilations"] != [1] or attributes["pads"] != [0, 0]:
                raise ValueError(f"unsupported convolution attributes in {node.name}")
            macs = (
                math.prod(shapes[node.output[0]])
                * int(weight.shape[1])
                * int(weight.shape[2])
            )
            convolutions.append(
                {
                    "graph_index": index,
                    "node": node.name,
                    "input_shape": shapes[node.input[0]],
                    "output_shape": shapes[node.output[0]],
                    "weight_shape": list(weight.shape),
                    "attributes": attributes,
                    "macs": macs,
                    "packed": {"weight": weight_id, "bias": bias_id},
                }
            )
        elif node.op_type == "InstanceNormalization":
            if len(node.input) != 3 or any(name not in static for name in node.input[1:]):
                raise ValueError(f"dynamic InstanceNorm affine in {node.name}")
            gamma = np.asarray(static[node.input[1]], dtype=np.float32)
            beta = np.asarray(static[node.input[2]], dtype=np.float32)
            packed = pack_instance_norm_affine(gamma, beta)
            section = builder.add(
                section_id=f"instance_norm:{len(normalizations)}:affine",
                kind="instance_norm_affine",
                array=packed,
                logical_shape=[int(gamma.size), 2],
                packed_shape=packed.shape,
                layout="C4_GAMMA_BETA",
                source_tensors=node.input[1:],
            )
            normalizations.append(
                {
                    "graph_index": index,
                    "node": node.name,
                    "input_shape": shapes[node.input[0]],
                    "epsilon": float(_attribute(node, "epsilon", 1e-5)),
                    "affine": section,
                }
            )
        elif node.op_type == "MaxPool":
            attributes = _pool_attributes(node)
            expected = {
                "kernel_shape": [3],
                "strides": [3],
                "dilations": [1],
                "pads": [0, 0],
                "ceil_mode": 0,
            }
            if attributes != expected:
                raise ValueError(f"unsupported pool attributes in {node.name}: {attributes}")
            pools.append(
                {
                    "graph_index": index,
                    "node": node.name,
                    "input_shape": shapes[node.input[0]],
                    "output_shape": shapes[node.output[0]],
                    "attributes": attributes,
                }
            )

    if len(convolutions) != 3 or len(normalizations) != 4 or len(pools) != 3:
        raise ValueError("unexpected lowered frontend inventory")
    if [item["weight_shape"] for item in convolutions] != [
        [80, 1, 251],
        [60, 80, 5],
        [60, 60, 5],
    ]:
        raise ValueError("pyannote convolution shapes changed")
    if [item["attributes"]["strides"] for item in convolutions] != [[10], [1], [1]]:
        raise ValueError("pyannote convolution strides changed")

    packed_sources = {
        source_name
        for section in builder.sections
        for source_name in section["source_tensors"]
    }
    initializer_names = {item.name for item in graph.initializer}
    if packed_sources != initializer_names:
        raise ValueError(
            "initializer coverage mismatch; "
            f"missing={sorted(initializer_names - packed_sources)}, "
            f"unexpected={sorted(packed_sources - initializer_names)}"
        )

    binary = builder.finish(source_sha256=source_hash, batch=batch)
    header = parse_header(binary)
    per_item_macs = sum(item["macs"] for item in convolutions) // batch
    metadata: dict[str, Any] = {
        "schema": "senko.pyannote-frontend.webgpu-pack",
        "format_version": FORMAT_VERSION,
        "source": {
            "file": source.name,
            "byte_length": len(source_bytes),
            "sha256": source_hash,
            "opset": [
                {"domain": item.domain, "version": int(item.version)}
                for item in model.opset_import
            ],
        },
        "binary": {
            "file": binary_name,
            "byte_length": len(binary),
            "sha256": sha256_bytes(binary),
            "payload_sha256": header["payload_sha256"],
            "header_bytes": HEADER_BYTES,
            "section_alignment": SECTION_ALIGNMENT,
            "section_count": len(builder.sections),
            "endianness": "little",
        },
        "contract": {
            "input": {"name": graph.input[0].name, "shape": shapes[graph.input[0].name], "dtype": "float32", "layout": "BCT"},
            "output": {"name": graph.output[0].name, "shape": shapes[graph.output[0].name], "dtype": "float32", "layout": "BTF"},
            "boundary_dtype": "float32",
            "intermediate_dtype": (
                "float16" if storage_dtype == np.float16 else "float32"
            ),
            "reduction_dtype": "float32",
            "weight_dtype": "float16" if storage_dtype == np.float16 else "float32",
            "channel_tile": CHANNEL_TILE,
        },
        "inventory": {
            "onnx_nodes": len(graph.node),
            "onnx_operators": dict(sorted(EXPECTED_OPERATORS.items())),
            "source_initializers": len(graph.initializer),
            "source_initializer_bytes": sum(len(item.raw_data) for item in graph.initializer),
            "lowered_convolutions": len(convolutions),
            "lowered_instance_normalizations": len(normalizations),
            "fused_max_pools": len(pools),
        },
        "compute": {
            "macs_per_batch_item": per_item_macs,
            "macs_full_batch": sum(item["macs"] for item in convolutions),
            "convolutions": convolutions,
            "pool_comparisons_full_batch": sum(
                math.prod(item["output_shape"]) * 2 for item in pools
            ),
        },
        "memory": {
            "onnx_graph": _graph_liveness(graph, shapes, types),
            "planned_webgpu": _activation_plan(
                batch, len(binary), storage_dtype.itemsize
            ),
        },
        "sections": builder.sections,
        "fused_program": {
            "dispatches": 7,
            "steps": [
                "waveform_instance_norm_statistics",
                "sinc_conv_abs_maxpool",
                "pool0_instance_norm_statistics",
                "normalized_leaky_conv1_maxpool",
                "pool1_instance_norm_statistics",
                "normalized_leaky_conv2_maxpool_btf",
                "features_instance_norm_leaky_in_place",
            ],
            "layouts": {"intermediate": "BCT", "output": "BTF", "conv_weight": "K_I_O4_O"},
            "sinc_workgroup": {
                "size": 64,
                "pooled_frames_per_workgroup": 64,
                "waveform_tile_samples": 2_161,
                "weight_vectors_per_output_group": 251,
                "workgroup_storage_bytes": 12_660,
                "note": "One workgroup handles 64 pooled frames for four filters; waveform and vec4 weights are staged once in workgroup memory.",
            },
        },
    }
    return binary, metadata


def write_package(
    source: Path,
    output_dir: Path,
    storage_dtype: np.dtype[Any] = np.dtype(np.float32),
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_dtype = np.dtype(storage_dtype)
    precision = "f16" if storage_dtype == np.float16 else "f32"
    binary_name = f"pyannote-segmentation-3.0-frontend-webgpu-{precision}.bin"
    metadata_name = f"pyannote-segmentation-3.0-frontend-webgpu-{precision}.json"
    binary, metadata = build_package(source, binary_name, storage_dtype)
    binary_path = output_dir / binary_name
    metadata_path = output_dir / metadata_name
    binary_path.write_bytes(binary)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return binary_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Static pyannote frontend ONNX")
    parser.add_argument("output_dir", type=Path, help="Output artifact directory")
    parser.add_argument(
        "--storage-precision",
        choices=("f16", "f32"),
        default="f32",
        help="Convolution weight and pooled-intermediate storage precision",
    )
    args = parser.parse_args()
    storage_dtype = np.float16 if args.storage_precision == "f16" else np.float32
    binary_path, metadata_path = write_package(
        args.source, args.output_dir, storage_dtype
    )
    print(binary_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
