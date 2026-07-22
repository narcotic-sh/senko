#!/usr/bin/env python3
"""Pack the static CAM++ ONNX graph for direct WGSL consumption.

The package deliberately contains no ONNX protobuf. Convolution weights are
retiled into 4x4 input/output-channel blocks, inference BatchNorm parameters
are reduced to affine scale/shift pairs, and all sections start at WebGPU's
portable 256-byte storage-buffer offset alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


FORMAT_MAGIC = b"SNKCAMW1"
FORMAT_VERSION = 1
HEADER_BYTES = 256
SECTION_ALIGNMENT = 256
CHANNEL_TILE = 4
SUPPORTED_OPERATORS = {
    "Add",
    "BatchNormalization",
    "Cast",
    "Concat",
    "Constant",
    "Conv",
    "Mul",
    "ReduceMean",
    "Relu",
    "Reshape",
    "Sigmoid",
    "Sqrt",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
}


def align_up(value: int, alignment: int = SECTION_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _little_endian_bytes(array: np.ndarray) -> bytes:
    dtype = array.dtype
    if dtype.kind not in "fiu":
        raise TypeError(f"unsupported packed dtype {dtype}")
    little = dtype.newbyteorder("<")
    return np.ascontiguousarray(array, dtype=little).tobytes(order="C")


def pack_conv_oihw4(weight: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Retile ONNX [O,I,spatial...] weights for vec4 WGSL loads.

    The physical order is [K, ceil(O/4), ceil(I/4), I-lane, O-lane]. A shader
    loads four consecutive output weights for one input lane as a vec4 and
    accumulates four output channels together.
    """

    if weight.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError(f"CAM++ convolution weights must be floating point, got {weight.dtype}")
    if weight.ndim not in (3, 4):
        raise ValueError(f"expected Conv1d/Conv2d OIHW tensor, got {weight.shape}")
    out_channels, in_channels = map(int, weight.shape[:2])
    kernel_elements = math.prod(weight.shape[2:])
    padded_out = align_up(out_channels, CHANNEL_TILE)
    padded_in = align_up(in_channels, CHANNEL_TILE)
    padded = np.zeros((padded_out, padded_in, kernel_elements), dtype=weight.dtype)
    padded[:out_channels, :in_channels] = weight.reshape(
        out_channels, in_channels, kernel_elements
    )
    packed = (
        padded.reshape(
            padded_out // CHANNEL_TILE,
            CHANNEL_TILE,
            padded_in // CHANNEL_TILE,
            CHANNEL_TILE,
            kernel_elements,
        )
        .transpose(4, 0, 2, 3, 1)
        .copy()
    )
    return packed, list(packed.shape)


def unpack_conv_oihw4(packed: np.ndarray, logical_shape: Iterable[int]) -> np.ndarray:
    """Test/reference inverse for :func:`pack_conv_oihw4`."""

    shape = tuple(int(value) for value in logical_shape)
    out_channels, in_channels = shape[:2]
    kernel_elements = math.prod(shape[2:])
    padded_out = align_up(out_channels, CHANNEL_TILE)
    padded_in = align_up(in_channels, CHANNEL_TILE)
    restored = packed.transpose(1, 4, 2, 3, 0).reshape(
        padded_out, padded_in, kernel_elements
    )
    return restored[:out_channels, :in_channels].reshape(shape).copy()


def compile_batch_norm_affine(
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    epsilon: float,
    storage_dtype: np.dtype[Any] = np.dtype(np.float32),
) -> tuple[np.ndarray, int]:
    """Compile inference BatchNorm to interleaved [scale4, shift4] groups.

    FP32 is intentional even though activations are FP16. Re-quantizing the
    combined affine to FP16 compounds through 56 normalizations; FP32 adds
    only about 128 KiB and is measurably closer to the source FP16 graph.
    """

    arrays = [np.asarray(value, dtype=np.float32).reshape(-1) for value in (gamma, beta, mean, variance)]
    channels = arrays[0].size
    if any(value.size != channels for value in arrays):
        raise ValueError("BatchNorm parameter lengths differ")
    scale = arrays[0] / np.sqrt(arrays[3] + np.float32(epsilon))
    shift = arrays[1] - arrays[2] * scale
    padded_channels = align_up(channels, CHANNEL_TILE)
    if np.dtype(storage_dtype) not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError("BatchNorm affine storage must be float16 or float32")
    groups = np.zeros(
        (padded_channels // CHANNEL_TILE, 2, CHANNEL_TILE), dtype=storage_dtype
    )
    groups[:, 0, :] = np.pad(scale, (0, padded_channels - channels)).reshape(-1, CHANNEL_TILE)
    groups[:, 1, :] = np.pad(shift, (0, padded_channels - channels)).reshape(-1, CHANNEL_TILE)
    return groups, channels


def apply_compiled_batch_norm(values: np.ndarray, packed: np.ndarray, channels: int) -> np.ndarray:
    """Reference evaluator used by tests; channels occupy the last axis."""

    scale = packed[:, 0, :].reshape(-1)[:channels].astype(np.float32)
    shift = packed[:, 1, :].reshape(-1)[:channels].astype(np.float32)
    return np.asarray(values, dtype=np.float32) * scale + shift


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
        if offset > len(self.payload):
            self.payload.extend(b"\0" * (offset - len(self.payload)))
        raw = _little_endian_bytes(array)
        self.payload.extend(raw)
        record = {
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
        self.sections.append(record)
        return section_id

    def finish(
        self,
        *,
        source_sha256: str,
        batch: int,
        frames: int,
        feature_dim: int,
        embedding_dim: int,
    ) -> bytes:
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
        struct.pack_into(
            "<IIII",
            self.payload,
            96,
            batch,
            frames,
            feature_dim,
            embedding_dim,
        )
        return bytes(self.payload)


def parse_header(data: bytes) -> dict[str, Any]:
    if len(data) < HEADER_BYTES:
        raise ValueError("truncated CAM++ package")
    magic, version, header_bytes, alignment, section_count, total_bytes = struct.unpack_from(
        "<8sIIIIQ", data, 0
    )
    batch, frames, feature_dim, embedding_dim = struct.unpack_from("<IIII", data, 96)
    return {
        "magic": magic.decode("ascii"),
        "format_version": version,
        "header_bytes": header_bytes,
        "section_alignment": alignment,
        "section_count": section_count,
        "total_bytes": total_bytes,
        "source_sha256": data[32:64].hex(),
        "payload_sha256": data[64:96].hex(),
        "source_batch": batch,
        "frames": frames,
        "feature_dim": feature_dim,
        "embedding_dim": embedding_dim,
    }


def _attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            value = helper.get_attribute_value(attribute)
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, tuple):
                return list(value)
            return value
    return default


def _static_tensors(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    result = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        tensor = next((attr.t for attr in node.attribute if attr.name == "value"), None)
        if tensor is None:
            raise ValueError(f"unsupported Constant encoding in {node.name}")
        result[node.output[0]] = numpy_helper.to_array(tensor)
    return result


def _tensor_shapes(model: onnx.ModelProto) -> tuple[dict[str, list[int]], dict[str, int]]:
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    shapes: dict[str, list[int]] = {}
    types: dict[str, int] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tensor_type = value.type.tensor_type
        shape: list[int] = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                raise ValueError(f"dynamic dimension remains for {value.name}")
            shape.append(int(dim.dim_value))
        shapes[value.name] = shape
        types[value.name] = tensor_type.elem_type
    return shapes, types


def _tensor_bytes(shape: Iterable[int], elem_type: int) -> int:
    widths = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
    }
    if elem_type not in widths:
        raise ValueError(f"unknown byte width for ONNX type {elem_type}")
    return math.prod(shape) * widths[elem_type]


def _conv_attributes(node: onnx.NodeProto, weight_shape: Iterable[int]) -> dict[str, Any]:
    shape = list(weight_shape)
    rank = len(shape) - 2
    kernel = shape[2:]
    return {
        "kernel_shape": [int(v) for v in _attribute(node, "kernel_shape", kernel)],
        "strides": [int(v) for v in _attribute(node, "strides", [1] * rank)],
        "dilations": [int(v) for v in _attribute(node, "dilations", [1] * rank)],
        "pads": [int(v) for v in _attribute(node, "pads", [0] * (2 * rank))],
        "group": int(_attribute(node, "group", 1)),
    }


def _classify_conv(name: str) -> str:
    if name.startswith("/head/"):
        return "fcm"
    if "/block" in name:
        return "cam_dense"
    if "/transit" in name:
        return "transit"
    if name == "/xvector/tdnn/linear/Conv":
        return "tdnn"
    if name == "/xvector/dense/linear/Conv":
        return "embedding"
    return "other"


def _graph_liveness(
    graph: onnx.GraphProto,
    shapes: dict[str, list[int]],
    types: dict[str, int],
    static_names: set[str],
) -> dict[str, Any]:
    last_use: dict[str, int] = {}
    for index, node in enumerate(graph.node):
        for name in node.input:
            if name:
                last_use[name] = index
    graph_outputs = {value.name for value in graph.output}
    live: dict[str, int] = {}
    for value in graph.input:
        if value.name not in static_names:
            live[value.name] = _tensor_bytes(shapes[value.name], types[value.name])
    peak = sum(live.values())
    peak_node = "graph-input"
    peak_live = dict(live)
    for index, node in enumerate(graph.node):
        if node.op_type != "Constant":
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
        for name in node.output:
            if name in live and name not in last_use and name not in graph_outputs:
                del live[name]
    largest = sorted(peak_live.items(), key=lambda item: item[1], reverse=True)[:8]
    return {
        "conservative_topological_peak_bytes": peak,
        "peak_after_node": peak_node,
        "largest_live_tensors_at_peak": [
            {"name": name, "bytes": size, "shape": shapes.get(name, [])}
            for name, size in largest
        ],
        "note": "Counts every ONNX value as materialized; view/in-place reuse can lower this number.",
    }


def _node_by_name(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    result = {node.name: node for node in graph.node if node.name}
    if len(result) != sum(bool(node.name) for node in graph.node):
        raise ValueError("duplicate node name")
    return result


def _fused_program(
    graph: onnx.GraphProto,
    shapes: dict[str, list[int]],
    conv_records: dict[str, dict[str, Any]],
    bn_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    nodes = _node_by_name(graph)
    dense_pattern = re.compile(r"^/xvector/block([123])/tdnnd(\d+)/linear1/Conv$")
    blocks: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    for name in conv_records:
        match = dense_pattern.match(name)
        if not match:
            continue
        block_index, layer_index = map(int, match.groups())
        prefix = f"/xvector/block{block_index}/tdnnd{layer_index}"
        bn_name = f"{prefix}/nonlinear1/batchnorm/BatchNormalization"
        local_name = f"{prefix}/cam_layer/linear_local/Conv"
        attention1_name = f"{prefix}/cam_layer/linear1/Conv"
        attention2_name = f"{prefix}/cam_layer/linear2/Conv"
        required = [name, local_name, attention1_name, attention2_name]
        if bn_name not in bn_records or any(item not in conv_records for item in required):
            raise ValueError(f"incomplete CAM dense layer {prefix}")
        bottleneck = conv_records[name]
        local = conv_records[local_name]
        input_shape = shapes[nodes[bn_name].input[0]]
        output_shape = shapes[nodes[local_name].output[0]]
        blocks[block_index].append(
            {
                "kind": "cam_dense_layer",
                "id": f"block{block_index}.layer{layer_index}",
                "layer": layer_index,
                "input_channels": input_shape[1],
                "append_channel": input_shape[1],
                "bottleneck_channels": bottleneck["weight_shape"][0],
                "output_channels": output_shape[1],
                "frames": output_shape[-1],
                "preactivation_affine": bn_records[bn_name]["affine"],
                "bottleneck": bottleneck["packed"],
                "local": local["packed"],
                "local_dilation": local["attributes"]["dilations"][0],
                "attention_reduce": "2 * mean(time)",
                "attention1": conv_records[attention1_name]["packed"],
                "attention2": conv_records[attention2_name]["packed"],
                "runtime_fusion": [
                    "bn_relu_pointwise_relu_and_mean",
                    "attention_mlp_local_conv_gate_append",
                ],
            }
        )
    expected_layers = {1: 12, 2: 24, 3: 16}
    for block_index, expected in expected_layers.items():
        blocks[block_index].sort(key=lambda item: item["layer"])
        if len(blocks[block_index]) != expected:
            raise ValueError(
                f"block {block_index}: expected {expected} dense layers, got {len(blocks[block_index])}"
            )

    transits: list[dict[str, Any]] = []
    for index in (1, 2, 3):
        bn_name = f"/xvector/transit{index}/nonlinear/batchnorm/BatchNormalization"
        conv_name = f"/xvector/transit{index}/linear/Conv"
        transits.append(
            {
                "kind": "transit",
                "id": f"transit{index}",
                "preactivation_affine": bn_records[bn_name]["affine"],
                "pointwise": conv_records[conv_name]["packed"],
                "runtime_fusion": "bn_relu_pointwise",
                "epilogue": "relu" if index == 3 else "identity",
            }
        )

    head_convs = [
        record for name, record in conv_records.items() if name.startswith("/head/")
    ]
    head_convs.sort(key=lambda record: record["graph_index"])
    return {
        "layout": {
            "fcm_activations": "BCFT (time-minor); final [B,32,10,150] is a zero-copy [B,320,150] view",
            "tdnn_activations": "BCT (time-minor within each channel)",
            "conv_weights": "K_O4_I4_I_O",
            "batch_norm_affine": "C4_SCALE_SHIFT",
        },
        "head": {
            "kind": "fcm_residual_frontend",
            "convolutions": [record["packed"] for record in head_convs],
            "dispatches_full_batch": 10,
            "fusion": "Fold shortcut convolution/add/ReLU into each residual conv2 epilogue.",
        },
        "tdnn": {
            "kind": "tdnn",
            "convolution": conv_records["/xvector/tdnn/linear/Conv"]["packed"],
            "epilogue": "relu",
        },
        "blocks": [
            {"kind": "cam_dense_block", "id": f"block{index}", "layers": blocks[index]}
            for index in (1, 2, 3)
        ],
        "transits": transits,
        "final": {
            "kind": "stats_pool_dense",
            "dense": conv_records["/xvector/dense/linear/Conv"]["packed"],
            "variance": "biased",
            "epsilon": 1e-5,
            "output_affine": bn_records[
                "/xvector/dense/nonlinear/batchnorm/BatchNormalization"
            ]["affine"],
            "epilogue": "batchnorm_affine_false_relu_f32_store",
            "runtime_fusion": "one persistent workgroup per batch item",
        },
        "dispatch_model": {
            "recommended_full_batch": 119,
            "breakdown": {
                "fcm": 10,
                "tdnn": 1,
                "cam_dense_layers": 104,
                "transits": 3,
                "stats_dense_output": 1,
            },
            "experimental_persistent_dense": 67,
            "persistent_note": (
                "A one-workgroup-per-(batch,layer) kernel can combine each dense layer into one "
                "dispatch, but must be benchmarked against the two-dispatch design for occupancy."
            ),
        },
    }


def _activation_plan(
    graph: onnx.GraphProto,
    shapes: dict[str, list[int]],
    binary_bytes: int,
    source_batch: int,
    storage_element_bytes: int = 2,
) -> dict[str, Any]:
    nodes = _node_by_name(graph)

    def output_bytes(name: str) -> int:
        output = nodes[name].output[0]
        return math.prod(shapes[output]) * storage_element_bytes

    def input_bytes(name: str) -> int:
        value = nodes[name].input[0]
        return math.prod(shapes[value]) * storage_element_bytes

    residuals = [
        "/head/layer1/layer1.0",
        "/head/layer1/layer1.1",
        "/head/layer2/layer2.0",
        "/head/layer2/layer2.1",
    ]
    head_candidates: list[dict[str, Any]] = []
    first = "/head/conv1/Conv"
    head_candidates.append(
        {
            "site": first,
            "bytes": math.prod(shapes[nodes[first].input[0]]) * storage_element_bytes
            + output_bytes(first),
        }
    )
    for prefix in residuals:
        conv1 = f"{prefix}/conv1/Conv"
        conv2 = f"{prefix}/conv2/Conv"
        head_candidates.append(
            {
                "site": prefix,
                "bytes": input_bytes(conv1) + output_bytes(conv1) + output_bytes(conv2),
            }
        )
    head_peak_record = max(head_candidates, key=lambda item: item["bytes"])
    head_peak = int(head_peak_record["bytes"])

    dense_candidates: list[dict[str, Any]] = []
    for block_index in (1, 2, 3):
        concats = [
            node
            for node in graph.node
            if node.name.startswith(f"/xvector/block{block_index}/Concat")
        ]
        final_concat = concats[-1]
        slab = math.prod(shapes[final_concat.output[0]]) * storage_element_bytes
        scratch_name = f"/xvector/block{block_index}/tdnnd1/linear1/Conv"
        scratch = output_bytes(scratch_name)
        dense_candidates.append(
            {"site": f"block{block_index}-append", "bytes": slab + scratch}
        )
        transit_name = f"/xvector/transit{block_index}/linear/Conv"
        dense_candidates.append(
            {
                "site": f"block{block_index}-transit",
                "bytes": slab + output_bytes(transit_name),
            }
        )
    dense_peak_record = max(dense_candidates, key=lambda item: item["bytes"])
    dense_peak = int(dense_peak_record["bytes"])
    full_peak = max(head_peak, dense_peak)
    tdnn_output = output_bytes("/xvector/tdnn/linear/Conv")
    microbatch: list[dict[str, int]] = []
    for batch in (4, 8, 16, source_batch):
        if batch > source_batch or source_batch % batch:
            continue
        if batch == source_batch:
            arena = full_peak
        else:
            arena = max(dense_peak, math.ceil(head_peak * batch / source_batch) + tdnn_output)
        microbatch.append(
            {
                "frontend_microbatch": batch,
                "activation_arena_bytes": int(arena),
                "minimum_resident_gpu_bytes": int(arena + binary_bytes),
                "frontend_tdnn_dispatches": 11 * (source_batch // batch),
            }
        )
    return {
        "recommended": {
            "frontend_microbatch": source_batch,
            "activation_arena_bytes": full_peak,
            "weight_buffer_bytes": binary_bytes,
            "minimum_resident_gpu_bytes": full_peak + binary_bytes,
            "peak_site": head_peak_record["site"] if head_peak >= dense_peak else dense_peak_record["site"],
        },
        "full_batch_head_peak_bytes": head_peak,
        "dense_backbone_peak_bytes": dense_peak,
        "frontend_microbatch_tradeoffs": microbatch,
        "assumptions": [
            f"{'FP16' if storage_element_bytes == 2 else 'FP32'} activations/convolution weights, FP32 BatchNorm affine, and FP32 final embeddings.",
            "Residual shortcut convolution/add/ReLU is fused into conv2 and has no output buffer.",
            "Dense blocks append into one maximum-channel slab; Concat is a metadata update.",
            "One reusable 128-channel bottleneck scratch tensor is retained.",
            "The activation arena is lifetime-aliased and includes input/output staging.",
        ],
    }


def build_package(
    source: Path,
    binary_name: str,
    internal_dtype: str = "float16",
) -> tuple[bytes, dict[str, Any]]:
    if internal_dtype not in ("float16", "float32"):
        raise ValueError("internal_dtype must be float16 or float32")
    storage_dtype = np.dtype(np.float16 if internal_dtype == "float16" else np.float32)
    storage_element_bytes = storage_dtype.itemsize
    source_bytes = source.read_bytes()
    source_hash = sha256_bytes(source_bytes)
    model = onnx.load_model_from_string(source_bytes)
    onnx.checker.check_model(model, full_check=True)
    graph = model.graph
    operators = Counter(node.op_type for node in graph.node)
    unsupported = set(operators) - SUPPORTED_OPERATORS
    if unsupported:
        raise ValueError(f"unsupported CAM++ operators: {sorted(unsupported)}")
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("CAM++ package requires one input and one output")
    shapes, types = _tensor_shapes(model)
    input_name = graph.input[0].name
    output_name = graph.output[0].name
    input_shape = shapes[input_name]
    output_shape = shapes[output_name]
    if len(input_shape) != 3 or input_shape[1:] != [150, 80]:
        raise ValueError(f"expected CAM++ [B,150,80] input, got {input_shape}")
    if output_shape != [input_shape[0], 192]:
        raise ValueError(f"expected CAM++ [B,192] output, got {output_shape}")
    if types[input_name] != TensorProto.FLOAT or types[output_name] != TensorProto.FLOAT:
        raise ValueError("the package expects FP32 API boundaries")

    static = _static_tensors(graph)
    builder = BinaryBuilder.create()
    conv_records: dict[str, dict[str, Any]] = {}
    bn_records: dict[str, dict[str, Any]] = {}
    conv_macs: Counter[str] = Counter()

    for graph_index, node in enumerate(graph.node):
        if node.op_type != "Conv":
            continue
        if len(node.input) < 2 or node.input[1] not in static:
            raise ValueError(f"dynamic convolution weight in {node.name}")
        weight = np.asarray(static[node.input[1]])
        if weight.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError(
                f"non-floating convolution weight {node.input[1]}: {weight.dtype}"
            )
        packed_weight, packed_shape = pack_conv_oihw4(
            weight.astype(storage_dtype, copy=False)
        )
        weight_id = builder.add(
            section_id=f"conv:{node.name}:weight",
            kind="conv_weight",
            array=packed_weight,
            logical_shape=weight.shape,
            packed_shape=packed_shape,
            layout="K_O4_I4_I_O",
            source_tensors=[node.input[1]],
        )
        out_channels = int(weight.shape[0])
        if len(node.input) >= 3 and node.input[2]:
            bias_source = np.asarray(static[node.input[2]], dtype=storage_dtype).reshape(-1)
            bias_names = [node.input[2]]
        else:
            bias_source = np.zeros(out_channels, dtype=storage_dtype)
            bias_names = []
        padded_bias = np.pad(
            bias_source,
            (0, align_up(out_channels, CHANNEL_TILE) - out_channels),
        ).astype(storage_dtype, copy=False)
        bias_id = builder.add(
            section_id=f"conv:{node.name}:bias",
            kind="conv_bias",
            array=padded_bias,
            logical_shape=[out_channels],
            packed_shape=[padded_bias.size // CHANNEL_TILE, CHANNEL_TILE],
            layout="O4",
            source_tensors=bias_names,
        )
        attributes = _conv_attributes(node, weight.shape)
        if attributes["group"] != 1:
            raise ValueError(f"grouped convolution is not in the WGSL contract: {node.name}")
        output_elements = math.prod(shapes[node.output[0]])
        macs = output_elements * int(weight.shape[1]) * math.prod(weight.shape[2:])
        category = _classify_conv(node.name)
        conv_macs[category] += macs
        conv_records[node.name] = {
            "graph_index": graph_index,
            "weight_shape": list(weight.shape),
            "output_shape": shapes[node.output[0]],
            "attributes": attributes,
            "macs": macs,
            "category": category,
            "packed": {"weight": weight_id, "bias": bias_id},
        }

    for node in graph.node:
        if node.op_type != "BatchNormalization":
            continue
        if len(node.input) != 5 or any(name not in static for name in node.input[1:]):
            raise ValueError(f"dynamic/incomplete BatchNorm in {node.name}")
        if int(_attribute(node, "training_mode", 0)) != 0:
            raise ValueError(f"training BatchNorm is not packable: {node.name}")
        epsilon = float(_attribute(node, "epsilon", 1e-5))
        packed_affine, channels = compile_batch_norm_affine(
            *(static[name] for name in node.input[1:]), epsilon
        )
        affine_id = builder.add(
            section_id=f"batchnorm:{node.name}:affine",
            kind="batch_norm_affine",
            array=packed_affine,
            logical_shape=[channels, 2],
            packed_shape=packed_affine.shape,
            layout="C4_SCALE_SHIFT",
            source_tensors=node.input[1:],
        )
        bn_records[node.name] = {
            "channels": channels,
            "epsilon": epsilon,
            "affine": affine_id,
        }

    initializer_names = {tensor.name for tensor in graph.initializer}
    packed_source_names = {
        name for section in builder.sections for name in section["source_tensors"]
    }
    missing = sorted(initializer_names - packed_source_names)
    unexpected = sorted(packed_source_names - set(static))
    if missing or unexpected:
        raise ValueError(
            f"initializer coverage mismatch; missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    if len(conv_records) != 225 or len(bn_records) != 56:
        raise ValueError(
            f"unexpected lowered inventory: {len(conv_records)} convolutions, "
            f"{len(bn_records)} BatchNorms"
        )
    binary = builder.finish(
        source_sha256=source_hash,
        batch=input_shape[0],
        frames=input_shape[1],
        feature_dim=input_shape[2],
        embedding_dim=output_shape[1],
    )
    header = parse_header(binary)
    if header["payload_sha256"] != sha256_bytes(binary[HEADER_BYTES:]):
        raise AssertionError("internal payload checksum mismatch")

    static_names = set(static)
    liveness = _graph_liveness(graph, shapes, types, static_names)
    fused_program = _fused_program(graph, shapes, conv_records, bn_records)
    activation_plan = _activation_plan(
        graph,
        shapes,
        len(binary),
        input_shape[0],
        storage_element_bytes,
    )
    total_macs = sum(conv_macs.values())
    metadata: dict[str, Any] = {
        "schema": "senko.campplus.webgpu-pack",
        "format_version": FORMAT_VERSION,
        "source": {
            "file": source.name,
            "byte_length": len(source_bytes),
            "sha256": source_hash,
            "opset": [
                {"domain": item.domain, "version": item.version}
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
            "stream_upload": (
                "Fetch metadata first, create one GPUBuffer, then queue.writeBuffer aligned response-body "
                "chunks while carrying at most three trailing bytes; never retain the complete binary in JS heap."
            ),
        },
        "contract": {
            "input": {"name": input_name, "shape": input_shape, "dtype": "float32"},
            "output": {"name": output_name, "shape": output_shape, "dtype": "float32"},
            "internal_dtype": internal_dtype,
            "required_webgpu_features":
                ["shader-f16"] if internal_dtype == "float16" else [],
            "channel_tile": CHANNEL_TILE,
            "weights_are_batch_independent": True,
        },
        "inventory": {
            "onnx_nodes": len(graph.node),
            "onnx_operators": dict(sorted(operators.items())),
            "initializers": len(graph.initializer),
            "lowered_convolutions": len(conv_records),
            "compiled_batch_norms": len(bn_records),
            "dense_cam_layers": 52,
            "packed_sections": len(builder.sections),
            "source_initializer_bytes": sum(
                int(numpy_helper.to_array(tensor).nbytes) for tensor in graph.initializer
            ),
            "packed_binary_bytes": len(binary),
        },
        "compute": {
            "conv_macs_per_source_batch": total_macs,
            "conv_macs_per_embedding": total_macs // input_shape[0],
            "by_stage_macs_per_source_batch": dict(sorted(conv_macs.items())),
            "note": "Counts convolution multiply-accumulates; elementwise/reduction/activation work is extra.",
        },
        "memory": {
            "onnx_reference": liveness,
            "planned_webgpu": activation_plan,
        },
        "sections": [
            {key: value for key, value in section.items() if key != "source_tensors"}
            for section in builder.sections
        ],
        "fused_program": fused_program,
    }
    return binary, metadata


def _json_bytes(metadata: dict[str, Any]) -> bytes:
    return (json.dumps(metadata, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_or_check(path: Path, expected: bytes, check: bool) -> None:
    if check:
        if not path.exists():
            raise SystemExit(f"missing generated artifact: {path}")
        actual = path.read_bytes()
        if actual != expected:
            raise SystemExit(
                f"generated artifact differs: {path} "
                f"(actual {sha256_bytes(actual)}, expected {sha256_bytes(expected)})"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("web/public/models/campplus-t150-b32-fp16.onnx"),
    )
    parser.add_argument(
        "--internal-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="packed convolution/activation storage type",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".research/campplus-webgpu-pack/campplus-t150-webgpu-fp16.bin"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(".research/campplus-webgpu-pack/campplus-t150-webgpu-fp16.json"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing output is byte-for-byte reproducible without writing",
    )
    args = parser.parse_args(argv)
    binary, metadata = build_package(
        args.input,
        args.output.name,
        args.internal_dtype,
    )
    _write_or_check(args.output, binary, args.check)
    _write_or_check(args.metadata, _json_bytes(metadata), args.check)
    summary = {
        "binary": str(args.output),
        "metadata": str(args.metadata),
        "binary_bytes": len(binary),
        "binary_sha256": metadata["binary"]["sha256"],
        "conv_macs_per_embedding": metadata["compute"]["conv_macs_per_embedding"],
        "activation_peak_bytes": metadata["memory"]["planned_webgpu"]["recommended"][
            "activation_arena_bytes"
        ],
        "minimum_resident_gpu_bytes": metadata["memory"]["planned_webgpu"]["recommended"][
            "minimum_resident_gpu_bytes"
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
