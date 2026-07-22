#!/usr/bin/env python3
"""Pack the eight-node pyannote dense tail as FP16 weights for one WGSL dispatch."""

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
from onnx import numpy_helper, shape_inference


MAGIC = b"SNKVADT1"
VERSION = 1
HEADER_BYTES = 256
ALIGNMENT = 256
TILE = 4
EXPECTED_OPERATORS = Counter({"MatMul": 3, "Add": 3, "LeakyRelu": 2})


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def pack_matrix_i_o4(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    value = np.asarray(matrix)
    if value.dtype != np.float16 or value.ndim != 2:
        raise TypeError(f"expected float16 [I,O] matrix, got {value.dtype} {value.shape}")
    input_features, output_features = map(int, value.shape)
    padded_output = align_up(output_features, TILE)
    padded = np.zeros((input_features, padded_output), dtype=np.float16)
    padded[:, :output_features] = value
    packed = padded.reshape(input_features, padded_output // TILE, TILE)
    return packed, list(packed.shape)


def unpack_matrix_i_o4(packed: np.ndarray, logical_shape: Iterable[int]) -> np.ndarray:
    input_features, output_features = (int(value) for value in logical_shape)
    return packed.reshape(input_features, -1)[:, :output_features].copy()


def pack_bias_o4(bias: np.ndarray) -> np.ndarray:
    value = np.asarray(bias, dtype=np.float16).reshape(-1)
    padded = np.zeros(align_up(value.size, TILE), dtype=np.float16)
    padded[: value.size] = value
    return padded.reshape(-1, TILE)


@dataclass
class Builder:
    payload: bytearray
    sections: list[dict[str, Any]]

    @classmethod
    def create(cls) -> "Builder":
        return cls(bytearray(HEADER_BYTES), [])

    def add(
        self,
        *,
        section_id: str,
        kind: str,
        array: np.ndarray,
        logical_shape: Iterable[int],
        layout: str,
        source_tensor: str,
    ) -> None:
        offset = align_up(len(self.payload))
        self.payload.extend(b"\0" * (offset - len(self.payload)))
        raw = np.ascontiguousarray(array, dtype="<f2").tobytes()
        self.payload.extend(raw)
        self.sections.append(
            {
                "id": section_id,
                "kind": kind,
                "byte_offset": offset,
                "byte_length": len(raw),
                "element_count": int(array.size),
                "dtype": "float16",
                "logical_shape": [int(value) for value in logical_shape],
                "packed_shape": list(array.shape),
                "layout": layout,
                "source_tensor": source_tensor,
            }
        )

    def finish(self, source_sha256: str, batch: int) -> bytes:
        size = align_up(len(self.payload))
        self.payload.extend(b"\0" * (size - len(self.payload)))
        payload_hash = hashlib.sha256(memoryview(self.payload)[HEADER_BYTES:]).digest()
        struct.pack_into(
            "<8sIIIIQ",
            self.payload,
            0,
            MAGIC,
            VERSION,
            HEADER_BYTES,
            ALIGNMENT,
            len(self.sections),
            len(self.payload),
        )
        self.payload[32:64] = bytes.fromhex(source_sha256)
        self.payload[64:96] = payload_hash
        struct.pack_into("<IIII", self.payload, 96, batch, 589, 256, 7)
        return bytes(self.payload)


def parse_header(data: bytes) -> dict[str, Any]:
    magic, version, header, alignment, sections, total = struct.unpack_from(
        "<8sIIIIQ", data, 0
    )
    batch, frames, inputs, classes = struct.unpack_from("<IIII", data, 96)
    return {
        "magic": magic.decode(),
        "version": version,
        "header_bytes": header,
        "alignment": alignment,
        "section_count": sections,
        "total_bytes": total,
        "source_sha256": data[32:64].hex(),
        "payload_sha256": data[64:96].hex(),
        "batch": batch,
        "frames": frames,
        "input_features": inputs,
        "classes": classes,
    }


def _shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    result: dict[str, list[int]] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        result[value.name] = [int(item.dim_value) for item in value.type.tensor_type.shape.dim]
    return result


def build_package(source: Path, binary_name: str) -> tuple[bytes, dict[str, Any]]:
    source_bytes = source.read_bytes()
    source_hash = sha256_bytes(source_bytes)
    model = onnx.load_model_from_string(source_bytes)
    onnx.checker.check_model(model, full_check=True)
    graph = model.graph
    if Counter(node.op_type for node in graph.node) != EXPECTED_OPERATORS:
        raise ValueError("tail operator inventory changed")
    if [node.op_type for node in graph.node] != [
        "MatMul",
        "Add",
        "LeakyRelu",
        "MatMul",
        "Add",
        "LeakyRelu",
        "MatMul",
        "Add",
    ]:
        raise ValueError("tail topology changed")
    shapes = _shapes(model)
    batch = shapes[graph.input[0].name][0]
    if shapes[graph.input[0].name] != [batch, 589, 256] or shapes[graph.output[0].name] != [batch, 589, 7]:
        raise ValueError("tail boundary shape changed")
    static = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    builder = Builder.create()
    layers: list[dict[str, Any]] = []
    macs = 0
    for layer, node_index in enumerate((0, 3, 6)):
        matmul = graph.node[node_index]
        add = graph.node[node_index + 1]
        weight_name = matmul.input[1]
        bias_name = add.input[0]
        source_weight = np.asarray(static[weight_name])
        source_bias = np.asarray(static[bias_name])
        if source_weight.dtype != np.float32 or source_bias.dtype != np.float32:
            raise ValueError("tail source parameters must be float32")
        weight = source_weight.astype(np.float16)
        bias = pack_bias_o4(source_bias)
        packed_weight, _ = pack_matrix_i_o4(weight)
        builder.add(
            section_id=f"linear:{layer}:weight",
            kind="matrix",
            array=packed_weight,
            logical_shape=source_weight.shape,
            layout="I_O4_O",
            source_tensor=weight_name,
        )
        builder.add(
            section_id=f"linear:{layer}:bias",
            kind="bias",
            array=bias,
            logical_shape=source_bias.shape,
            layout="O4",
            source_tensor=bias_name,
        )
        layer_macs = math.prod(shapes[matmul.output[0]]) * int(source_weight.shape[0])
        macs += layer_macs
        layers.append(
            {
                "layer": layer,
                "input_features": int(source_weight.shape[0]),
                "output_features": int(source_weight.shape[1]),
                "macs": layer_macs,
                "leaky_relu": layer < 2,
            }
        )
    if {(item["input_features"], item["output_features"]) for item in layers} != {
        (256, 128),
        (128, 128),
        (128, 7),
    }:
        raise ValueError("tail affine dimensions changed")
    if {item["source_tensor"] for item in builder.sections} != set(static):
        raise ValueError("tail initializer coverage mismatch")
    binary = builder.finish(source_hash, batch)
    header = parse_header(binary)
    metadata = {
        "schema": "senko.pyannote-tail.webgpu-pack",
        "format_version": VERSION,
        "source": {
            "file": source.name,
            "byte_length": len(source_bytes),
            "sha256": source_hash,
        },
        "binary": {
            "file": binary_name,
            "byte_length": len(binary),
            "sha256": sha256_bytes(binary),
            "payload_sha256": header["payload_sha256"],
            "header_bytes": HEADER_BYTES,
            "section_alignment": ALIGNMENT,
            "section_count": len(builder.sections),
        },
        "contract": {
            "input_shape": [batch, 589, 256],
            "output_shape": [batch, 589, 7],
            "boundary_dtype": "float32",
            "weight_dtype": "float16",
            "accumulator_dtype": "float32",
        },
        "compute": {
            "macs_per_batch_item": macs // batch,
            "macs_full_batch": macs,
            "layers": layers,
            "dispatches": 1,
        },
        "memory": {
            "weight_buffer_bytes": len(binary),
            "output_buffer_bytes": batch * 589 * 7 * 4,
            "readback_buffer_bytes": batch * 589 * 7 * 4,
            "uniform_bytes": 64,
            "explicit_gpu_bytes": len(binary) + 2 * batch * 589 * 7 * 4 + 64,
            "workgroup_bytes": (256 + 128 + 128) * 4,
        },
        "sections": builder.sections,
    }
    return binary, metadata


def write_package(source: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_name = "pyannote-segmentation-3.0-tail-webgpu-f16.bin"
    binary, metadata = build_package(source, binary_name)
    binary_path = output_dir / binary_name
    metadata_path = output_dir / "pyannote-segmentation-3.0-tail-webgpu-f16.json"
    binary_path.write_bytes(binary)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return binary_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for path in write_package(args.source, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
