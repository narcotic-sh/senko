#!/usr/bin/env python3
"""Verify compiled FP32 BatchNorm affine sections against the source FP16 graph."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from pack import _attribute, _static_tensors, compile_batch_norm_affine


def replace_batch_norms(model: onnx.ModelProto) -> onnx.ModelProto:
    transformed = copy.deepcopy(model)
    static = _static_tensors(transformed.graph)
    nodes: list[onnx.NodeProto] = []
    added_initializers: list[onnx.TensorProto] = []
    for node in transformed.graph.node:
        if node.op_type != "BatchNormalization":
            nodes.append(copy.deepcopy(node))
            continue
        affine, channels = compile_batch_norm_affine(
            *(static[name] for name in node.input[1:]),
            float(_attribute(node, "epsilon", 1e-5)),
        )
        scale = affine[:, 0, :].reshape(-1)[:channels]
        shift = affine[:, 1, :].reshape(-1)[:channels]
        rank = 2 if node.name.endswith("dense/nonlinear/batchnorm/BatchNormalization") else 3
        broadcast_shape = [1, channels] if rank == 2 else [1, channels, 1]
        scale_name = f"{node.name}/compiled_scale"
        shift_name = f"{node.name}/compiled_shift"
        fp32_input = f"{node.name}/fp32_input"
        multiplied = f"{node.name}/multiplied"
        fp32_output = f"{node.name}/fp32_output"
        added_initializers.extend(
            [
                numpy_helper.from_array(scale.reshape(broadcast_shape), scale_name),
                numpy_helper.from_array(shift.reshape(broadcast_shape), shift_name),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Cast",
                    [node.input[0]],
                    [fp32_input],
                    name=f"{node.name}/cast_to_fp32",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Mul",
                    [fp32_input, scale_name],
                    [multiplied],
                    name=f"{node.name}/compiled_mul",
                ),
                helper.make_node(
                    "Add",
                    [multiplied, shift_name],
                    [fp32_output],
                    name=f"{node.name}/compiled_add",
                ),
                helper.make_node(
                    "Cast",
                    [fp32_output],
                    list(node.output),
                    name=f"{node.name}/cast_to_fp16",
                    to=TensorProto.FLOAT16,
                ),
            ]
        )
    del transformed.graph.node[:]
    transformed.graph.node.extend(nodes)
    transformed.graph.initializer.extend(added_initializers)
    used = {name for node in nodes for name in node.input}
    retained = [tensor for tensor in transformed.graph.initializer if tensor.name in used]
    del transformed.graph.initializer[:]
    transformed.graph.initializer.extend(retained)
    onnx.checker.check_model(transformed, full_check=True)
    return transformed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("web/public/models/campplus-t150-b32-fp16.onnx"),
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--maximum-absolute-error", type=float, default=0.005)
    parser.add_argument("--minimum-cosine-similarity", type=float, default=0.99999)
    args = parser.parse_args(argv)

    source = onnx.load(args.input)
    compiled = replace_batch_norms(source)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.log_severity_level = 4
    providers = ["CPUExecutionProvider"]
    source_session = ort.InferenceSession(str(args.input), options, providers=providers)
    compiled_session = ort.InferenceSession(
        compiled.SerializeToString(), options, providers=providers
    )
    input_meta = source_session.get_inputs()[0]
    values = np.random.default_rng(args.seed).normal(size=input_meta.shape).astype(np.float32)
    reference = source_session.run(None, {input_meta.name: values})[0]
    actual = compiled_session.run(None, {input_meta.name: values})[0]
    difference = np.abs(reference - actual)
    cosine = np.sum(reference * actual, axis=1) / (
        np.linalg.norm(reference, axis=1) * np.linalg.norm(actual, axis=1)
    )
    metrics = {
        "seed": args.seed,
        "batch": input_meta.shape[0],
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "minimum_cosine_similarity": float(cosine.min()),
        "mean_cosine_similarity": float(cosine.mean()),
    }
    print(json.dumps(metrics, indent=2))
    if metrics["maximum_absolute_error"] > args.maximum_absolute_error:
        return 1
    if metrics["minimum_cosine_similarity"] < args.minimum_cosine_similarity:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
