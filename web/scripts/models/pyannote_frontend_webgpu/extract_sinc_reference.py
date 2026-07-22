#!/usr/bin/env python3
"""Extract the first four frontend nodes as an ignored ORT parity artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, shape_inference


OUTPUT_NAME = "/sincnet/pool1d.0/MaxPool_output_0"


def extract(source: Path, destination: Path) -> None:
    model = onnx.load(source)
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    nodes = list(inferred.graph.node[:4])
    if [node.op_type for node in nodes] != [
        "InstanceNormalization",
        "Conv",
        "Abs",
        "MaxPool",
    ] or nodes[-1].output[0] != OUTPUT_NAME:
        raise ValueError("source graph no longer starts with the expected Sinc stage")
    required = {name for node in nodes for name in node.input}
    initializers = [item for item in inferred.graph.initializer if item.name in required]
    output = next(
        value for value in inferred.graph.value_info if value.name == OUTPUT_NAME
    )
    graph = helper.make_graph(
        nodes,
        "senko-pyannote-sinc-reference",
        list(inferred.graph.input),
        [output],
        initializers,
    )
    extracted = helper.make_model(
        graph,
        producer_name="senko-pyannote-sinc-reference",
        opset_imports=list(inferred.opset_import),
        ir_version=inferred.ir_version,
    )
    onnx.checker.check_model(extracted, full_check=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(extracted, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    extract(args.source, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
