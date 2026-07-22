#!/usr/bin/env python3
"""Generate the compact deterministic B32 final-embedding oracle."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL = REPO_ROOT / "web/public/models/campplus-t150-b32-fp16.onnx"
DEFAULT_OUTPUT = REPO_ROOT / "web/public/models/campplus-t150-b32-reference.f32"


def deterministic_features() -> np.ndarray:
    # Match JavaScript's binary64 Math.sin/Math.cos followed by Float32Array storage.
    values = np.empty((32, 150, 80), dtype=np.float32)
    for batch in range(32):
        for frame in range(150):
            for feature in range(80):
                values[batch, frame, feature] = (
                    math.sin(batch * 0.17 + frame * 0.041 + feature * 0.013) * 0.65
                    + math.cos(batch * 0.07 - frame * 0.023 + feature * 0.009) * 0.3
                )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    session = ort.InferenceSession(
        args.model.read_bytes(),
        providers=["CPUExecutionProvider"],
    )
    result = session.run(["embeddings"], {"features": deterministic_features()})[0]
    output = np.asarray(result, dtype="<f4", order="C")
    if output.shape != (32, 192):
        raise RuntimeError(f"unexpected CAM++ output shape: {output.shape}")
    payload = output.tobytes(order="C")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote={args.output}")
    print(f"bytes={len(payload)}")
    print(f"sha256={hashlib.sha256(payload).hexdigest()}")
    print(f"sum={float(output.astype(np.float64).sum()):.17g}")
    print(f"l2={float(np.linalg.norm(output.astype(np.float64))):.17g}")
    print(f"max={float(output.max()):.17g}")


if __name__ == "__main__":
    main()
