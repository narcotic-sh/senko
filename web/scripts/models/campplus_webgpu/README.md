# CAM++ direct-WebGPU packer

This tool converts either checked B32 CAM++ ONNX artifact into a deterministic
weight package designed for raw WGSL kernels. FP16 remains the default; the
FP32 mode produces a package that requires no optional WebGPU feature.

From the repository root:

```bash
uv run --project web/scripts/models --locked \
  python3 web/scripts/models/campplus_webgpu/pack.py

uv run --project web/scripts/models --locked \
  python3 web/scripts/models/campplus_webgpu/pack.py --check

uv run --project web/scripts/models --locked \
  python3 web/scripts/models/campplus_webgpu/pack.py \
  --internal-dtype float32 \
  --input web/public/models/campplus-t150-b32.onnx \
  --output web/public/models/campplus-t150-webgpu-fp32.bin \
  --metadata web/public/models/campplus-t150-webgpu-fp32.json
```

The default output is under the ignored
`.research/campplus-webgpu-pack/` directory. Supply `--output` and `--metadata`
when promoting the package to the browser model directory.

The output consists of:

- one 256-byte-aligned binary containing retiled FP16 or FP32 convolution
  weights, same-precision padded biases, and compiled FP32 BatchNorm affine
  pairs;
- a minified, deterministic JSON program/section map governed by
  [`metadata.schema.json`](metadata.schema.json).

Run the packer tests and the full-graph BatchNorm compilation check with:

```bash
uv run --project web/scripts/models --locked \
  python3 -m unittest discover \
  -s web/scripts/models/campplus_webgpu -p 'test_*.py' -v

uv run --project web/scripts/models --locked \
  python3 web/scripts/models/campplus_webgpu/verify_affine.py
```

The runtime/kernel design and measured package characteristics are in
[`web/docs/CAMPPLUS_WEBGPU.md`](../../../docs/CAMPPLUS_WEBGPU.md).

Regenerate the two compact tracked graph oracles with the same Python 3.13
environment:

```bash
uv run --project web/scripts/models --locked \
  python3 web/scripts/models/campplus_webgpu/generate_graph_reference.py

uv run --project web/scripts/models --locked \
  python3 web/scripts/models/campplus_webgpu/generate_graph_reference.py \
  --model web/public/models/campplus-t150-b32.onnx \
  --output web/public/models/campplus-t150-b32-fp32-reference.f32
```
