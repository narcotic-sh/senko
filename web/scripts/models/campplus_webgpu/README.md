# CAM++ direct-WebGPU packer

This tool converts the checked B32 FP16 CAM++ ONNX artifact into a deterministic
weight package designed for raw WGSL kernels. It does not modify the ONNX
exporter or runtime manifest.

From the repository root:

```bash
uv run --project web/scripts/models --locked \
  python3 web/scripts/models/campplus_webgpu/pack.py

uv run --project web/scripts/models --locked \
  python3 web/scripts/models/campplus_webgpu/pack.py --check
```

The default output is under the ignored
`.research/campplus-webgpu-pack/` directory. Supply `--output` and `--metadata`
when promoting the package to the browser model directory.

The output consists of:

- one 256-byte-aligned binary containing retiled FP16 convolution weights,
  padded FP16 biases, and compiled FP32 BatchNorm affine pairs;
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
