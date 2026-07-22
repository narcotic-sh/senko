# Browser model exports

This directory owns the reproducible, static-shape ONNX exports used by the
browser runtime. It requires `uv` and Python 3.13; model artifacts themselves
are deliberately ignored by git.

From the repository root:

```bash
uv sync --project web/scripts/models --python 3.13 --locked
uv run --project web/scripts/models --locked \
  python3 web/scripts/models/export_models.py all
```

The command creates these deployment artifacts in `web/public/models/`:

- pyannote segmentation-3.0 raw logits at static batches 1, 8, 16, and 32,
  input `[B, 1, 160000]`, output `[B, 589, 7]`;
- matching pyannote frontend `[B,1,160000] -> [B,589,60]` and tail
  `[B,589,256] -> [B,589,7]` ONNX graphs with no LSTM nodes, plus a deterministic
  GPU-ready production FP16-weight package for the four-layer bidirectional
  LSTM and an exact FP32 package for explicit browser A/B tests;
- CAM++ FP32 at static batches 32, 64, and 128 with input `[B, 150, 80]` and
  output `[B, 192]`, plus a production B32 internal-FP16 graph retaining FP32
  input/output boundaries (`campplus-t150-b32-fp16.onnx`);
- `manifest.json`, containing runtime-facing `version` and
  `models.{segmentation,campplus}.batches` maps as well as source/artifact
  SHA-256 checksums, exact shapes, operator inventories, tool versions, and
numerical-check results.

CAM++ FP16 conversion uses the pinned `onnxconverter-common` dependency with
`keep_io_types=True`. Every converted artifact passes ONNX's full checker,
stores all floating initializers as FP16, and is compared independently with
both the PyTorch model and its corresponding FP32 ONNX graph. Acceptance
requires maximum absolute error at most `0.03` and minimum cosine similarity
at least `0.9999`.

ONNX Runtime Web/JSEP must load these FP16 graphs with
`graphOptimizationLevel: "basic"`. Extended/all optimization fuses FP16
convolution patterns into `FusedConv(float16)`, for which JSEP has no kernel.
The required setting and reason are recorded in
`models.campplus.precision_variants.float16.ort_web` in the manifest. The
existing `models.campplus.batches` map remains the FP32 map unchanged; FP16
artifacts live under the sibling `precision_variants.float16.batches` map.

The final pyannote `LogSoftmax` is replaced with `Identity` before export. VAD
decoding only takes `argmax`, so logits have identical class decisions while
removing an unsupported WebGPU operation. The exporter asserts that the graph
contains exactly four ONNX `LSTM` nodes and no `LogSoftmax` or `Softmax` node.

The trained `ParamSincFB` filters are also materialized once as the weights of a
static `Conv1d`. This is exactly equivalent at inference time and removes the
runtime filter-construction subgraph (trigonometry, concatenation, and reverse
slicing). In addition to being faster, it avoids an ONNX Runtime Web/JSEP bug
where the negative-step slice loses one tap and turns a 251-tap filter into an
invalid 250-element reshape input. The exporter rejects pyannote artifacts that
still contain those dynamic Sinc operators.

With ONNX Runtime Web 1.27's JSEP backend, the resulting graph assigns every
operator except the four ONNX `LSTM` nodes to WebGPU. JSEP does not register an
LSTM WebGPU kernel, so the monolithic graph is retained only as a diagnostic
CPU-fallback reference. Production uses the split frontend and tail as strict
WebGPU ORT sessions and runs the four recurrent layers through Senko's
persistent WGSL kernel. There is no production CPU execution fallback.

The split artifacts implement that custom-kernel boundary. Their BTF layouts,
PyTorch IFGO equations, forward/reverse conventions, binary offsets, exact GPU
buffer sizes, persistent dispatch, and browser parity results are documented in
[`PYANNOTE_LSTM_WEBGPU.md`](PYANNOTE_LSTM_WEBGPU.md). The split contract and all
artifact hashes are also embedded at `models.segmentation.split` in the runtime
manifest. ORT/JSEP is initialized from the selected `GPUAdapter`, and every
external tensor and custom buffer is allocated from the exact device exposed by
`ort.env.webgpu.device` after the first session is created.

Every export runs ONNX's full checker and ONNX Runtime CPU against its PyTorch
source before it is accepted. Re-run those checks without rewriting models:

```bash
uv run --project web/scripts/models --locked \
  python3 web/scripts/models/export_models.py all --verify-only
```

Prove the deployed split composition against the full frozen model:

```bash
uv run --project web/scripts/models --locked \
  python3 web/scripts/models/verify_pyannote_split.py --batch 1
```

Verify the optional FP16-weight package while keeping every recurrent
accumulation and activation in float32:

```bash
uv run --project web/scripts/models --locked \
  python3 web/scripts/models/verify_pyannote_split.py \
  --batch 1 --lstm-precision float16
```

The runtime manifest selects FP16 after its Chrome/M3 parity, throughput, and
memory gates passed; FP32 remains a diagnostic baseline. See
[`PYANNOTE_LSTM_WEBGPU.md`](PYANNOTE_LSTM_WEBGPU.md) for the Chrome diagnostic
URLs used to compare FP32 and FP16 parity, throughput, and owned GPU memory.

To export only a fast development bucket:

```bash
uv run --project web/scripts/models --locked \
  python3 web/scripts/models/export_models.py pyannote --pyannote-batches 1
```

The source checkpoints are pinned in `export_models.py`. A checksum mismatch is
fatal by design: review an updated checkpoint before changing the pin.

Run the lightweight tooling/manifest contract tests with:

```bash
uv run --project web/scripts/models --locked \
  python3 -m unittest web/scripts/models/test_export_models.py
```

CAM++ uses the static/export-safe implementation already used by Senko's Core
ML backend. With `T=150`, the CAM blocks see fewer than 100 frames, so its
`2 * global mean` context is equivalent to the reference segmented pooling.
Its final statistics use biased variance plus epsilon, matching the shipping
Mac backend. Each export is also checked for cosine parity against the reference
implementation in `senko/camplusplus.py`.
