# Pyannote frontend direct-WebGPU package

This directory removes the remaining ONNX Runtime frontend from pyannote
segmentation. The pinned graph is a static 15-node SincNet:

```text
InstanceNorm(waveform)
  -> Conv1d 1x80, kernel 251, stride 10 -> Abs -> MaxPool 3
  -> InstanceNorm -> LeakyReLU -> Conv1d 80x60, kernel 5 -> MaxPool 3
  -> InstanceNorm -> LeakyReLU -> Conv1d 60x60, kernel 5 -> MaxPool 3
  -> InstanceNorm -> LeakyReLU -> BCT-to-BTF
```

`pack.py` checks that exact contract, covers every initializer, retiles all
convolution weights as `[kernel,input,output-group,output-lane]`, and emits
headered, 256-byte-aligned FP16 or FP32 binaries plus JSON metadata. The FP16
production path keeps reductions, state, boundaries, and accumulators in FP32.
The shader-f16-free fallback keeps weights, intermediates, scratch, and math in
FP32. No ONNX protobuf is needed by either raw runtime path.

Generate the ignored deployment package from the B8 export:

```bash
cd web/scripts/models
uv run --python 3.13 python3 pyannote_frontend_webgpu/pack.py \
  ../../public/models/pyannote-segmentation-3.0-frontend-b8.onnx \
  ../../public/models \
  --storage-precision f16

uv run --python 3.13 python3 pyannote_frontend_webgpu/pack.py \
  ../../public/models/pyannote-segmentation-3.0-frontend-b8.onnx \
  ../../public/models \
  --storage-precision f32
```

Run the Python 3.13 parity and package tests:

```bash
cd web/scripts/models
uv run --python 3.13 python3 -m unittest pyannote_frontend_webgpu.test_pack -v
```

## Compute inventory

The three convolutions execute 480,324,000 MACs per 10-second chunk and
3,842,592,000 MACs per B8 call:

| Layer | Output per item | MAC/item | MAC/B8 |
|---|---:|---:|---:|
| Sinc Conv251 | `80 x 15975` | 320,778,000 | 2,566,224,000 |
| Conv5 80->60 | `60 x 5321` | 127,704,000 | 1,021,632,000 |
| Conv5 60->60 | `60 x 1769` | 31,842,000 | 254,736,000 |

The raw lowering is seven dispatches: waveform statistics; fused Sinc, Abs,
and pool; norm-0 statistics; fused normalized/leaky Conv1 and pool; norm-1
statistics; fused normalized/leaky Conv2 and pool; final in-place norm/leaky.
InstanceNorm statistics need their own dispatch because later convolution
workgroups all depend on a channel-wide mean and variance.

## Memory plan

The topological ONNX graph has an 81,792,000-byte conservative activation
peak when Conv0 and Abs outputs are both live. Even the first Conv tensor alone
is 40,896,000 bytes. Fusing Conv -> Abs -> Pool writes only the 13,632,000-byte
logical pooled tensor, stored in 6,816,000 bytes as FP16.

The complete mixed-precision B8 runtime uses two aliased activation slots:

| Allocation | Bytes | Aliased lifetimes |
|---|---:|---|
| Slot A | 6,816,000 | FP16 pool0, then FP32 final `[B,589,60]` features |
| Slot B | 5,120,000 | waveform, then pool1 |
| Stats scratch | 5,120 | max 80 channels x B8 x scale/shift |
| Packed weights | 127,488 | all four norms and three convolutions |
| Total excluding uniforms | 12,068,608 | excludes later LSTM buffers |

The seven dispatches own 384 uniform bytes, so the exact frontend GPU allocation
is 12,068,992 bytes.

The FP32 fallback uses 13,632,000-byte slot A, the same 5,120,000-byte slot B,
5,120-byte statistics, 251,904-byte weights, and 384 bytes of uniforms: exactly
19,009,408 owned GPU-buffer bytes. Its Sinc scratch is 12,660 bytes. Conv5 uses
16-channel blocks and 26,112 scratch bytes when the adapter reports enough
workgroup storage; otherwise it selects an 8-channel, 13,056-byte lowering that
stays below WebGPU's 16 KiB core floor.

Workgroup scratch is transient and does not add to that persistent allocation.
The production Sinc kernel uses 10,652 bytes: 2,161 FP32 signal values and 251
FP16 `vec4` filters. Each Conv5 kernel uses 13,056 bytes: 6,208 FP16 normalized
activations and 80 FP16 `vec4` filters. Keeping the Conv5 accumulators FP32 while
halving its scratch from the 25,472-byte FP32-tile baseline materially improves
M3 occupancy without allocating another GPU buffer.

This is lifetime aliasing, not an estimate based on garbage collection. The
runtime retains two GPU buffers for the full frontend and overwrites them only
after their previous values' last use. CPU residency after upload is the JSON
metadata and small JavaScript objects; the 127,488-byte response buffer is not
retained by `PyannoteFrontendGpuPackage`.

## Kernel lowering

The Sinc kernel interleaves the three independent convolution accumulators that
feed each MaxPool result. Each accumulator still executes its 251 FP32 FMAs in
the identical order, so this scheduling change is byte-exact relative to the
serial kernel while removing its single dependency chain. Conv1 applies the
norm0 scale/shift as it loads pool0, so no normalized activation is written.
Conv2 reuses the same kernel, writes BTF directly, and the final InstanceNorm
plus LeakyReLU runs in place.

The raw VAD tail uses one workgroup per `(batch, frame)`: two 128-wide
affine/Leaky stages in workgroup memory followed by the seven-way classifier.
The tail is 29,478,272 MACs per item (235,826,176 per B8) and needs no globally
materialized hidden tensors.

FluidAudio and Mobius confirm the same 160,000-sample/589-frame contract, but
their optimization target is Core ML/ANE. Their useful transferable decisions
are static Sinc filter materialization, batched 10-second windows, zero-copy
input buffers, and pooled model allocations. They do not contain GPU kernels
that can replace this WGSL lowering.

## M3 Chrome measurement

The complete frontend was measured twice in isolated Chrome on the target M3
with deterministic B8 input:

- FP32 scratch with serial Sinc: 20.61--20.63 ms mean per B8;
- FP32 scratch with byte-exact interleaved Sinc: 15.23--15.27 ms;
- production interleaved Sinc plus FP16 Conv5 scratch: 11.12--11.14 ms;
- production frontend versus the FP32 baseline: max absolute `1.519e-3`, RMS
  `8.564e-5`, cosine `0.999999831`, and zero non-finite values.

The full raw frontend -> FP16-weight/FP32-state LSTM -> raw tail now uses the
eight-frame input-affine LSTM split. A production-build B8 A/B pooled 14 settled
samples per geometry: tile-4 measured 36.9775 ms wall / 35.389440 ms GPU and
tile-8 measured 35.9500 / 33.914880 ms. The four-layer input-affine subgroup
fell from 9.0075 to 7.7225 ms (14.27%); 47 steady calls project to 48.29 ms less
wall time before dual-device overlap. Maximum logit error versus the split ORT
reference remains `5.141e-3`, RMS remains `1.183e-3`, all 4712/4712 argmax
decisions match, and tile-4/tile-8 output SHA-256 is byte-identical. Both use
the same 19,300,352-byte FP32 preactivation arena, so exact LSTM ownership stays
31,711,360 GPU bytes and complete VAD ownership stays 44,145,664 bytes. Tile-8
halves four-layer input-affine workgroups from 9472 to 4736; only local
workgroup storage rises, from 4096 to 8192 bytes.

Rounding the Sinc signal tile to FP16 was rejected. It saved only about 0.9 ms
per B8 beyond the production configuration (roughly 49 ms across 47 calls when
combined with the Conv5 change) while raising the full raw maximum logit error
to `9.380e-3`. It did not justify weakening numerical agreement for a sub-1%
whole-pipeline projection.
