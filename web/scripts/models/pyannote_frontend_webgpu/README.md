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
convolution weights as `[kernel,input,output-group,output-lane]`, and emits a
headered, 256-byte-aligned FP16-storage binary plus JSON metadata. Reductions,
state, boundaries, and convolution accumulators remain FP32. The two Conv5
kernels round only their normalized workgroup-local input tiles to FP16; their
global inputs and outputs retain the package contract. No ONNX protobuf is
needed by the raw runtime.

Generate the ignored deployment package from the B8 export:

```bash
cd web/scripts/models
.venv/bin/python pyannote_frontend_webgpu/pack.py \
  ../../../.research/.typed-umap-dist/models/pyannote-segmentation-3.0-frontend-b8.onnx \
  ../../public/models \
  --storage-precision f16
```

Run the Python 3.13 parity and package tests:

```bash
cd web/scripts/models
.venv/bin/python -m unittest pyannote_frontend_webgpu/test_pack.py -v
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
four-frame input-affine LSTM split. Across balanced isolated-Chrome runs it
settled at 35.2525 ms wall / 33.6200 ms GPU per synthetic B8 call, versus
48.7475 / 46.8910 ms for the retained persistent-kernel baseline. Maximum
logit error versus the split ORT reference remains `5.141e-3`, RMS remains
`1.183e-3`, all 4712/4712 argmax decisions match, and the output SHA-256 is
byte-identical. The 19,300,352-byte FP32 preactivation arena raises exact VAD
ownership to 44,145,664 GPU bytes. Forty-seven steady calls project to about
1.67 seconds before dual-device scheduler overlap.

Rounding the Sinc signal tile to FP16 was rejected. It saved only about 0.9 ms
per B8 beyond the production configuration (roughly 49 ms across 47 calls when
combined with the Conv5 change) while raising the full raw maximum logit error
to `9.380e-3`. It did not justify weakening numerical agreement for a sub-1%
whole-pipeline projection.
