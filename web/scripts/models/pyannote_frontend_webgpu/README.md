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
state, boundaries, and convolution accumulators remain FP32. No ONNX protobuf
is needed by the raw runtime.

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

This is lifetime aliasing, not an estimate based on garbage collection. The
runtime retains two GPU buffers for the full frontend and overwrites them only
after their previous values' last use. CPU residency after upload is the JSON
metadata and small JavaScript objects; the 127,488-byte response buffer is not
retained by `PyannoteFrontendGpuPackage`.

## Kernel order

1. Validate the implemented Sinc/Abs/Pool kernel against an ONNX intermediate
   on Chrome/M3, including silence and deterministic nonzero input.
2. Generalize its output-channel vec4 accumulator to Conv1. Norm0 scale/shift
   is applied while loading pool0, so no normalized activation is written.
3. Reuse the same kernel for Conv2, write BTF directly, and run final
   InstanceNorm plus LeakyReLU in place.
4. Replace the eight-node tail with one workgroup per `(batch, frame)`: two
   128-wide affine/Leaky stages in workgroup memory followed by the seven-way
   classifier. The tail is 29,478,272 MACs per item (235,826,176 per B8) and
   needs no globally materialized hidden tensors.

FluidAudio and Mobius confirm the same 160,000-sample/589-frame contract, but
their optimization target is Core ML/ANE. Their useful transferable decisions
are static Sinc filter materialization, batched 10-second windows, zero-copy
input buffers, and pooled model allocations. They do not contain GPU kernels
that can replace this WGSL lowering.

## M3 Chrome measurement

The complete frontend was measured in Chrome on the target M3 with B8 random
input and the ORT reference session released before timing:

- settled mean: 26.708 ms per B8;
- settled median: 26.375 ms per B8;
- ORT-frontend parity: max absolute `2.924e-3`, RMS `2.011e-4`, and zero
  non-finite values.

The full raw frontend -> FP16-weight/FP32-state LSTM -> raw tail settled at
58.728 ms per synthetic B8 call, with maximum logit error `5.015e-3`, RMS
`1.137e-3`, and 4712/4712 matching argmax decisions versus the split ORT
reference. It owns exactly 24,845,312 GPU bytes. On `test_audio.wav` (3696.043
seconds, 370 chunks), 47 B8 calls complete in roughly 3.1--3.6 seconds in
Chrome/M3; run-to-run GPU scheduling is the dominant spread.
