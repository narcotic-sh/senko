# Historical CAM++ direct WebGPU design

> **Historical design record (superseded).** This document captured the plan
> before the raw runtime was implemented. Production now uses the complete
> 119-dispatch direct-WGSL graph at B16 with 39,843,072 explicitly owned GPU
> buffer bytes; FCM, dense/CAM, transit, and final statistics/projection are all
> implemented. B32 recommendations, ONNX timing, future-tense implementation
> steps, and the “runtime is not complete” statement below are retained only as
> engineering history. See
> [`src/pipeline/campplus-webgpu/README.md`](../src/pipeline/campplus-webgpu/README.md)
> for current production status.

## Decision

CAM++ should move from ONNX Runtime Web to a small static raw-WGSL runtime if
the goal is comfortably below 30 seconds for the one-hour fixture, and
especially if the goal remains below 15 seconds. The network is unusually well
suited to this: all shapes are static, the graph has only convolutional,
reduction, and elementwise math, and 52 dense layers repeat the same structure.

Keep the FP16 ONNX path as the numerical oracle while the kernels are built.
It is already a useful performance baseline. It should not dictate the final
execution plan: its 845-node graph exposes many short-lived values and misses
the most valuable CAM++-specific fusions.

The packer is implemented in
[`web/scripts/models/campplus_webgpu/pack.py`](../scripts/models/campplus_webgpu/pack.py).
Its current ignored artifacts are:

```text
.research/campplus-webgpu-pack/campplus-t150-webgpu-fp16.bin
  13,852,416 bytes
  sha256 05ade874c8225035b7c6c966f6ba6e13ab14b0833336fee5a3746e645735186d

.research/campplus-webgpu-pack/campplus-t150-webgpu-fp16.json
  178,095 bytes
```

The binary is deterministic; `pack.py --check` compares both generated files
byte for byte. The source ONNX is 14,108,188 bytes and its initializers total
13,815,360 bytes. The packed binary is slightly larger than the raw initializer
payload because the compiled BatchNorm affine values deliberately remain FP32,
but it is still smaller than the protobuf and is directly uploadable.

## What the exported graph contains

The inspected artifact is `campplus-t150-b32-fp16.onnx`, with FP32 boundaries
`[32,150,80] -> [32,192]` and FP16 internals.

| ONNX operator | Count |
| --- | ---: |
| Conv | 225 |
| Relu | 172 |
| Constant | 113 |
| Mul | 105 |
| BatchNormalization | 56 |
| ReduceMean | 55 |
| Concat | 53 |
| Sigmoid | 52 |
| Add | 5 |
| Cast | 2 |
| Unsqueeze | 2 |
| Transpose, Reshape, Sub, Sqrt, Squeeze | 1 each |

All 617 ONNX initializers are consumed by the pack. The 225 convolutions lower
to 225 retiled weights plus 225 biases, and the 56 BatchNorms lower to 56
affine sections, for 506 aligned sections total. There are no unsupported or
dynamic operators hidden in the model.

The convolution work for one B32 invocation is 25,761,775,616 MACs, or
805,055,488 MACs per embedding:

| Work | B32 MACs | Share |
| --- | ---: | ---: |
| FCM main convolutions | 11,169,792,000 | 43.36% |
| Dense bottleneck 1x1 convolutions | 9,417,523,200 | 36.56% |
| TDNN, transits, final dense | 3,328,966,656 | 12.92% |
| Dense local dilated convolutions | 1,533,542,400 | 5.95% |
| FCM shortcut convolutions | 294,912,000 | 1.14% |
| CAM attention MLPs | 17,039,360 | 0.07% |

This makes the priority clear: optimize the FCM front end and dense bottleneck
kernels first. Attention arithmetic is nearly free; its value is as a fusion
boundary that avoids extra dispatches and storage traffic.

## Package format

The binary starts with a fixed 256-byte little-endian header. It contains the
eight-byte magic `SNKCAMW1`, format/header/alignment fields, section count,
total length, source SHA-256, payload SHA-256, and source B/T/F/output sizes.
Every following section begins at a 256-byte boundary.

Convolution weights change from ONNX `[O,I,spatial...]` into:

```text
[kernel-element, ceil(O/4), ceil(I/4), input-lane, output-lane]
```

The physical layout name is `K_O4_I4_I_O`. Four consecutive FP16 values are
the weights for four output channels at one input lane. WGSL can therefore
accumulate a `vec4` of outputs per input scalar, while a workgroup cooperatively
caches the weights that all time/spatial lanes reuse. Non-multiple channel
counts are zero padded; the source shape remains in metadata.

Biases use padded `O4`. BatchNorm becomes interleaved
`[channel-group, scale-or-shift, lane]` (`C4_SCALE_SHIFT`). The affine is:

```text
scale = gamma / sqrt(running_variance + epsilon)
shift = beta - running_mean * scale
y = x * scale + shift
```

The combined affine is stored as FP32. Storing it as FP16 saved only about
128 KiB but accumulated visible error through 56 normalizations. The included
full-graph verifier replaces every BatchNorm with the packed FP32 equation and
measured, for deterministic seed 31:

| Metric vs source FP16 ONNX | Result |
| --- | ---: |
| Maximum absolute error | 0.0029296875 |
| Mean absolute error | 0.0002163431 |
| Minimum cosine similarity | 0.99999958 |
| Mean cosine similarity | 0.99999976 |

The JSON is a compact typed sidecar rather than a second graph format. It has a
JSON Schema, source/binary hashes, every section range and layout, exact
operator/MAC/memory inventories, and a discriminated fused program. It is only
174 KiB minified, avoiding the several-megabyte JS object graph that a verbose
ONNX-node mirror would create.

Load metadata first, create one read-only storage `GPUBuffer`, then stream the
binary response into it. Keep at most three trailing bytes between response
chunks so every `queue.writeBuffer` offset and size is four-byte aligned. Do
not call `arrayBuffer()` on the complete response; that would temporarily add
13.2 MiB to the JS heap. The same read-only buffer can be bound through FP16
and FP32 views, or through section-relative bindings, because all offsets are
256-byte aligned.

## Execution layout and fusions

Use channel-first activations with time as the innermost dimension:

```text
FCM:   [B, C, F, T]
TDNN:  [B, C, T]
```

For FCM, threads spanning adjacent time samples read coalesced data. Its final
`[B,32,10,150]` buffer is already a zero-copy `[B,320,150]` view for TDNN; no
transpose or reshape kernel is needed. The first convolution indexes the FP32
`[B,T,F]` upload directly, rounds each loaded feature to FP16, and folds
Cast/Transpose/Unsqueeze into address math.

The recommended first implementation is 119 dispatches per full batch:

| Kernel family | Dispatches | Fusion |
| --- | ---: | --- |
| FCM front end | 10 | Conv/bias/activation; residual conv2 also evaluates shortcut, adds, and applies ReLU |
| Initial TDNN | 1 | K5 stride-2 convolution, folded bias, ReLU |
| Dense bottleneck | 52 | pre-BN affine, ReLU, 1x1 convolution, folded post-BN bias, ReLU, time mean |
| Dense local/CAM | 52 | attention MLP, sigmoid, dilated K3 convolution, gate, append |
| Transit | 3 | BN affine, ReLU, 1x1 convolution; transit 3 also applies its folded output ReLU |
| Statistics/final dense | 1 | mean, biased variance + 1e-5, dense, output affine, ReLU, FP32 store |

The 53 graph `Concat` nodes become no-ops. Allocate each dense block once at
its maximum channel count. A layer reads the initialized prefix and writes its
32 output channels at `append_channel`; the next layer simply increases the
logical prefix. Only one `[B,128,T]` bottleneck scratch buffer is reused.

### Dense bottleneck kernel

Use `workgroup_size(128)` and one workgroup per `(batch, output-channel-vec4)`.
Lanes 0-74 own time positions; the remaining lanes help load shared data. The
largest dense input has 992 channels. Caching four output columns of FP16
weights, FP32 affine scale/shift, and a padded `vec4<f32>` time-reduction array
requires about 17,920 bytes of workgroup storage. Check the M3 adapter limit at
startup. If it is below that number, leave affine parameters in read-only
storage/L2 rather than splitting the kernel.

With B32 this yields 1,024 workgroups per dense layer, enough occupancy. For
each input channel, adjacent time lanes access adjacent addresses. Each lane
accumulates four outputs, writes FP16 bottleneck values, and the workgroup
reduces the four time means needed by CAM.

Start with FP32 accumulators and FP16 inputs/weights. A separate FP16-accumulate
pipeline is worth benchmarking, but it should be accepted only on final
embedding cosine and diarization parity, not on a single layer's error.
During the correctness implementation, explicitly round fused results at every
tensor boundary that the source graph stores as FP16 (after BatchNorm affine
and after convolution bias, before the following activation). Removing one of
those roundings is a later benchmark/quality decision; silently carrying FP32
through an entire fused block would not reproduce the verified graph.

### Dense local/CAM kernel

Use one 128-thread workgroup per `(batch, output-channel-vec4)`, 256 workgroups
per layer. First compute the 64-value hidden attention vector from the 128
bottleneck means, then the workgroup's four sigmoid gates. Do not cache the
shared 128x64 attention-1 matrix: it is 16 KiB by itself and its arithmetic is
only 0.05% of the model. Rely on the GPU cache even though the eight output
groups repeat those reads.

Cache only the local K3 x 128 x 4 FP16 weights (6,144 bytes), apply the correct
dilation, multiply by the four gates, and store directly into the dense slab's
append range. This removes ReduceMean, scalar multiply, two attention Conv
dispatches, Sigmoid, gate multiply, and Concat as materialized operations.

An experimental persistent version can execute an entire dense layer with one
workgroup per batch, reducing the whole model to about 67 dispatches. It has
only 32 workgroups and reads the large bottleneck matrix from global memory, so
it may lose more occupancy than it saves in launch/storage overhead. Implement
it only after the two-dispatch kernel establishes correctness and a timestamp
baseline.

### FCM kernels

FCM is the largest compute stage and the activation peak. Use time-lane spatial
tiles in BCFT, cache the small K3 input-channel slices used by an output vec4,
and specialize stride-1 and frequency-stride-2 variants. There are only 12
logical FCM convolutions and 173 KiB of FCM packed parameters, so specialization
is preferable to a generic convolution interpreter.

For each residual block, dispatch conv1 and then a combined conv2/shortcut
kernel. The combined kernel reads conv1 scratch plus the block input, evaluates
the learned shortcut only on the two downsampling blocks, sums, applies ReLU,
and writes one result. This prevents a third full-size shortcut output.

## Memory plan

The proposed full-B32 arena has an exact theoretical live peak of 49,152,000
bytes (46.875 MiB), at the first FCM residual block:

```text
24,576,000-byte block input
+ 12,288,000-byte conv1 scratch
+ 12,288,000-byte fused conv2/residual output
= 49,152,000 bytes
```

The dense backbone peaks at only 7,372,800 bytes. With the 13,852,416-byte
weight buffer, the CAM++-only resident GPU floor is 63,004,416 bytes
(60.09 MiB). Pipeline caches, compiled pipelines/driver state, VAD, and browser
overhead are not included and must continue to be measured externally.

The front end can be microbatched while retaining the B32 dense backbone:

| FCM/TDNN microbatch | Activation arena | Weights + arena | FCM/TDNN dispatches |
| ---: | ---: | ---: | ---: |
| 32 | 46.875 MiB | 60.09 MiB | 11 |
| 16 | 24.02 MiB | 37.23 MiB | 22 |
| 8 | 12.31 MiB | 25.52 MiB | 44 |
| 4 | 7.03 MiB | 20.24 MiB | 88 |

Start performance work at B32. Benchmark B16 immediately afterward: it still
provides a large grid on M3, and if its latency is within roughly 5%, its
23 MiB resident-memory saving is worthwhile. Raw kernels are batch-independent,
so B64 can also be tested without repacking, but its roughly doubled FCM arena
must earn a meaningful throughput gain.

Use two small input staging regions to overlap WASM FBank production with the
GPU's preceding CAM++ batch. The extra staging cost is about 1.5 MiB per B32
FP32 feature batch. Return only the `[B,192]` FP32 embeddings; no intermediate
readback is necessary.

## Throughput target and bottlenecks

The 5,713 useful embeddings in `test_audio.wav` require about 4.60 trillion
MACs (4.61 trillion with the last B32 batch padded). The measured ONNX Runtime
FP16 B32 path is about 132 ms per batch, approximately 195 GMAC/s and 23.6
seconds for CAM++ alone.

Approximate CAM++ budgets are:

| CAM++ time | Required sustained throughput |
| ---: | ---: |
| 20 s | 230 GMAC/s |
| 15 s | 307 GMAC/s |
| 12 s | 383 GMAC/s |
| 10 s | 460 GMAC/s |
| 8 s | 575 GMAC/s |

The under-30-second pipeline target is plausible with the current FP16 graph
but leaves little margin for VAD, FBank, clustering, initialization, and browser
variance. A comfortable under-15-second total likely needs CAM++ near 8-10
seconds, or roughly 2.4-3.0x the current effective ORT throughput. Eliminating
hundreds of graph dispatches helps, but the raw kernels must also improve the
two real arithmetic bottlenecks: FCM spatial convolution and dense 1x1
convolution. Timestamp both families separately; optimizing attention cannot
move the end-to-end number.

Raw WGSL is preferable to jax-js for this model. A graph compiler can remove
some elementwise nodes, but it is unlikely to infer the in-place dense append,
the residual shortcut output elimination, the fixed BCFT/BCT lifetime arena,
or the CAM-specific mean/attention/local fusion. Those transformations are the
reason to leave ORT. A framework-generated kernel layer would also make exact
storage accounting and workgroup tuning harder. jax-js remains useful as an
experimentation/reference tool, not as the performance endpoint.

## Implementation and validation order

1. Implement the streaming package loader, validate header/source hashes, and
   allocate one weight buffer plus one lifetime-aliased activation arena.
2. Implement a packed vec4 pointwise/TDNN kernel and validate individual TDNN
   and transit outputs against exposed ONNX intermediates. This proves binary
   addressing, FP16 reads, FP32 affine values, and BCT indexing.
3. Implement the two dense kernels. Validate after every one of the 52 append
   steps; this is the most reusable and second-largest compute family.
4. Implement specialized FCM stride/residual kernels in BCFT. Validate each
   residual boundary and prove the final reshape is a view.
5. Fuse statistics pooling/final dense and compare final embeddings with the
   source FP16 ONNX on deterministic random data and real FBank windows.
6. Run B32 and B16 with GPU timestamps, then tune workgroup size, cached data,
   FP32 versus FP16 accumulation, and optional subgroup paths.
7. Double-buffer FBank/CAM++ work, run the one-hour browser benchmark, and
   report exact owned GPU bytes, JS heap checkpoints, renderer RSS, GPU-process
   RSS, model load peak, steady-state peak, and memory after disposal.

The runtime is not complete until it passes both numerical and diarization
checks. Initial raw-vs-FP16-ONNX targets should be maximum embedding error below
0.005 and minimum cosine above 0.99999. Then compare speaker count, label ARI,
raw turns, and merged turns on `test_audio.wav`; cosine alone can hide a small
boundary-changing regression.
