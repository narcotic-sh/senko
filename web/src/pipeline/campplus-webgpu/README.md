# Raw WebGPU CAM++ runtime

This directory contains the complete production CAM++ inference path. It runs
the model directly in WGSL and deliberately has no ONNX Runtime fallback.

## Production graph

`CampPlusRawGraph` implements the entire fixed-shape `[B, 150, 80] -> [B, 192]`
network:

- the FCM front end;
- initial TDNN;
- all 52 dense bottleneck and local/CAM append layers;
- all three transit layers; and
- statistics pooling plus the final dense projection.

The static schedule has 119 dispatches. They are encoded into one command
buffer and submitted once per batch; dependent kernels use compute-pass
boundaries as WebGPU storage hazards. Dense-block concatenations are logical
views into one append-only slab rather than copied tensors.

The production FCM kernels use the measured `tile4-fold` geometry: each
workgroup computes four adjacent output channels and folds the final time tap
into the same shader invocation. The original `tile1-split` geometry remains
available only through the raw-graph diagnostic for direct parity and
performance comparisons. Across the ten FCM dispatches this reduces aggregate
workgroups from 84,480 to 10,560 at B16; maximum workgroup storage is 10,240
bytes and explicit GPU-buffer residency is unchanged.

Dense bottlenecks use `direct-tile4-wg128`. One 128-lane workgroup shares each
activation load across four adjacent output channels while reading packed
weights directly, so no extra activation or weight-cache buffer is needed.
The FP32 FMA/reduction order for each output remains identical to the tile-1
oracle. The three pointwise transit layers use `chunk512`: their tile-4 shaders
strip-mine input channels in 512-channel chunks, reducing workgroup storage
from 32 KiB to 16 KiB while preserving accumulation order. Production compiles
only these selected dense and transit pipelines.

The initial TDNN uses `direct-tile8-wg96`. It replaces the cached tile-1
geometry with a 96-lane workgroup that evaluates each input once for eight
adjacent output vec4 groups. Reading packed weights directly removes the
12,800-byte workgroup cache and its synchronization barrier while preserving
the kernel/channel FMA order of every output exactly.

The packed package loader validates metadata, section ranges, fixed headers,
source/payload hashes, and the whole-binary hash. It streams the 13,852,416-byte
weight package directly into one GPU buffer, retaining no full weight copy on
the JavaScript heap. Activations use one fixed, lifetime-aliased arena.

## Production batch and memory

The pinned manifest and pipeline worker select B16. The graph explicitly owns
exactly 39,855,360 GPU-buffer bytes:

| Allocation | Bytes |
| --- | ---: |
| Packed weights | 13,852,416 |
| Activation arena | 25,190,400 |
| FP32 input | 768,000 |
| FP32 output | 12,288 |
| Two FP32 readback slots | 24,576 |
| Dispatch uniforms | 7,680 |
| Timestamp queries in production | 0 |
| **Total** | **39,855,360** |

The pipeline overlaps FBank extraction with the preceding WebGPU run using at
most two B16 host input arrays (1,536,000 bytes total). Its strict two-slot
readback ring can queue batch N+1 while the CPU maps batch N, without creating
an unbounded set of pending mappings. The second slot adds 12,288 GPU-buffer
bytes. Host accounting includes at most two returned output arrays (24,576
bytes total) in addition to the two inputs. The pipeline otherwise retains only
the final 192-value embedding per speech window.

CAM++ is stage-scoped: VAD buffers are destroyed before this graph is loaded,
and every CAM++ buffer is destroyed before CPU clustering begins. A subsequent
recording reconstructs VAD on the same worker device. Disposing the worker's
model set also destroys the `GPUDevice`.

All supported graph batches and their exact explicit buffer totals are pinned
in `public/models/manifest.json`; B4, B8, and B32 remain useful diagnostics,
but they are not the production choice.

## Current validation

On the target M3 Mac, the production combination (`tile4-fold`,
`direct-tile8-wg96` TDNN, `direct-tile4-wg128` dense bottlenecks, and
`chunk512` transits) measured a 22.806528 ms nine-run pooled B16 whole-graph GPU
median. The TDNN itself fell from 3.080192 ms for cached tile-1 to
0.655360-0.720896 ms for the winner. The preceding transit A/B measured
25.100288 ms for `chunk512` versus 27.590656 ms for the full-cache shader, and
the dense A/B measured 27.7217 ms for direct tile-4 versus 30.8019 ms for
direct tile-2. Output fingerprints and parity results were identical, and none
of these optimizations changes explicit GPU-buffer residency.

An FCM tile-8 follow-up was measured and rejected rather than retained. In its
contemporaneous paired run, its representative whole-graph time was
approximately 26.782 ms versus 24.969216 ms for tile-4, and its FCM group took
approximately 8.389 ms versus 6.750 ms. The larger 20 KiB workgroup footprint
reduced occupancy enough to outweigh the lower dispatch count. TDNN tile-16
and tile-8/WG128 were likewise tested and rejected without retaining their
experimental code.

The latest cooled production acceptance of the one-hour fixture completed in
12.939250 seconds: VAD 2.890530 seconds, CAM++ embedding 8.242635 seconds,
clustering 1.766425 seconds, and postprocessing 0.004110 seconds. FBank took
3.000765 seconds but overlaps CAM++. The short fixture completed in 1.973510
seconds with the exact expected 4-speaker/49-segment result. The long result
passed the offline-Senko timeline gates with 9 speakers and 137 segments.
Exact-boundary windows compared with native Core ML had mean embedding cosine
0.99983 and median 0.99987; the standalone FCM variant check reported cosine
0.99999941 against the retained baseline.

The retained browser diagnostics are intentionally separate from production:

- `?raw-campplus-graph-diagnostic=1` runs the complete graph. Add
  `&batch=16&fcm-variant=tile1-split`, `tile1-fold`, `tile2-fold`, or
  `tile4-fold` for the retained FCM A/B variants. Dense bottlenecks accept
  `&dense-bottleneck-variant=direct-tile1-wg128`, `direct-tile2-wg128`, or
  `direct-tile4-wg128`; the initial TDNN accepts `&tdnn-variant=` followed by
  any retained packed-convolution variant, including `cached-tile1-wg128` and
  production `direct-tile8-wg96`; transits accept
  `&pointwise-transit-variant=full-cache` or `chunk512`;
- `?raw-campplus-file-parity=1` compares real FBank windows with the retained
  ORT reference;
- `?raw-campplus-dense-diagnostic=1` profiles the earlier B32 kernel geometry;
  and
- `?raw-campplus-diagnostic=1` checks the packed convolution foundation.

The B32 diagnostics are development oracles, not statements about current
production batching or residency.

Run graph diagnostics through the isolated Chrome launcher from the repository
root (after starting the production preview):

```bash
node web/scripts/benchmark/run-browser-diagnostic.mjs \
  --url 'http://127.0.0.1:4173/?raw-campplus-graph-diagnostic=1&batch=16&fcm-variant=tile4-fold&tdnn-variant=direct-tile8-wg96&dense-bottleneck-variant=direct-tile4-wg128&pointwise-transit-variant=chunk512' \
  --remove-profile
```

It installs the result listener before navigation, uses a fresh extension-free
profile and one tab, emits only the parsed diagnostic JSON, and always tears
down that isolated Chrome process.
