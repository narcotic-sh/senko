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

On the target M3 Mac, clean standalone raw-graph measurements gave
`tile4-fold` a 37.748736 ms pooled whole-graph GPU median, versus 45.842432 ms
for `tile1-split`. Its FCM profile median was 8.060928 ms, versus 11.993088 ms.
Both variants produced the same output fingerprint and parity result, and the
selected kernel does not increase explicit GPU-buffer residency. One
system-contended sample was excluded because every unchanged graph stage and
the Codex GPU process slowed by approximately 2x; it is recorded in the raw
A/B artifact rather than silently discarded.

Three independent cooled production-pipeline runs placed the one-hour CAM++
embedding stage, including package load, validation, pipeline compilation, and
warm-up, at 12.756735-12.768055 seconds. Their complete pipeline wall times
were 17.362080-17.382200 seconds. Exact-boundary windows compared with native
Core ML had mean embedding cosine 0.99983 and median 0.99987; the standalone
FCM variant check reported cosine 0.99999941 against the retained baseline.

The retained browser diagnostics are intentionally separate from production:

- `?raw-campplus-graph-diagnostic=1` runs the complete graph. Add
  `&batch=16&fcm-variant=tile1-split`, `tile1-fold`, `tile2-fold`, or
  `tile4-fold` for the retained FCM A/B variants;
- `?raw-campplus-file-parity=1` compares real FBank windows with the retained
  ORT reference;
- `?raw-campplus-dense-diagnostic=1` profiles the earlier B32 kernel geometry;
  and
- `?raw-campplus-diagnostic=1` checks the packed convolution foundation.

The B32 diagnostics are development oracles, not statements about current
production batching or residency.
