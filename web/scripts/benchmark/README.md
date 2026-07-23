# Browser benchmark runner

## Timing acceptance: production build only

Build once and serve the immutable production bundle from `web/`:

```bash
cd web
pnpm build
pnpm preview --host 127.0.0.1 --port 4173 --strictPort
```

Then run the one-hour acceptance from the repository root:

```bash
node web/scripts/benchmark/run-browser-pipeline.mjs \
  --mode timing \
  --url http://127.0.0.1:4173/ \
  --audio ./test_audio.wav \
  --offline-reference ./.research/native-reference/test-audio-reference.json \
  --remove-profile
```

Port 4173 and the production preview are the defaults, so `--url` is shown
only to make the acceptance protocol explicit. Timing mode inspects the served
HTML and refuses Vite's development client or an unrecognized artifact. It also
requires cross-origin isolation, WebGPU, a dedicated worker, WASM SIMD, and
WASM threads. An FP16 run additionally requires `shader-f16`; an automatically
selected or explicitly forced FP32 run does not. The record is therefore labeled
`timingAcceptanceEligible: true` only when the full accelerated environment is
present, page-memory instrumentation is absent, and the selected precision is
the adapter's automatic production path.
It also verifies the canonical `test_audio.wav` byte length and SHA-256, then
requires a complete six-stage result with the expected 3696.0426875-second
duration and internally consistent speakers and segments.
The SHA-256 pass happens after pipeline completion so verification does not
pre-warm the WAV in the filesystem cache before `wallMs` starts.

To force the complete FP32 model/kernel path on an adapter that also exposes
`shader-f16`, add `--precision float32`. The runner sets
`?precision=float32`, verifies that the worker actually initialized FP32
models, and records `runtime.modelPrecision` in its output:

```bash
node web/scripts/benchmark/run-browser-pipeline.mjs \
  --mode timing \
  --precision float32 \
  --audio ./test_audio.wav \
  --remove-profile
```

On an adapter that exposes `shader-f16`, this forced run is labeled
`timing-fp32-compatibility-diagnostic` and is not a production timing
acceptance. On an adapter without `shader-f16`, automatically selected FP32 is
the production path and remains timing-acceptance eligible.

Omitting the flag keeps production selection automatic: FP16 is preferred
when both model-device adapter handles expose `shader-f16`; otherwise the
fully custom FP32 WebGPU path is selected. WebGPU itself remains mandatory.

Offline Senko is the correctness oracle. With `--offline-reference`, the
runner scores the browser segments after timing has stopped and records the
reference file's path, size, and SHA-256. It reports label-permutation-invariant
10 ms and 50 ms timeline speech IoU, optimal one-to-one mapped-speaker
agreement on frames where both timelines contain speech, and signed
speaker/segment-count deltas. Scoring is dependency-free and outside
`wallMs`; retain these metrics with every performance result so a faster but
incorrect pipeline cannot be mistaken for an improvement.
When supplied, the oracle is also an eligibility gate: both resolutions require
speech IoU ≥ 0.995 and mapped agreement ≥ 0.98, with the absolute segment delta
capped at 10. Speaker-count delta is recorded but is deliberately not a hard
gate: native Senko's unseeded UMAP varies between runs, and the pinned
seven-speaker result is known to undercount this recording. Extra or split
speakers still reduce the one-to-one mapped frame-agreement score. A failed
gate is emitted as `timing-correctness-rejected` with
`timingAcceptanceEligible: false`, preserving the timing and detailed failure
metrics for diagnosis.

For a final performance claim, use multiple independent timing invocations and
report every `wallMs` plus the median. Each invocation creates a clean profile;
pyannote VAD and CAM++ are loaded and warmed concurrently on separate WebGPU
devices before the run. Both remain resident across runs. Avoid other GPU-heavy
work during acceptance because another app can still contend for the M3 GPU
even though it cannot enter Senko's memory accounting.

## Current native-parity snapshot (2026-07-22)

These are isolated Chrome 150 production-build runs on the target M3. VAD,
FBank, and CAM++ overlap, so stage times must not be summed to derive wall
time. Only the canonical one-hour row is a timing acceptance; the other two
are correctness diagnostics over the same production artifact.

| Fixture | Worker wall | VAD | FBank | CAM++ | Clustering | Postprocess | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `test_audio_short.wav` | **1.522200 s** | 0.354905 s | 0.391690 s | 1.253450 s | 0.202090 s | 0.001830 s | 4 speakers, 49 segments |
| `test_audio.wav` | **15.708715 s** | 2.554340 s | 2.885770 s | 8.876305 s | 6.762460 s | 0.003780 s | 7 speakers, 131 segments |
| `test_audio_long.wav` | **166.726970 s** | 21.822615 s | 36.716525 s | 121.058300 s | 45.013010 s | 0.013085 s | 6 speakers, 1,084 segments |

The one-hour run passed the offline gate with 99.9696% speech IoU and 99.9153%
mapped-speaker agreement at 10 ms; its segment delta was -1. The 31,054-second
(8 h 37 min) stress run matched the offline six-speaker partition, removing
the former phantom seventh speaker. It reached 99.9550% speech IoU and 99.9055%
mapped agreement at 10 ms, with a +7 segment delta.

Logical memory accounting for short/one-hour/eight-hour was respectively
7,351,392 / 14,481,956 / 111,145,788 known CPU bytes, plus 12,058,624 /
29,491,200 / 184,942,592 bytes of ordinary and shared WASM high-water. Explicit
GPU buffers remained 84,001,024 bytes in every run. The 8.6-hour fixture is a
stress case, not a hard 200 MB combined-memory gate.

Native clustering uses offline Senko's 40-neighbor, 60-dimensional UMAP,
500/200 epoch policy, unseeded stochastic behavior, HDBSCAN 20/10, and exact
centroid post-processing. The threaded layout's CSR ABI cut its eight-hour
shared allocation from 125,698,048 to 91,815,936 bytes with byte-identical
one-worker output and no material eight-worker slowdown.

The first post-snapshot clustering optimization checkpoint is intentionally
reported as an adjacent warm-machine A/B rather than a new cooled median.
Exact persistent PyNNDescent pair deduplication plus dynamic row scheduling
moved worker wall from 17.276480 to 14.588160 seconds and clustering from
7.953425 to 5.513930 seconds. Within clustering, neighbor search moved from
1.634440 to 0.574550 seconds and layout from 4.412300 to 3.380050 seconds.
The candidate returned seven speakers and exactly 132 segments, passing the
offline gate with 99.9927% speech IoU and 99.9771% mapped agreement at 10 ms.
Ordinary WASM high-water increased by 2,031,616 bytes while shared layout
memory and explicit GPU buffers remained unchanged.

## Historical browser-specific clustering checkpoint (2026-07-22)

These numbers are the latest cooled checkpoint on the target M3 Mac, not a
promise that every run will reproduce the same wall time. VAD, FBank, and
CAM++ overlap, so stage times must not be summed to derive the end-to-end time.
The long row is the median-wall run from three fresh isolated-Chrome timing
acceptances: **9.911170, 9.922655, and 9.934825 seconds**. Every run passed the
offline-correctness gate and produced the same merged-segment payload.

| Fixture | Worker wall | VAD | FBank | CAM++ | Clustering | Postprocess | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `test_audio_short.wav` | **1.487360 s** | 0.356200 s | 0.395445 s | 1.222820 s | 0.199085 s | 0.001735 s | 4 speakers, 49 segments |
| `test_audio.wav` | **9.922655 s** | 2.448150 s | 2.924735 s | 8.736230 s | 1.104210 s | 0.003465 s | 9 speakers, 137 segments |

This section records the replaced deterministic browser-specific clustering
path and is retained as optimization history; it is not the current production
correctness path. Its long median was 5.077 seconds below the 15-second stretch target and
0.077 seconds below the aspirational 10-second target. The production CAM++
graph now combines `tile4-fold` FCM, `direct-tile8-wg96` initial TDNN,
`direct-tile4-wg128` dense bottlenecks, and `chunk512` pointwise transits. The
TDNN replaces cached tile-1 with a 96-lane workgroup that shares each input
evaluation across eight adjacent output vec4 groups. Direct packed-weight
reads remove its 12,800-byte workgroup cache and barrier while preserving the
exact kernel/channel FMA order for every output.

The final 2026-07-22 promotions keep the three pointwise-transit accumulator
vectors in FP16 and radix-sort HDBSCAN hierarchy edges without boxed indices.
The transit change first crossed 10 seconds at 9.972000 seconds. The hierarchy
sort then reduced its controlled 5,713-row substage from 48–63 ms to
10.6–12.4 ms with byte-identical labels and lower transient memory, producing
the three-run median above. The short fixture's 4-speaker/49-segment payload is
byte-identical to the preceding tile-8 VAD checkpoint. On the long fixture, the
small embedding-numeric change retained 9 speakers and 137 segments and agreed
with that preceding browser result at 99.9838% speech IoU and 99.9156%
mapped-speaker agreement at 10 ms. It passed every offline-Senko gate at both
10 ms and 50 ms. GPU-buffer and WASM totals stayed exactly 84,001,024 and 12,058,624
bytes. The logical CPU peak changed by only 56 data-dependent bytes (9,822,864
to 9,822,920), entirely in clustering working-size accounting; neither final
optimization allocates a persistent buffer.

The latest gain comes from exact incremental VAD reduction and dual-device
streaming. The first B8 VAD batch establishes an immutable speech-window
prefix. For every later VAD dispatch, the worker submits VAD first and then up
to two full B16 CAM++ batches on the second device. Partial CAM batches remain
staged until EOF, so inference count and output ordering stay identical to the
sequential pipeline. Against the preceding cooled 12.05-12.21-second checkpoint,
the dual-device scheduler's 11.062-second median was roughly 9% faster.

The split LSTM initially tiled its input-affine term across four independent
frames and left only the hidden-state term in the serial recurrent kernel. That
moved the raw B8 VAD call from 48.7475 to 35.2525 ms wall (27.68%) with a
byte-identical output. Production now shares each packed input weight across
eight frames; tile-4 remains the explicit diagnostic baseline. In a later
production-build A/B, pooled tile-8 medians improved whole-call wall/GPU from
36.9775/35.389440 ms to 35.9500/33.914880 ms and reduced the four-layer
input-affine profile from 9.0075 to 7.7225 ms. The 47-call long-file projection
is 48.29 ms. Its exact short/long acceptance is included in the final snapshot
above.

Tile-8 and tile-4 have identical output SHA-256 and ORT parity, and both retain
the same 19,300,352-byte FP32 preactivation arena. Exact VAD ownership remains
44,145,664 GPU-buffer bytes; only local workgroup storage changes from 4096 to
8192 bytes while four-layer B8 input-affine workgroups halve from 9472 to 4736.
A persistent unordered-pair bitset also lowers long-file clustering by about
0.1 seconds. Together the earlier split-LSTM and clustering changes moved the
full median from 11.061990 to 10.573895 seconds while the measured logical CPU
peak remained 9.82 MB.

The TDNN's cached baseline measured 3.080192 ms; `direct-tile8-wg96` measured
0.655360-0.720896 ms and delivered a 22.806528 ms nine-run pooled full-graph
GPU median. Its fingerprint was identical to baseline and explicit GPU buffers
did not change. Tile-16 and tile-8/WG128 were measured and rejected without
retaining their experimental code. The earlier dense, transit, and FCM A/B
results remain documented in the raw runtime README and retained artifacts.

Long-file UMAP now performs row normalization and deterministic 64-neighbor
seed discovery as one warmed operation in the clustering WASM arena. On
the 5,713-row reference fixture, seed construction fell from 395.021 ms to
179.750 ms while returning byte-identical indices and similarities. Refinement
now records each evaluated unordered row pair in a 2,039,544-byte bitset. Its
six-trial median fell from 253.45 to 160.24 ms (36.8%) because reverse and
cross-iteration duplicates can be skipped exactly; fixture labels remain
byte-identical with ARI 1.0. Reusing one UMAP distance power in the positive
edge update also removes a redundant transcendental. Stable typed-array radix
ordering subsequently removed another 36–50 ms from controlled hierarchy
construction. The accepted long-file median clustering stage is now 1.104210
seconds versus 1.224510 seconds at the preceding checkpoint.

A thermally contended raw-graph A/B measured B16 at 59.5317 ms per 16 items
and B32 at 115.5783 ms per 32 items. B32 improved per-item throughput by only
about 3.0% while increasing explicit diagnostic GPU buffers from 39,843,104 to
64,597,280 bytes, a 24,754,176-byte increase. Those standalone pre-ring
diagnostic totals each include a 32-byte timestamp-query buffer. The current
two-slot production B16 graph uses 39,855,360 explicit GPU-buffer bytes and no
timestamp buffer. The B32 tradeoff is not worthwhile for the pipeline's memory
target, so B16 remains the production batch.

Correctness against the pinned offline result is:

| Fixture | Speech IoU (10 ms / 50 ms) | Mapped agreement (10 ms / 50 ms) | Segment delta | Speaker delta |
| --- | ---: | ---: | ---: | ---: |
| `test_audio_short.wav` | 0.999960 / 1.000000 | 1.000000 / 1.000000 | 0 | 0 |
| `test_audio.wav` | 0.998514 / 0.998591 | 0.988287 / 0.988223 | +5 | +2 |

The short result matches the offline partition and all 49 merged segments. On
the long file, the pinned offline run reports seven speakers and 132 segments,
while the browser reports nine and 137. A manual audit of the recording has
established that it contains at least eight speakers, so seven is known
under-clustering rather than a speaker-count target. Native UMAP is also
unseeded: repeated offline runs have produced seven speakers most often and
eight occasionally. Consequently, speaker-count delta remains diagnostic
only. The two timeline metrics and segment delta above are the actual quality
gates, and the long result passes them.

Long-file clustering forensics support eight meaningful speakers rather than
using either raw count as ground truth. Browser `SPEAKER_09` is one coherent
13.05-second turn that remains separated from `SPEAKER_06` in both browser and
native embedding space, demonstrating that the native seven-speaker result
under-clusters. In contrast, browser `SPEAKER_07` is a heterogeneous
50-window outlier bucket spread over 32 runs: only five fragments totaling
14.696 seconds survive postprocessing, its pairwise cosine cohesion is low,
and its native labels scatter. The evidence therefore favors eight meaningful
speakers. Senko does not add fixture-specific label or count forcing to obtain
that result; the browser retains the native postprocessing semantics.

For these reference fixtures, the clustering pair bitset adds 2 MiB to the
initial WASM heap while the LSTM input-affine arena adds 19,300,352 bytes to
exact GPU ownership. Known CPU
peaks remain small:

| Fixture | Known CPU peak | Explicit GPU-buffer peak | WASM heaps | Isolated page + worker diagnostic |
| --- | ---: | ---: | ---: | ---: |
| `test_audio_short.wav` | 7,351,392 B | 84,001,024 B | 12,058,624 B | not sampled at this checkpoint |
| `test_audio.wav` | 9,822,920 B | 84,001,024 B | 12,058,624 B | 14,430,058 B post-run-1 baseline |

These columns are complementary measurements and must not be added together.
The page value is Chrome's coarse `measureUserAgentSpecificMemory()` result for
the Senko page agent cluster, including its dedicated worker. The runner uses a
new isolated Chrome profile with one tab, so it excludes the user's other
Chrome windows and tabs. The 118,273,444-byte long-file `Blob` is externally
held and streamed without making a file-sized copy; it is therefore reported
separately from `knownCpuPeakBytes`. The long run's largest named CPU working
allocation is 5,116,104 bytes for clustering.

Production's reference-file ownership is 9,822,920 bytes at the known CPU
peak, 84,001,024 bytes of explicit GPU buffers, and 12,058,624 bytes across the
WASM heaps. The GPU total is exactly 44,145,664 bytes for B8 VAD plus
39,855,360 bytes for B16 CAM++. The external zero-copy audio `Blob` is
118,273,444 bytes. The retained-memory protocol below measured the comparable
post-GC page-agent baseline near 14.43 MB; Chrome's coarse page samples and the
exact logical allocation ledger answer different questions.

The ring adds the named `camOutputBatchBytes = 24,576` host allocation for two
returned B16 output arrays. That lifetime is below the existing VAD peak on the
short file and clustering peak on the long file, so the known CPU peaks in the
table do not increase. The explicit GPU peak does increase by 12,288 bytes for
the second readback slot.

The current long-file retained-memory diagnostic processed the same input twice
in one isolated page and worker while both WebGPU models remained resident.
Chrome measured 14,430,058 bytes after run 1 and 14,444,841 bytes after run 2:
a **+14,783-byte delta (0.102%)**. The two non-acceptance runs were 9.931720 and
9.932335 seconds. This is small enough to be Chrome measurement/collection
noise and shows no material per-run accumulation. Page-memory and
retained-memory modes are diagnostic only; their timings are not used in the
three-run acceptance median.

### FP32 compatibility validation

The complete fallback was forced on the same M3 with `?precision=float32`.
Both model devices were requested without `shader-f16`, so WebGPU validation
proved that no FP16 shader syntax leaked into the fallback. This is a
compatibility diagnostic on M3, not a replacement for the faster automatic
FP16 production path.

| Fixture | Worker wall | Explicit GPU buffers | Result | Offline agreement |
| --- | ---: | ---: | --- | --- |
| `test_audio_short.wav` | 2.465 s | 132,535,040 B | 4 speakers, 49 segments | 0.999960/1.000000 speech IoU; 1.000000/1.000000 mapped agreement |
| `test_audio.wav` | 16.650–16.760 s | 132,535,040 B | 9 speakers, 136 segments | 0.998660/0.998744 speech IoU; 0.988457/0.988393 mapped agreement |

The long page-scoped run measured 25,146,688 bytes for the isolated Senko page
and dedicated worker at completion, alongside 12,058,624 WASM bytes and
9,815,976 bytes of known CPU working state. The input Blob remained externally
backed. A two-run short-file retained-memory diagnostic grew by only 73,864
bytes from post-run 1 to post-run 2, within the coarse measurement noise. The
automatic FP16 regression run remained at 9.972 s with its original
84,001,024-byte GPU total and passed the same offline correctness gate.

Do not use `pnpm dev` for acceptance. HMR, source transforms, development module
loading, and concurrent code edits make its timings non-reproducible. The dev
server on port 5173 remains useful for implementation diagnostics:

```bash
cd web
pnpm dev

# In another terminal; diagnostic mode is intentionally non-acceptance.
node web/scripts/benchmark/run-browser-pipeline.mjs \
  --mode page-memory \
  --url http://127.0.0.1:5173/ \
  --audio ./test_audio.wav
```

## What the runner isolates and records

The runner launches a new Chrome process with a unique `--user-data-dir`, no
extensions/background mode/sync, and exactly one Senko tab. It connects through
raw CDP using Node's built-in WebSocket, waits for the FP16/FP32 model-ready
marker, sets
the absolute WAV path with `DOM.setFileInputFiles`, and captures the page's
result JSON at the exact `Pipeline complete.` DOM mutation. Chrome and only its
isolated process group are terminated in `finally`, including on errors. The
unique profile is removed by default; pass `--keep-profile` to retain it and
receive its path in the output.

Standard output is one concise JSON record. `wallMs` is the real end-to-end
worker wall clock; `stageAttributedTotalMs` can be larger because FBank and
CAM++ overlap. `logicalMemory` contains Senko's exact/lower-bound accounting for
owned CPU buffers, GPU buffers, the current WASM heaps, and named allocations. The
full segment list is intentionally omitted. Use `--raw-result <path>` to also
save the unmodified completion-time result; its byte length and SHA-256 are
always recorded.

## Memory-only diagnostics

Page-scoped sampling is a separate, single-run diagnostic:

```bash
node web/scripts/benchmark/run-browser-pipeline.mjs \
  --mode page-memory \
  --url http://127.0.0.1:4173/ \
  --audio ./test_audio.wav \
  --remove-profile
```

This mode adds `memory=1`, waits for Chrome's final coarse-cadence sample, and
emits `timingAcceptanceEligible: false`. Its measurement covers the Senko page
agent cluster, including the dedicated pipeline worker, rather than other
Chrome windows or tabs.

Use retained-memory mode to look for growth across repeated use of one tab and
worker:

```bash
node web/scripts/benchmark/run-browser-pipeline.mjs \
  --mode retained-memory \
  --url http://127.0.0.1:4173/ \
  --audio ./test_audio.wav \
  --remove-profile
```

Senko keeps VAD and CAM++ resident on separate devices from initialization until
the worker/model set is disposed. Retained-memory mode records the initial
dual-resident state as context, treats the post-run-1 dual-resident measurement
as the comparable baseline, and reports post-run 2 minus post-run 1 as the
retained-growth signal. Both runs use the same page, worker, devices, and model
instances. The app's `?memory=1` sampler is disabled in this mode, so only one
`measureUserAgentSpecificMemory()` request is in flight at a time. Chrome's
measurement is coarse and garbage-collection timing can add noise; the
second-run delta is a leak signal, not proof by itself. This mode processes the
input twice and is always non-acceptance. With `--raw-result result.json`, its
two exact captures are written as `result.run-1.json` and
`result.run-2.json`.

Run the small runner/helper tests without loading models or processing audio:

```bash
node --test web/scripts/benchmark/*.test.mjs
```

## Isolated raw-graph diagnostics

Use the dedicated diagnostic launcher for kernel A/B work rather than opening
the URL in an existing Chrome session:

```bash
node web/scripts/benchmark/run-browser-diagnostic.mjs \
  --url 'http://127.0.0.1:4173/?raw-campplus-graph-diagnostic=1&batch=16&fcm-variant=tile4-fold&tdnn-variant=direct-tile8-wg96&dense-bottleneck-variant=direct-tile4-wg128&pointwise-transit-variant=chunk512' \
  --remove-profile
```

The runner installs the diagnostic event listener before navigation, launches
one tab in a fresh extension-free profile, and falls back to the terminal DOM
result if a diagnostic does not dispatch its event. Standard output is only
the parsed result JSON, making it safe to redirect directly into a retained A/B
artifact. The isolated Chrome process and profile are removed in `finally`
unless `--keep-profile` is requested.

The retained dual-device diagnostic is query-gated; its measurements selected
the production dual-resident scheduler. It runs production B8 VAD and B16
CAM++ backends on two devices reporting the same Apple Metal adapter.
`cam-runs=1..4` selects the work ratio and `cam-inflight=1..2` selects CAM++
submission depth:

```bash
node web/scripts/benchmark/run-browser-diagnostic.mjs \
  --url 'http://127.0.0.1:4173/?dual-device-concurrency-diagnostic=1&rounds=12&warmups=3&cam-runs=2&cam-inflight=1' \
  --event senko-dual-device-concurrency-diagnostic \
  --selector '#dual-device-concurrency-result' \
  --remove-profile
```

At the pre-input-affine-LSTM checkpoint, three initial two-CAM isolated runs
saved 15.17-22.41 ms per balanced group
(1.187-1.266x; median 19.98 ms and 1.238x), with bit-exact outputs, stable
late/early timing, 64,700,672 bytes of summed explicit GPU buffers, and about
11.75 MB of page-agent memory. Chrome consumes each `GPUAdapter` handle after
one `requestDevice`, so the page requests two handles and verifies identical
reported adapter identity, features, and limits.

An eight-round work-ratio sweep saved 14.42, 16.92, and 16.28 ms for one, two,
and three sequential CAM++ calls per VAD call. The concurrent three-call trace
placed CAM++ call 3 fully after VAD completion, so two calls maximize hidden
work. Queueing both calls immediately into the production graph's two readback
slots saved 17.08 ms (1.212x), only 0.16 ms above sequential CAM submission.
The resulting production scheduler has 46 overlap opportunities on the
one-hour file and lowered the cooled end-to-end checkpoint from 12.05-12.21
seconds to an 11.062-second median.

## Standalone RSS monitors

For manual measurements, launch a clean Chrome process with a unique
`--user-data-dir`, no extensions, and exactly one Senko tab. Then pass that
browser PID to the process-tree monitor:

```bash
node web/scripts/benchmark/isolated-chrome-memory-monitor.mjs <browser-pid>
```

Write a newline to standard input after the run. The script follows only the
given browser PID and its descendants, so the report includes the isolated
profile's one renderer, dedicated-worker residency, browser process, and GPU
process while excluding every unrelated Chrome profile, window, and tab.
Report the increment over the clean-profile baseline rather than treating
Chrome's own idle RSS as Senko memory.

`chrome-memory-monitor.mjs` is intentionally broader: it samples every Chrome
process on macOS. Do not use its aggregate as a Senko/tab memory result when the
user has other Chrome work open. It is retained for machines where the entire
Chrome instance is dedicated to the benchmark.

OS RSS, user-agent memory, and Senko's exact owned-buffer accounting complement
one another. Browser/driver internals remain opaque to WebGPU page JavaScript.
