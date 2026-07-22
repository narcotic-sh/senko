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
requires cross-origin isolation, WebGPU, a dedicated worker, WASM SIMD, WASM
threads, and `shader-f16`. The record is therefore labeled
`timingAcceptanceEligible: true` only when the full accelerated environment is
present and page-memory instrumentation is absent.
It also verifies the canonical `test_audio.wav` byte length and SHA-256, then
requires a complete six-stage result with the expected 3696.0426875-second
duration and internally consistent speakers and segments.
The SHA-256 pass happens after pipeline completion so verification does not
pre-warm the WAV in the filesystem cache before `wallMs` starts.

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
pyannote VAD is loaded and warmed before the run, while CAM++ asset loading,
validation, pipeline compilation, and warmup deliberately occur inside the
measured embedding stage after VAD is released. Avoid other GPU-heavy work
during acceptance because another app can still contend for the M3 GPU even
though it cannot enter Senko's memory accounting.

## Current measured snapshot (2026-07-21)

These numbers are the latest cooled checkpoint on the target M3 Mac, not a
promise that every run will reproduce the same wall time. FBank and CAM++
overlap, so stage times must not be summed to derive the end-to-end time. Each
row is one fresh isolated-Chrome validation of the production bundle. The long
run passed the offline-correctness gate; repeated final acceptance runs remain
appropriate before treating this single sample as a stable median.

| Fixture | Worker wall | VAD | FBank | CAM++ | Clustering | Postprocess | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `test_audio_short.wav` | **2.094945 s** | 0.442270 s | 0.388905 s | 1.434480 s | 0.201285 s | 0.001445 s | 4 speakers, 49 segments |
| `test_audio.wav` | **13.680845 s** | 2.885215 s | 3.067940 s | 9.091250 s | 1.683555 s | 0.003610 s | 9 speakers, 137 segments |

The long checkpoint is 1.319 seconds below the 15-second stretch target and
3.681 seconds above the aspirational 10-second target. The production CAM++
graph now combines `tile4-fold` FCM, `direct-tile4-wg128` dense bottlenecks,
and `chunk512` pointwise transits. Dense tile-4 shares each activation across
four output channels without changing per-output FMA order. Chunked transits
strip-mine 512 input channels at a time, cutting their workgroup cache from
32 KiB to 16 KiB while preserving accumulation order.

A current standalone graph invocation contained an exact 24.969216 ms GPU
sample. Across the interleaved transit A/B, `chunk512` had a 25.100288 ms
pooled graph median versus 27.590656 ms for `full-cache`; the preceding dense
A/B measured 27.7217 ms for direct tile-4 versus 30.8019 ms for direct tile-2.
All retained winners had identical output fingerprints and parity, with no
increase in explicit GPU memory. FCM tile-8 was also tested and rejected: its
representative graph time was approximately 26.782 ms versus 24.969216 ms, and
its FCM group took approximately 8.389 ms versus 6.750 ms for tile-4. The
experimental tile-8 code was not retained.

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
| `test_audio.wav` | 0.998676 / 0.998744 | 0.988583 / 0.988518 | +5 | +2 |

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

The current exact/lower-bound memory accounting is unchanged by the dense and
transit kernel optimizations:

| Fixture | Known CPU peak | Explicit GPU-buffer peak | Fixed WASM heaps | Isolated page + worker diagnostic |
| --- | ---: | ---: | ---: | ---: |
| `test_audio_short.wav` | 5,571,936 B | 39,855,360 B | 9,961,472 B | 11,899,769 B post-run-1 baseline |
| `test_audio.wav` | 12,250,848 B | 39,855,360 B | 9,961,472 B | 17,448,302 B peak; 12,002,711 B complete |

These columns are complementary measurements and must not be added together.
The page value is Chrome's coarse `measureUserAgentSpecificMemory()` result for
the Senko page agent cluster, including its dedicated worker. The runner uses a
new isolated Chrome profile with one tab, so it excludes the user's other
Chrome windows and tabs. The 118,273,444-byte long-file `Blob` is externally
held and streamed without making a file-sized copy; it is therefore reported
separately from `knownCpuPeakBytes`. The long run's largest named CPU working
allocation is 7,544,032 bytes for clustering.

Production's fixed ownership remains 12,250,848 bytes at the known long-file
CPU peak, 39,855,360 bytes of explicit GPU buffers, and 9,961,472 bytes across
the fixed WASM heaps. The external zero-copy audio `Blob` is 118,273,444 bytes.

The ring adds the named `camOutputBatchBytes = 24,576` host allocation for two
returned B16 output arrays. That lifetime is below the existing VAD peak on the
short file and clustering peak on the long file, so the known CPU peaks in the
table do not increase. The explicit GPU peak does increase by 12,288 bytes for
the second readback slot.

The current post-ring short-file retained-memory diagnostic processed the same
input twice in one isolated page and worker. Its 4,018.825 ms and 4,079.590 ms
runs were thermally throttled and are non-acceptance timings. With both stage
models released after each run, Chrome measured 11,899,769 bytes after run 1
and 11,999,323 bytes after run 2: a +99,554-byte retained delta. This supersedes
the pre-ring lifecycle result for current code and shows no material retained
growth; the small difference is consistent with garbage-collection timing and
Chrome's coarse measurement. For historical comparison, the pre-ring
diagnostic measured 11,893,545 and 12,037,500 bytes, a +143,955-byte delta,
across 2,933.05 ms and 3,221.89 ms non-acceptance runs.

The current long-file retained-memory diagnostic measured 11,986,837 bytes
after run 1 and 12,081,119 bytes after run 2, a +94,282-byte delta (0.79%).
Both passes completed with neither model resident, so this is the comparable
per-run growth signal rather than the larger one-time transition from the
initial VAD-resident state. The difference is small enough to be consistent
with Chrome measurement and garbage-collection noise and shows no material
per-run accumulation. The page-memory and retained-memory modes are diagnostic
only; their timings are not used in the three-run acceptance median.

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
raw CDP using Node's built-in WebSocket, waits for `WebGPU models ready.`, sets
the absolute WAV path with `DOM.setFileInputFiles`, and captures the page's
result JSON at the exact `Pipeline complete.` DOM mutation. Chrome and only its
isolated process group are terminated in `finally`, including on errors. The
unique profile is removed by default; pass `--keep-profile` to retain it and
receive its path in the output.

Standard output is one concise JSON record. `wallMs` is the real end-to-end
worker wall clock; `stageAttributedTotalMs` can be larger because FBank and
CAM++ overlap. `logicalMemory` contains Senko's exact/lower-bound accounting for
owned CPU buffers, GPU buffers, the fixed WASM heap, and named allocations. The
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

Senko deliberately keeps VAD and CAM++ mutually exclusive: VAD is resident at
`WebGPU models ready.`, then each run releases VAD, constructs and warms CAM++,
and releases CAM++ before completion. The mode records that initial VAD-resident
state only as lifecycle context. It treats post-run 1 (neither model resident)
as the comparable baseline and reports post-run 2 minus post-run 1 as the
retained-growth signal. Both runs use the same page, worker, GPU device, and
model metadata, but their stage backends are reconstructed by design. The app's
`?memory=1` sampler is disabled in this mode, so only one
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
  --url 'http://127.0.0.1:4173/?raw-campplus-graph-diagnostic=1&batch=16&fcm-variant=tile4-fold&dense-bottleneck-variant=direct-tile4-wg128&pointwise-transit-variant=chunk512' \
  --remove-profile
```

The runner installs the diagnostic event listener before navigation, launches
one tab in a fresh extension-free profile, and falls back to the terminal DOM
result if a diagnostic does not dispatch its event. Standard output is only
the parsed result JSON, making it safe to redirect directly into a retained A/B
artifact. The isolated Chrome process and profile are removed in `finally`
unless `--keep-profile` is requested.

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
