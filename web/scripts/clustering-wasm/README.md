# Adaptive-memory clustering kernels

This module accelerates the deterministic numeric hotspots in browser
clustering while TypeScript retains graph construction, hierarchy selection,
and post-processing. The production path uses three numeric operations:

- Fused row normalization and deterministic SimHash seed-neighbor discovery.
- Euclidean neighbor-descent refinement over the original 192-dimensional
  CAM++ embeddings. Normal recording sizes retain the fast triangular bitset,
  which removes repeated unordered pairs across initialization and every
  snapshot pass. If that exact operation would exceed the initial 10 MiB
  arena, refinement switches to the predecessor's exact per-pass/per-row stamp
  deduplication. Stamp scratch is linear in the row count instead of quadratic.
- Exact Euclidean 40-nearest-neighbor construction over UMAP's 10-dimensional
  layout.

The fused seed operation copies the original embeddings once, normalizes them
in place, and builds the graph before resetting the arena. Only the seed
indices and similarities return to JavaScript. A WASM layout kernel was
benchmarked and rejected: its `pow` was slower and different libm rounding
changed a small number of final labels. Keeping layout in TypeScript preserves
byte-exact labels.

## Build and benchmark

The checked-in binary was built with Emscripten 6.0.3-git, `-O3`, and
`-msimd128`; rebuilding with that local toolchain reproduces it byte for byte:

```sh
pnpm build:clustering-wasm
pnpm benchmark:clustering-wasm
```

The build has no generated JavaScript glue or filesystem. Linear memory starts
at 11 MiB with a 10 MiB aligned resettable arena. Before creating any typed
array view, the TypeScript wrapper computes the operation's checked high-water
mark and asks WASM to reserve it. Larger reservations grow in exact 64 KiB
WebAssembly pages (`MEMORY_GROWTH_GEOMETRIC_STEP=0`), are retained for reuse,
and never shrink during the instance's lifetime. There is no arbitrary row- or
audio-duration cap; practical limits are wasm32, browser memory, and runtime.

The benchmark reads `.research/native-reference/embeddings.f32` (5,713 rows by
192 dimensions), verifies the fused seed graph byte for byte, then runs the
TypeScript reference and three warmed WASM trials. It requires byte-identical
final `Int32Array` labels plus ARI 1.0. A measured M3 run was:

| Measurement | TypeScript | WASM hybrid |
| --- | ---: | ---: |
| Whole clustering | 2,260–2,330 ms | 1,225–1,271 ms warmed |
| Normalization + seed k-NN | 394 ms | 184–194 ms; byte-identical graph |
| Euclidean refinement | 754–761 ms | 156–166 ms warmed |
| Exact post-UMAP k-NN | 366–420 ms | 120–127 ms warmed |
| Final labels | reference | byte-identical; ARI 1.0 |

Against the immediately preceding WASM binary, the unordered-pair bitset moved
the six-trial refinement median from 253.45 ms to 160.24 ms: a 93.21 ms, 36.8%
reduction attributable to this kernel change. The whole-clustering range above
is a current-tree measurement that also includes an independent layout
arithmetic optimization, so it is not used to attribute the bitset's gain.

The bounded initialization warm-up took 50–56 ms in these runs and occurs
alongside model initialization, before user audio is processed.

## Memory contract

- Initial clustering linear memory: 11,534,336 bytes.
- Initial reusable arena capacity: 10,485,760 bytes.
- Fixture peak arena cursor: 9,489,316 bytes (Euclidean refinement).
- Fixture unordered-pair bitset: 2,039,544 bytes for 16,316,328 possible pairs.
- Fixture fused-seed arena cursor: 7,565,888 bytes.
- At the native `dim=192`, seed-64, neighbor-20 shape, rows through 6,199 use
  the dense bitset (916 bytes remain). Row 6,200 switches to linear stamps and
  needs only 8,109,600 bytes, so this boundary is a strategy cutoff rather than
  a support limit. The 5,713-row benchmark retains 996,444 bytes of headroom
  and continues to execute the unchanged dense path.
- A checked opt-in 47,999-row native-shape refinement uses 62,782,700 bytes,
  reserves a 62,783,488-byte (958-page) arena, produces valid deterministic
  output, and reuses the same heap on a second call.
- A complete deterministic 47,999-row synthetic clustering run finished in
  31.85 seconds on the target M3, including UMAP, exact 40-NN, hierarchy, and
  post-processing. It used a 63,383,408-byte arena high-water mark, reserved
  63,438,848 bytes, and produced the stable label hash `0a3d1ee4`.
- Largest returned JS array set: 2,925,056 bytes (seed graph).
- Returned JS refinement arrays: 1,028,340 bytes at fixture shape.
- Returned JS exact-graph arrays: 1,828,160 bytes at fixture shape.

Inputs are copied directly from their existing typed arrays into the arena; the
wrapper does not create an extra JS staging copy. Output arrays are copied once
into JS because the next operation resets the arena. The fused seed path avoids
a 4,387,584-byte normalized JavaScript matrix on the fixture. Memory statistics
read the current `memory.buffer.byteLength` and arena capacity, so growth is
visible to pipeline accounting instead of being frozen at the initial 11 MiB.

The ordinary suite includes a cheap 47,999-row stamp-path test. Run the native
192-dimensional scale acceptance explicitly when changing allocation or
refinement logic:

```sh
SENKO_RUN_CLUSTERING_SCALE_TEST=1 pnpm vitest run \
  src/clustering/wasm-kernels.test.ts \
  -t 'scales native-shape refinement'

SENKO_RUN_CLUSTERING_FULL_SCALE_TEST=1 pnpm vitest run \
  src/clustering/wasm-kernels.test.ts \
  -t 'completes full clustering'
```
