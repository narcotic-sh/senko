# Fixed-memory clustering kernels

This module accelerates the deterministic numeric hotspots in browser
clustering while TypeScript retains graph construction, hierarchy selection,
and post-processing. The production path uses three numeric operations:

- Fused row normalization and deterministic SimHash seed-neighbor discovery.
- Euclidean neighbor-descent refinement over the original 192-dimensional
  CAM++ embeddings. A triangular bitset removes repeated unordered pairs
  across initialization and every snapshot pass. Each attempt updates both
  endpoint heaps and their bounded maximum distances only decrease, so every
  removed attempt would have returned zero without changing flags or
  convergence.
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

The build has no JavaScript glue, filesystem, or memory growth. Its linear
memory is fixed at 11 MiB, containing a 10 MiB aligned resettable arena. Every
operation resets and reuses that arena; there is one instance per pipeline
worker and no per-recording WASM-memory growth.

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

- Fixed clustering linear memory: 11,534,336 bytes.
- Reusable arena capacity: 10,485,760 bytes.
- Fixture peak arena cursor: 9,489,316 bytes (Euclidean refinement).
- Fixture unordered-pair bitset: 2,039,544 bytes for 16,316,328 possible pairs.
- Fixture fused-seed arena cursor: 7,565,888 bytes.
- Production refinement supports at most 6,199 rows at the native
  `dim=192`, seed-64, neighbor-20 shape (916 bytes remain). The wrapper
  preflights this exact high-water mark and reports the required bytes before
  copying inputs; the 5,713-row benchmark retains 996,444 bytes of headroom.
- Largest returned JS array set: 2,925,056 bytes (seed graph).
- Returned JS refinement arrays: 1,028,340 bytes at fixture shape.
- Returned JS exact-graph arrays: 1,828,160 bytes at fixture shape.

Inputs are copied directly from their existing typed arrays into the fixed
arena; the wrapper does not create an extra JS staging copy. Output arrays are
copied once into JS because the next operation resets the arena. The fused seed
path avoids a 4,387,584-byte normalized JavaScript matrix on the fixture. The
pipeline's logical UMAP accounting covers returned arrays and TypeScript graph
state; the 11 MiB WASM reservation is reported separately. Since the clustering
instance is preloaded while FBank is live, the pipeline reports 11 MiB +
512 KiB = 12,058,624 bytes of fixed WASM memory rather than taking the maximum
of the two heaps.
