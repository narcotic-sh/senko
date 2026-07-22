# Fixed-memory clustering kernels

This module accelerates the deterministic numeric hotspots in browser
clustering while TypeScript retains graph construction, hierarchy selection,
and post-processing. The production path currently uses two exports:

- Euclidean neighbor-descent refinement over the original 192-dimensional
  CAM++ embeddings.
- Exact Euclidean 40-nearest-neighbor construction over UMAP's 10-dimensional
  layout.

Normalization and SimHash candidate generation are also implemented and
regression-tested in the module, but are deliberately not used by the default
UMAP path because the TypeScript LSH is already competitive. A WASM layout
kernel was benchmarked and rejected: its `pow` was slower and different libm
rounding changed a small number of final labels. Keeping layout in TypeScript
preserves byte-exact labels.

## Build and benchmark

The checked-in binary is reproducibly built with Emscripten, `-O3`, and
`-msimd128`:

```sh
pnpm build:clustering-wasm
pnpm benchmark:clustering-wasm
```

The build has no JavaScript glue, filesystem, or memory growth. Its linear
memory is fixed at 9 MiB, containing an 8 MiB aligned resettable arena. Every
operation resets and reuses that arena; there is one instance per pipeline
worker and no per-recording WASM-memory growth.

The benchmark reads `.research/native-reference/embeddings.f32` (5,713 rows by
192 dimensions), runs the TypeScript reference and three warmed WASM trials,
and requires byte-identical final `Int32Array` labels plus ARI 1.0. A measured
M3 run was:

| Measurement | TypeScript | WASM hybrid |
| --- | ---: | ---: |
| Whole clustering | 2,563 ms | 2,063 ms first trial; 1,805 ms best |
| Euclidean refinement | 877 ms | 477 ms first trial |
| Exact post-UMAP k-NN | 431 ms | 135 ms first trial |
| Labels | reference | byte-identical; ARI 1.0 |

The bounded initialization warm-up took 85 ms in that run and occurs alongside
model initialization, before user audio is processed.

## Memory contract

- Fixed clustering linear memory: 9,437,184 bytes.
- Reusable arena capacity: 8,388,608 bytes.
- Fixture peak arena cursor: 7,449,764 bytes (refinement).
- Returned JS refinement arrays: 1,028,340 bytes at fixture shape.
- Returned JS exact-graph arrays: 1,828,160 bytes at fixture shape.

Inputs are copied directly from their existing typed arrays into the fixed
arena; the wrapper does not create an extra JS staging copy. Output arrays are
copied once into JS because the next operation resets the arena. The pipeline's
logical UMAP accounting continues to cover these returned arrays and the
TypeScript LSH/fuzzy-graph state; the 9 MiB WASM reservation is reported
separately. Since the clustering instance is preloaded while FBank is live, the
pipeline reports 9 MiB + 512 KiB = 9,961,472 bytes of fixed WASM memory rather
than taking the maximum of the two heaps.
