# Fixed-memory clustering kernels

This module accelerates the deterministic numeric hotspots in browser
clustering while TypeScript retains graph construction, hierarchy selection,
and post-processing. The production path uses three numeric operations:

- Fused row normalization and deterministic SimHash seed-neighbor discovery.
- Euclidean neighbor-descent refinement over the original 192-dimensional
  CAM++ embeddings. Repeated row/candidate pairs within one snapshot pass are
  removed with a count-sized stamp array before distance evaluation; the
  bounded heap threshold only decreases, so every removed attempt would have
  returned zero without changing flags or convergence.
- Exact Euclidean 40-nearest-neighbor construction over UMAP's 10-dimensional
  layout.

The fused seed operation copies the original embeddings once, normalizes them
in place, and builds the graph before resetting the arena. Only the seed
indices and similarities return to JavaScript. A WASM layout kernel was
benchmarked and rejected: its `pow` was slower and different libm rounding
changed a small number of final labels. Keeping layout in TypeScript preserves
byte-exact labels.

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
192 dimensions), verifies the fused seed graph byte for byte, then runs the
TypeScript reference and three warmed WASM trials. It requires byte-identical
final `Int32Array` labels plus ARI 1.0. A measured M3 run was:

| Measurement | TypeScript | WASM hybrid |
| --- | ---: | ---: |
| Whole clustering | 2,265 ms | 1,346–1,357 ms warmed |
| Normalization + seed k-NN | 398 ms | 182 ms first trial; byte-identical graph |
| Euclidean refinement | 770 ms | 251–258 ms warmed |
| Exact post-UMAP k-NN | 359 ms | 121–123 ms warmed |
| Final labels | reference | byte-identical; ARI 1.0 |

The bounded initialization warm-up took 72–86 ms in these runs and occurs
alongside model initialization, before user audio is processed.

## Memory contract

- Fixed clustering linear memory: 9,437,184 bytes.
- Reusable arena capacity: 8,388,608 bytes.
- Fixture peak arena cursor: 7,565,888 bytes (fused seed graph).
- Fixture refinement arena cursor: 7,472,628 bytes, including the 22,852-byte
  candidate stamp array.
- Largest returned JS array set: 2,925,056 bytes (seed graph).
- Returned JS refinement arrays: 1,028,340 bytes at fixture shape.
- Returned JS exact-graph arrays: 1,828,160 bytes at fixture shape.

Inputs are copied directly from their existing typed arrays into the fixed
arena; the wrapper does not create an extra JS staging copy. Output arrays are
copied once into JS because the next operation resets the arena. The fused seed
path avoids a 4,387,584-byte normalized JavaScript matrix on the fixture. The
pipeline's logical UMAP accounting covers returned arrays and TypeScript graph
state; the 9 MiB WASM reservation is reported separately. Since the clustering
instance is preloaded while FBank is live, the pipeline reports 9 MiB +
512 KiB = 9,961,472 bytes of fixed WASM memory rather than taking the maximum
of the two heaps.
