# Browser clustering implementations

`clusterEmbeddings` is the retained deterministic browser-specific baseline.
Production long recordings use the native-parity implementation described
below. The baseline preserves the density hierarchy and post-processing
behavior without materializing an `N x N` matrix:

1. Project the 192-dimensional CAM++ embeddings to 10 dimensions with a
   specialized typed-array UMAP: 20 neighbors, minimum distance 0, 50 epochs,
   and a fixed Mulberry32 seed.
2. Seed UMAP neighborhoods with deterministic random-hyperplane LSH, re-rank
   candidates exactly, then refine them with Euclidean neighbor descent. The
   production WASM path normalizes and builds this seed graph in one arena
   operation, so it never returns a separate normalized embedding matrix.
3. Construct the fuzzy simplicial graph using radix-sorted typed edge arrays
   and optimize the layout in one flat `Float32Array`. Recordings through
   65,536 embeddings retain the benchmarked packed-Uint32 edge path; larger
   graphs switch to a two-endpoint radix order so packed keys cannot wrap.
4. Build an exact Euclidean 40-nearest-neighbor graph over the 10-dimensional
   layout without allocating an `N x N` distance matrix.
5. Compute `min_samples=20` core distances and a sparse mutual-reachability MST.
6. Condense the hierarchy at `min_cluster_size=10` and select clusters by
   excess of mass, following HDBSCAN's flat-cluster strategy. Condensation and
   selection use explicit traversal stacks, so degenerate long-recording trees
   cannot exhaust the JavaScript call stack.
7. Reassign every label population smaller than ten using original CAM++
   embedding centroids, then repeatedly merge centroid pairs whose cosine
   similarity is at least `0.875`. HDBSCAN's `-1` label is deliberately treated
   like an ordinary label here, exactly matching offline Senko.

The default LSH workload is bounded by six 8-bit tables and 64 candidates per
bucket. Large same-speaker buckets are sampled across the full recording rather
than exhaustively compared. Refinement retains one bit per unordered row pair
while the exact operation fits the initial 10 MiB arena, so a 192-dimensional
distance is never recomputed on the production fixture. That triangular bitset
is 2.04 MB for 5,713 rows. Larger shapes switch to exact per-pass/per-row stamps
whose scratch is linear in the row count. The post-UMAP exact graph is
quadratic in distance computations but linear in retained memory.

Hierarchy construction orders its 228,520 fixture edges with stable 16-bit LSD
radix passes over `to`, `from`, and the unsigned bits of each non-negative
Float64 mutual-reachability weight. This is exactly the former comparator order,
including stable ordering of duplicate edges; a focused test compares every
edge against that comparator. On the 5,713-row controlled fixture, warmed
hierarchy time fell from 48–63 ms to 10.6–12.4 ms while final labels remained
byte-identical with ARI 1.0.

At that shape, radix ordering holds 3,918,464 explicitly sized bytes: the
1,828,160-byte weight array, two 914,080-byte Uint32 order arrays, and a
262,144-byte count table. The old path held 3,656,320 typed-array bytes plus a
228,520-element boxed-number order, so the replacement saves at least 651,936
transient bytes even if each boxed slot is counted as only four bytes. Pipeline
`clusteringPeakWorkingBytes` takes the larger of UMAP's measured typed-array
peak and the exact graph/hierarchy allocation plan, rather than omitting this
later transient state.

`onUmapStats` exposes stage timings and deterministic typed-array allocation
accounting. On the 5,713-row reference, UMAP's logical peak is about 5.1 MB;
the whole-clustering ledger reports 6,020,848 bytes after also accounting for
the retained projection, exact graph, hierarchy radix state, and dendrogram.
The initial 11 MiB WASM heap and caller-owned embeddings are reported
separately. At the
native 192D/seed-64/neighbor-20 shape, rows through 6,199 retain the fast dense
bitset. Larger recordings use linear stamps and grow the reusable arena in
exact 64 KiB pages; production preflights the exact requirement before copying
data and reports the live grown heap to pipeline memory accounting.
Process-level profiling is higher because it also includes V8 and downstream
clustering allocations.

Production workers preload one reusable WASM kernel instance. Fused
normalization plus seed k-NN, Euclidean neighbor refinement, and exact
post-UMAP k-NN run there; TypeScript retains layout, hierarchy selection, and
Senko's `CommonClustering` orchestration.
See [`scripts/clustering-wasm/README.md`](../../scripts/clustering-wasm/README.md)
for the reproducible benchmark and adaptive-memory accounting.

## Production native-parity path

`native-umap.ts` assembles the offline Senko parameters used in production:
cosine k-NN 40, 60 output dimensions, spectral initialization, 500
epochs through 10,000 embeddings (otherwise 200), native `a`/`b`, HDBSCAN
20/10, minor-cluster reassignment, and repeated centroid merging at 0.875.
Its WASM stages independently cover PyNNDescent, fuzzy-union CSR, the
normalized-Laplacian eigenspace, legacy NumPy RNG/noise, UMAP layout, and
native approximate-Borůvka HDBSCAN.

The 500/200-epoch Hogwild layout runs in a persistent eight-worker shared-WASM
pool. ABI v2 passes CSR row offsets rather than a repeated COO head array,
derives the negative-sample period exactly on active updates, and dynamically
distributes row-aligned work across heterogeneous CPU cores. The optimizer
retains float64 sample clocks while evaluating its gradient coefficients and
coordinate updates in float32. This reduces the 43,804-row layout allocation
from 125,698,048 to 91,815,936 bytes while the one-hour shape remains at the
16 MiB minimum. On the one-hour fixture, isolated Chrome layout medians fell
from about 3.38 seconds to 1.69–1.75 seconds with seven clusters and no noise
across six stochastic trials.

The serial assembled path remains a correctness oracle, not a production
fallback. Query-independent differential tests live under
`scripts/clustering-wasm` and are opt-in through their `SENKO_RUN_*_PARITY`
environment variables. Final isolated-Chrome acceptance produced:

- `test_audio_short.wav`: unchanged spectral branch, 4 speakers/49 segments,
  with exact mapped agreement to offline Senko.
- `test_audio.wav`: the latest accepted optimization run returned 7
  speakers/131 segments, 99.949% mapped agreement at 10 ms, and a
  13.225-second full-pipeline wall time. This is one isolated run rather than
  a cooled multi-run median.
- `test_audio_long.wav` (31,054 seconds): 6 speakers/1,084 segments versus
  offline Senko's 6/1,077, with 99.906% mapped agreement at 10 ms.
