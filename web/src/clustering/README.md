# Browser clustering baseline

`clusterEmbeddings` is a deterministic, worker-friendly implementation of
Senko's long-recording clustering stage. It preserves the density hierarchy and
post-processing behavior without materializing an `N x N` matrix:

1. Project the 192-dimensional CAM++ embeddings to 10 dimensions with a
   specialized typed-array UMAP: 20 neighbors, minimum distance 0, 50 epochs,
   and a fixed Mulberry32 seed.
2. Seed UMAP neighborhoods with deterministic random-hyperplane LSH, re-rank
   candidates exactly, then refine them with Euclidean neighbor descent. The
   production WASM path normalizes and builds this seed graph in one arena
   operation, so it never returns a separate normalized embedding matrix.
3. Construct the fuzzy simplicial graph using radix-sorted typed edge arrays
   and optimize the layout in one flat `Float32Array`.
4. Build an exact Euclidean 40-nearest-neighbor graph over the 10-dimensional
   layout without allocating an `N x N` distance matrix.
5. Compute `min_samples=20` core distances and a sparse mutual-reachability MST.
6. Condense the hierarchy at `min_cluster_size=10` and select clusters by
   excess of mass, following HDBSCAN's flat-cluster strategy.
7. Reassign every label population smaller than ten using original CAM++
   embedding centroids, then repeatedly merge centroid pairs whose cosine
   similarity is at least `0.875`. HDBSCAN's `-1` label is deliberately treated
   like an ordinary label here, exactly matching offline Senko.

The default LSH workload is bounded by six 8-bit tables and 64 candidates per
bucket. Large same-speaker buckets are sampled across the full recording rather
than exhaustively compared. The post-UMAP exact graph is quadratic in distance
computations but linear in retained memory.

`onUmapStats` exposes stage timings and deterministic typed-array allocation
accounting. On the native hour-long reference, the production WASM-assisted
path's logical peak is about 5.1 MB including its 0.23 MB output. The fixed
9 MiB WASM heap and caller-owned embeddings are reported separately.
Process-level profiling is higher because it also includes V8 and downstream
clustering allocations.

Production workers preload one fixed-memory WASM kernel instance. Fused
normalization plus seed k-NN, Euclidean neighbor refinement, and exact
post-UMAP k-NN run there; TypeScript retains layout, hierarchy selection, and
Senko's `CommonClustering` orchestration.
See [`scripts/clustering-wasm/README.md`](../../scripts/clustering-wasm/README.md)
for the reproducible benchmark and fixed-memory accounting.
