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
than exhaustively compared. Refinement retains one bit per unordered row pair
so a 192-dimensional distance is never recomputed; that triangular bitset is
2.04 MB for the 5,713-row fixture and grows quadratically in bits, not in
floating-point distances. The post-UMAP exact graph is quadratic in distance
computations but linear in retained memory.

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
`clusteringPeakWorkingBytes` does not decrease: that ledger reports the earlier,
larger UMAP allocation peak rather than hierarchy's later transient storage.

`onUmapStats` exposes stage timings and deterministic typed-array allocation
accounting. On the native hour-long reference, the production WASM-assisted
path's logical peak is about 5.1 MB including its 0.23 MB output. The fixed
11 MiB WASM heap and caller-owned embeddings are reported separately.
At the native 192D/seed-64/neighbor-20 shape, the fixed arena admits up to
6,199 rows; production preflights the exact requirement before copying data.
Process-level profiling is higher because it also includes V8 and downstream
clustering allocations.

Production workers preload one fixed-memory WASM kernel instance. Fused
normalization plus seed k-NN, Euclidean neighbor refinement, and exact
post-UMAP k-NN run there; TypeScript retains layout, hierarchy selection, and
Senko's `CommonClustering` orchestration.
See [`scripts/clustering-wasm/README.md`](../../scripts/clustering-wasm/README.md)
for the reproducible benchmark and fixed-memory accounting.
