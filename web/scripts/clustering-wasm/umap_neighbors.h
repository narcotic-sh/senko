#ifndef SENKO_WEB_CLUSTERING_UMAP_NEIGHBORS_H_
#define SENKO_WEB_CLUSTERING_UMAP_NEIGHBORS_H_

#include <stddef.h>
#include <stdint.h>

/*
 * Dense cosine-neighbor search for Senko's native-parity UMAP path.
 *
 * The algorithmic behavior follows umap-learn 0.5.12 (BSD-3-Clause,
 * Copyright 2017 Leland McInnes) and PyNNDescent 0.6.0 (BSD-2-Clause,
 * Copyright Leland McInnes). This is a standalone C++/WASM implementation;
 * it does not embed Python, NumPy, Numba, or either project's source.
 */

namespace senko::umap_neighbors {

constexpr int kExactThreshold = 4096;
constexpr int kDefaultNeighborCount = 40;
constexpr int kDefaultMaxCandidates = 60;
constexpr int kDefaultLeafSize = 200;
constexpr float kDefaultDelta = 0.001f;

enum Status : int {
  kSuccess = 1,
  kInvalidArgument = -1,
  kInsufficientWorkspace = -2,
  kSizeOverflow = -3,
  kTreeDepthExceeded = -4,
};

struct ApproximateOptions {
  /*
   * A NumPy RandomState-compatible seed. A fresh, entropy-derived seed keeps
   * production stochastic; a fixed value enables native differential tests.
   */
  uint32_t random_seed = 42u;
  /* Zero selects UMAP's n_trees formula. */
  int n_trees = 0;
  /* Zero selects UMAP's n_iters formula. */
  int n_iters = 0;
  int max_candidates = kDefaultMaxCandidates;
  int leaf_size = kDefaultLeafSize;
  int max_tree_depth = 200;
  float delta = kDefaultDelta;
};

/*
 * Workspace excludes caller-owned input and output arrays. All functions are
 * allocation-free after entry and are safe to use with the existing reusable
 * WebAssembly arena. `workspace` must be aligned to at least 16 bytes.
 */
size_t ExactWorkspaceBytes(int count, int dimension);
size_t ApproximateWorkspaceBytes(int count, int dimension, int neighbor_count,
                                 const ApproximateOptions& options);

/*
 * Reproduce UMAP's small-data branch without allocating its dense N-by-N
 * distance matrix. Output rows are ordered by (cosine distance, input index),
 * which is equivalent to the stable mergesort used by umap-learn.
 */
int ExactCosineKnn(const float* values, int count, int dimension,
                   int neighbor_count, void* workspace,
                   size_t workspace_bytes, int32_t* output_indices,
                   float* output_distances);

/*
 * Angular random-projection forest plus bounded-memory NNDescent for UMAP's
 * N>=4096 branch. Proxy log-cosine distances are converted back to ordinary
 * cosine distances before return, matching PyNNDescent's neighbor_graph API.
 */
int ApproximateCosineKnn(const float* values, int count, int dimension,
                         int neighbor_count,
                         const ApproximateOptions& options, void* workspace,
                         size_t workspace_bytes, int32_t* output_indices,
                         float* output_distances);

}  // namespace senko::umap_neighbors

#endif  // SENKO_WEB_CLUSTERING_UMAP_NEIGHBORS_H_
