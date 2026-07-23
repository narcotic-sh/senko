#ifndef SENKO_WEB_CLUSTERING_UMAP_LAYOUT_HPP_
#define SENKO_WEB_CLUSTERING_UMAP_LAYOUT_HPP_

#include <stddef.h>
#include <stdint.h>

/*
 * Serial Euclidean layout optimizer for Senko's native-parity UMAP path.
 *
 * The update schedule and objective follow umap-learn 0.5.12
 * (BSD-3-Clause, Copyright 2017 Leland McInnes). This is a standalone
 * allocation-free C++ implementation; it does not embed Python, NumPy,
 * Numba, or umap-learn source.
 */

namespace senko::umap_layout {

enum Status : int {
  kSuccess = 1,
  kInvalidArgument = -1,
  kInsufficientWorkspace = -2,
  kSizeOverflow = -3,
};

/*
 * Workspace excludes every caller-owned input and the embedding output. It is
 * reusable across calls and must be aligned to at least 16 bytes.
 */
size_t SerialWorkspaceBytes(int vertex_count, int edge_count);

/*
 * Mutate `embedding` in-place using the deterministic serial optimizer selected
 * by umap-learn when random_state is set.
 *
 * `embedding` is vertex-major float32 [vertex_count, dimension]. `head`,
 * `tail`, and `epochs_per_sample` each contain `edge_count` entries in the COO
 * order supplied to umap-learn. `rng_state` is the base three-int64 state
 * captured immediately before optimize_layout_euclidean.
 */
int OptimizeSerial(float* embedding, int vertex_count, int dimension,
                   const int32_t* head, const int32_t* tail,
                   const double* epochs_per_sample, int edge_count,
                   const int64_t rng_state[3], int epoch_count, double a,
                   double b, double gamma, double negative_sample_rate,
                   void* workspace, size_t workspace_bytes);

}  // namespace senko::umap_layout

#endif  // SENKO_WEB_CLUSTERING_UMAP_LAYOUT_HPP_
