#ifndef SENKO_WEB_CLUSTERING_UMAP_FUZZY_GRAPH_H_
#define SENKO_WEB_CLUSTERING_UMAP_FUZZY_GRAPH_H_

#include <stddef.h>
#include <stdint.h>

/*
 * Fuzzy simplicial-set construction for Senko's native-parity UMAP path.
 *
 * The algorithmic behavior follows umap-learn 0.5.12 (BSD-3-Clause,
 * Copyright 2017 Leland McInnes). This standalone C++ implementation uses
 * caller-owned buffers and does not embed Python, NumPy, Numba, or SciPy.
 */

namespace senko::umap_fuzzy_graph {

enum Status : int {
  kSuccess = 1,
  kInvalidArgument = -1,
  kInsufficientWorkspace = -2,
  kInsufficientOutput = -3,
  kSizeOverflow = -4,
};

/*
 * The symmetric fuzzy union contains at most two entries for every directed
 * k-NN entry. The actual count is returned by BuildCsr.
 */
size_t MaximumCsrEntries(int count, int neighbor_count);

/*
 * Workspace excludes inputs, sigma/rho outputs, and CSR outputs. `workspace`
 * must be aligned to at least 16 bytes.
 */
size_t WorkspaceBytes(int count, int neighbor_count);

/*
 * Match UMAP 0.5.12's default fuzzy_simplicial_set path:
 *   smooth_knn_dist(k=neighbor_count, n_iter=64, local_connectivity=1,
 *                   bandwidth=1)
 *   membership strengths
 *   set_op_mix_ratio=1 fuzzy union with the transpose
 *   explicit-zero elimination
 *
 * Inputs are row-major [count, neighbor_count]. Output is canonical CSR:
 * row offsets have count+1 entries and columns are strictly increasing within
 * each row. `output_capacity` is measured in column/value entries.
 */
int BuildCsr(const int32_t* knn_indices, const float* knn_distances, int count,
             int neighbor_count, void* workspace, size_t workspace_bytes,
             float* output_sigmas, float* output_rhos,
             int32_t* output_row_offsets, int32_t* output_column_indices,
             float* output_values, size_t output_capacity,
             size_t* output_entry_count);

}  // namespace senko::umap_fuzzy_graph

#endif  // SENKO_WEB_CLUSTERING_UMAP_FUZZY_GRAPH_H_
