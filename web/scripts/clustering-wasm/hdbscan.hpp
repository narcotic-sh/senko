#pragma once

#include <stdint.h>

namespace senko_hdbscan {

// Return the exact scratch space consumed by run_f64_semantics, excluding the
// caller-owned input and output arrays. Zero means that an argument or size is
// invalid for the 32-bit WebAssembly address space.
uint32_t workspace_bytes(int count, int dimension, int min_samples,
                         int min_cluster_size);

// HDBSCAN 0.8.44's fixed Senko subset:
//   * Euclidean distances after an f32 -> f64 input conversion
//   * alpha = 1
//   * excess-of-mass cluster selection
//   * allow_single_cluster = false
//   * cluster_selection_epsilon = 0
//
// Inputs with at least 1,024 rows use an exact KD-tree/Boruvka provider;
// smaller inputs retain the implicit complete-graph Prim implementation as a
// compact correctness oracle. Both feed the same condensed-tree and label
// implementation.
//
// Returns 1 on success, -1 for invalid arguments, -2 when the supplied
// workspace is too small, and -3 if an internal tree invariant is violated.
int run_f64_semantics(const float* projection, int count, int dimension,
                      int min_samples, int min_cluster_size, int32_t* labels,
                      void* workspace, uint32_t workspace_size);

// Diagnostic variant that additionally copies core distances and the raw
// unsorted MST as rows [from, to, mutual_reachability_distance]. Integer
// endpoints are represented exactly as Float64 values to match the upstream
// fixture format.
int run_f64_semantics_diagnostic(
    const float* projection, int count, int dimension, int min_samples,
    int min_cluster_size, int32_t* labels, double* core_distances,
    double* mst_rows, void* workspace, uint32_t workspace_size);

}  // namespace senko_hdbscan
