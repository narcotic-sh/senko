#ifndef SENKO_WEB_CLUSTERING_UMAP_SPECTRAL_HPP_
#define SENKO_WEB_CLUSTERING_UMAP_SPECTRAL_HPP_

#include <cstddef>
#include <cstdint>

namespace senko::umap_spectral {

/*
 * Sparse spectral initialization for the connected-graph path in
 * umap-learn 0.5.12 (BSD-3-Clause). This is an independent implementation
 * of the documented numerical procedure, not copied SciPy/ARPACK source.
 *
 * The input is a symmetric fuzzy graph in CSR form. Callers must apply
 * UMAP's embedding-time weight cutoff first:
 *
 *   weight < max_weight / n_epochs  ->  remove the edge
 *
 * Native UMAP forms D^(-1/2) G D^(-1/2) in float32, promotes those rounded
 * values to float64, and asks ARPACK for the dim + 1 smallest eigenpairs of
 * I - D^(-1/2) G D^(-1/2). This implementation preserves that operator and
 * performs dot products, recurrence arithmetic, the projected solve, Ritz
 * reconstruction, and residual checks in float64. Only the orthonormal basis
 * is retained as float32. Thick restart retains converged low Ritz vectors,
 * further reducing the dominant long-recording allocation while preserving
 * the same low eigenspace.
 */

enum class Status : std::int32_t {
  kSuccess = 1,
  kInvalidArgument = -1,
  kInvalidGraph = -2,
  kDisconnectedGraph = -3,
  kNumericalFailure = -4,
  kDidNotConverge = -5,
};

struct Options {
  /*
   * Maximum number of simultaneous Lanczos/Ritz vectors. Zero selects
   * 2 * (dim + 1) + 38, which is 160 vectors for Senko's 60-dimensional
   * UMAP projection.
   */
  std::int32_t maximum_basis_size = 0;
  /*
   * Low Ritz vectors retained at a restart. Zero selects dim + 12 (72 for
   * Senko), capped so at least two new Krylov vectors fit.
   */
  std::int32_t retained_ritz_vectors = 0;
  /** Maximum thick restarts after the initial Lanczos sweep. */
  std::int32_t maximum_restarts = 4;
  double tolerance = 1.0e-4;
  /*
   * Return a usable result whose residual exceeds tolerance. This is useful
   * for diagnostic sweeps over basis sizes. Production should leave it false.
   */
  bool accept_unconverged = false;
};

struct Stats {
  std::int32_t requested_eigenpairs = 0;
  std::int32_t basis_size = 0;
  std::int32_t restart_count = 0;
  std::int32_t converged_eigenpairs = 0;
  double maximum_residual = 0.0;
  double smallest_eigenvalue = 0.0;
  double largest_returned_eigenvalue = 0.0;
  std::size_t peak_working_bytes = 0;
};

/*
 * Compute dim non-trivial normalized-Laplacian eigenvectors.
 *
 * output_vectors is row-major [count, dim], matching UMAP's spectral_layout
 * result shape. output_eigenvalues contains dim values when non-null. The
 * trivial first eigenvector/eigenvalue is omitted from both outputs.
 *
 * Eigenvector signs are canonicalized by making the largest-magnitude entry
 * in each column non-negative. ARPACK does not specify signs, and UMAP's
 * Euclidean objective is invariant to a sign flip before noise is added.
 */
Status initialize_connected_graph(
    const std::int32_t* row_offsets,
    const std::int32_t* columns,
    const float* weights,
    std::int32_t count,
    std::int32_t edge_count,
    std::int32_t dim,
    double* output_vectors,
    double* output_eigenvalues,
    const Options& options,
    Stats* stats);

const char* status_message(Status status);

}  // namespace senko::umap_spectral

#endif  // SENKO_WEB_CLUSTERING_UMAP_SPECTRAL_HPP_
