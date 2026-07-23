#ifndef SENKO_WEB_CLUSTERING_UMAP_INITIALIZATION_HPP_
#define SENKO_WEB_CLUSTERING_UMAP_INITIALIZATION_HPP_

#include <cstddef>
#include <cstdint>

namespace senko::umap_initialization {

enum class Status : std::int32_t {
  kSuccess = 1,
  kInvalidArgument = -1,
  kInsufficientWorkspace = -2,
  kNumericalFailure = -3,
};

/*
 * Scratch excludes the spectral input, Float32 output, and three-word layout
 * RNG state. The workspace must be aligned to at least 16 bytes.
 */
std::size_t WorkspaceBytes(std::int32_t dimension);

/*
 * Reproduce UMAP 0.5.12's connected spectral initialization between eigsh and
 * optimize_layout_euclidean:
 *
 *   * advance the shared legacy NumPy RandomState past PyNNDescent
 *   * consume the otherwise-unused spectral eigensolver normal guess
 *   * globally scale the eigenspace, cast to Float32, and add Gaussian noise
 *   * rescale each output column to [0, 10]
 *   * draw the three-word base tau RNG state for layout
 *
 * `approximate_neighbors` must be true exactly when UMAP selected its
 * N>=4096 PyNNDescent branch. A fresh entropy-derived seed gives native
 * stochastic production behavior; seed 42 supports differential tests.
 */
Status Initialize(const double* spectral_embedding,
                  std::int32_t count,
                  std::int32_t dimension,
                  std::uint32_t random_seed,
                  bool approximate_neighbors,
                  void* workspace,
                  std::size_t workspace_bytes,
                  float* output_embedding,
                  std::int64_t* output_layout_rng_state);

}  // namespace senko::umap_initialization

#endif  // SENKO_WEB_CLUSTERING_UMAP_INITIALIZATION_HPP_
