#include "umap_spectral.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace senko::umap_spectral {
namespace {

constexpr double kBreakdownTolerance = 1.0e-14;

bool checked_add(std::size_t left, std::size_t right, std::size_t* result) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    return false;
  }
  *result = left + right;
  return true;
}

bool checked_product(std::size_t left, std::size_t right, std::size_t* result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

class ScratchArena {
 public:
  ScratchArena(void* workspace, std::size_t size)
      : bytes_(static_cast<std::uint8_t*>(workspace)), size_(size) {}

  template <typename T>
  T* allocate(std::size_t count) {
    std::size_t byte_count = 0;
    if (!checked_product(count, sizeof(T), &byte_count)) {
      valid_ = false;
      return nullptr;
    }
    const std::size_t alignment = alignof(T);
    if (cursor_ > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
      valid_ = false;
      return nullptr;
    }
    const std::size_t start = (cursor_ + alignment - 1) & ~(alignment - 1);
    std::size_t end = 0;
    if (!checked_add(start, byte_count, &end) || end > size_) {
      valid_ = false;
      return nullptr;
    }
    cursor_ = end;
    return reinterpret_cast<T*>(bytes_ + start);
  }

  bool valid() const { return valid_; }
  std::size_t used() const { return cursor_; }

 private:
  std::uint8_t* bytes_;
  std::size_t size_;
  std::size_t cursor_ = 0;
  bool valid_ = true;
};

class ScratchSizer {
 public:
  template <typename T>
  bool add(std::size_t count) {
    std::size_t byte_count = 0;
    if (!checked_product(count, sizeof(T), &byte_count) ||
        cursor_ > std::numeric_limits<std::size_t>::max() -
                      (alignof(T) - 1)) {
      valid_ = false;
      return false;
    }
    const std::size_t start =
        (cursor_ + alignof(T) - 1) & ~(alignof(T) - 1);
    if (!checked_add(start, byte_count, &cursor_)) {
      valid_ = false;
      return false;
    }
    return true;
  }

  bool valid() const { return valid_; }
  std::size_t bytes() const { return valid_ ? cursor_ : 0; }

 private:
  std::size_t cursor_ = 0;
  bool valid_ = true;
};

struct ResolvedOptions {
  std::int32_t requested = 0;
  std::int32_t basis_size = 0;
  std::int32_t retained = 0;
  std::int32_t maximum_restarts = 0;
};

bool resolve_options(std::int32_t count,
                     std::int32_t edge_count,
                     std::int32_t dim,
                     const Options& options,
                     ResolvedOptions* result) {
  if (count < 3 || edge_count < 1 || dim < 1 || dim > count - 2 ||
      !std::isfinite(options.tolerance) || options.tolerance <= 0.0 ||
      options.maximum_basis_size < 0 || options.retained_ritz_vectors < 0 ||
      options.maximum_restarts < 0) {
    return false;
  }
  const std::int64_t requested = static_cast<std::int64_t>(dim) + 1;
  const std::int64_t automatic_basis = 2 * requested + 38;
  result->requested = static_cast<std::int32_t>(requested);
  result->basis_size = static_cast<std::int32_t>(
      std::min<std::int64_t>(
          count, options.maximum_basis_size == 0
                     ? automatic_basis
                     : std::max<std::int64_t>(options.maximum_basis_size,
                                              requested + 1)));
  if (result->basis_size <= result->requested) {
    return false;
  }
  const std::int64_t requested_retained =
      options.retained_ritz_vectors == 0
          ? requested + 11
          : static_cast<std::int64_t>(options.retained_ritz_vectors);
  result->retained = static_cast<std::int32_t>(
      std::min<std::int64_t>(requested_retained, result->basis_size - 2));
  result->maximum_restarts =
      result->retained >= result->requested ? options.maximum_restarts : 0;
  return true;
}

std::size_t graph_workspace_bytes(std::int32_t count) {
  ScratchSizer sizer;
  sizer.add<float>(static_cast<std::size_t>(count));
  sizer.add<float>(static_cast<std::size_t>(count));
  sizer.add<std::int32_t>(static_cast<std::size_t>(count));
  sizer.add<std::uint8_t>(static_cast<std::size_t>(count));
  return sizer.bytes();
}

std::size_t solver_workspace_bytes(std::int32_t count,
                                   const ResolvedOptions& resolved) {
  std::size_t basis_elements = 0;
  std::size_t projected_elements = 0;
  if (!checked_product(static_cast<std::size_t>(count),
                       static_cast<std::size_t>(resolved.basis_size),
                       &basis_elements) ||
      !checked_product(static_cast<std::size_t>(resolved.basis_size),
                       static_cast<std::size_t>(resolved.basis_size),
                       &projected_elements)) {
    return 0;
  }
  ScratchSizer sizer;
  sizer.add<float>(basis_elements);
  sizer.add<double>(static_cast<std::size_t>(count));
  sizer.add<double>(static_cast<std::size_t>(count));
  sizer.add<double>(static_cast<std::size_t>(count));
  sizer.add<double>(projected_elements);
  sizer.add<double>(projected_elements);
  sizer.add<double>(projected_elements);
  sizer.add<double>(static_cast<std::size_t>(resolved.basis_size));
  sizer.add<std::int32_t>(static_cast<std::size_t>(resolved.basis_size));
  if (resolved.maximum_restarts > 0) {
    std::size_t restart_elements = 0;
    if (!checked_product(static_cast<std::size_t>(count),
                         static_cast<std::size_t>(resolved.retained + 1),
                         &restart_elements)) {
      return 0;
    }
    sizer.add<float>(restart_elements);
  }
  return sizer.bytes();
}

std::int32_t find_root(std::int32_t* parents, std::int32_t vertex) {
  std::int32_t root = vertex;
  while (parents[root] != root) {
    root = parents[root];
  }
  while (parents[vertex] != vertex) {
    const std::int32_t next = parents[vertex];
    parents[vertex] = root;
    vertex = next;
  }
  return root;
}

void unite(std::int32_t* parents,
           std::uint8_t* ranks,
           std::int32_t left,
           std::int32_t right) {
  left = find_root(parents, left);
  right = find_root(parents, right);
  if (left == right) {
    return;
  }
  if (ranks[left] < ranks[right]) {
    std::swap(left, right);
  }
  parents[right] = left;
  if (ranks[left] == ranks[right]) {
    ++ranks[left];
  }
}

Status build_normalized_graph(const std::int32_t* row_offsets,
                              const std::int32_t* columns,
                              float* weights,
                              std::int32_t count,
                              std::int32_t edge_count,
                              void* workspace,
                              std::size_t workspace_size) {
  if (row_offsets[0] != 0 || row_offsets[count] != edge_count) {
    return Status::kInvalidGraph;
  }

  ScratchArena scratch(workspace, workspace_size);
  float* const degree = scratch.allocate<float>(static_cast<std::size_t>(count));
  float* const inverse_sqrt_degree =
      scratch.allocate<float>(static_cast<std::size_t>(count));
  std::int32_t* const parents =
      scratch.allocate<std::int32_t>(static_cast<std::size_t>(count));
  std::uint8_t* const ranks =
      scratch.allocate<std::uint8_t>(static_cast<std::size_t>(count));
  if (!scratch.valid()) {
    return Status::kInvalidArgument;
  }
  std::fill_n(degree, count, 0.0f);
  std::fill_n(ranks, count, static_cast<std::uint8_t>(0));
  for (std::int32_t row = 0; row < count; ++row) {
    parents[row] = row;
  }

  for (std::int32_t row = 0; row < count; ++row) {
    const std::int32_t begin = row_offsets[row];
    const std::int32_t end = row_offsets[row + 1];
    if (begin < 0 || end < begin || end > edge_count) {
      return Status::kInvalidGraph;
    }
    for (std::int32_t edge = begin; edge < end; ++edge) {
      const std::int32_t column = columns[edge];
      const float weight = weights[edge];
      if (column < 0 || column >= count || !std::isfinite(weight) ||
          weight < 0.0f) {
        return Status::kInvalidGraph;
      }
      /*
       * scipy.sparse graph.sum(axis=0) accumulates the float32 graph into a
       * float32 result. Keep the same rounding point before sqrt.
       */
      degree[column] += weight;
      if (weight > 0.0f) {
        unite(parents, ranks, row, column);
      }
    }
  }

  const std::int32_t first_root = find_root(parents, 0);
  for (std::int32_t row = 0; row < count; ++row) {
    if (!(degree[row] > 0.0f) || !std::isfinite(degree[row])) {
      return Status::kInvalidGraph;
    }
    if (find_root(parents, row) != first_root) {
      return Status::kDisconnectedGraph;
    }
  }

  for (std::int32_t row = 0; row < count; ++row) {
    inverse_sqrt_degree[row] = 1.0f / std::sqrt(degree[row]);
  }

  for (std::int32_t row = 0; row < count; ++row) {
    const float row_scale = inverse_sqrt_degree[row];
    for (std::int32_t edge = row_offsets[row]; edge < row_offsets[row + 1];
         ++edge) {
      /*
       * scipy's D * graph * D performs two float32 sparse multiplications.
       * Keep the intermediate assignment so native compilers cannot reassociate
       * the expression into a differently rounded three-factor product.
       */
      const float row_scaled = row_scale * weights[edge];
      weights[edge] = row_scaled * inverse_sqrt_degree[columns[edge]];
    }
  }
  return Status::kSuccess;
}

void apply_laplacian(const std::int32_t* row_offsets,
                     const std::int32_t* columns,
                     const float* normalized_weights,
                     std::int32_t count,
                     const double* input,
                     double* output) {
  for (std::int32_t row = 0; row < count; ++row) {
    double value = input[row];
    for (std::int32_t edge = row_offsets[row]; edge < row_offsets[row + 1];
         ++edge) {
      value -=
          static_cast<double>(normalized_weights[edge]) *
          input[columns[edge]];
    }
    output[row] = value;
  }
}

void apply_laplacian_mixed(const std::int32_t* row_offsets,
                           const std::int32_t* columns,
                           const float* normalized_weights,
                           std::int32_t count,
                           const float* input,
                           double* output) {
  for (std::int32_t row = 0; row < count; ++row) {
    double value = static_cast<double>(input[row]);
    for (std::int32_t edge = row_offsets[row]; edge < row_offsets[row + 1];
         ++edge) {
      value -=
          static_cast<double>(normalized_weights[edge]) *
          static_cast<double>(input[columns[edge]]);
    }
    output[row] = value;
  }
}

double dot_mixed(const float* left, const double* right, std::int32_t count) {
  double result = 0.0;
  for (std::int32_t row = 0; row < count; ++row) {
    result += static_cast<double>(left[row]) * right[row];
  }
  return result;
}

double squared_norm(const double* values, std::int32_t count) {
  double result = 0.0;
  for (std::int32_t row = 0; row < count; ++row) {
    result += values[row] * values[row];
  }
  return result;
}

void subtract_scaled_mixed(double* target,
                           const float* source,
                           double scale,
                           std::int32_t count) {
  for (std::int32_t row = 0; row < count; ++row) {
    target[row] -= scale * static_cast<double>(source[row]);
  }
}

/*
 * Cyclic Jacobi diagonalization of the compact Rayleigh-Ritz problem.
 * Thick restart makes the projected matrix arrowhead-plus-tridiagonal rather
 * than purely tridiagonal. At Senko's default basis size (160), this dense
 * float64 solve is small compared with sparse matvec and reorthogonalization.
 */
bool diagonalize_symmetric(double* matrix,
                           std::int32_t size,
                           double* eigenvalues,
                           double* eigenvectors) {
  std::fill_n(eigenvectors, static_cast<std::size_t>(size) * size, 0.0);
  for (std::int32_t index = 0; index < size; ++index) {
    eigenvectors[static_cast<std::size_t>(index) * size + index] = 1.0;
  }

  bool converged = false;
  for (std::int32_t sweep = 0; sweep < 40; ++sweep) {
    double largest = 0.0;
    double diagonal_scale = 0.0;
    for (std::int32_t index = 0; index < size; ++index) {
      diagonal_scale =
          std::max(diagonal_scale,
                   std::abs(matrix[static_cast<std::size_t>(index) * size +
                                   index]));
    }
    for (std::int32_t left = 0; left < size - 1; ++left) {
      for (std::int32_t right = left + 1; right < size; ++right) {
        const std::size_t left_right =
            static_cast<std::size_t>(left) * size + right;
        const double off_diagonal = matrix[left_right];
        largest = std::max(largest, std::abs(off_diagonal));
        const double left_diagonal =
            matrix[static_cast<std::size_t>(left) * size + left];
        const double right_diagonal =
            matrix[static_cast<std::size_t>(right) * size + right];
        if (std::abs(off_diagonal) <=
            std::numeric_limits<double>::epsilon() * 8.0 *
                (std::abs(left_diagonal) + std::abs(right_diagonal) + 1.0)) {
          continue;
        }

        const double tau =
            (right_diagonal - left_diagonal) / (2.0 * off_diagonal);
        const double tangent =
            std::copysign(1.0, tau) /
            (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
        const double sine = tangent * cosine;
        matrix[static_cast<std::size_t>(left) * size + left] =
            left_diagonal - tangent * off_diagonal;
        matrix[static_cast<std::size_t>(right) * size + right] =
            right_diagonal + tangent * off_diagonal;
        matrix[left_right] = 0.0;
        matrix[static_cast<std::size_t>(right) * size + left] = 0.0;

        for (std::int32_t index = 0; index < size; ++index) {
          if (index == left || index == right) {
            continue;
          }
          const std::size_t index_left =
              static_cast<std::size_t>(index) * size + left;
          const std::size_t index_right =
              static_cast<std::size_t>(index) * size + right;
          const double value_left = matrix[index_left];
          const double value_right = matrix[index_right];
          const double rotated_left =
              cosine * value_left - sine * value_right;
          const double rotated_right =
              sine * value_left + cosine * value_right;
          matrix[index_left] = rotated_left;
          matrix[static_cast<std::size_t>(left) * size + index] =
              rotated_left;
          matrix[index_right] = rotated_right;
          matrix[static_cast<std::size_t>(right) * size + index] =
              rotated_right;
        }
        for (std::int32_t row = 0; row < size; ++row) {
          const std::size_t vector_left =
              static_cast<std::size_t>(row) * size + left;
          const std::size_t vector_right =
              static_cast<std::size_t>(row) * size + right;
          const double value_left = eigenvectors[vector_left];
          const double value_right = eigenvectors[vector_right];
          eigenvectors[vector_left] = cosine * value_left - sine * value_right;
          eigenvectors[vector_right] = sine * value_left + cosine * value_right;
        }
      }
    }
    if (largest <= std::max(1.0, diagonal_scale) * 1.0e-13) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    return false;
  }
  for (std::int32_t index = 0; index < size; ++index) {
    eigenvalues[index] =
        matrix[static_cast<std::size_t>(index) * size + index];
  }
  return true;
}

std::int32_t extend_krylov_basis(
    const std::int32_t* row_offsets,
    const std::int32_t* columns,
    const float* normalized_weights,
    std::int32_t count,
    std::int32_t maximum_basis_size,
    std::int32_t initial_columns,
    float* basis,
    double* projected,
    double* candidate) {
  std::int32_t completed = initial_columns;
  while (completed < maximum_basis_size) {
    const std::int32_t source_column = completed - 1;
    const float* source =
        basis + static_cast<std::size_t>(source_column) * count;
    apply_laplacian_mixed(row_offsets, columns, normalized_weights, count,
                          source, candidate);

    for (std::int32_t prior = 0; prior < completed; ++prior) {
      const float* prior_vector =
          basis + static_cast<std::size_t>(prior) * count;
      const double projection = dot_mixed(prior_vector, candidate, count);
      projected[static_cast<std::size_t>(prior) * maximum_basis_size +
                source_column] = projection;
      projected[static_cast<std::size_t>(source_column) * maximum_basis_size +
                prior] = projection;
      subtract_scaled_mixed(candidate, prior_vector, projection, count);
    }
    /*
     * A correction pass keeps a float32-retained basis orthogonal enough for
     * the float64 Rayleigh-Ritz solve. Accumulate the tiny corrections into
     * the projected column rather than discarding them.
     */
    for (std::int32_t prior = 0; prior < completed; ++prior) {
      const float* prior_vector =
          basis + static_cast<std::size_t>(prior) * count;
      const double correction = dot_mixed(prior_vector, candidate, count);
      projected[static_cast<std::size_t>(prior) * maximum_basis_size +
                source_column] += correction;
      projected[static_cast<std::size_t>(source_column) * maximum_basis_size +
                prior] += correction;
      subtract_scaled_mixed(candidate, prior_vector, correction, count);
    }

    const double norm = std::sqrt(squared_norm(candidate, count));
    if (!std::isfinite(norm) || !(norm > kBreakdownTolerance)) {
      break;
    }
    float* next =
        basis + static_cast<std::size_t>(completed) * count;
    const double inverse_norm = 1.0 / norm;
    for (std::int32_t row = 0; row < count; ++row) {
      next[row] = static_cast<float>(
          candidate[row] * inverse_norm);
    }
    projected[static_cast<std::size_t>(source_column) * maximum_basis_size +
              completed] = norm;
    projected[static_cast<std::size_t>(completed) * maximum_basis_size +
              source_column] = norm;
    ++completed;
  }

  if (completed > 0) {
    const std::int32_t last = completed - 1;
    const float* source =
        basis + static_cast<std::size_t>(last) * count;
    apply_laplacian_mixed(row_offsets, columns, normalized_weights, count,
                          source, candidate);
    for (std::int32_t prior = 0; prior < completed; ++prior) {
      const double projection = dot_mixed(
          basis + static_cast<std::size_t>(prior) * count, candidate, count);
      projected[static_cast<std::size_t>(prior) * maximum_basis_size + last] =
          projection;
      projected[static_cast<std::size_t>(last) * maximum_basis_size + prior] =
          projection;
    }
  }
  return completed;
}

void reconstruct_ritz_vector(const float* basis,
                             std::int32_t count,
                             std::int32_t basis_size,
                             const double* small_vectors,
                             std::int32_t small_column,
                             double* output) {
  std::fill_n(output, count, 0.0);
  for (std::int32_t basis_column = 0; basis_column < basis_size;
       ++basis_column) {
    const double scale =
        small_vectors[static_cast<std::size_t>(basis_column) * basis_size +
                      small_column];
    if (scale == 0.0) {
      continue;
    }
    const float* source =
        basis + static_cast<std::size_t>(basis_column) * count;
    for (std::int32_t row = 0; row < count; ++row) {
      output[row] += static_cast<double>(source[row]) * scale;
    }
  }
}

void canonicalize_sign(double* vectors,
                       std::int32_t count,
                       std::int32_t dim,
                       std::int32_t column) {
  std::int32_t pivot = 0;
  double maximum = 0.0;
  for (std::int32_t row = 0; row < count; ++row) {
    const double magnitude =
        std::abs(vectors[static_cast<std::size_t>(row) * dim + column]);
    if (magnitude > maximum) {
      maximum = magnitude;
      pivot = row;
    }
  }
  if (vectors[static_cast<std::size_t>(pivot) * dim + column] >= 0.0) {
    return;
  }
  for (std::int32_t row = 0; row < count; ++row) {
    const std::size_t index =
        static_cast<std::size_t>(row) * dim + column;
    vectors[index] = -vectors[index];
  }
}

}  // namespace

std::size_t workspace_bytes(std::int32_t count,
                            std::int32_t edge_count,
                            std::int32_t dim,
                            const Options& options) {
  ResolvedOptions resolved;
  if (!resolve_options(count, edge_count, dim, options, &resolved)) {
    return 0;
  }
  const std::size_t graph_bytes = graph_workspace_bytes(count);
  const std::size_t solver_bytes = solver_workspace_bytes(count, resolved);
  if (graph_bytes == 0 || solver_bytes == 0) {
    return 0;
  }
  return std::max(graph_bytes, solver_bytes);
}

Status initialize_connected_graph(const std::int32_t* row_offsets,
                                  const std::int32_t* columns,
                                  float* weights,
                                  std::int32_t count,
                                  std::int32_t edge_count,
                                  std::int32_t dim,
                                  double* output_vectors,
                                  double* output_eigenvalues,
                                  void* workspace,
                                  std::size_t workspace_size,
                                  const Options& options,
                                  Stats* stats) {
  if (stats != nullptr) {
    *stats = {};
  }
  ResolvedOptions resolved;
  const std::size_t required_workspace =
      workspace_bytes(count, edge_count, dim, options);
  if (row_offsets == nullptr || columns == nullptr || weights == nullptr ||
      output_vectors == nullptr || workspace == nullptr ||
      required_workspace == 0 || workspace_size < required_workspace ||
      reinterpret_cast<std::uintptr_t>(workspace) % alignof(double) != 0 ||
      !resolve_options(count, edge_count, dim, options, &resolved)) {
    return Status::kInvalidArgument;
  }

  std::size_t basis_elements = 0;
  std::size_t projected_elements = 0;
  if (!checked_product(
          static_cast<std::size_t>(count),
          static_cast<std::size_t>(resolved.basis_size), &basis_elements) ||
      !checked_product(
          static_cast<std::size_t>(resolved.basis_size),
          static_cast<std::size_t>(resolved.basis_size),
          &projected_elements)) {
    return Status::kInvalidArgument;
  }

  const Status graph_status = build_normalized_graph(
      row_offsets, columns, weights, count, edge_count, workspace,
      workspace_size);
  if (graph_status != Status::kSuccess) {
    return graph_status;
  }

  ScratchArena scratch(workspace, workspace_size);
  float* const basis = scratch.allocate<float>(basis_elements);
  double* const candidate =
      scratch.allocate<double>(static_cast<std::size_t>(count));
  double* const residual_product =
      scratch.allocate<double>(static_cast<std::size_t>(count));
  double* const restart_direction =
      scratch.allocate<double>(static_cast<std::size_t>(count));
  double* const projected = scratch.allocate<double>(projected_elements);
  double* const compact_projected =
      scratch.allocate<double>(projected_elements);
  double* const small_vectors = scratch.allocate<double>(projected_elements);
  double* const small_values = scratch.allocate<double>(
      static_cast<std::size_t>(resolved.basis_size));
  std::int32_t* const order = scratch.allocate<std::int32_t>(
      static_cast<std::size_t>(resolved.basis_size));
  float* restart_basis = nullptr;
  if (resolved.maximum_restarts > 0) {
    std::size_t restart_elements = 0;
    if (!checked_product(
            static_cast<std::size_t>(count),
            static_cast<std::size_t>(resolved.retained + 1),
            &restart_elements)) {
      return Status::kInvalidArgument;
    }
    restart_basis = scratch.allocate<float>(restart_elements);
  }
  if (!scratch.valid()) {
    return Status::kInvalidArgument;
  }
  std::fill_n(basis, basis_elements, 0.0f);
  std::fill_n(projected, projected_elements, 0.0);

  const float initial_value =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(count)));
  std::fill_n(basis, count, initial_value);

  std::int32_t initial_columns = 1;
  std::int32_t actual_basis_size = 0;
  std::int32_t restart_count = 0;
  std::int32_t converged = 0;
  double maximum_residual = std::numeric_limits<double>::infinity();
  double smallest_eigenvalue = 0.0;
  double largest_returned_eigenvalue = 0.0;

  for (std::int32_t cycle = 0; cycle <= resolved.maximum_restarts; ++cycle) {
    actual_basis_size = extend_krylov_basis(
        row_offsets, columns, weights, count, resolved.basis_size,
        initial_columns, basis, projected, candidate);
    if (actual_basis_size < resolved.requested) {
      return Status::kNumericalFailure;
    }

    for (std::int32_t row = 0; row < actual_basis_size; ++row) {
      for (std::int32_t column = 0; column < actual_basis_size; ++column) {
        compact_projected[static_cast<std::size_t>(row) *
                              actual_basis_size +
                          column] =
            projected[static_cast<std::size_t>(row) *
                          resolved.basis_size +
                      column];
      }
    }
    if (!diagonalize_symmetric(compact_projected, actual_basis_size,
                               small_values, small_vectors)) {
      return Status::kNumericalFailure;
    }

    /*
     * Stable insertion sort is allocation-free. At the fixed maximum of 160
     * entries it is negligible next to a sparse matvec, and the strict
     * comparison preserves stable_sort's tie ordering.
     */
    for (std::int32_t index = 0; index < actual_basis_size; ++index) {
      order[index] = index;
    }
    for (std::int32_t index = 1; index < actual_basis_size; ++index) {
      const std::int32_t value = order[index];
      std::int32_t position = index;
      while (position > 0 &&
             small_values[value] < small_values[order[position - 1]]) {
        order[position] = order[position - 1];
        --position;
      }
      order[position] = value;
    }

    const bool can_restart =
        cycle < resolved.maximum_restarts &&
        resolved.retained >= resolved.requested &&
        resolved.retained < actual_basis_size;
    const std::int32_t evaluated =
        can_restart ? resolved.retained : resolved.requested;
    if (can_restart && restart_basis == nullptr) {
      return Status::kInvalidArgument;
    }

    converged = 0;
    maximum_residual = 0.0;
    double largest_restart_residual = -1.0;
    for (std::int32_t output_column = 0; output_column < evaluated;
         ++output_column) {
      const std::int32_t small_column = order[output_column];
      reconstruct_ritz_vector(basis, count, actual_basis_size, small_vectors,
                              small_column, candidate);
      const double eigenvalue = small_values[small_column];
      apply_laplacian(row_offsets, columns, weights, count, candidate,
                      residual_product);
      double squared_residual = 0.0;
      for (std::int32_t row = 0; row < count; ++row) {
        residual_product[row] -= eigenvalue * candidate[row];
        const double value = residual_product[row];
        squared_residual += value * value;
      }
      const double residual = std::sqrt(squared_residual);
      if (!std::isfinite(residual)) {
        return Status::kNumericalFailure;
      }

      if (output_column < resolved.requested) {
        maximum_residual = std::max(maximum_residual, residual);
        if (residual <= options.tolerance) {
          ++converged;
        }
        if (output_column == 0) {
          smallest_eigenvalue = eigenvalue;
        } else {
          const std::int32_t target_column = output_column - 1;
          for (std::int32_t row = 0; row < count; ++row) {
            output_vectors[static_cast<std::size_t>(row) * dim +
                           target_column] = candidate[row];
          }
          canonicalize_sign(output_vectors, count, dim, target_column);
          if (output_eigenvalues != nullptr) {
            output_eigenvalues[target_column] = eigenvalue;
          }
          if (output_column == resolved.requested - 1) {
            largest_returned_eigenvalue = eigenvalue;
          }
        }
      }

      if (can_restart) {
        float* restart_column =
            restart_basis + static_cast<std::size_t>(output_column) * count;
        for (std::int32_t row = 0; row < count; ++row) {
          restart_column[row] = static_cast<float>(candidate[row]);
        }
        if (squared_residual > largest_restart_residual) {
          largest_restart_residual = squared_residual;
          std::copy_n(residual_product, count, restart_direction);
        }
      }
    }

    if (converged == resolved.requested || !can_restart) {
      break;
    }

    /*
     * All Ritz residuals of exact Lanczos arithmetic share one direction.
     * With a float32 retained basis they differ slightly, so retain the
     * largest residual and explicitly orthogonalize it against every kept
     * vector.
     */
    for (std::int32_t pass = 0; pass < 2; ++pass) {
      for (std::int32_t prior = 0; prior < resolved.retained; ++prior) {
        const float* prior_vector =
            restart_basis + static_cast<std::size_t>(prior) * count;
        const double projection =
            dot_mixed(prior_vector, restart_direction, count);
        subtract_scaled_mixed(restart_direction, prior_vector, projection,
                              count);
      }
    }
    const double restart_norm = std::sqrt(squared_norm(restart_direction,
                                                       count));
    if (!std::isfinite(restart_norm) ||
        !(restart_norm > kBreakdownTolerance)) {
      return Status::kNumericalFailure;
    }
    float* restart_column =
        restart_basis + static_cast<std::size_t>(resolved.retained) * count;
    const double inverse_restart_norm = 1.0 / restart_norm;
    for (std::int32_t row = 0; row < count; ++row) {
      restart_column[row] = static_cast<float>(
          restart_direction[row] * inverse_restart_norm);
    }

    std::fill_n(basis, basis_elements, 0.0f);
    std::copy_n(
        restart_basis,
        static_cast<std::size_t>(count) * (resolved.retained + 1), basis);
    std::fill_n(projected, projected_elements, 0.0);
    for (std::int32_t index = 0; index < resolved.retained; ++index) {
      const std::int32_t small_column = order[index];
      projected[static_cast<std::size_t>(index) * resolved.basis_size +
                index] = small_values[small_column];
    }
    initial_columns = resolved.retained + 1;
    ++restart_count;
  }

  if (stats != nullptr) {
    stats->requested_eigenpairs = resolved.requested;
    stats->basis_size = actual_basis_size;
    stats->restart_count = restart_count;
    stats->converged_eigenpairs = converged;
    stats->maximum_residual = maximum_residual;
    stats->smallest_eigenvalue = smallest_eigenvalue;
    stats->largest_returned_eigenvalue = largest_returned_eigenvalue;
    stats->peak_working_bytes = required_workspace;
  }

  if (converged != resolved.requested && !options.accept_unconverged) {
    return Status::kDidNotConverge;
  }
  return Status::kSuccess;
}

const char* status_message(Status status) {
  switch (status) {
    case Status::kSuccess:
      return "success";
    case Status::kInvalidArgument:
      return "invalid argument";
    case Status::kInvalidGraph:
      return "invalid graph";
    case Status::kDisconnectedGraph:
      return "disconnected graph requires UMAP's multi-component layout";
    case Status::kNumericalFailure:
      return "numerical failure";
    case Status::kDidNotConverge:
      return "eigenpair residual exceeded tolerance";
  }
  return "unknown status";
}

}  // namespace senko::umap_spectral
