#include "umap_spectral.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace senko::umap_spectral {
namespace {

constexpr double kBreakdownTolerance = 1.0e-14;

bool checked_product(std::size_t left, std::size_t right, std::size_t* result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

std::int32_t find_root(std::vector<std::int32_t>& parents,
                       std::int32_t vertex) {
  std::int32_t root = vertex;
  while (parents[static_cast<std::size_t>(root)] != root) {
    root = parents[static_cast<std::size_t>(root)];
  }
  while (parents[static_cast<std::size_t>(vertex)] != vertex) {
    const std::int32_t next = parents[static_cast<std::size_t>(vertex)];
    parents[static_cast<std::size_t>(vertex)] = root;
    vertex = next;
  }
  return root;
}

void unite(std::vector<std::int32_t>& parents,
           std::vector<std::uint8_t>& ranks,
           std::int32_t left,
           std::int32_t right) {
  left = find_root(parents, left);
  right = find_root(parents, right);
  if (left == right) {
    return;
  }
  if (ranks[static_cast<std::size_t>(left)] <
      ranks[static_cast<std::size_t>(right)]) {
    std::swap(left, right);
  }
  parents[static_cast<std::size_t>(right)] = left;
  if (ranks[static_cast<std::size_t>(left)] ==
      ranks[static_cast<std::size_t>(right)]) {
    ++ranks[static_cast<std::size_t>(left)];
  }
}

struct NormalizedGraph {
  std::vector<float> values;
  std::size_t working_bytes = 0;
};

Status build_normalized_graph(const std::int32_t* row_offsets,
                              const std::int32_t* columns,
                              const float* weights,
                              std::int32_t count,
                              std::int32_t edge_count,
                              NormalizedGraph* result) {
  if (row_offsets[0] != 0 || row_offsets[count] != edge_count) {
    return Status::kInvalidGraph;
  }

  std::vector<float> degree(static_cast<std::size_t>(count), 0.0f);
  std::vector<std::int32_t> parents(static_cast<std::size_t>(count));
  std::vector<std::uint8_t> ranks(static_cast<std::size_t>(count), 0);
  std::iota(parents.begin(), parents.end(), 0);

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
      degree[static_cast<std::size_t>(column)] += weight;
      if (weight > 0.0f) {
        unite(parents, ranks, row, column);
      }
    }
  }

  const std::int32_t first_root = find_root(parents, 0);
  for (std::int32_t row = 0; row < count; ++row) {
    if (!(degree[static_cast<std::size_t>(row)] > 0.0f) ||
        !std::isfinite(degree[static_cast<std::size_t>(row)])) {
      return Status::kInvalidGraph;
    }
    if (find_root(parents, row) != first_root) {
      return Status::kDisconnectedGraph;
    }
  }

  std::vector<float> inverse_sqrt_degree(static_cast<std::size_t>(count));
  for (std::int32_t row = 0; row < count; ++row) {
    inverse_sqrt_degree[static_cast<std::size_t>(row)] =
        1.0f / std::sqrt(degree[static_cast<std::size_t>(row)]);
  }

  result->values.resize(static_cast<std::size_t>(edge_count));
  for (std::int32_t row = 0; row < count; ++row) {
    const float row_scale =
        inverse_sqrt_degree[static_cast<std::size_t>(row)];
    for (std::int32_t edge = row_offsets[row]; edge < row_offsets[row + 1];
         ++edge) {
      /*
       * scipy's D * graph * D performs two float32 sparse multiplications.
       * Keep the intermediate assignment so native compilers cannot reassociate
       * the expression into a differently rounded three-factor product.
       */
      const float row_scaled = row_scale * weights[edge];
      result->values[static_cast<std::size_t>(edge)] =
          row_scaled *
          inverse_sqrt_degree[static_cast<std::size_t>(columns[edge])];
    }
  }
  result->working_bytes =
      degree.size() * sizeof(float) +
      inverse_sqrt_degree.size() * sizeof(float) +
      parents.size() * sizeof(std::int32_t) +
      ranks.size() * sizeof(std::uint8_t) +
      result->values.size() * sizeof(float);
  return Status::kSuccess;
}

void apply_laplacian(const std::int32_t* row_offsets,
                     const std::int32_t* columns,
                     const std::vector<float>& normalized_weights,
                     std::int32_t count,
                     const double* input,
                     double* output) {
  for (std::int32_t row = 0; row < count; ++row) {
    double value = input[row];
    for (std::int32_t edge = row_offsets[row]; edge < row_offsets[row + 1];
         ++edge) {
      value -=
          static_cast<double>(
              normalized_weights[static_cast<std::size_t>(edge)]) *
          input[columns[edge]];
    }
    output[row] = value;
  }
}

void apply_laplacian_mixed(const std::int32_t* row_offsets,
                           const std::int32_t* columns,
                           const std::vector<float>& normalized_weights,
                           std::int32_t count,
                           const float* input,
                           double* output) {
  for (std::int32_t row = 0; row < count; ++row) {
    double value = static_cast<double>(input[row]);
    for (std::int32_t edge = row_offsets[row]; edge < row_offsets[row + 1];
         ++edge) {
      value -=
          static_cast<double>(
              normalized_weights[static_cast<std::size_t>(edge)]) *
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
bool diagonalize_symmetric(const std::vector<double>& input,
                           std::int32_t size,
                           std::vector<double>* eigenvalues,
                           std::vector<double>* eigenvectors) {
  std::vector<double> matrix = input;
  eigenvectors->assign(static_cast<std::size_t>(size) * size, 0.0);
  for (std::int32_t index = 0; index < size; ++index) {
    (*eigenvectors)[static_cast<std::size_t>(index) * size + index] = 1.0;
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
          const double value_left = (*eigenvectors)[vector_left];
          const double value_right = (*eigenvectors)[vector_right];
          (*eigenvectors)[vector_left] =
              cosine * value_left - sine * value_right;
          (*eigenvectors)[vector_right] =
              sine * value_left + cosine * value_right;
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
  eigenvalues->resize(static_cast<std::size_t>(size));
  for (std::int32_t index = 0; index < size; ++index) {
    (*eigenvalues)[static_cast<std::size_t>(index)] =
        matrix[static_cast<std::size_t>(index) * size + index];
  }
  return true;
}

std::int32_t extend_krylov_basis(
    const std::int32_t* row_offsets,
    const std::int32_t* columns,
    const std::vector<float>& normalized_weights,
    std::int32_t count,
    std::int32_t maximum_basis_size,
    std::int32_t initial_columns,
    std::vector<float>* basis,
    std::vector<double>* projected,
    std::vector<double>* candidate) {
  std::int32_t completed = initial_columns;
  while (completed < maximum_basis_size) {
    const std::int32_t source_column = completed - 1;
    const float* source =
        basis->data() + static_cast<std::size_t>(source_column) * count;
    apply_laplacian_mixed(row_offsets, columns, normalized_weights, count,
                          source, candidate->data());

    for (std::int32_t prior = 0; prior < completed; ++prior) {
      const float* prior_vector =
          basis->data() + static_cast<std::size_t>(prior) * count;
      const double projection =
          dot_mixed(prior_vector, candidate->data(), count);
      (*projected)[static_cast<std::size_t>(prior) * maximum_basis_size +
                   source_column] = projection;
      (*projected)[static_cast<std::size_t>(source_column) *
                       maximum_basis_size +
                   prior] = projection;
      subtract_scaled_mixed(candidate->data(), prior_vector, projection, count);
    }
    /*
     * A correction pass keeps a float32-retained basis orthogonal enough for
     * the float64 Rayleigh-Ritz solve. Accumulate the tiny corrections into
     * the projected column rather than discarding them.
     */
    for (std::int32_t prior = 0; prior < completed; ++prior) {
      const float* prior_vector =
          basis->data() + static_cast<std::size_t>(prior) * count;
      const double correction =
          dot_mixed(prior_vector, candidate->data(), count);
      (*projected)[static_cast<std::size_t>(prior) * maximum_basis_size +
                   source_column] += correction;
      (*projected)[static_cast<std::size_t>(source_column) *
                       maximum_basis_size +
                   prior] += correction;
      subtract_scaled_mixed(candidate->data(), prior_vector, correction, count);
    }

    const double norm =
        std::sqrt(squared_norm(candidate->data(), count));
    if (!std::isfinite(norm) || !(norm > kBreakdownTolerance)) {
      break;
    }
    float* next =
        basis->data() + static_cast<std::size_t>(completed) * count;
    const double inverse_norm = 1.0 / norm;
    for (std::int32_t row = 0; row < count; ++row) {
      next[row] = static_cast<float>(
          (*candidate)[static_cast<std::size_t>(row)] * inverse_norm);
    }
    (*projected)[static_cast<std::size_t>(source_column) *
                     maximum_basis_size +
                 completed] = norm;
    (*projected)[static_cast<std::size_t>(completed) * maximum_basis_size +
                 source_column] = norm;
    ++completed;
  }

  if (completed > 0) {
    const std::int32_t last = completed - 1;
    const float* source =
        basis->data() + static_cast<std::size_t>(last) * count;
    apply_laplacian_mixed(row_offsets, columns, normalized_weights, count,
                          source, candidate->data());
    for (std::int32_t prior = 0; prior < completed; ++prior) {
      const double projection = dot_mixed(
          basis->data() + static_cast<std::size_t>(prior) * count,
          candidate->data(), count);
      (*projected)[static_cast<std::size_t>(prior) * maximum_basis_size +
                   last] = projection;
      (*projected)[static_cast<std::size_t>(last) * maximum_basis_size +
                   prior] = projection;
    }
  }
  return completed;
}

void reconstruct_ritz_vector(const std::vector<float>& basis,
                             std::int32_t count,
                             std::int32_t basis_size,
                             const std::vector<double>& small_vectors,
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
        basis.data() + static_cast<std::size_t>(basis_column) * count;
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

Status initialize_connected_graph(const std::int32_t* row_offsets,
                                  const std::int32_t* columns,
                                  const float* weights,
                                  std::int32_t count,
                                  std::int32_t edge_count,
                                  std::int32_t dim,
                                  double* output_vectors,
                                  double* output_eigenvalues,
                                  const Options& options,
                                  Stats* stats) {
  if (stats != nullptr) {
    *stats = {};
  }
  if (row_offsets == nullptr || columns == nullptr || weights == nullptr ||
      output_vectors == nullptr || count < 3 || edge_count < 1 || dim < 1 ||
      dim > count - 2 || !std::isfinite(options.tolerance) ||
      options.tolerance <= 0.0 || options.maximum_basis_size < 0 ||
      options.retained_ritz_vectors < 0 || options.maximum_restarts < 0) {
    return Status::kInvalidArgument;
  }

  const std::int32_t requested = dim + 1;
  const std::int32_t automatic_basis = 2 * requested + 38;
  const std::int32_t basis_size =
      std::min(count,
               options.maximum_basis_size == 0
                   ? automatic_basis
                   : std::max(options.maximum_basis_size, requested + 1));
  if (basis_size <= requested) {
    return Status::kInvalidArgument;
  }

  std::size_t basis_elements = 0;
  if (!checked_product(static_cast<std::size_t>(count),
                       static_cast<std::size_t>(basis_size),
                       &basis_elements)) {
    return Status::kInvalidArgument;
  }
  const std::int32_t requested_retained =
      options.retained_ritz_vectors == 0 ? requested + 11
                                         : options.retained_ritz_vectors;
  const std::int32_t retained =
      std::min(requested_retained, basis_size - 2);
  const std::int32_t maximum_restarts =
      retained >= requested ? options.maximum_restarts : 0;

  NormalizedGraph graph;
  const Status graph_status = build_normalized_graph(
      row_offsets, columns, weights, count, edge_count, &graph);
  if (graph_status != Status::kSuccess) {
    return graph_status;
  }

  std::vector<float> basis(basis_elements, 0.0f);
  std::vector<double> candidate(static_cast<std::size_t>(count), 0.0);
  std::vector<double> residual_product(static_cast<std::size_t>(count), 0.0);
  std::vector<double> restart_direction(static_cast<std::size_t>(count), 0.0);
  std::vector<double> projected(static_cast<std::size_t>(basis_size) *
                                    basis_size,
                                0.0);

  const float initial_value =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(count)));
  std::fill_n(basis.data(), count, initial_value);

  std::int32_t initial_columns = 1;
  std::int32_t actual_basis_size = 0;
  std::int32_t restart_count = 0;
  std::int32_t converged = 0;
  double maximum_residual = std::numeric_limits<double>::infinity();
  double smallest_eigenvalue = 0.0;
  double largest_returned_eigenvalue = 0.0;
  std::size_t maximum_restart_basis_bytes = 0;
  std::size_t maximum_compact_eigen_bytes = 0;

  for (std::int32_t cycle = 0; cycle <= maximum_restarts; ++cycle) {
    actual_basis_size = extend_krylov_basis(
        row_offsets, columns, graph.values, count, basis_size, initial_columns,
        &basis, &projected, &candidate);
    if (actual_basis_size < requested) {
      return Status::kNumericalFailure;
    }

    std::vector<double> compact_projected(
        static_cast<std::size_t>(actual_basis_size) * actual_basis_size);
    for (std::int32_t row = 0; row < actual_basis_size; ++row) {
      for (std::int32_t column = 0; column < actual_basis_size; ++column) {
        compact_projected[static_cast<std::size_t>(row) *
                              actual_basis_size +
                          column] =
            projected[static_cast<std::size_t>(row) * basis_size + column];
      }
    }
    std::vector<double> small_values;
    std::vector<double> small_vectors;
    if (!diagonalize_symmetric(compact_projected, actual_basis_size,
                               &small_values, &small_vectors)) {
      return Status::kNumericalFailure;
    }
    maximum_compact_eigen_bytes =
        std::max(maximum_compact_eigen_bytes,
                 compact_projected.size() * sizeof(double) * 2 +
                     small_vectors.size() * sizeof(double) +
                     small_values.size() * sizeof(double));

    std::vector<std::int32_t> order(
        static_cast<std::size_t>(actual_basis_size));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::int32_t left, std::int32_t right) {
                       return small_values[static_cast<std::size_t>(left)] <
                              small_values[static_cast<std::size_t>(right)];
                     });

    const bool can_restart =
        cycle < maximum_restarts && retained >= requested &&
        retained < actual_basis_size;
    const std::int32_t evaluated =
        can_restart ? retained : requested;
    std::vector<float> restart_basis;
    if (can_restart) {
      restart_basis.resize(static_cast<std::size_t>(count) * (retained + 1));
      maximum_restart_basis_bytes =
          std::max(maximum_restart_basis_bytes,
                   restart_basis.size() * sizeof(float));
    }

    converged = 0;
    maximum_residual = 0.0;
    double largest_restart_residual = -1.0;
    for (std::int32_t output_column = 0; output_column < evaluated;
         ++output_column) {
      const std::int32_t small_column =
          order[static_cast<std::size_t>(output_column)];
      reconstruct_ritz_vector(basis, count, actual_basis_size, small_vectors,
                              small_column, candidate.data());
      const double eigenvalue =
          small_values[static_cast<std::size_t>(small_column)];
      apply_laplacian(row_offsets, columns, graph.values, count,
                      candidate.data(), residual_product.data());
      double squared_residual = 0.0;
      for (std::int32_t row = 0; row < count; ++row) {
        residual_product[static_cast<std::size_t>(row)] -=
            eigenvalue * candidate[static_cast<std::size_t>(row)];
        const double value =
            residual_product[static_cast<std::size_t>(row)];
        squared_residual += value * value;
      }
      const double residual = std::sqrt(squared_residual);
      if (!std::isfinite(residual)) {
        return Status::kNumericalFailure;
      }

      if (output_column < requested) {
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
                           target_column] =
                candidate[static_cast<std::size_t>(row)];
          }
          canonicalize_sign(output_vectors, count, dim, target_column);
          if (output_eigenvalues != nullptr) {
            output_eigenvalues[target_column] = eigenvalue;
          }
          if (output_column == requested - 1) {
            largest_returned_eigenvalue = eigenvalue;
          }
        }
      }

      if (can_restart) {
        float* restart_column =
            restart_basis.data() +
            static_cast<std::size_t>(output_column) * count;
        for (std::int32_t row = 0; row < count; ++row) {
          restart_column[row] = static_cast<float>(
              candidate[static_cast<std::size_t>(row)]);
        }
        if (squared_residual > largest_restart_residual) {
          largest_restart_residual = squared_residual;
          std::copy(residual_product.begin(), residual_product.end(),
                    restart_direction.begin());
        }
      }
    }

    if (converged == requested || !can_restart) {
      break;
    }

    /*
     * All Ritz residuals of exact Lanczos arithmetic share one direction.
     * With a float32 retained basis they differ slightly, so retain the
     * largest residual and explicitly orthogonalize it against every kept
     * vector.
     */
    for (std::int32_t pass = 0; pass < 2; ++pass) {
      for (std::int32_t prior = 0; prior < retained; ++prior) {
        const float* prior_vector =
            restart_basis.data() +
            static_cast<std::size_t>(prior) * count;
        const double projection =
            dot_mixed(prior_vector, restart_direction.data(), count);
        subtract_scaled_mixed(restart_direction.data(), prior_vector,
                              projection, count);
      }
    }
    const double restart_norm =
        std::sqrt(squared_norm(restart_direction.data(), count));
    if (!std::isfinite(restart_norm) ||
        !(restart_norm > kBreakdownTolerance)) {
      return Status::kNumericalFailure;
    }
    float* restart_column =
        restart_basis.data() + static_cast<std::size_t>(retained) * count;
    const double inverse_restart_norm = 1.0 / restart_norm;
    for (std::int32_t row = 0; row < count; ++row) {
      restart_column[row] = static_cast<float>(
          restart_direction[static_cast<std::size_t>(row)] *
          inverse_restart_norm);
    }

    /*
     * Free the old full basis before growing the retained vectors back to the
     * configured capacity. The live transition is one full basis plus the
     * compact retained basis, not two full bases.
     */
    std::vector<float>().swap(basis);
    basis.assign(basis_elements, 0.0f);
    std::copy(restart_basis.begin(), restart_basis.end(), basis.begin());
    std::fill(projected.begin(), projected.end(), 0.0);
    for (std::int32_t index = 0; index < retained; ++index) {
      const std::int32_t small_column =
          order[static_cast<std::size_t>(index)];
      projected[static_cast<std::size_t>(index) * basis_size + index] =
          small_values[static_cast<std::size_t>(small_column)];
    }
    initial_columns = retained + 1;
    ++restart_count;
  }

  if (stats != nullptr) {
    stats->requested_eigenpairs = requested;
    stats->basis_size = actual_basis_size;
    stats->restart_count = restart_count;
    stats->converged_eigenpairs = converged;
    stats->maximum_residual = maximum_residual;
    stats->smallest_eigenvalue = smallest_eigenvalue;
    stats->largest_returned_eigenvalue = largest_returned_eigenvalue;
    stats->peak_working_bytes =
        graph.working_bytes + basis.size() * sizeof(float) +
        maximum_restart_basis_bytes +
        candidate.size() * sizeof(double) +
        residual_product.size() * sizeof(double) +
        restart_direction.size() * sizeof(double) +
        projected.size() * sizeof(double) + maximum_compact_eigen_bytes +
        static_cast<std::size_t>(basis_size) * sizeof(std::int32_t);
  }

  if (converged != requested && !options.accept_unconverged) {
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
