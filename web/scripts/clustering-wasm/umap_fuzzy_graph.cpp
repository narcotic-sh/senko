#include "umap_fuzzy_graph.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(SENKO_UMAP_FUZZY_GRAPH_HOST_TEST)
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#endif

/*
 * Algorithmic reference:
 *   umap-learn 0.5.12, umap/umap_.py:
 *     smooth_knn_dist, compute_membership_strengths, fuzzy_simplicial_set
 *   BSD-3-Clause, Copyright (c) 2017 Leland McInnes.
 *
 * The sparse construction and memory layout here are purpose-built for
 * WebAssembly. Two compact CSR views replace SciPy's temporary COO matrices.
 */

namespace senko::umap_fuzzy_graph {
namespace {

constexpr size_t kAlignment = 16;
constexpr float kSmoothTolerance = 1.0e-5f;
constexpr float kMinimumDistanceScale = 1.0e-3f;
constexpr int kSmoothIterations = 64;

bool CheckedAdd(size_t left, size_t right, size_t* result) {
  if (right > std::numeric_limits<size_t>::max() - left) return false;
  *result = left + right;
  return true;
}

bool CheckedMultiply(size_t left, size_t right, size_t* result) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool AlignSize(size_t value, size_t* result) {
  constexpr size_t mask = kAlignment - 1;
  if (value > std::numeric_limits<size_t>::max() - mask) return false;
  *result = (value + mask) & ~mask;
  return true;
}

bool AddArrayBytes(size_t* total, size_t count, size_t item_bytes) {
  size_t aligned = 0;
  size_t bytes = 0;
  if (!AlignSize(*total, &aligned) ||
      !CheckedMultiply(count, item_bytes, &bytes) ||
      !CheckedAdd(aligned, bytes, total)) {
    return false;
  }
  return true;
}

class Workspace {
 public:
  Workspace(void* data, size_t size)
      : data_(static_cast<uint8_t*>(data)), size_(size) {}

  template <typename T>
  T* Allocate(size_t count) {
    size_t aligned = 0;
    size_t bytes = 0;
    size_t end = 0;
    if (!AlignSize(cursor_, &aligned) ||
        !CheckedMultiply(count, sizeof(T), &bytes) ||
        !CheckedAdd(aligned, bytes, &end) || end > size_) {
      return nullptr;
    }
    cursor_ = end;
    return reinterpret_cast<T*>(data_ + aligned);
  }

 private:
  uint8_t* data_;
  size_t size_;
  size_t cursor_ = 0;
};

bool IsAligned(const void* pointer) {
  return (reinterpret_cast<uintptr_t>(pointer) & (kAlignment - 1)) == 0;
}

bool ValidShape(int count, int neighbor_count, size_t* input_length) {
  if (count <= 0 || neighbor_count <= 1 || neighbor_count > count ||
      !CheckedMultiply(static_cast<size_t>(count),
                       static_cast<size_t>(neighbor_count), input_length)) {
    return false;
  }
  return *input_length <= static_cast<size_t>(INT32_MAX);
}

float MeanFloat32(const float* values, int count) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  int index = 0;
  for (; index + 3 < count; index += 4) {
    sum0 += values[index];
    sum1 += values[index + 1];
    sum2 += values[index + 2];
    sum3 += values[index + 3];
  }
  float result = (sum0 + sum1) + (sum2 + sum3);
  for (; index < count; ++index) result += values[index];
  return result / static_cast<float>(count);
}

bool SmoothKnnDistances(const float* distances, int count, int neighbor_count,
                        float* sigmas, float* rhos) {
  const size_t input_length =
      static_cast<size_t>(count) * neighbor_count;
  for (size_t index = 0; index < input_length; ++index) {
    if (std::isnan(distances[index]) || distances[index] < 0.0f) return false;
  }
  const float global_mean =
      MeanFloat32(distances, static_cast<int>(input_length));
  const double target = std::log2(static_cast<double>(neighbor_count));

  for (int row = 0; row < count; ++row) {
    const float* row_distances =
        distances + static_cast<size_t>(row) * neighbor_count;
    float rho = 0.0f;
    for (int rank = 0; rank < neighbor_count; ++rank) {
      if (row_distances[rank] > 0.0f) {
        rho = row_distances[rank];
        break;
      }
    }
    rhos[row] = rho;

    float lower = 0.0f;
    float upper = FLT_MAX;
    float middle = 1.0f;
    for (int iteration = 0; iteration < kSmoothIterations; ++iteration) {
      float probability_sum = 0.0f;
      for (int rank = 1; rank < neighbor_count; ++rank) {
        const float adjusted = row_distances[rank] - rho;
        probability_sum +=
            adjusted > 0.0f ? std::exp(-(adjusted / middle)) : 1.0f;
      }
      if (std::fabs(static_cast<double>(probability_sum) - target) <
          kSmoothTolerance) {
        break;
      }
      if (probability_sum > target) {
        upper = middle;
        middle = (lower + upper) / 2.0f;
      } else {
        lower = middle;
        if (upper >= FLT_MAX) {
          middle *= 2.0f;
        } else {
          middle = (lower + upper) / 2.0f;
        }
      }
    }

    const float scale_mean =
        rho > 0.0f ? MeanFloat32(row_distances, neighbor_count) : global_mean;
    const float minimum = kMinimumDistanceScale * scale_mean;
    sigmas[row] = middle < minimum ? minimum : middle;
  }
  return true;
}

float MembershipStrength(int row, int column, float distance, float sigma,
                         float rho) {
  if (row == column) return 0.0f;
  const float adjusted = distance - rho;
  if (adjusted <= 0.0f || sigma == 0.0f) return 1.0f;
  return std::exp(-(adjusted / sigma));
}

void SortRow(int32_t* columns, float* values, int begin, int end) {
  for (int position = begin + 1; position < end; ++position) {
    const int32_t column = columns[position];
    const float value = values[position];
    int insertion = position;
    while (insertion > begin && column < columns[insertion - 1]) {
      columns[insertion] = columns[insertion - 1];
      values[insertion] = values[insertion - 1];
      --insertion;
    }
    columns[insertion] = column;
    values[insertion] = value;
  }
}

float FuzzyUnion(float forward, float reverse) {
  const float sum = forward + reverse;
  const float product = forward * reverse;
  return sum - product;
}

}  // namespace

size_t MaximumCsrEntries(int count, int neighbor_count) {
  size_t input_length = 0;
  size_t result = 0;
  if (!ValidShape(count, neighbor_count, &input_length) ||
      !CheckedMultiply(input_length, size_t{2}, &result)) {
    return 0;
  }
  return result;
}

size_t WorkspaceBytes(int count, int neighbor_count) {
  size_t input_length = 0;
  if (!ValidShape(count, neighbor_count, &input_length)) return 0;
  const size_t row_offsets = static_cast<size_t>(count) + 1;
  size_t result = 0;
  if (!AddArrayBytes(&result, row_offsets, sizeof(int32_t)) ||
      !AddArrayBytes(&result, input_length, sizeof(int32_t)) ||
      !AddArrayBytes(&result, input_length, sizeof(float)) ||
      !AddArrayBytes(&result, row_offsets, sizeof(int32_t)) ||
      !AddArrayBytes(&result, input_length, sizeof(int32_t)) ||
      !AddArrayBytes(&result, input_length, sizeof(float)) ||
      !AddArrayBytes(&result, static_cast<size_t>(count), sizeof(int32_t))) {
    return 0;
  }
  return result;
}

int BuildCsr(const int32_t* knn_indices, const float* knn_distances, int count,
             int neighbor_count, void* workspace, size_t workspace_bytes,
             float* output_sigmas, float* output_rhos,
             int32_t* output_row_offsets, int32_t* output_column_indices,
             float* output_values, size_t output_capacity,
             size_t* output_entry_count) {
  size_t input_length = 0;
  if (!knn_indices || !knn_distances || !workspace ||
      !IsAligned(workspace) || !output_sigmas || !output_rhos ||
      !output_row_offsets || !output_column_indices || !output_values ||
      !output_entry_count ||
      !ValidShape(count, neighbor_count, &input_length)) {
    return kInvalidArgument;
  }
  const size_t required_workspace = WorkspaceBytes(count, neighbor_count);
  if (required_workspace == 0) return kSizeOverflow;
  if (workspace_bytes < required_workspace) return kInsufficientWorkspace;
  if (!SmoothKnnDistances(knn_distances, count, neighbor_count, output_sigmas,
                          output_rhos)) {
    return kInvalidArgument;
  }

  Workspace allocator(workspace, workspace_bytes);
  const size_t offset_count = static_cast<size_t>(count) + 1;
  int32_t* outgoing_offsets = allocator.Allocate<int32_t>(offset_count);
  int32_t* outgoing_columns = allocator.Allocate<int32_t>(input_length);
  float* outgoing_values = allocator.Allocate<float>(input_length);
  int32_t* incoming_offsets = allocator.Allocate<int32_t>(offset_count);
  int32_t* incoming_columns = allocator.Allocate<int32_t>(input_length);
  float* incoming_values = allocator.Allocate<float>(input_length);
  int32_t* counts = allocator.Allocate<int32_t>(static_cast<size_t>(count));
  if (!outgoing_offsets || !outgoing_columns || !outgoing_values ||
      !incoming_offsets || !incoming_columns || !incoming_values || !counts) {
    return kInsufficientWorkspace;
  }

  outgoing_offsets[0] = 0;
  size_t directed_count = 0;
  for (int row = 0; row < count; ++row) {
    int row_count = 0;
    const size_t input_offset = static_cast<size_t>(row) * neighbor_count;
    for (int rank = 0; rank < neighbor_count; ++rank) {
      const int column = knn_indices[input_offset + rank];
      if (column < -1 || column >= count) return kInvalidArgument;
      if (column < 0 || column == row) continue;
      for (int previous = 0; previous < rank; ++previous) {
        if (knn_indices[input_offset + previous] == column) {
          return kInvalidArgument;
        }
      }
      const float value = MembershipStrength(
          row, column, knn_distances[input_offset + rank], output_sigmas[row],
          output_rhos[row]);
      if (value > 0.0f) ++row_count;
    }
    directed_count += static_cast<size_t>(row_count);
    if (directed_count > static_cast<size_t>(INT32_MAX)) return kSizeOverflow;
    outgoing_offsets[row + 1] = static_cast<int32_t>(directed_count);
  }

  for (int row = 0; row < count; ++row) {
    int write = outgoing_offsets[row];
    const size_t input_offset = static_cast<size_t>(row) * neighbor_count;
    for (int rank = 0; rank < neighbor_count; ++rank) {
      const int column = knn_indices[input_offset + rank];
      if (column < 0 || column == row) continue;
      const float value = MembershipStrength(
          row, column, knn_distances[input_offset + rank], output_sigmas[row],
          output_rhos[row]);
      if (value == 0.0f) continue;
      outgoing_columns[write] = column;
      outgoing_values[write] = value;
      ++write;
    }
    SortRow(outgoing_columns, outgoing_values, outgoing_offsets[row],
            outgoing_offsets[row + 1]);
  }

  std::fill_n(counts, static_cast<size_t>(count), 0);
  for (size_t edge = 0; edge < directed_count; ++edge) {
    ++counts[outgoing_columns[edge]];
  }
  incoming_offsets[0] = 0;
  size_t incoming_count = 0;
  for (int row = 0; row < count; ++row) {
    incoming_count += static_cast<size_t>(counts[row]);
    if (incoming_count > static_cast<size_t>(INT32_MAX)) return kSizeOverflow;
    incoming_offsets[row + 1] = static_cast<int32_t>(incoming_count);
    counts[row] = incoming_offsets[row];
  }
  for (int source = 0; source < count; ++source) {
    for (int edge = outgoing_offsets[source];
         edge < outgoing_offsets[source + 1]; ++edge) {
      const int target = outgoing_columns[edge];
      const int write = counts[target]++;
      incoming_columns[write] = source;
      incoming_values[write] = outgoing_values[edge];
    }
  }

  output_row_offsets[0] = 0;
  size_t required_entries = 0;
  for (int row = 0; row < count; ++row) {
    int outgoing = outgoing_offsets[row];
    const int outgoing_end = outgoing_offsets[row + 1];
    int incoming = incoming_offsets[row];
    const int incoming_end = incoming_offsets[row + 1];
    while (outgoing < outgoing_end || incoming < incoming_end) {
      const int outgoing_column =
          outgoing < outgoing_end ? outgoing_columns[outgoing] : INT32_MAX;
      const int incoming_column =
          incoming < incoming_end ? incoming_columns[incoming] : INT32_MAX;
      ++required_entries;
      if (outgoing_column <= incoming_column) ++outgoing;
      if (incoming_column <= outgoing_column) ++incoming;
    }
    if (required_entries > static_cast<size_t>(INT32_MAX)) return kSizeOverflow;
    output_row_offsets[row + 1] = static_cast<int32_t>(required_entries);
  }
  *output_entry_count = required_entries;
  if (output_capacity < required_entries) return kInsufficientOutput;

  size_t write = 0;
  for (int row = 0; row < count; ++row) {
    int outgoing = outgoing_offsets[row];
    const int outgoing_end = outgoing_offsets[row + 1];
    int incoming = incoming_offsets[row];
    const int incoming_end = incoming_offsets[row + 1];
    while (outgoing < outgoing_end || incoming < incoming_end) {
      const int outgoing_column =
          outgoing < outgoing_end ? outgoing_columns[outgoing] : INT32_MAX;
      const int incoming_column =
          incoming < incoming_end ? incoming_columns[incoming] : INT32_MAX;
      if (outgoing_column == incoming_column) {
        output_column_indices[write] = outgoing_column;
        output_values[write] =
            FuzzyUnion(outgoing_values[outgoing], incoming_values[incoming]);
        ++outgoing;
        ++incoming;
      } else if (outgoing_column < incoming_column) {
        output_column_indices[write] = outgoing_column;
        output_values[write] = outgoing_values[outgoing++];
      } else {
        output_column_indices[write] = incoming_column;
        output_values[write] = incoming_values[incoming++];
      }
      ++write;
    }
  }
  return kSuccess;
}

}  // namespace senko::umap_fuzzy_graph

#if defined(SENKO_UMAP_FUZZY_GRAPH_HOST_TEST)

namespace {

template <typename T>
bool ReadExact(const char* path, size_t count, std::vector<T>* output) {
  output->resize(count);
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  input.read(reinterpret_cast<char*>(output->data()),
             static_cast<std::streamsize>(count * sizeof(T)));
  return input.good() && input.peek() == std::ifstream::traits_type::eof();
}

struct ErrorSummary {
  double sum = 0.0;
  double maximum = 0.0;
  size_t count = 0;

  void Add(float left, float right) {
    const double error =
        std::fabs(static_cast<double>(left) - static_cast<double>(right));
    sum += error;
    maximum = std::max(maximum, error);
    ++count;
  }
};

}  // namespace

/*
 * Opt-in native-fixture diagnostic:
 *
 * clang++ -std=c++20 -O3 -DSENKO_UMAP_FUZZY_GRAPH_HOST_TEST \
 *   web/scripts/clustering-wasm/umap_fuzzy_graph.cpp \
 *   -o .research/umap-fuzzy-graph-host-test
 *
 * .research/umap-fuzzy-graph-host-test KNN_I KNN_D N K SIGMA RHO \
 *   GRAPH_INDPTR GRAPH_INDICES GRAPH_DATA
 */
int main(int argc, char** argv) {
  if (argc != 10) {
    std::cerr << "usage: KNN_I KNN_D N K SIGMA RHO GRAPH_INDPTR "
                 "GRAPH_INDICES GRAPH_DATA\n";
    return 2;
  }
  const int count = std::stoi(argv[3]);
  const int neighbor_count = std::stoi(argv[4]);
  if (count <= 0 || neighbor_count <= 1) return 2;
  const size_t input_length =
      static_cast<size_t>(count) * neighbor_count;
  std::vector<int32_t> knn_indices;
  std::vector<float> knn_distances;
  std::vector<float> reference_sigmas;
  std::vector<float> reference_rhos;
  std::vector<int32_t> reference_offsets;
  if (!ReadExact(argv[1], input_length, &knn_indices) ||
      !ReadExact(argv[2], input_length, &knn_distances) ||
      !ReadExact(argv[5], static_cast<size_t>(count), &reference_sigmas) ||
      !ReadExact(argv[6], static_cast<size_t>(count), &reference_rhos) ||
      !ReadExact(argv[7], static_cast<size_t>(count) + 1,
                 &reference_offsets)) {
    std::cerr << "failed to read fixed-size fixture input\n";
    return 3;
  }
  if (reference_offsets.front() != 0 || reference_offsets.back() < 0) return 3;
  const size_t reference_entries =
      static_cast<size_t>(reference_offsets.back());
  std::vector<int32_t> reference_columns;
  std::vector<float> reference_values;
  if (!ReadExact(argv[8], reference_entries, &reference_columns) ||
      !ReadExact(argv[9], reference_entries, &reference_values)) {
    std::cerr << "failed to read graph fixture\n";
    return 3;
  }

  const size_t workspace_bytes =
      senko::umap_fuzzy_graph::WorkspaceBytes(count, neighbor_count);
  const size_t output_capacity =
      senko::umap_fuzzy_graph::MaximumCsrEntries(count, neighbor_count);
  std::vector<uint8_t> workspace(workspace_bytes);
  std::vector<float> sigmas(count);
  std::vector<float> rhos(count);
  std::vector<int32_t> offsets(static_cast<size_t>(count) + 1);
  std::vector<int32_t> columns(output_capacity);
  std::vector<float> values(output_capacity);
  size_t entries = 0;
  const auto started = std::chrono::steady_clock::now();
  const int status = senko::umap_fuzzy_graph::BuildCsr(
      knn_indices.data(), knn_distances.data(), count, neighbor_count,
      workspace.data(), workspace.size(), sigmas.data(), rhos.data(),
      offsets.data(), columns.data(), values.data(), values.size(), &entries);
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  if (status != senko::umap_fuzzy_graph::kSuccess) {
    std::cerr << "fuzzy graph failed with status " << status << "\n";
    return 4;
  }

  ErrorSummary sigma_error;
  ErrorSummary rho_error;
  for (int row = 0; row < count; ++row) {
    sigma_error.Add(sigmas[row], reference_sigmas[row]);
    rho_error.Add(rhos[row], reference_rhos[row]);
  }
  size_t support_intersection = 0;
  size_t support_union = 0;
  size_t exact_weight_count = 0;
  size_t reference_zero_count = 0;
  ErrorSummary shared_weight_error;
  for (int row = 0; row < count; ++row) {
    int candidate = offsets[row];
    const int candidate_end = offsets[row + 1];
    int reference = reference_offsets[row];
    const int reference_end = reference_offsets[row + 1];
    while (candidate < candidate_end || reference < reference_end) {
      const int candidate_column =
          candidate < candidate_end ? columns[candidate] : INT32_MAX;
      const int reference_column =
          reference < reference_end ? reference_columns[reference] : INT32_MAX;
      ++support_union;
      if (candidate_column == reference_column) {
        ++support_intersection;
        shared_weight_error.Add(values[candidate], reference_values[reference]);
        exact_weight_count +=
            values[candidate] == reference_values[reference];
        reference_zero_count += reference_values[reference] == 0.0f;
        ++candidate;
        ++reference;
      } else if (candidate_column < reference_column) {
        ++candidate;
      } else {
        ++reference;
      }
    }
  }
  const double support_jaccard =
      static_cast<double>(support_intersection) / support_union;
  std::cout << "{"
            << "\"status\":" << status << ","
            << "\"elapsedMs\":" << elapsed_ms << ","
            << "\"workspaceBytes\":" << workspace_bytes << ","
            << "\"outputCapacityBytes\":"
            << output_capacity * (sizeof(int32_t) + sizeof(float)) << ","
            << "\"entries\":" << entries << ","
            << "\"referenceEntries\":" << reference_entries << ","
            << "\"supportJaccard\":" << support_jaccard << ","
            << "\"referenceStoredZeros\":" << reference_zero_count << ","
            << "\"sigmaMeanAbsoluteError\":"
            << sigma_error.sum / sigma_error.count << ","
            << "\"sigmaMaxAbsoluteError\":" << sigma_error.maximum << ","
            << "\"rhoMeanAbsoluteError\":"
            << rho_error.sum / rho_error.count << ","
            << "\"rhoMaxAbsoluteError\":" << rho_error.maximum << ","
            << "\"sharedWeightExactFraction\":"
            << static_cast<double>(exact_weight_count) /
                   shared_weight_error.count
            << ","
            << "\"sharedWeightMeanAbsoluteError\":"
            << shared_weight_error.sum / shared_weight_error.count << ","
            << "\"sharedWeightMaxAbsoluteError\":"
            << shared_weight_error.maximum << "}\n";
  return support_jaccard >= 0.99 && sigma_error.maximum <= 1.0e-4 &&
                 rho_error.maximum <= 1.0e-6
             ? 0
             : 5;
}

#endif  // SENKO_UMAP_FUZZY_GRAPH_HOST_TEST
