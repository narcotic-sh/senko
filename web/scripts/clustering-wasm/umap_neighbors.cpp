#include "umap_neighbors.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <limits>

#if defined(SENKO_UMAP_NEIGHBORS_HOST_TEST)
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#endif

/*
 * Algorithmic references:
 *   umap-learn 0.5.12, umap/umap_.py and umap/utils.py
 *     BSD-3-Clause, Copyright (c) 2017 Leland McInnes.
 *   pynndescent 0.6.0, pynndescent_.py, rp_trees.py, utils.py, distances.py
 *     BSD-2-Clause, Copyright Leland McInnes.
 *
 * This file is a purpose-built C++ implementation for dense float32 CAM++
 * embeddings. Memory ownership, data layout, and execution are original and
 * designed for a caller-provided bounded WebAssembly arena.
 */

namespace senko::umap_neighbors {
namespace {

constexpr size_t kAlignment = 16;
constexpr float kRpEpsilon = 1.0e-8f;
constexpr int kCandidateBlockSize = 16384;
// Keep the quadratic optimization bounded. This covers the one-hour fixture
// (2,039,544 bytes) and inputs through 8,192 rows; larger inputs retain the
// predecessor's linear-memory candidate traversal.
constexpr size_t kMaximumPairBitsetBytes = 4u * 1024u * 1024u;
constexpr uint64_t kUint32Mask = 0xffff'ffffULL;
constexpr int64_t kRandomIntLow = -2147483647LL;
constexpr uint64_t kRandomIntRangeMax = 4294967292ULL;

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

bool ValidShape(int count, int dimension, int neighbor_count) {
  if (count <= 0 || dimension <= 0 || neighbor_count <= 0 ||
      neighbor_count > count) {
    return false;
  }
  size_t ignored = 0;
  return CheckedMultiply(static_cast<size_t>(count),
                         static_cast<size_t>(dimension), &ignored) &&
         CheckedMultiply(static_cast<size_t>(count),
                         static_cast<size_t>(neighbor_count), &ignored);
}

bool IsWorkspaceAligned(const void* workspace) {
  return (reinterpret_cast<uintptr_t>(workspace) & (kAlignment - 1)) == 0;
}

constexpr size_t PairBitsetWordCount(int count) {
  if (count <= 1) return 0;
  const uint64_t pair_count =
      static_cast<uint64_t>(count) * static_cast<uint64_t>(count - 1) / 2u;
  const uint64_t word_count = (pair_count + 31u) / 32u;
  if (word_count >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      word_count * sizeof(uint32_t) > kMaximumPairBitsetBytes) {
    return 0;
  }
  return static_cast<size_t>(word_count);
}

static_assert(PairBitsetWordCount(8192) != 0);
static_assert(PairBitsetWordCount(8193) == 0);

bool ShouldEvaluateUnorderedPair(uint32_t* evaluated_pairs, int count,
                                 int left, int right) {
  // Preserve PyNNDescent's self-comparison behavior. For distinct endpoints,
  // every attempt pushes the same distance into both heaps. Heap thresholds
  // only decrease, so a repeated pair can never produce an update after its
  // first attempt, even in a later snapshot iteration.
  if (!evaluated_pairs || left == right) return true;
  const uint64_t lower =
      static_cast<uint64_t>(left < right ? left : right);
  const uint64_t upper =
      static_cast<uint64_t>(left < right ? right : left);
  const uint64_t pair_index =
      lower * (static_cast<uint64_t>(count) * 2u - lower - 1u) / 2u +
      upper - lower - 1u;
  const size_t word = static_cast<size_t>(pair_index >> 5u);
  const uint32_t mask = 1u << static_cast<uint32_t>(pair_index & 31u);
  if ((evaluated_pairs[word] & mask) != 0u) return false;
  evaluated_pairs[word] |= mask;
  return true;
}

float SumSquares(const float* values, int dimension) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  int column = 0;
  for (; column + 3 < dimension; column += 4) {
    sum0 += values[column] * values[column];
    sum1 += values[column + 1] * values[column + 1];
    sum2 += values[column + 2] * values[column + 2];
    sum3 += values[column + 3] * values[column + 3];
  }
  float result = (sum0 + sum1) + (sum2 + sum3);
  for (; column < dimension; ++column) {
    result += values[column] * values[column];
  }
  return result;
}

float Dot(const float* left, const float* right, int dimension) {
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  int column = 0;
  for (; column + 3 < dimension; column += 4) {
    sum0 += left[column] * right[column];
    sum1 += left[column + 1] * right[column + 1];
    sum2 += left[column + 2] * right[column + 2];
    sum3 += left[column + 3] * right[column + 3];
  }
  float result = (sum0 + sum1) + (sum2 + sum3);
  for (; column < dimension; ++column) {
    result += left[column] * right[column];
  }
  return result;
}

bool NormalizeRows(const float* values, int count, int dimension,
                   float* normalized) {
  for (int row = 0; row < count; ++row) {
    const float* source = values + static_cast<size_t>(row) * dimension;
    float* target = normalized + static_cast<size_t>(row) * dimension;
    const float squared_norm = SumSquares(source, dimension);
    if (!std::isfinite(squared_norm)) return false;
    const float norm = std::sqrt(squared_norm);
    const float scale = norm == 0.0f ? 1.0f : 1.0f / norm;
    for (int column = 0; column < dimension; ++column) {
      target[column] = source[column] * scale;
    }
  }
  return true;
}

void InsertSortedNeighbor(int32_t* indices, float* distances, int size,
                          int candidate, float distance) {
  const int last = size - 1;
  if (distance > distances[last] ||
      (distance == distances[last] && indices[last] >= 0 &&
       candidate >= indices[last])) {
    return;
  }
  int position = last;
  while (position > 0 &&
         (distance < distances[position - 1] ||
          (distance == distances[position - 1] &&
           candidate < indices[position - 1]))) {
    distances[position] = distances[position - 1];
    indices[position] = indices[position - 1];
    --position;
  }
  distances[position] = distance;
  indices[position] = candidate;
}

float ClippedCosineDistance(const float* normalized_left,
                            const float* normalized_right, int dimension) {
  float result = 1.0f - Dot(normalized_left, normalized_right, dimension);
  if (result < 0.0f) result = 0.0f;
  if (result > 2.0f) result = 2.0f;
  return result;
}

/*
 * Legacy NumPy RandomState's MT19937 and bounded uint32 sampling. This is
 * intentionally not C++ std::mt19937: the seeding is the same, but keeping the
 * implementation here makes the diagnostic sequence stable across toolchains.
 */
class NumpyRandomState {
 public:
  explicit NumpyRandomState(uint32_t seed) { Seed(seed); }

  int64_t NextSignedRangeValue() {
    uint32_t sample = 0;
    do {
      sample = NextUint32();
    } while (static_cast<uint64_t>(sample) > kRandomIntRangeMax);
    return kRandomIntLow + static_cast<int64_t>(sample);
  }

 private:
  void Seed(uint32_t seed) {
    state_[0] = seed;
    for (int index = 1; index < 624; ++index) {
      state_[index] =
          1812433253u * (state_[index - 1] ^ (state_[index - 1] >> 30)) +
          static_cast<uint32_t>(index);
    }
    cursor_ = 624;
  }

  void Twist() {
    constexpr uint32_t upper_mask = 0x80000000u;
    constexpr uint32_t lower_mask = 0x7fffffffu;
    for (int index = 0; index < 624; ++index) {
      const uint32_t bits =
          (state_[index] & upper_mask) |
          (state_[(index + 1) % 624] & lower_mask);
      state_[index] = state_[(index + 397) % 624] ^ (bits >> 1) ^
                      ((bits & 1u) == 0 ? 0u : 0x9908b0dfu);
    }
    cursor_ = 0;
  }

  uint32_t NextUint32() {
    if (cursor_ == 624) Twist();
    uint32_t result = state_[cursor_++];
    result ^= result >> 11;
    result ^= (result << 7) & 0x9d2c5680u;
    result ^= (result << 15) & 0xefc60000u;
    result ^= result >> 18;
    return result;
  }

  uint32_t state_[624]{};
  int cursor_ = 624;
};

struct TauState {
  int64_t values[3];
};

int64_t ArithmeticShiftRight(uint64_t bits, int shift) {
  if ((bits & (uint64_t{1} << 63)) == 0) {
    return static_cast<int64_t>(bits >> shift);
  }
  const uint64_t fill = ~uint64_t{0} << (64 - shift);
  return static_cast<int64_t>((bits >> shift) | fill);
}

int32_t TauRandInt(TauState* state) {
  const uint64_t original0 = static_cast<uint64_t>(state->values[0]);
  const uint64_t original1 = static_cast<uint64_t>(state->values[1]);
  const uint64_t original2 = static_cast<uint64_t>(state->values[2]);
  const uint64_t left0 = ((original0 & 4294967294ULL) << 12) & kUint32Mask;
  const uint64_t mixed0 =
      (((original0 << 13) & kUint32Mask) ^ original0);
  state->values[0] =
      static_cast<int64_t>(left0) ^ ArithmeticShiftRight(mixed0, 19);

  const uint64_t left1 = ((original1 & 4294967288ULL) << 4) & kUint32Mask;
  const uint64_t mixed1 =
      (((original1 << 2) & kUint32Mask) ^ original1);
  state->values[1] =
      static_cast<int64_t>(left1) ^ ArithmeticShiftRight(mixed1, 25);

  const uint64_t left2 = ((original2 & 4294967280ULL) << 17) & kUint32Mask;
  const uint64_t mixed2 =
      (((original2 << 3) & kUint32Mask) ^ original2);
  state->values[2] =
      static_cast<int64_t>(left2) ^ ArithmeticShiftRight(mixed2, 11);

  return static_cast<int32_t>(
      static_cast<uint64_t>(state->values[0]) ^
      static_cast<uint64_t>(state->values[1]) ^
      static_cast<uint64_t>(state->values[2]));
}

float TauRand(TauState* state) {
  const int32_t integer = TauRandInt(state);
  return static_cast<float>(
      std::fabs(static_cast<double>(integer) / 2147483647.0));
}

int PositiveModulo(int32_t value, int divisor) {
  int result = value % divisor;
  if (result < 0) result += divisor;
  return result;
}

int AbsoluteModulo(int32_t value, int divisor) {
  const uint32_t magnitude =
      value < 0 ? uint32_t{0} - static_cast<uint32_t>(value)
                : static_cast<uint32_t>(value);
  return static_cast<int>(magnitude % static_cast<uint32_t>(divisor));
}

void FillTauState(NumpyRandomState* random, TauState* state) {
  for (int index = 0; index < 3; ++index) {
    state->values[index] = random->NextSignedRangeValue();
  }
}

int ResolveTreeCount(int count, const ApproximateOptions& options) {
  if (options.n_trees > 0) return options.n_trees;
  const double suggested =
      5.0 + std::nearbyint(std::sqrt(static_cast<double>(count)) / 20.0);
  return std::min(64, static_cast<int>(suggested));
}

int ResolveIterationCount(int count, const ApproximateOptions& options) {
  if (options.n_iters > 0) return options.n_iters;
  return std::max(
      5, static_cast<int>(std::nearbyint(std::log2(static_cast<double>(count)))));
}

bool ValidApproximateOptions(int count, const ApproximateOptions& options) {
  const int tree_count = ResolveTreeCount(count, options);
  const int iteration_count = ResolveIterationCount(count, options);
  return tree_count > 0 && tree_count <= 64 && iteration_count > 0 &&
         iteration_count <= 256 && options.max_candidates > 0 &&
         options.max_candidates <= 256 && options.leaf_size > 1 &&
         options.leaf_size <= 1024 && options.max_tree_depth > 0 &&
         options.max_tree_depth <= 1024 && std::isfinite(options.delta) &&
         options.delta >= 0.0f;
}

int HeapPush(int32_t* indices, float* distances, uint8_t* flags, int size,
             float distance, int candidate, uint8_t flag) {
  if (distance >= distances[0]) return 0;
  for (int position = 0; position < size; ++position) {
    if (indices[position] == candidate) return 0;
  }
  distances[0] = distance;
  indices[0] = candidate;
  flags[0] = flag;
  int position = 0;
  while (true) {
    const int left = position * 2 + 1;
    const int right = left + 1;
    if (left >= size) break;
    int swap = left;
    if (right < size && distances[right] > distances[left]) swap = right;
    if (distance >= distances[swap]) break;
    distances[position] = distances[swap];
    indices[position] = indices[swap];
    flags[position] = flags[swap];
    position = swap;
  }
  distances[position] = distance;
  indices[position] = candidate;
  flags[position] = flag;
  return 1;
}

int CandidatePush(int32_t* indices, float* priorities, int size, float priority,
                  int candidate) {
  if (priority >= priorities[0]) return 0;
  for (int position = 0; position < size; ++position) {
    if (indices[position] == candidate) return 0;
  }
  priorities[0] = priority;
  indices[0] = candidate;
  int position = 0;
  while (true) {
    const int left = position * 2 + 1;
    const int right = left + 1;
    if (left >= size) break;
    int swap = left;
    if (right < size && priorities[right] > priorities[left]) swap = right;
    if (priority >= priorities[swap]) break;
    priorities[position] = priorities[swap];
    indices[position] = indices[swap];
    position = swap;
  }
  priorities[position] = priority;
  indices[position] = candidate;
  return 1;
}

float AlternativeCosineDistance(const float* values, const float* norms_squared,
                                int dimension, int left, int right) {
  const float left_norm = norms_squared[left];
  const float right_norm = norms_squared[right];
  if (left_norm == 0.0f && right_norm == 0.0f) return 0.0f;
  if (left_norm == 0.0f || right_norm == 0.0f) return FLT_MAX;
  const float product =
      Dot(values + static_cast<size_t>(left) * dimension,
          values + static_cast<size_t>(right) * dimension, dimension);
  if (product <= 0.0f) return FLT_MAX;
  const float ratio = std::sqrt(left_norm * right_norm) / product;
  return std::log2(ratio);
}

struct AngularTreeContext {
  const float* values;
  const float* norms_squared;
  int count;
  int dimension;
  int neighbor_count;
  int leaf_size;
  int max_depth;
  int32_t* graph_indices;
  float* graph_distances;
  uint8_t* graph_flags;
  int32_t* tree_indices;
  int32_t* partition_scratch;
  uint8_t* sides;
  float* hyperplane;
  uint32_t* evaluated_pairs;
  TauState* random;
};

void ProcessLeaf(AngularTreeContext* context, int begin, int end) {
  for (int left_position = begin; left_position < end; ++left_position) {
    const int left = context->tree_indices[left_position];
    for (int right_position = left_position + 1; right_position < end;
         ++right_position) {
      const int right = context->tree_indices[right_position];
      if (!ShouldEvaluateUnorderedPair(context->evaluated_pairs,
                                       context->count, left, right)) {
        continue;
      }
      const float distance = AlternativeCosineDistance(
          context->values, context->norms_squared, context->dimension, left,
          right);
      const size_t left_offset =
          static_cast<size_t>(left) * context->neighbor_count;
      const size_t right_offset =
          static_cast<size_t>(right) * context->neighbor_count;
      HeapPush(context->graph_indices + left_offset,
               context->graph_distances + left_offset,
               context->graph_flags + left_offset, context->neighbor_count,
               distance, right, 1);
      HeapPush(context->graph_indices + right_offset,
               context->graph_distances + right_offset,
               context->graph_flags + right_offset, context->neighbor_count,
               distance, left, 1);
    }
  }
}

int BuildAngularTree(AngularTreeContext* context, int begin, int end,
                     int depth) {
  const int size = end - begin;
  if (size <= context->leaf_size || depth >= context->max_depth) {
    ProcessLeaf(context, begin, end);
    return kSuccess;
  }

  const int left_position =
      begin + PositiveModulo(TauRandInt(context->random), size);
  int right_relative = PositiveModulo(TauRandInt(context->random), size);
  right_relative += begin + right_relative == left_position;
  right_relative %= size;
  const int right_position = begin + right_relative;
  const int left_index = context->tree_indices[left_position];
  const int right_index = context->tree_indices[right_position];
  const float left_norm_squared = context->norms_squared[left_index];
  const float right_norm_squared = context->norms_squared[right_index];
  const float left_norm =
      std::fabs(left_norm_squared) < kRpEpsilon
          ? 1.0f
          : std::sqrt(left_norm_squared);
  const float right_norm =
      std::fabs(right_norm_squared) < kRpEpsilon
          ? 1.0f
          : std::sqrt(right_norm_squared);
  const float* left_values =
      context->values + static_cast<size_t>(left_index) * context->dimension;
  const float* right_values =
      context->values + static_cast<size_t>(right_index) * context->dimension;
  for (int column = 0; column < context->dimension; ++column) {
    context->hyperplane[column] =
        left_values[column] / left_norm - right_values[column] / right_norm;
  }
  float hyperplane_norm = std::sqrt(
      SumSquares(context->hyperplane, context->dimension));
  if (std::fabs(hyperplane_norm) < kRpEpsilon) hyperplane_norm = 1.0f;
  for (int column = 0; column < context->dimension; ++column) {
    context->hyperplane[column] /= hyperplane_norm;
  }

  int left_count = 0;
  int right_count = 0;
  for (int position = begin; position < end; ++position) {
    const int point = context->tree_indices[position];
    const float margin =
        Dot(context->hyperplane,
            context->values + static_cast<size_t>(point) * context->dimension,
            context->dimension);
    uint8_t side = 0;
    if (std::fabs(margin) < kRpEpsilon) {
      side = static_cast<uint8_t>(
          PositiveModulo(TauRandInt(context->random), 2));
    } else {
      side = margin > 0.0f ? 0u : 1u;
    }
    context->sides[position] = side;
    left_count += side == 0;
    right_count += side != 0;
  }
  if (left_count == 0 || right_count == 0) {
    left_count = 0;
    right_count = 0;
    for (int position = begin; position < end; ++position) {
      const uint8_t side = static_cast<uint8_t>(
          PositiveModulo(TauRandInt(context->random), 2));
      context->sides[position] = side;
      left_count += side == 0;
      right_count += side != 0;
    }
  }
  if (left_count == 0 || right_count == 0) {
    /*
     * PyNNDescent would recurse into an empty split here. It is vanishingly
     * unlikely for a real leaf; retaining it as one leaf is the only safe,
     * bounded behavior.
     */
    ProcessLeaf(context, begin, end);
    return kSuccess;
  }

  int left_write = begin;
  int right_write = begin + left_count;
  for (int position = begin; position < end; ++position) {
    if (context->sides[position] == 0) {
      context->partition_scratch[left_write++] =
          context->tree_indices[position];
    } else {
      context->partition_scratch[right_write++] =
          context->tree_indices[position];
    }
  }
  std::memcpy(context->tree_indices + begin,
              context->partition_scratch + begin,
              static_cast<size_t>(size) * sizeof(int32_t));
  const int middle = begin + left_count;
  int status = BuildAngularTree(context, begin, middle, depth + 1);
  if (status != kSuccess) return status;
  return BuildAngularTree(context, middle, end, depth + 1);
}

void InitializeRandomNeighbors(const float* values,
                               const float* norms_squared, int count,
                               int dimension, int neighbor_count,
                               TauState* random, int32_t* graph_indices,
                               float* graph_distances, uint8_t* graph_flags) {
  for (int row = 0; row < count; ++row) {
    const size_t offset = static_cast<size_t>(row) * neighbor_count;
    int occupied = 0;
    for (int rank = 0; rank < neighbor_count; ++rank) {
      occupied += graph_indices[offset + rank] >= 0;
    }
    const int missing = neighbor_count - occupied;
    for (int sample = 0; sample < missing; ++sample) {
      const int candidate = AbsoluteModulo(TauRandInt(random), count);
      const float distance = AlternativeCosineDistance(
          values, norms_squared, dimension, candidate, row);
      HeapPush(graph_indices + offset, graph_distances + offset,
               graph_flags + offset, neighbor_count, distance, candidate, 1);
    }
  }
}

void BuildCandidates(int count, int neighbor_count, int max_candidates,
                     const int32_t* graph_indices, uint8_t* graph_flags,
                     const TauState& random_state, int32_t* new_indices,
                     float* new_priorities, int32_t* old_indices,
                     float* old_priorities) {
  const size_t candidate_length =
      static_cast<size_t>(count) * max_candidates;
  std::fill_n(new_indices, candidate_length, -1);
  std::fill_n(old_indices, candidate_length, -1);
  std::fill_n(new_priorities, candidate_length,
              std::numeric_limits<float>::infinity());
  std::fill_n(old_priorities, candidate_length,
              std::numeric_limits<float>::infinity());

  TauState local_random = random_state;
  for (int row = 0; row < count; ++row) {
    const size_t graph_offset = static_cast<size_t>(row) * neighbor_count;
    for (int rank = 0; rank < neighbor_count; ++rank) {
      const size_t graph_position = graph_offset + rank;
      const int candidate = graph_indices[graph_position];
      if (candidate < 0) continue;
      const float priority = TauRand(&local_random);
      int32_t* selected_indices =
          graph_flags[graph_position] != 0 ? new_indices : old_indices;
      float* selected_priorities =
          graph_flags[graph_position] != 0 ? new_priorities : old_priorities;
      const size_t row_offset = static_cast<size_t>(row) * max_candidates;
      const size_t candidate_offset =
          static_cast<size_t>(candidate) * max_candidates;
      CandidatePush(selected_indices + row_offset,
                    selected_priorities + row_offset, max_candidates, priority,
                    candidate);
      CandidatePush(selected_indices + candidate_offset,
                    selected_priorities + candidate_offset, max_candidates,
                    priority, row);
    }
  }

  for (int row = 0; row < count; ++row) {
    const size_t graph_offset = static_cast<size_t>(row) * neighbor_count;
    const size_t candidate_offset =
        static_cast<size_t>(row) * max_candidates;
    for (int rank = 0; rank < neighbor_count; ++rank) {
      const int candidate = graph_indices[graph_offset + rank];
      for (int position = 0; position < max_candidates; ++position) {
        if (new_indices[candidate_offset + position] == candidate) {
          graph_flags[graph_offset + rank] = 0;
          break;
        }
      }
    }
  }
}

int ProcessCandidateBlock(
    const float* values, const float* norms_squared, int count, int dimension,
    int neighbor_count, int max_candidates, int block_begin, int block_end,
    const int32_t* new_indices, const int32_t* old_indices,
    const float* threshold_snapshot, uint32_t* evaluated_pairs,
    int32_t* graph_indices,
    float* graph_distances, uint8_t* graph_flags) {
  int changes = 0;
  for (int row = block_begin; row < block_end; ++row) {
    const size_t candidate_offset =
        static_cast<size_t>(row) * max_candidates;
    for (int left_position = 0; left_position < max_candidates;
         ++left_position) {
      const int left = new_indices[candidate_offset + left_position];
      if (left < 0) continue;
      const float left_threshold = threshold_snapshot[left];
      for (int right_position = left_position;
           right_position < max_candidates; ++right_position) {
        const int right = new_indices[candidate_offset + right_position];
        if (right < 0) continue;
        if (!ShouldEvaluateUnorderedPair(evaluated_pairs, count, left, right)) {
          continue;
        }
        const float distance = AlternativeCosineDistance(
            values, norms_squared, dimension, left, right);
        if (distance <=
            std::max(left_threshold, threshold_snapshot[right])) {
          const size_t left_offset =
              static_cast<size_t>(left) * neighbor_count;
          const size_t right_offset =
              static_cast<size_t>(right) * neighbor_count;
          changes += HeapPush(
              graph_indices + left_offset, graph_distances + left_offset,
              graph_flags + left_offset, neighbor_count, distance, right, 1);
          changes += HeapPush(
              graph_indices + right_offset, graph_distances + right_offset,
              graph_flags + right_offset, neighbor_count, distance, left, 1);
        }
      }
      for (int right_position = 0; right_position < max_candidates;
           ++right_position) {
        const int right = old_indices[candidate_offset + right_position];
        if (right < 0) continue;
        if (!ShouldEvaluateUnorderedPair(evaluated_pairs, count, left, right)) {
          continue;
        }
        const float distance = AlternativeCosineDistance(
            values, norms_squared, dimension, left, right);
        if (distance <=
            std::max(left_threshold, threshold_snapshot[right])) {
          const size_t left_offset =
              static_cast<size_t>(left) * neighbor_count;
          const size_t right_offset =
              static_cast<size_t>(right) * neighbor_count;
          changes += HeapPush(
              graph_indices + left_offset, graph_distances + left_offset,
              graph_flags + left_offset, neighbor_count, distance, right, 1);
          changes += HeapPush(
              graph_indices + right_offset, graph_distances + right_offset,
              graph_flags + right_offset, neighbor_count, distance, left, 1);
        }
      }
    }
  }
  return changes;
}

void SiftDown(float* distances, int32_t* indices, int length, int position) {
  while (position * 2 + 1 < length) {
    const int left = position * 2 + 1;
    const int right = left + 1;
    int swap = position;
    if (distances[swap] < distances[left]) swap = left;
    if (right < length && distances[swap] < distances[right]) swap = right;
    if (swap == position) break;
    std::swap(distances[position], distances[swap]);
    std::swap(indices[position], indices[swap]);
    position = swap;
  }
}

void DeheapSortAndCorrect(int count, int neighbor_count, int32_t* indices,
                          float* distances) {
  for (int row = 0; row < count; ++row) {
    const size_t offset = static_cast<size_t>(row) * neighbor_count;
    int32_t* row_indices = indices + offset;
    float* row_distances = distances + offset;
    for (int end = neighbor_count - 1; end > 0; --end) {
      std::swap(row_indices[0], row_indices[end]);
      std::swap(row_distances[0], row_distances[end]);
      SiftDown(row_distances, row_indices, end, 0);
    }
    for (int rank = 0; rank < neighbor_count; ++rank) {
      row_distances[rank] =
          1.0f - std::exp2(-row_distances[rank]);
    }
  }
}

}  // namespace

size_t ExactWorkspaceBytes(int count, int dimension) {
  if (count <= 0 || dimension <= 0) return 0;
  size_t element_count = 0;
  size_t result = 0;
  if (!CheckedMultiply(static_cast<size_t>(count),
                       static_cast<size_t>(dimension), &element_count) ||
      !AddArrayBytes(&result, element_count, sizeof(float))) {
    return 0;
  }
  return result;
}

size_t ApproximateWorkspaceBytes(int count, int dimension, int neighbor_count,
                                 const ApproximateOptions& options) {
  if (!ValidShape(count, dimension, neighbor_count) ||
      !ValidApproximateOptions(count, options)) {
    return 0;
  }
  const size_t rows = static_cast<size_t>(count);
  size_t graph_length = 0;
  size_t candidate_length = 0;
  const size_t pair_bitset_word_count = PairBitsetWordCount(count);
  if (!CheckedMultiply(rows, static_cast<size_t>(neighbor_count),
                       &graph_length) ||
      !CheckedMultiply(rows, static_cast<size_t>(options.max_candidates),
                       &candidate_length)) {
    return 0;
  }
  size_t result = 0;
  if (!AddArrayBytes(&result, graph_length, sizeof(uint8_t)) ||
      !AddArrayBytes(&result, rows, sizeof(float)) ||
      !AddArrayBytes(&result, candidate_length, sizeof(int32_t)) ||
      !AddArrayBytes(&result, candidate_length, sizeof(float)) ||
      !AddArrayBytes(&result, candidate_length, sizeof(int32_t)) ||
      !AddArrayBytes(&result, candidate_length, sizeof(float)) ||
      !AddArrayBytes(&result, rows, sizeof(float)) ||
      !AddArrayBytes(&result, rows, sizeof(int32_t)) ||
      !AddArrayBytes(&result, rows, sizeof(int32_t)) ||
      !AddArrayBytes(&result, rows, sizeof(uint8_t)) ||
      !AddArrayBytes(&result, static_cast<size_t>(dimension), sizeof(float)) ||
      (pair_bitset_word_count != 0 &&
       !AddArrayBytes(&result, pair_bitset_word_count, sizeof(uint32_t)))) {
    return 0;
  }
  return result;
}

int ExactCosineKnn(const float* values, int count, int dimension,
                   int neighbor_count, void* workspace,
                   size_t workspace_bytes, int32_t* output_indices,
                   float* output_distances) {
  if (!values || !workspace || !output_indices || !output_distances ||
      !IsWorkspaceAligned(workspace) ||
      !ValidShape(count, dimension, neighbor_count)) {
    return kInvalidArgument;
  }
  const size_t required = ExactWorkspaceBytes(count, dimension);
  if (required == 0) return kSizeOverflow;
  if (workspace_bytes < required) return kInsufficientWorkspace;
  Workspace allocator(workspace, workspace_bytes);
  const size_t value_count =
      static_cast<size_t>(count) * static_cast<size_t>(dimension);
  float* normalized = allocator.Allocate<float>(value_count);
  if (!normalized) return kInsufficientWorkspace;
  if (!NormalizeRows(values, count, dimension, normalized)) {
    return kInvalidArgument;
  }

  const size_t output_length =
      static_cast<size_t>(count) * neighbor_count;
  std::fill_n(output_indices, output_length, -1);
  std::fill_n(output_distances, output_length,
              std::numeric_limits<float>::infinity());
  for (int row = 0; row < count; ++row) {
    const size_t row_offset = static_cast<size_t>(row) * neighbor_count;
    InsertSortedNeighbor(output_indices + row_offset,
                         output_distances + row_offset, neighbor_count, row,
                         0.0f);
    const float* left =
        normalized + static_cast<size_t>(row) * dimension;
    for (int candidate = row + 1; candidate < count; ++candidate) {
      const float distance = ClippedCosineDistance(
          left, normalized + static_cast<size_t>(candidate) * dimension,
          dimension);
      const size_t candidate_offset =
          static_cast<size_t>(candidate) * neighbor_count;
      InsertSortedNeighbor(output_indices + row_offset,
                           output_distances + row_offset, neighbor_count,
                           candidate, distance);
      InsertSortedNeighbor(output_indices + candidate_offset,
                           output_distances + candidate_offset, neighbor_count,
                           row, distance);
    }
  }
  return kSuccess;
}

int ApproximateCosineKnn(const float* values, int count, int dimension,
                         int neighbor_count,
                         const ApproximateOptions& options, void* workspace,
                         size_t workspace_bytes, int32_t* output_indices,
                         float* output_distances) {
  if (!values || !workspace || !output_indices || !output_distances ||
      !IsWorkspaceAligned(workspace) ||
      !ValidShape(count, dimension, neighbor_count) ||
      !ValidApproximateOptions(count, options)) {
    return kInvalidArgument;
  }
  const size_t required =
      ApproximateWorkspaceBytes(count, dimension, neighbor_count, options);
  if (required == 0) return kSizeOverflow;
  if (workspace_bytes < required) return kInsufficientWorkspace;

  const size_t rows = static_cast<size_t>(count);
  size_t graph_length = 0;
  size_t candidate_length = 0;
  if (!CheckedMultiply(rows, static_cast<size_t>(neighbor_count),
                       &graph_length) ||
      !CheckedMultiply(rows, static_cast<size_t>(options.max_candidates),
                       &candidate_length)) {
    return kSizeOverflow;
  }
  Workspace allocator(workspace, workspace_bytes);
  uint8_t* graph_flags = allocator.Allocate<uint8_t>(graph_length);
  float* norms_squared = allocator.Allocate<float>(rows);
  int32_t* new_indices = allocator.Allocate<int32_t>(candidate_length);
  float* new_priorities = allocator.Allocate<float>(candidate_length);
  int32_t* old_indices = allocator.Allocate<int32_t>(candidate_length);
  float* old_priorities = allocator.Allocate<float>(candidate_length);
  float* threshold_snapshot = allocator.Allocate<float>(rows);
  int32_t* tree_indices = allocator.Allocate<int32_t>(rows);
  int32_t* partition_scratch = allocator.Allocate<int32_t>(rows);
  uint8_t* sides = allocator.Allocate<uint8_t>(rows);
  float* hyperplane =
      allocator.Allocate<float>(static_cast<size_t>(dimension));
  const size_t pair_bitset_word_count = PairBitsetWordCount(count);
  uint32_t* evaluated_pairs =
      pair_bitset_word_count == 0
          ? nullptr
          : allocator.Allocate<uint32_t>(pair_bitset_word_count);
  if (!graph_flags || !norms_squared || !new_indices || !new_priorities ||
      !old_indices || !old_priorities || !threshold_snapshot ||
      !tree_indices || !partition_scratch || !sides || !hyperplane ||
      (pair_bitset_word_count != 0 && !evaluated_pairs)) {
    return kInsufficientWorkspace;
  }
  if (evaluated_pairs) {
    std::fill_n(evaluated_pairs, pair_bitset_word_count, uint32_t{0});
  }

  std::fill_n(output_indices, graph_length, -1);
  std::fill_n(output_distances, graph_length,
              std::numeric_limits<float>::infinity());
  std::fill_n(graph_flags, graph_length, uint8_t{0});
  for (int row = 0; row < count; ++row) {
    const float squared_norm =
        SumSquares(values + static_cast<size_t>(row) * dimension, dimension);
    if (!std::isfinite(squared_norm)) return kInvalidArgument;
    norms_squared[row] = squared_norm;
  }

  NumpyRandomState numpy_random(options.random_seed);
  TauState descent_random{};
  TauState unused_search_random{};
  FillTauState(&numpy_random, &descent_random);
  FillTauState(&numpy_random, &unused_search_random);
  for (int warmup = 0; warmup < 10; ++warmup) {
    TauRandInt(&unused_search_random);
  }

  const int tree_count = ResolveTreeCount(count, options);
  for (int tree = 0; tree < tree_count; ++tree) {
    TauState tree_random{};
    FillTauState(&numpy_random, &tree_random);
    for (int row = 0; row < count; ++row) tree_indices[row] = row;
    AngularTreeContext tree_context{
        values,
        norms_squared,
        count,
        dimension,
        neighbor_count,
        options.leaf_size,
        options.max_tree_depth,
        output_indices,
        output_distances,
        graph_flags,
        tree_indices,
        partition_scratch,
        sides,
        hyperplane,
        evaluated_pairs,
        &tree_random,
    };
    const int status = BuildAngularTree(&tree_context, 0, count, 0);
    if (status != kSuccess) return status;
  }

  InitializeRandomNeighbors(values, norms_squared, count, dimension,
                            neighbor_count, &descent_random, output_indices,
                            output_distances, graph_flags);

  const int iteration_count = ResolveIterationCount(count, options);
  const double convergence_limit =
      static_cast<double>(options.delta) * neighbor_count * count;
  for (int iteration = 0; iteration < iteration_count; ++iteration) {
    BuildCandidates(count, neighbor_count, options.max_candidates,
                    output_indices, graph_flags, descent_random, new_indices,
                    new_priorities, old_indices, old_priorities);
    int changes = 0;
    for (int block_begin = 0; block_begin < count;
         block_begin += kCandidateBlockSize) {
      for (int row = 0; row < count; ++row) {
        threshold_snapshot[row] =
            output_distances[static_cast<size_t>(row) * neighbor_count];
      }
      const int block_end = std::min(count, block_begin + kCandidateBlockSize);
      changes += ProcessCandidateBlock(
          values, norms_squared, count, dimension, neighbor_count,
          options.max_candidates, block_begin, block_end, new_indices,
          old_indices, threshold_snapshot, evaluated_pairs, output_indices,
          output_distances, graph_flags);
    }
    if (changes <= convergence_limit) break;
  }

  DeheapSortAndCorrect(count, neighbor_count, output_indices,
                       output_distances);
  return kSuccess;
}

}  // namespace senko::umap_neighbors

#if defined(SENKO_UMAP_NEIGHBORS_HOST_TEST)

namespace {

template <typename T>
bool ReadHostFixture(const char* path, size_t count, std::vector<T>* result) {
  result->resize(count);
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  input.read(reinterpret_cast<char*>(result->data()),
             static_cast<std::streamsize>(count * sizeof(T)));
  return input.good() && input.peek() == std::ifstream::traits_type::eof();
}

}  // namespace

/*
 * Reproducible, opt-in native diagnostic:
 *
 * clang++ -std=c++20 -O3 -DSENKO_UMAP_NEIGHBORS_HOST_TEST \
 *   web/scripts/clustering-wasm/umap_neighbors.cpp \
 *   -o .research/umap-neighbors-host-test
 *
 * .research/umap-neighbors-host-test INPUT.f32 COUNT DIM K APPROX SEED \
 *   REFERENCE_INDICES.i32 REFERENCE_DISTANCES.f32
 */
int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr
        << "usage: INPUT.f32 COUNT DIM K APPROX SEED REFERENCE_INDICES.i32 "
           "REFERENCE_DISTANCES.f32\n";
    return 2;
  }
  const int count = std::stoi(argv[2]);
  const int dimension = std::stoi(argv[3]);
  const int neighbor_count = std::stoi(argv[4]);
  const bool approximate = std::stoi(argv[5]) != 0;
  const uint32_t seed = static_cast<uint32_t>(std::stoul(argv[6]));
  if (count <= 0 || dimension <= 0 || neighbor_count <= 0) return 2;
  const size_t value_count =
      static_cast<size_t>(count) * static_cast<size_t>(dimension);
  const size_t output_count =
      static_cast<size_t>(count) * static_cast<size_t>(neighbor_count);
  std::vector<float> values;
  std::vector<int32_t> reference_indices;
  std::vector<float> reference_distances;
  if (!ReadHostFixture(argv[1], value_count, &values) ||
      !ReadHostFixture(argv[7], output_count, &reference_indices) ||
      !ReadHostFixture(argv[8], output_count, &reference_distances)) {
    std::cerr << "failed to read an exact-size fixture\n";
    return 3;
  }

  senko::umap_neighbors::ApproximateOptions options;
  options.random_seed = seed;
  const size_t workspace_bytes =
      approximate
          ? senko::umap_neighbors::ApproximateWorkspaceBytes(
                count, dimension, neighbor_count, options)
          : senko::umap_neighbors::ExactWorkspaceBytes(count, dimension);
  std::vector<uint8_t> workspace(workspace_bytes);
  std::vector<int32_t> candidate_indices(output_count);
  std::vector<float> candidate_distances(output_count);
  const auto started = std::chrono::steady_clock::now();
  const int status =
      approximate
          ? senko::umap_neighbors::ApproximateCosineKnn(
                values.data(), count, dimension, neighbor_count, options,
                workspace.data(), workspace.size(), candidate_indices.data(),
                candidate_distances.data())
          : senko::umap_neighbors::ExactCosineKnn(
                values.data(), count, dimension, neighbor_count,
                workspace.data(), workspace.size(), candidate_indices.data(),
                candidate_distances.data());
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  if (status != senko::umap_neighbors::kSuccess) {
    std::cerr << "neighbor search failed with status " << status << "\n";
    return 4;
  }

  size_t exact_entries = 0;
  size_t exact_rows = 0;
  size_t shared_neighbors = 0;
  size_t missing_neighbors = 0;
  double distance_error_sum = 0.0;
  double distance_error_max = 0.0;
  double recall_sum = 0.0;
  double minimum_recall = 1.0;
  for (int row = 0; row < count; ++row) {
    const size_t offset = static_cast<size_t>(row) * neighbor_count;
    bool exact_row = true;
    int shared_in_row = 0;
    for (int rank = 0; rank < neighbor_count; ++rank) {
      const size_t position = offset + rank;
      exact_entries +=
          candidate_indices[position] == reference_indices[position];
      exact_row &= candidate_indices[position] == reference_indices[position];
      missing_neighbors += candidate_indices[position] < 0;
      for (int reference_rank = 0; reference_rank < neighbor_count;
           ++reference_rank) {
        const size_t reference_position = offset + reference_rank;
        if (candidate_indices[position] ==
            reference_indices[reference_position]) {
          const double error = std::fabs(
              static_cast<double>(candidate_distances[position]) -
              reference_distances[reference_position]);
          distance_error_sum += error;
          distance_error_max = std::max(distance_error_max, error);
          ++shared_neighbors;
          ++shared_in_row;
          break;
        }
      }
    }
    exact_rows += exact_row;
    const double recall =
        static_cast<double>(shared_in_row) / neighbor_count;
    recall_sum += recall;
    minimum_recall = std::min(minimum_recall, recall);
  }
  const double recall = recall_sum / count;
  std::cout << "{"
            << "\"status\":" << status << ","
            << "\"elapsedMs\":" << elapsed_ms << ","
            << "\"workspaceBytes\":" << workspace_bytes << ","
            << "\"exactRowFraction\":"
            << static_cast<double>(exact_rows) / count << ","
            << "\"exactIndexFraction\":"
            << static_cast<double>(exact_entries) / output_count << ","
            << "\"meanRecallAtK\":" << recall << ","
            << "\"minimumRecallAtK\":" << minimum_recall << ","
            << "\"missingNeighbors\":" << missing_neighbors << ","
            << "\"sharedDistanceMeanAbsoluteError\":"
            << (shared_neighbors == 0
                    ? 0.0
                    : distance_error_sum / shared_neighbors)
            << ","
            << "\"sharedDistanceMaxAbsoluteError\":" << distance_error_max
            << "}\n";
  const double required_recall = approximate ? 0.999 : 1.0;
  return recall >= required_recall && missing_neighbors == 0 &&
                 distance_error_max <= 1.0e-5
             ? 0
             : 5;
}

#endif  // SENKO_UMAP_NEIGHBORS_HOST_TEST
