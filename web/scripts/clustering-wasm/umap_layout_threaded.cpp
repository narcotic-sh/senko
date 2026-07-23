#include "umap_layout_threaded.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__wasm__)
#define SENKO_EXPORT(name) __attribute__((export_name(name)))
#else
#define SENKO_EXPORT(name)
#endif

namespace senko::umap_layout_threaded {
namespace {

constexpr std::uint64_t kUint32Mask = 0xffff'ffffULL;
constexpr std::uint32_t kMaximumWorkerCount = 32;
constexpr std::uint32_t kAlignment = 64;
constexpr std::uint32_t kRowsPerChunk = 16;
constexpr std::int64_t kBarrierWaitNanoseconds = 1'000'000'000;

struct Plan {
  std::uint32_t offsets[kTotalBytes + 1] = {};
};

struct TauState {
  std::int64_t values[3];
};

static_assert(sizeof(TauState) == sizeof(std::int64_t) * 3);

bool CheckedAdd(std::uint64_t left,
                std::uint64_t right,
                std::uint64_t* result) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
  *result = left + right;
  return true;
}

bool CheckedMultiply(std::uint64_t left,
                     std::uint64_t right,
                     std::uint64_t* result) {
  if (left != 0 &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool Align(std::uint64_t value,
           std::uint64_t alignment,
           std::uint64_t* result) {
  const std::uint64_t mask = alignment - 1;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) return false;
  *result = (value + mask) & ~mask;
  return true;
}

bool AddArray(std::uint64_t* cursor,
              std::uint64_t count,
              std::uint64_t item_bytes,
              std::uint32_t* offset) {
  std::uint64_t aligned = 0;
  std::uint64_t bytes = 0;
  std::uint64_t end = 0;
  if (!Align(*cursor, kAlignment, &aligned) ||
      !CheckedMultiply(count, item_bytes, &bytes) ||
      !CheckedAdd(aligned, bytes, &end) ||
      aligned > std::numeric_limits<std::uint32_t>::max() ||
      end > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  *offset = static_cast<std::uint32_t>(aligned);
  *cursor = end;
  return true;
}

bool BuildPlan(std::uint32_t worker_count,
               std::uint32_t vertex_count,
               std::uint32_t dimension,
               std::uint32_t edge_count,
               Plan* plan) {
  if (plan == nullptr || worker_count == 0 ||
      worker_count > kMaximumWorkerCount || vertex_count == 0 ||
      vertex_count >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      dimension == 0 ||
      dimension >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::int32_t>::max()) ||
      edge_count == 0 ||
      edge_count >
          static_cast<std::uint32_t>(
              std::numeric_limits<std::int32_t>::max())) {
    return false;
  }

  std::uint64_t stack_bytes = 0;
  std::uint64_t cursor = 0;
  if (!CheckedMultiply(worker_count, kWorkerStackBytes, &stack_bytes) ||
      !CheckedAdd(kStackRegionOffset, stack_bytes, &cursor) ||
      !Align(cursor, kAlignment, &cursor) ||
      cursor > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }

  plan->offsets[kHeader] = static_cast<std::uint32_t>(cursor);
  if (!CheckedAdd(cursor, sizeof(RunHeader), &cursor)) return false;

  std::uint64_t embedding_values = 0;
  if (!CheckedMultiply(vertex_count, dimension, &embedding_values) ||
      !AddArray(&cursor, embedding_values, sizeof(float),
                &plan->offsets[kEmbedding]) ||
      !AddArray(&cursor, static_cast<std::uint64_t>(vertex_count) + 1,
                sizeof(std::uint32_t), &plan->offsets[kRowOffsets]) ||
      !AddArray(&cursor, edge_count, sizeof(std::int32_t),
                &plan->offsets[kTail]) ||
      !AddArray(&cursor, edge_count, sizeof(double),
                &plan->offsets[kEpochsPerSample]) ||
      !AddArray(&cursor, 3, sizeof(std::int64_t),
                &plan->offsets[kRngSeed]) ||
      !AddArray(&cursor, edge_count, sizeof(double),
                &plan->offsets[kEpochOfNextNegativeSample]) ||
      !AddArray(&cursor, edge_count, sizeof(double),
                &plan->offsets[kEpochOfNextSample])) {
    return false;
  }

  std::uint64_t rng_values = 0;
  if (!CheckedMultiply(vertex_count, 3, &rng_values) ||
      !AddArray(&cursor, rng_values, sizeof(std::int64_t),
                &plan->offsets[kRngStatePerVertex])) {
    return false;
  }

  cursor = cursor < kMinimumMemoryBytes ? kMinimumMemoryBytes : cursor;
  if (!Align(cursor, kPageBytes, &cursor) ||
      cursor > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  plan->offsets[kTotalBytes] = static_cast<std::uint32_t>(cursor);
  return true;
}

template <typename T>
T* OffsetPointer(std::uint32_t offset) {
  return reinterpret_cast<T*>(static_cast<std::uintptr_t>(offset));
}

std::uint32_t AtomicLoad(const std::uint32_t* value) {
  return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

std::int32_t AtomicLoad(const std::int32_t* value) {
  return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

void AtomicStore(std::uint32_t* value, std::uint32_t desired) {
  __atomic_store_n(value, desired, __ATOMIC_RELEASE);
}

void AtomicStore(std::int32_t* value, std::int32_t desired) {
  __atomic_store_n(value, desired, __ATOMIC_RELEASE);
}

std::uint32_t AtomicFetchAdd(std::uint32_t* value,
                             std::uint32_t increment) {
  return __atomic_fetch_add(value, increment, __ATOMIC_ACQ_REL);
}

void WaitWhileEqual(std::uint32_t* address, std::uint32_t expected) {
#if defined(__wasm__)
  (void)__builtin_wasm_memory_atomic_wait32(
      reinterpret_cast<int*>(address),
      static_cast<int>(expected),
      kBarrierWaitNanoseconds);
#else
  (void)address;
  (void)expected;
#endif
}

void NotifyAll(std::uint32_t* address) {
#if defined(__wasm__)
  (void)__builtin_wasm_memory_atomic_notify(
      reinterpret_cast<int*>(address),
      0x7fffffff);
#else
  (void)address;
#endif
}

void Abort(RunHeader* header, Status status) {
  AtomicStore(&header->status, static_cast<std::int32_t>(status));
  AtomicStore(&header->cancelled, 1);
  NotifyAll(&header->generation);
}

bool Barrier(RunHeader* header) {
  const std::uint32_t generation = AtomicLoad(&header->generation);
  const std::uint32_t previous = AtomicFetchAdd(&header->arrived, 1);
  if (previous + 1 == header->worker_count) {
    AtomicStore(&header->next_row, 0);
    AtomicStore(&header->arrived, 0);
    AtomicStore(&header->generation, generation + 1);
    NotifyAll(&header->generation);
    return AtomicLoad(&header->cancelled) == 0;
  }
  while (AtomicLoad(&header->generation) == generation) {
    if (AtomicLoad(&header->cancelled) != 0) return false;
    WaitWhileEqual(&header->generation, generation);
  }
  return AtomicLoad(&header->cancelled) == 0;
}

bool IsFiniteEmbedding(const float* embedding,
                       std::uint32_t vertex_count,
                       std::uint32_t dimension) {
  const std::uint64_t value_count =
      static_cast<std::uint64_t>(vertex_count) * dimension;
  for (std::uint64_t index = 0; index < value_count; ++index) {
    if (!std::isfinite(embedding[index])) return false;
  }
  return true;
}

bool InitializeSharedState(RunHeader* header, const Plan& plan) {
  if (header->magic != kHeaderMagic ||
      header->version != kHeaderVersion ||
      header->total_bytes != plan.offsets[kTotalBytes] ||
      header->embedding_offset != plan.offsets[kEmbedding] ||
      header->row_offsets_offset != plan.offsets[kRowOffsets] ||
      header->tail_offset != plan.offsets[kTail] ||
      header->epochs_per_sample_offset !=
          plan.offsets[kEpochsPerSample] ||
      header->rng_seed_offset != plan.offsets[kRngSeed] ||
      header->epoch_of_next_negative_sample_offset !=
          plan.offsets[kEpochOfNextNegativeSample] ||
      header->epoch_of_next_sample_offset !=
          plan.offsets[kEpochOfNextSample] ||
      header->rng_state_per_vertex_offset !=
          plan.offsets[kRngStatePerVertex] ||
      reinterpret_cast<std::uintptr_t>(header) != plan.offsets[kHeader] ||
      header->epoch_count == 0 || !std::isfinite(header->a) ||
      !std::isfinite(header->b) || !std::isfinite(header->gamma) ||
      !std::isfinite(header->negative_sample_rate) || header->a <= 0.0 ||
      header->b <= 0.0 || header->gamma < 0.0 ||
      header->negative_sample_rate <= 0.0 ||
      AtomicLoad(&header->generation) != 0 ||
      AtomicLoad(&header->cancelled) != 0 ||
      AtomicLoad(&header->next_row) != 0) {
    return false;
  }

  float* embedding = OffsetPointer<float>(header->embedding_offset);
  const std::uint32_t* row_offsets =
      OffsetPointer<std::uint32_t>(header->row_offsets_offset);
  const std::int32_t* tail =
      OffsetPointer<std::int32_t>(header->tail_offset);
  const double* epochs_per_sample =
      OffsetPointer<double>(header->epochs_per_sample_offset);
  const std::int64_t* rng_seed =
      OffsetPointer<std::int64_t>(header->rng_seed_offset);
  double* epoch_of_next_negative_sample =
      OffsetPointer<double>(header->epoch_of_next_negative_sample_offset);
  double* epoch_of_next_sample =
      OffsetPointer<double>(header->epoch_of_next_sample_offset);
  TauState* rng_state_per_vertex =
      OffsetPointer<TauState>(header->rng_state_per_vertex_offset);

  if (!IsFiniteEmbedding(embedding, header->vertex_count,
                         header->dimension)) {
    return false;
  }
  if (row_offsets[0] != 0 ||
      row_offsets[header->vertex_count] != header->edge_count) {
    return false;
  }
  for (std::uint32_t vertex = 0; vertex < header->vertex_count; ++vertex) {
    if (row_offsets[vertex] > row_offsets[vertex + 1]) return false;
  }
  for (std::uint32_t edge = 0; edge < header->edge_count; ++edge) {
    if (tail[edge] < 0 ||
        static_cast<std::uint32_t>(tail[edge]) >= header->vertex_count ||
        !std::isfinite(epochs_per_sample[edge]) ||
        epochs_per_sample[edge] <= 0.0) {
      return false;
    }
    const double negative_epoch =
        epochs_per_sample[edge] / header->negative_sample_rate;
    epoch_of_next_negative_sample[edge] = negative_epoch;
    epoch_of_next_sample[edge] = epochs_per_sample[edge];
  }

  for (std::uint32_t vertex = 0; vertex < header->vertex_count; ++vertex) {
    const float first_coordinate =
        embedding[static_cast<std::uint64_t>(vertex) * header->dimension];
    const std::int64_t coordinate_bits =
        std::bit_cast<std::int64_t>(static_cast<double>(first_coordinate));
    for (std::uint32_t state_index = 0; state_index < 3; ++state_index) {
      rng_state_per_vertex[vertex].values[state_index] =
          std::bit_cast<std::int64_t>(
              static_cast<std::uint64_t>(rng_seed[state_index]) +
              static_cast<std::uint64_t>(coordinate_bits));
    }
  }
  AtomicStore(&header->completed_epochs, 0);
  AtomicStore(&header->status, static_cast<std::int32_t>(kRunning));
  return true;
}

float ReducedSquaredDistance(const float* left,
                             const float* right,
                             std::uint32_t dimension) {
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;
  float accumulator4 = 0.0f;
  float accumulator5 = 0.0f;
  float accumulator6 = 0.0f;
  float accumulator7 = 0.0f;
  float accumulator8 = 0.0f;
  float accumulator9 = 0.0f;
  float accumulator10 = 0.0f;
  float accumulator11 = 0.0f;
  float accumulator12 = 0.0f;
  float accumulator13 = 0.0f;
  float accumulator14 = 0.0f;
  float accumulator15 = 0.0f;

  const std::uint32_t vector_end = dimension & ~std::uint32_t{15};
  std::uint32_t coordinate = 0;
  for (; coordinate < vector_end; coordinate += 16) {
    const float difference0 = left[coordinate] - right[coordinate];
    const float difference1 = left[coordinate + 1] - right[coordinate + 1];
    const float difference2 = left[coordinate + 2] - right[coordinate + 2];
    const float difference3 = left[coordinate + 3] - right[coordinate + 3];
    const float difference4 = left[coordinate + 4] - right[coordinate + 4];
    const float difference5 = left[coordinate + 5] - right[coordinate + 5];
    const float difference6 = left[coordinate + 6] - right[coordinate + 6];
    const float difference7 = left[coordinate + 7] - right[coordinate + 7];
    const float difference8 = left[coordinate + 8] - right[coordinate + 8];
    const float difference9 = left[coordinate + 9] - right[coordinate + 9];
    const float difference10 =
        left[coordinate + 10] - right[coordinate + 10];
    const float difference11 =
        left[coordinate + 11] - right[coordinate + 11];
    const float difference12 =
        left[coordinate + 12] - right[coordinate + 12];
    const float difference13 =
        left[coordinate + 13] - right[coordinate + 13];
    const float difference14 =
        left[coordinate + 14] - right[coordinate + 14];
    const float difference15 =
        left[coordinate + 15] - right[coordinate + 15];
    accumulator0 += difference0 * difference0;
    accumulator1 += difference1 * difference1;
    accumulator2 += difference2 * difference2;
    accumulator3 += difference3 * difference3;
    accumulator4 += difference4 * difference4;
    accumulator5 += difference5 * difference5;
    accumulator6 += difference6 * difference6;
    accumulator7 += difference7 * difference7;
    accumulator8 += difference8 * difference8;
    accumulator9 += difference9 * difference9;
    accumulator10 += difference10 * difference10;
    accumulator11 += difference11 * difference11;
    accumulator12 += difference12 * difference12;
    accumulator13 += difference13 * difference13;
    accumulator14 += difference14 * difference14;
    accumulator15 += difference15 * difference15;
  }

  const float lane0 =
      (accumulator12 + accumulator8) + (accumulator4 + accumulator0);
  const float lane1 =
      (accumulator13 + accumulator9) + (accumulator5 + accumulator1);
  const float lane2 =
      (accumulator14 + accumulator10) + (accumulator6 + accumulator2);
  const float lane3 =
      (accumulator15 + accumulator11) + (accumulator7 + accumulator3);
  float result = (lane0 + lane1) + (lane2 + lane3);

  const std::uint32_t lane_end = dimension & ~std::uint32_t{3};
  float tail0 = result;
  float tail1 = 0.0f;
  float tail2 = 0.0f;
  float tail3 = 0.0f;
  for (; coordinate < lane_end; coordinate += 4) {
    const float difference0 = left[coordinate] - right[coordinate];
    const float difference1 = left[coordinate + 1] - right[coordinate + 1];
    const float difference2 = left[coordinate + 2] - right[coordinate + 2];
    const float difference3 = left[coordinate + 3] - right[coordinate + 3];
    tail0 += difference0 * difference0;
    tail1 += difference1 * difference1;
    tail2 += difference2 * difference2;
    tail3 += difference3 * difference3;
  }
  result = (tail0 + tail1) + (tail2 + tail3);
  for (; coordinate < dimension; ++coordinate) {
    const float difference = left[coordinate] - right[coordinate];
    result += difference * difference;
  }
  return result;
}

float Clip(float value) {
  if (value > 4.0f) return 4.0f;
  if (value < -4.0f) return -4.0f;
  return value;
}

std::int64_t ArithmeticShiftRight(std::uint64_t bits, int shift) {
  if ((bits & (std::uint64_t{1} << 63)) == 0) {
    return static_cast<std::int64_t>(bits >> shift);
  }
  const std::uint64_t fill = ~std::uint64_t{0} << (64 - shift);
  return std::bit_cast<std::int64_t>((bits >> shift) | fill);
}

std::int32_t TauRandInt(TauState* state) {
  const std::uint64_t original0 =
      static_cast<std::uint64_t>(state->values[0]);
  const std::uint64_t original1 =
      static_cast<std::uint64_t>(state->values[1]);
  const std::uint64_t original2 =
      static_cast<std::uint64_t>(state->values[2]);

  const std::uint64_t left0 =
      ((original0 & 4294967294ULL) << 12) & kUint32Mask;
  const std::uint64_t mixed0 =
      ((original0 << 13) & kUint32Mask) ^ original0;
  state->values[0] = std::bit_cast<std::int64_t>(
      left0 ^
      static_cast<std::uint64_t>(ArithmeticShiftRight(mixed0, 19)));

  const std::uint64_t left1 =
      ((original1 & 4294967288ULL) << 4) & kUint32Mask;
  const std::uint64_t mixed1 =
      ((original1 << 2) & kUint32Mask) ^ original1;
  state->values[1] = std::bit_cast<std::int64_t>(
      left1 ^
      static_cast<std::uint64_t>(ArithmeticShiftRight(mixed1, 25)));

  const std::uint64_t left2 =
      ((original2 & 4294967280ULL) << 17) & kUint32Mask;
  const std::uint64_t mixed2 =
      ((original2 << 3) & kUint32Mask) ^ original2;
  state->values[2] = std::bit_cast<std::int64_t>(
      left2 ^
      static_cast<std::uint64_t>(ArithmeticShiftRight(mixed2, 11)));

  const std::uint32_t result_bits = static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(state->values[0]) ^
      static_cast<std::uint64_t>(state->values[1]) ^
      static_cast<std::uint64_t>(state->values[2]));
  return std::bit_cast<std::int32_t>(result_bits);
}

int PositiveModulo(std::int32_t value, std::uint32_t divisor) {
  int result = value % static_cast<std::int32_t>(divisor);
  if (result < 0) result += static_cast<int>(divisor);
  return result;
}

float PositiveGradientCoefficient(float distance_squared,
                                  float a,
                                  float b) {
  if (!(distance_squared > 0.0f)) return 0.0f;
  float result =
      -2.0f * a * b * std::pow(distance_squared, b - 1.0f);
  result /= a * std::pow(distance_squared, b) + 1.0f;
  return result;
}

float NegativeGradientCoefficient(float distance_squared,
                                  float a,
                                  float b,
                                  float gamma) {
  if (!(distance_squared > 0.0f)) return 0.0f;
  float result = 2.0f * gamma * b;
  result /=
      (0.001f + distance_squared) *
      (a * std::pow(distance_squared, b) + 1.0f);
  return result;
}

std::int32_t OptimizeShard(std::uint32_t worker_id, RunHeader* header) {
  float* embedding = OffsetPointer<float>(header->embedding_offset);
  const std::uint32_t* row_offsets =
      OffsetPointer<std::uint32_t>(header->row_offsets_offset);
  const std::int32_t* tail =
      OffsetPointer<std::int32_t>(header->tail_offset);
  const double* epochs_per_sample =
      OffsetPointer<double>(header->epochs_per_sample_offset);
  double* epoch_of_next_negative_sample =
      OffsetPointer<double>(header->epoch_of_next_negative_sample_offset);
  double* epoch_of_next_sample =
      OffsetPointer<double>(header->epoch_of_next_sample_offset);
  TauState* rng_state_per_vertex =
      OffsetPointer<TauState>(header->rng_state_per_vertex_offset);

  const float a = static_cast<float>(header->a);
  const float b = static_cast<float>(header->b);
  const float gamma = static_cast<float>(header->gamma);
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(gamma) ||
      a <= 0.0f || b <= 0.0f || gamma < 0.0f) {
    return kInvalidArgument;
  }
  float alpha = 1.0f;
  for (std::uint32_t epoch = 0; epoch < header->epoch_count; ++epoch) {
    while (true) {
      const std::uint32_t row_begin =
          AtomicFetchAdd(&header->next_row, kRowsPerChunk);
      if (row_begin >= header->vertex_count) break;
      const std::uint32_t row_end =
          std::min(row_begin + kRowsPerChunk, header->vertex_count);
      for (std::uint32_t head_index = row_begin;
           head_index < row_end;
           ++head_index) {
        const std::uint32_t edge_end = row_offsets[head_index + 1];
        for (std::uint32_t edge = row_offsets[head_index];
             edge < edge_end;
             ++edge) {
          if ((edge & 1023U) == 0 &&
              AtomicLoad(&header->cancelled) != 0) {
            return kCancelled;
          }
          if (epoch_of_next_sample[edge] > static_cast<double>(epoch)) {
            continue;
          }

          const std::uint32_t tail_index =
              static_cast<std::uint32_t>(tail[edge]);
          float* current = embedding +
              static_cast<std::uint64_t>(head_index) * header->dimension;
          float* other = embedding +
              static_cast<std::uint64_t>(tail_index) * header->dimension;

          float distance_squared =
              ReducedSquaredDistance(current, other, header->dimension);
          const float positive_coefficient =
              PositiveGradientCoefficient(distance_squared, a, b);
          for (std::uint32_t coordinate = 0;
               coordinate < header->dimension;
               ++coordinate) {
            const float difference =
                current[coordinate] - other[coordinate];
            const float gradient = Clip(positive_coefficient * difference);
            current[coordinate] += gradient * alpha;
            other[coordinate] -= gradient * alpha;
          }
          epoch_of_next_sample[edge] += epochs_per_sample[edge];

          const double negative_epoch =
              epochs_per_sample[edge] / header->negative_sample_rate;
          const std::int64_t negative_sample_count =
              static_cast<std::int64_t>(
                  (static_cast<double>(epoch) -
                   epoch_of_next_negative_sample[edge]) /
                  negative_epoch);
          for (std::int64_t sample = 0;
               sample < negative_sample_count;
               ++sample) {
            const std::uint32_t negative_tail =
                static_cast<std::uint32_t>(PositiveModulo(
                    TauRandInt(&rng_state_per_vertex[head_index]),
                    header->vertex_count));
            other = embedding +
                static_cast<std::uint64_t>(negative_tail) * header->dimension;
            distance_squared =
                ReducedSquaredDistance(current, other, header->dimension);
            if (!(distance_squared > 0.0f) &&
                head_index == negative_tail) {
              continue;
            }

            const float negative_coefficient =
                NegativeGradientCoefficient(distance_squared, a, b, gamma);
            for (std::uint32_t coordinate = 0;
                 coordinate < header->dimension;
                 ++coordinate) {
              float gradient = 0.0f;
              if (negative_coefficient > 0.0f) {
                const float difference =
                    current[coordinate] - other[coordinate];
                gradient = Clip(negative_coefficient * difference);
              }
              current[coordinate] += gradient * alpha;
            }
          }
          epoch_of_next_negative_sample[edge] +=
              static_cast<double>(negative_sample_count) * negative_epoch;
        }
      }
    }

    if (!Barrier(header)) return kCancelled;
    if (worker_id == 0) {
      AtomicStore(&header->completed_epochs, epoch + 1);
    }
    alpha =
        1.0f - (static_cast<float>(epoch) /
                static_cast<float>(header->epoch_count));
  }
  return kSuccess;
}

}  // namespace
}  // namespace senko::umap_layout_threaded

extern "C" {

SENKO_EXPORT("umap_layout_threaded_plan_offset")
std::uint32_t umap_layout_threaded_plan_offset(
    std::uint32_t section,
    std::uint32_t worker_count,
    std::uint32_t vertex_count,
    std::uint32_t dimension,
    std::uint32_t edge_count) {
  using namespace senko::umap_layout_threaded;
  if (section > kTotalBytes) return 0;
  Plan plan;
  if (!BuildPlan(worker_count, vertex_count, dimension, edge_count, &plan)) {
    return 0;
  }
  return plan.offsets[section];
}

SENKO_EXPORT("umap_layout_threaded_stack_top")
std::uint32_t umap_layout_threaded_stack_top(std::uint32_t worker_id) {
  using namespace senko::umap_layout_threaded;
  if (worker_id >= kMaximumWorkerCount) return 0;
  return kStackRegionOffset + (worker_id + 1) * kWorkerStackBytes;
}

SENKO_EXPORT("umap_layout_threaded_run")
std::int32_t umap_layout_threaded_run(
    std::uint32_t worker_id,
    senko::umap_layout_threaded::RunHeader* header) {
  using namespace senko::umap_layout_threaded;
  if (header == nullptr || header->worker_count == 0 ||
      worker_id >= header->worker_count) {
    return kInvalidArgument;
  }
  Plan plan;
  if (!BuildPlan(header->worker_count, header->vertex_count,
                 header->dimension, header->edge_count, &plan)) {
    if (worker_id == 0) Abort(header, kInvalidArgument);
    return kInvalidArgument;
  }

  if (worker_id == 0) {
    if (!InitializeSharedState(header, plan)) {
      Abort(header, kMemoryLayoutMismatch);
      return kMemoryLayoutMismatch;
    }
  }
  if (!Barrier(header)) {
    const std::int32_t status = AtomicLoad(&header->status);
    return status < 0 ? status : kCancelled;
  }

  const std::int32_t status = OptimizeShard(worker_id, header);
  if (status != kSuccess) {
    if (AtomicLoad(&header->cancelled) == 0) {
      Abort(header, static_cast<Status>(status));
    }
    return status;
  }
  if (worker_id == 0) {
    AtomicStore(&header->status, static_cast<std::int32_t>(kSuccess));
  }
  return kSuccess;
}

}
