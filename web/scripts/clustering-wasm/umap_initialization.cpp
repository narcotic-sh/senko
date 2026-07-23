#include "umap_initialization.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace senko::umap_initialization {
namespace {

constexpr std::size_t kAlignment = 16;
constexpr std::int64_t kRandomIntLow = -2147483647LL;
constexpr std::uint64_t kRandomIntRangeMaximum = 4294967292ULL;
constexpr double kNoiseScale = 0.0001;

bool CheckedMultiply(std::size_t left,
                     std::size_t right,
                     std::size_t* result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool IsAligned(const void* pointer) {
  return (reinterpret_cast<std::uintptr_t>(pointer) & (kAlignment - 1)) == 0;
}

class NumpyRandomState {
 public:
  explicit NumpyRandomState(std::uint32_t seed) { Seed(seed); }

  std::int64_t NextSignedRangeValue() {
    std::uint32_t sample = 0;
    do {
      sample = NextUint32();
    } while (static_cast<std::uint64_t>(sample) >
             kRandomIntRangeMaximum);
    return kRandomIntLow + static_cast<std::int64_t>(sample);
  }

  double NextStandardNormal() {
    if (has_gaussian_) {
      has_gaussian_ = false;
      const double result = gaussian_;
      gaussian_ = 0.0;
      return result;
    }
    double first = 0.0;
    double second = 0.0;
    double radius_squared = 0.0;
    do {
      first = 2.0 * NextDouble() - 1.0;
      second = 2.0 * NextDouble() - 1.0;
      radius_squared = first * first + second * second;
    } while (radius_squared >= 1.0 || radius_squared == 0.0);
    const double factor =
        std::sqrt(-2.0 * std::log(radius_squared) / radius_squared);
    gaussian_ = factor * first;
    has_gaussian_ = true;
    return factor * second;
  }

 private:
  void Seed(std::uint32_t seed) {
    state_[0] = seed;
    for (std::int32_t index = 1; index < 624; ++index) {
      state_[index] =
          1812433253u * (state_[index - 1] ^ (state_[index - 1] >> 30)) +
          static_cast<std::uint32_t>(index);
    }
    cursor_ = 624;
  }

  void Twist() {
    constexpr std::uint32_t upper_mask = 0x80000000u;
    constexpr std::uint32_t lower_mask = 0x7fffffffu;
    for (std::int32_t index = 0; index < 624; ++index) {
      const std::uint32_t bits =
          (state_[index] & upper_mask) |
          (state_[(index + 1) % 624] & lower_mask);
      state_[index] =
          state_[(index + 397) % 624] ^ (bits >> 1) ^
          ((bits & 1u) == 0 ? 0u : 0x9908b0dfu);
    }
    cursor_ = 0;
  }

  std::uint32_t NextUint32() {
    if (cursor_ == 624) Twist();
    std::uint32_t result = state_[cursor_++];
    result ^= result >> 11;
    result ^= (result << 7) & 0x9d2c5680u;
    result ^= (result << 15) & 0xefc60000u;
    result ^= result >> 18;
    return result;
  }

  double NextDouble() {
    const std::uint32_t first = NextUint32() >> 5;
    const std::uint32_t second = NextUint32() >> 6;
    return (static_cast<double>(first) * 67108864.0 +
            static_cast<double>(second)) /
           9007199254740992.0;
  }

  std::uint32_t state_[624]{};
  std::int32_t cursor_ = 624;
  bool has_gaussian_ = false;
  double gaussian_ = 0.0;
};

std::int32_t ResolveTreeCount(std::int32_t count) {
  const double suggested =
      5.0 + std::nearbyint(std::sqrt(static_cast<double>(count)) / 20.0);
  return std::min(64, static_cast<std::int32_t>(suggested));
}

void AdvancePastNeighborSearch(NumpyRandomState* random,
                               std::int32_t count,
                               bool approximate_neighbors) {
  if (!approximate_neighbors) return;
  const std::int32_t values =
      6 + 3 * ResolveTreeCount(count);
  for (std::int32_t index = 0; index < values; ++index) {
    random->NextSignedRangeValue();
  }
}

}  // namespace

std::size_t WorkspaceBytes(std::int32_t dimension) {
  if (dimension <= 0) return 0;
  std::size_t bytes = 0;
  if (!CheckedMultiply(static_cast<std::size_t>(dimension),
                       sizeof(float) * 2, &bytes)) {
    return 0;
  }
  return bytes;
}

Status Initialize(const double* spectral_embedding,
                  std::int32_t count,
                  std::int32_t dimension,
                  std::uint32_t random_seed,
                  bool approximate_neighbors,
                  void* workspace,
                  std::size_t workspace_bytes,
                  float* output_embedding,
                  std::int64_t* output_layout_rng_state) {
  std::size_t value_count = 0;
  const std::size_t required = WorkspaceBytes(dimension);
  if (spectral_embedding == nullptr || workspace == nullptr ||
      output_embedding == nullptr || output_layout_rng_state == nullptr ||
      !IsAligned(workspace) || count < 3 || dimension < 1 ||
      dimension > count - 2 ||
      !CheckedMultiply(static_cast<std::size_t>(count),
                       static_cast<std::size_t>(dimension), &value_count)) {
    return Status::kInvalidArgument;
  }
  if (workspace_bytes < required) return Status::kInsufficientWorkspace;

  NumpyRandomState random(random_seed);
  AdvancePastNeighborSearch(&random, count, approximate_neighbors);

  std::size_t discarded_normals = 0;
  if (!CheckedMultiply(static_cast<std::size_t>(count),
                       static_cast<std::size_t>(dimension + 1),
                       &discarded_normals)) {
    return Status::kInvalidArgument;
  }
  for (std::size_t index = 0; index < discarded_normals; ++index) {
    static_cast<void>(random.NextStandardNormal());
  }

  double maximum_absolute_coordinate = 0.0;
  for (std::size_t index = 0; index < value_count; ++index) {
    const double value = spectral_embedding[index];
    if (!std::isfinite(value)) return Status::kInvalidArgument;
    maximum_absolute_coordinate =
        std::max(maximum_absolute_coordinate, std::abs(value));
  }
  if (!(maximum_absolute_coordinate > 0.0) ||
      !std::isfinite(maximum_absolute_coordinate)) {
    return Status::kNumericalFailure;
  }
  const double expansion = 10.0 / maximum_absolute_coordinate;
  for (std::size_t index = 0; index < value_count; ++index) {
    const float scaled =
        static_cast<float>(spectral_embedding[index] * expansion);
    const float noise =
        static_cast<float>(random.NextStandardNormal() * kNoiseScale);
    output_embedding[index] = scaled + noise;
  }

  auto* minimum = static_cast<float*>(workspace);
  auto* maximum = minimum + dimension;
  std::fill_n(minimum, dimension, std::numeric_limits<float>::infinity());
  std::fill_n(maximum, dimension, -std::numeric_limits<float>::infinity());
  for (std::int32_t row = 0; row < count; ++row) {
    const float* source =
        output_embedding + static_cast<std::size_t>(row) * dimension;
    for (std::int32_t column = 0; column < dimension; ++column) {
      minimum[column] = std::min(minimum[column], source[column]);
      maximum[column] = std::max(maximum[column], source[column]);
    }
  }
  for (std::int32_t column = 0; column < dimension; ++column) {
    if (!(maximum[column] > minimum[column])) {
      return Status::kNumericalFailure;
    }
  }
  for (std::int32_t row = 0; row < count; ++row) {
    float* target =
        output_embedding + static_cast<std::size_t>(row) * dimension;
    for (std::int32_t column = 0; column < dimension; ++column) {
      const float shifted = target[column] - minimum[column];
      const float numerator = 10.0f * shifted;
      target[column] = numerator / (maximum[column] - minimum[column]);
    }
  }

  for (std::int32_t index = 0; index < 3; ++index) {
    output_layout_rng_state[index] = random.NextSignedRangeValue();
  }
  return Status::kSuccess;
}

}  // namespace senko::umap_initialization
