#include "umap_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
std::vector<T> ReadBinary(const std::filesystem::path& path) {
  static_assert(std::is_trivially_copyable_v<T>);
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("Could not open " + path.string());
  const std::streamsize bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
    throw std::runtime_error("Invalid byte length for " + path.string());
  }
  stream.seekg(0);
  std::vector<T> values(static_cast<size_t>(bytes) / sizeof(T));
  if (bytes > 0 &&
      !stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
    throw std::runtime_error("Could not read " + path.string());
  }
  return values;
}

template <typename T>
void WriteBinary(const std::filesystem::path& path,
                 const std::vector<T>& values) {
  static_assert(std::is_trivially_copyable_v<T>);
  std::ofstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Could not create " + path.string());
  const size_t byte_count = values.size() * sizeof(T);
  if (byte_count > 0) {
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(byte_count));
  }
  if (!stream) throw std::runtime_error("Could not write " + path.string());
}

double SquaredDistance(const std::vector<float>& values, int left, int right,
                       int dimension) {
  const float* left_row =
      values.data() + static_cast<size_t>(left) * dimension;
  const float* right_row =
      values.data() + static_cast<size_t>(right) * dimension;
  double result = 0.0;
  for (int coordinate = 0; coordinate < dimension; ++coordinate) {
    const double difference =
        static_cast<double>(left_row[coordinate]) - right_row[coordinate];
    result += difference * difference;
  }
  return result;
}

struct ErrorMetrics {
  double maximum_absolute = 0.0;
  double root_mean_square = 0.0;
  double relative_l2 = 0.0;
};

ErrorMetrics CoordinateError(const std::vector<float>& actual,
                             const std::vector<float>& expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("Coordinate vector sizes differ");
  }
  long double squared_error = 0.0;
  long double squared_expected = 0.0;
  double maximum_absolute = 0.0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const double difference =
        static_cast<double>(actual[index]) - expected[index];
    maximum_absolute = std::max(maximum_absolute, std::abs(difference));
    squared_error += static_cast<long double>(difference) * difference;
    squared_expected +=
        static_cast<long double>(expected[index]) * expected[index];
  }
  return {
      maximum_absolute,
      std::sqrt(static_cast<double>(squared_error / actual.size())),
      std::sqrt(static_cast<double>(squared_error / squared_expected)),
  };
}

double SampledDistanceRelativeL2(const std::vector<float>& actual,
                                 const std::vector<float>& expected,
                                 int vertex_count, int dimension) {
  constexpr int kSampleCount = 50000;
  uint64_t random = 0x243f6a8885a308d3ULL;
  long double squared_error = 0.0;
  long double squared_expected = 0.0;
  for (int sample = 0; sample < kSampleCount; ++sample) {
    random = random * 6364136223846793005ULL + 1442695040888963407ULL;
    const int left = static_cast<int>((random >> 32) %
                                      static_cast<uint64_t>(vertex_count));
    random = random * 6364136223846793005ULL + 1442695040888963407ULL;
    const int right = static_cast<int>((random >> 32) %
                                       static_cast<uint64_t>(vertex_count));
    const double actual_distance =
        std::sqrt(SquaredDistance(actual, left, right, dimension));
    const double expected_distance =
        std::sqrt(SquaredDistance(expected, left, right, dimension));
    const double difference = actual_distance - expected_distance;
    squared_error += static_cast<long double>(difference) * difference;
    squared_expected +=
        static_cast<long double>(expected_distance) * expected_distance;
  }
  return std::sqrt(static_cast<double>(squared_error / squared_expected));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 4) {
      std::cerr << "usage: " << argv[0]
                << " FIXTURE_DIRECTORY [EPOCH_COUNT] [OUTPUT_F32]\n";
      return 2;
    }
    const std::filesystem::path fixture = argv[1];
    const int epoch_count = argc >= 3 ? std::stoi(argv[2]) : 500;

    std::vector<float> embedding =
        ReadBinary<float>(fixture / "umap-layout-initial-embedding.f32");
    const std::vector<float> expected =
        ReadBinary<float>(fixture / "umap-projection.f32");
    const std::vector<int32_t> head =
        ReadBinary<int32_t>(fixture / "umap-layout-head.i32");
    const std::vector<int32_t> tail =
        ReadBinary<int32_t>(fixture / "umap-layout-tail.i32");
    const std::vector<double> epochs_per_sample =
        ReadBinary<double>(fixture / "umap-layout-epochs-per-sample.f64");
    const std::vector<int64_t> rng_state =
        ReadBinary<int64_t>(fixture / "umap-layout-rng-state.i64");

    if (head.empty() || head.size() != tail.size() ||
        head.size() != epochs_per_sample.size() || rng_state.size() != 3 ||
        embedding.size() != expected.size()) {
      throw std::runtime_error("Fixture arrays have inconsistent shapes");
    }
    const int vertex_count =
        *std::max_element(head.begin(), head.end()) + 1;
    if (vertex_count <= 0 ||
        embedding.size() % static_cast<size_t>(vertex_count) != 0) {
      throw std::runtime_error("Could not infer fixture shape");
    }
    const size_t inferred_dimension =
        embedding.size() / static_cast<size_t>(vertex_count);
    if (inferred_dimension >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("Fixture dimension exceeds int range");
    }
    const int dimension = static_cast<int>(inferred_dimension);
    const int edge_count = static_cast<int>(head.size());

    const size_t workspace_bytes =
        senko::umap_layout::SerialWorkspaceBytes(vertex_count, edge_count);
    std::vector<uint8_t> workspace_storage(workspace_bytes + 15);
    const uintptr_t workspace_address =
        (reinterpret_cast<uintptr_t>(workspace_storage.data()) + 15) &
        ~uintptr_t{15};
    void* workspace = reinterpret_cast<void*>(workspace_address);
    const size_t aligned_workspace_bytes =
        workspace_storage.size() -
        static_cast<size_t>(workspace_address -
                            reinterpret_cast<uintptr_t>(
                                workspace_storage.data()));

    const auto started = std::chrono::steady_clock::now();
    const int status = senko::umap_layout::OptimizeSerial(
        embedding.data(), vertex_count, dimension, head.data(), tail.data(),
        epochs_per_sample.data(), edge_count, rng_state.data(), epoch_count,
        1.932808397545408, 0.7904949735905139, 1.0, 5.0,
        workspace, aligned_workspace_bytes);
    const auto ended = std::chrono::steady_clock::now();
    if (status != senko::umap_layout::kSuccess) {
      throw std::runtime_error("OptimizeSerial failed with status " +
                               std::to_string(status));
    }

    const ErrorMetrics error = CoordinateError(embedding, expected);
    const double sampled_distance_error =
        SampledDistanceRelativeL2(embedding, expected, vertex_count, dimension);
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(ended - started).count();

    std::cout << std::setprecision(12)
              << "vertices=" << vertex_count << "\n"
              << "dimension=" << dimension << "\n"
              << "edges=" << edge_count << "\n"
              << "epochs=" << epoch_count << "\n"
              << "workspaceBytes=" << workspace_bytes << "\n"
              << "elapsedMs=" << elapsed_ms << "\n"
              << "coordinateMaxAbs=" << error.maximum_absolute << "\n"
              << "coordinateRmse=" << error.root_mean_square << "\n"
              << "coordinateRelativeL2=" << error.relative_l2 << "\n"
              << "sampledPairDistanceRelativeL2="
              << sampled_distance_error << "\n";

    if (argc == 4) WriteBinary<float>(argv[3], embedding);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
