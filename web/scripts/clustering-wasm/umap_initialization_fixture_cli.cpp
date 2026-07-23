#include "umap_initialization.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename T>
bool ReadBinary(const std::filesystem::path& path, std::vector<T>* output) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return false;
  const std::streamsize bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
    return false;
  }
  output->resize(static_cast<std::size_t>(bytes) / sizeof(T));
  stream.seekg(0);
  return static_cast<bool>(
      stream.read(reinterpret_cast<char*>(output->data()), bytes));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: umap_initialization_fixture_cli FIXTURE_DIR COUNT "
                 "DIM SEED APPROXIMATE\n";
    return 2;
  }
  const std::filesystem::path fixture(argv[1]);
  const int count = std::stoi(argv[2]);
  const int dimension = std::stoi(argv[3]);
  const std::uint32_t seed =
      static_cast<std::uint32_t>(std::stoul(argv[4]));
  const bool approximate = std::stoi(argv[5]) != 0;
  std::vector<double> spectral;
  std::vector<float> reference_embedding;
  std::vector<std::int64_t> reference_rng;
  if (!ReadBinary(fixture / "umap-spectral-embedding.f64", &spectral) ||
      !ReadBinary(fixture / "umap-layout-initial-embedding.f32",
                  &reference_embedding) ||
      !ReadBinary(fixture / "umap-layout-rng-state.i64", &reference_rng) ||
      spectral.size() != static_cast<std::size_t>(count) * dimension ||
      reference_embedding.size() != spectral.size() ||
      reference_rng.size() != 3) {
    std::cerr << "failed to read exact-size fixture artifacts\n";
    return 3;
  }

  const std::size_t workspace_bytes =
      senko::umap_initialization::WorkspaceBytes(dimension);
  std::vector<std::uint8_t> workspace(workspace_bytes + 15);
  const std::uintptr_t aligned_address =
      (reinterpret_cast<std::uintptr_t>(workspace.data()) +
       std::uintptr_t{15}) &
      ~std::uintptr_t{15};
  void* aligned_workspace = reinterpret_cast<void*>(aligned_address);
  std::vector<float> candidate(reference_embedding.size());
  std::int64_t candidate_rng[3]{};
  const auto started = std::chrono::steady_clock::now();
  const auto status = senko::umap_initialization::Initialize(
      spectral.data(), count, dimension, seed, approximate, aligned_workspace,
      workspace_bytes, candidate.data(), candidate_rng);
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  if (status != senko::umap_initialization::Status::kSuccess) {
    std::cerr << "initialization failed with status "
              << static_cast<int>(status) << "\n";
    return 4;
  }

  std::size_t exact_values = 0;
  double maximum_error = 0.0;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    const double difference =
        static_cast<double>(candidate[index]) - reference_embedding[index];
    exact_values += candidate[index] == reference_embedding[index];
    maximum_error = std::max(maximum_error, std::abs(difference));
    squared_error += difference * difference;
    squared_reference +=
        static_cast<double>(reference_embedding[index]) *
        reference_embedding[index];
  }
  const bool exact_rng =
      std::equal(reference_rng.begin(), reference_rng.end(), candidate_rng);
  std::cout << "{\"elapsedMs\":" << elapsed_ms
            << ",\"exactValueFraction\":"
            << static_cast<double>(exact_values) / candidate.size()
            << ",\"maximumAbsoluteError\":" << maximum_error
            << ",\"relativeFrobeniusError\":"
            << std::sqrt(squared_error / squared_reference)
            << ",\"exactRngState\":" << (exact_rng ? "true" : "false")
            << "}\n";
  return maximum_error <= 1.0e-5 && exact_rng ? 0 : 5;
}
