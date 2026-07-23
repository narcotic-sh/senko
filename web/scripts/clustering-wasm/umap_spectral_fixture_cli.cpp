#include "umap_spectral.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

template <typename T>
bool read_binary(const std::filesystem::path& path, std::vector<T>* output) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    return false;
  }
  const std::streamsize bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
    return false;
  }
  output->resize(static_cast<std::size_t>(bytes) / sizeof(T));
  stream.seekg(0);
  return static_cast<bool>(
      stream.read(reinterpret_cast<char*>(output->data()), bytes));
}

template <typename T>
bool write_binary(const std::filesystem::path& path,
                  const std::vector<T>& values) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    return false;
  }
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  return static_cast<bool>(stream);
}

bool parse_positive(const char* text, std::int32_t* value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<std::int32_t>::max()) {
    return false;
  }
  *value = static_cast<std::int32_t>(parsed);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5 || argc > 6) {
    std::cerr
        << "usage: umap_spectral_fixture_cli FIXTURE_DIR OUTPUT_PREFIX "
           "DIM EPOCHS [BASIS]\n";
    return 2;
  }

  std::int32_t dim = 0;
  std::int32_t epochs = 0;
  std::int32_t basis = 0;
  if (!parse_positive(argv[3], &dim) || !parse_positive(argv[4], &epochs) ||
      (argc == 6 && !parse_positive(argv[5], &basis))) {
    std::cerr << "DIM, EPOCHS, and BASIS must be positive integers\n";
    return 2;
  }

  const std::filesystem::path fixture(argv[1]);
  const std::filesystem::path output_prefix(argv[2]);
  std::vector<std::int32_t> source_offsets;
  std::vector<std::int32_t> source_columns;
  std::vector<float> source_weights;
  if (!read_binary(fixture / "umap-graph-indptr.i32", &source_offsets) ||
      !read_binary(fixture / "umap-graph-indices.i32", &source_columns) ||
      !read_binary(fixture / "umap-graph-data.f32", &source_weights) ||
      source_offsets.size() < 2 ||
      source_columns.size() != source_weights.size()) {
    std::cerr << "failed to read UMAP graph fixture\n";
    return 2;
  }

  const std::int32_t count =
      static_cast<std::int32_t>(source_offsets.size() - 1);
  const float maximum =
      *std::max_element(source_weights.begin(), source_weights.end());
  const float cutoff = maximum / static_cast<float>(epochs);
  std::vector<std::int32_t> offsets(static_cast<std::size_t>(count) + 1);
  std::vector<std::int32_t> columns;
  std::vector<float> weights;
  columns.reserve(source_columns.size());
  weights.reserve(source_weights.size());
  for (std::int32_t row = 0; row < count; ++row) {
    for (std::int32_t edge = source_offsets[static_cast<std::size_t>(row)];
         edge < source_offsets[static_cast<std::size_t>(row + 1)]; ++edge) {
      if (source_weights[static_cast<std::size_t>(edge)] < cutoff) {
        continue;
      }
      columns.push_back(source_columns[static_cast<std::size_t>(edge)]);
      weights.push_back(source_weights[static_cast<std::size_t>(edge)]);
    }
    offsets[static_cast<std::size_t>(row + 1)] =
        static_cast<std::int32_t>(columns.size());
  }

  std::vector<double> vectors(static_cast<std::size_t>(count) * dim);
  std::vector<double> eigenvalues(static_cast<std::size_t>(dim));
  senko::umap_spectral::Options options;
  options.maximum_basis_size = basis;
  const std::size_t workspace_bytes = senko::umap_spectral::workspace_bytes(
      count, static_cast<std::int32_t>(columns.size()), dim, options);
  if (workspace_bytes == 0) {
    std::cerr << "invalid spectral workspace dimensions\n";
    return 2;
  }
  std::vector<std::uint64_t> workspace(
      (workspace_bytes + sizeof(std::uint64_t) - 1) /
      sizeof(std::uint64_t));
  senko::umap_spectral::Stats stats;
  const auto started = std::chrono::steady_clock::now();
  const auto status = senko::umap_spectral::initialize_connected_graph(
      offsets.data(), columns.data(), weights.data(), count,
      static_cast<std::int32_t>(columns.size()), dim, vectors.data(),
      eigenvalues.data(), workspace.data(),
      workspace.size() * sizeof(std::uint64_t), options, &stats);
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();

  std::cout << "{\n"
            << "  \"status\": " << static_cast<std::int32_t>(status) << ",\n"
            << "  \"message\": \""
            << senko::umap_spectral::status_message(status) << "\",\n"
            << "  \"count\": " << count << ",\n"
            << "  \"retainedEdges\": " << columns.size() << ",\n"
            << "  \"basisSize\": " << stats.basis_size << ",\n"
            << "  \"restartCount\": " << stats.restart_count << ",\n"
            << "  \"convergedEigenpairs\": " << stats.converged_eigenpairs
            << ",\n"
            << "  \"maximumResidual\": " << stats.maximum_residual << ",\n"
            << "  \"smallestEigenvalue\": " << stats.smallest_eigenvalue
            << ",\n"
            << "  \"largestReturnedEigenvalue\": "
            << stats.largest_returned_eigenvalue << ",\n"
            << "  \"peakWorkingBytes\": " << stats.peak_working_bytes << ",\n"
            << "  \"elapsedMs\": " << elapsed_ms << "\n"
            << "}\n";
  if (status != senko::umap_spectral::Status::kSuccess) {
    return 1;
  }
  if (!write_binary(output_prefix.string() + "-vectors.f64", vectors) ||
      !write_binary(output_prefix.string() + "-eigenvalues.f64",
                    eigenvalues)) {
    std::cerr << "failed to write spectral output\n";
    return 2;
  }
  return 0;
}
