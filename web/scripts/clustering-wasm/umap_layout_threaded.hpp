#ifndef SENKO_WEB_CLUSTERING_UMAP_LAYOUT_THREADED_HPP_
#define SENKO_WEB_CLUSTERING_UMAP_LAYOUT_THREADED_HPP_

#include <cstddef>
#include <cstdint>

/*
 * Shared-memory, allocation-free UMAP 0.5.12 layout optimizer.
 *
 * This module is deliberately separate from Senko's ordinary clustering Wasm:
 * WebAssembly.Memory must be declared shared when a module is compiled, so the
 * serial/growable arena cannot be upgraded in place. The browser coordinator
 * asks PlanOffset for the exact run layout, allocates a fixed shared memory,
 * copies the immutable inputs, fills RunHeader, and instantiates one module per
 * nested worker against that same memory.
 */

namespace senko::umap_layout_threaded {

constexpr std::uint32_t kHeaderMagic = 0x534b554d;  // "SKUM"
constexpr std::uint32_t kHeaderVersion = 1;
constexpr std::uint32_t kPageBytes = 65'536;
constexpr std::uint32_t kStackRegionOffset = 131'072;
constexpr std::uint32_t kWorkerStackBytes = 65'536;
constexpr std::uint32_t kMinimumMemoryBytes = 16 * 1024 * 1024;

enum Status : std::int32_t {
  kSuccess = 1,
  kRunning = 2,
  kInvalidArgument = -1,
  kCancelled = -2,
  kMemoryLayoutMismatch = -3,
};

enum PlanSection : std::uint32_t {
  kHeader = 0,
  kEmbedding = 1,
  kHead = 2,
  kTail = 3,
  kEpochsPerSample = 4,
  kRngSeed = 5,
  kEpochsPerNegativeSample = 6,
  kEpochOfNextNegativeSample = 7,
  kEpochOfNextSample = 8,
  kRngStatePerVertex = 9,
  kTotalBytes = 10,
};

/*
 * The byte offsets below are part of the JS/Wasm ABI. Immutable fields are
 * written by the coordinator before workers start. The five final words are
 * accessed atomically by Wasm and JavaScript.
 */
struct alignas(8) RunHeader {
  std::uint32_t magic;                              // 0
  std::uint32_t version;                            // 4
  std::uint32_t total_bytes;                        // 8
  std::uint32_t worker_count;                       // 12
  std::uint32_t vertex_count;                       // 16
  std::uint32_t dimension;                          // 20
  std::uint32_t edge_count;                         // 24
  std::uint32_t epoch_count;                        // 28
  std::uint32_t embedding_offset;                   // 32
  std::uint32_t head_offset;                        // 36
  std::uint32_t tail_offset;                        // 40
  std::uint32_t epochs_per_sample_offset;           // 44
  std::uint32_t rng_seed_offset;                    // 48
  std::uint32_t epochs_per_negative_sample_offset;  // 52
  std::uint32_t epoch_of_next_negative_sample_offset;  // 56
  std::uint32_t epoch_of_next_sample_offset;        // 60
  std::uint32_t rng_state_per_vertex_offset;        // 64
  std::uint32_t reserved0;                          // 68
  double a;                                         // 72
  double b;                                         // 80
  double gamma;                                     // 88
  double negative_sample_rate;                      // 96
  std::uint32_t arrived;                            // 104
  std::uint32_t generation;                         // 108
  std::uint32_t cancelled;                          // 112
  std::int32_t status;                              // 116
  std::uint32_t completed_epochs;                   // 120
  std::uint32_t reserved1;                          // 124
};

static_assert(sizeof(RunHeader) == 128);
static_assert(offsetof(RunHeader, a) == 72);
static_assert(offsetof(RunHeader, arrived) == 104);

}  // namespace senko::umap_layout_threaded

extern "C" {

std::uint32_t umap_layout_threaded_plan_offset(
    std::uint32_t section,
    std::uint32_t worker_count,
    std::uint32_t vertex_count,
    std::uint32_t dimension,
    std::uint32_t edge_count);

std::uint32_t umap_layout_threaded_stack_top(std::uint32_t worker_id);

std::int32_t umap_layout_threaded_run(
    std::uint32_t worker_id,
    senko::umap_layout_threaded::RunHeader* header);

}

#endif  // SENKO_WEB_CLUSTERING_UMAP_LAYOUT_THREADED_HPP_
