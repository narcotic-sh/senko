// Generates native-fbank-1p5s.f32 from Senko's production C++ extractor.
// See README.md for the reproducible command.

#include "feature_computer.h"

#include <cstdint>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr int kSamples = 24000;

std::vector<float> MakeSamples() {
  std::vector<float> samples(kSamples);
  uint32_t state = 0x12345678u;
  for (int i = 0; i < kSamples; ++i) {
    state = state * 1664525u + 1013904223u;
    const int32_t pcm = (static_cast<int32_t>(state >> 16) - 32768) >> 2;
    samples[i] = static_cast<float>(pcm) / 32768.0f;
  }
  return samples;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  auto samples = MakeSamples();
  FeatureComputer computer;
  auto features = computer.compute_feature(std::span<float>(samples));

  std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
  if (!output) return 3;
  for (const auto& frame : features) {
    output.write(reinterpret_cast<const char*>(frame.data()),
                 static_cast<std::streamsize>(frame.size() * sizeof(float)));
  }
  return output ? 0 : 4;
}
