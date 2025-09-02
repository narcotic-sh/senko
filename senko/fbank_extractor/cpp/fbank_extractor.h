#pragma once

#include <string>
#include <vector>

struct FbankResult {
    std::vector<float> features;
    std::vector<size_t> frames_per_subsegment;
    std::vector<size_t> subsegment_offsets;
};

class FbankExtractor {
public:
    FbankExtractor();
    ~FbankExtractor() = default;

    FbankResult extract_features(const std::string& wav_path, const std::vector<std::pair<float, float>>& subsegments);
};