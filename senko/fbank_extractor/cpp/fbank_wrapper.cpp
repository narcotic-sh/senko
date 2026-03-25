#include "fbank_interface.h"
#include "fbank_extractor.h"

#include <vector>
#include <cstring>

extern "C" {

    static std::vector<std::pair<float, float>> build_subsegments(float* subsegments_array, size_t num_subsegments) {
        std::vector<std::pair<float, float>> subsegments;
        subsegments.reserve(num_subsegments);
        for (size_t i = 0; i < num_subsegments; ++i) {
            subsegments.emplace_back(subsegments_array[i*2], subsegments_array[i*2 + 1]);
        }
        return subsegments;
    }

    static FbankFeatures build_feature_result(const FbankResult& result, size_t num_subsegments) {
        FbankFeatures features;

        features.data = new float[result.features.size()];
        std::memcpy(features.data, result.features.data(), result.features.size() * sizeof(float));

        features.frames_per_subsegment = new size_t[result.frames_per_subsegment.size()];
        std::memcpy(features.frames_per_subsegment, result.frames_per_subsegment.data(),
                   result.frames_per_subsegment.size() * sizeof(size_t));

        features.subsegment_offsets = new size_t[result.subsegment_offsets.size()];
        std::memcpy(features.subsegment_offsets, result.subsegment_offsets.data(),
                   result.subsegment_offsets.size() * sizeof(size_t));

        features.num_subsegments = num_subsegments;
        features.feature_dim = 80;
        features.total_frames = 0;
        for (size_t frames : result.frames_per_subsegment) {
            features.total_frames += frames;
        }

        return features;
    }

    FbankExtractorHandle create_fbank_extractor() {
        auto* extractor = new FbankExtractor();
        return reinterpret_cast<FbankExtractorHandle>(extractor);
    }

    void destroy_fbank_extractor(FbankExtractorHandle handle) {
        if (handle) {
            auto* extractor = reinterpret_cast<FbankExtractor*>(handle);
            delete extractor;
        }
    }

    FbankFeatures extract_fbank_features(FbankExtractorHandle handle,
                                        const char* wav_path,
                                        float* subsegments_array,
                                        size_t num_subsegments) {
        auto* extractor = reinterpret_cast<FbankExtractor*>(handle);
        auto subsegments = build_subsegments(subsegments_array, num_subsegments);
        auto result = extractor->extract_features(wav_path, subsegments);
        return build_feature_result(result, num_subsegments);
    }

    FbankFeatures extract_fbank_features_from_memory(FbankExtractorHandle handle,
                                                    const float* samples,
                                                    size_t num_samples,
                                                    float* subsegments_array,
                                                    size_t num_subsegments) {
        auto* extractor = reinterpret_cast<FbankExtractor*>(handle);
        auto subsegments = build_subsegments(subsegments_array, num_subsegments);
        auto result = extractor->extract_features_from_memory(samples, num_samples, subsegments);
        return build_feature_result(result, num_subsegments);
    }

    void free_fbank_features(FbankFeatures* features) {
        if (features) {
            delete[] features->data;
            delete[] features->frames_per_subsegment;
            delete[] features->subsegment_offsets;
        }
    }

}
