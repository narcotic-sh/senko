#include "fbank_interface.h"
#include "fbank_extractor.h"

#include <vector>
#include <cstring>

extern "C" {

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

        // Convert subsegments array to vector of pairs
        std::vector<std::pair<float, float>> subsegments;
        subsegments.reserve(num_subsegments);
        for (size_t i = 0; i < num_subsegments; ++i) {
            subsegments.emplace_back(subsegments_array[i*2], subsegments_array[i*2 + 1]);
        }

        // Extract features
        auto result = extractor->extract_features(wav_path, subsegments);

        FbankFeatures features;

        // Copy flattened features
        features.data = new float[result.features.size()];
        std::memcpy(features.data, result.features.data(), result.features.size() * sizeof(float));

        // Copy frame counts
        features.frames_per_subsegment = new size_t[result.frames_per_subsegment.size()];
        std::memcpy(features.frames_per_subsegment, result.frames_per_subsegment.data(),
                   result.frames_per_subsegment.size() * sizeof(size_t));

        // Copy offsets
        features.subsegment_offsets = new size_t[result.subsegment_offsets.size()];
        std::memcpy(features.subsegment_offsets, result.subsegment_offsets.data(),
                   result.subsegment_offsets.size() * sizeof(size_t));

        // Set metadata
        features.num_subsegments = num_subsegments;
        features.feature_dim = 80;

        // Calculate total frames
        features.total_frames = 0;
        for (size_t frames : result.frames_per_subsegment) {
            features.total_frames += frames;
        }

        return features;
    }

    void free_fbank_features(FbankFeatures* features) {
        if (features) {
            delete[] features->data;
            delete[] features->frames_per_subsegment;
            delete[] features->subsegment_offsets;
        }
    }

}
