#pragma once
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        float* data;                    // Flattened feature array
        size_t* frames_per_subsegment;  // Number of frames for each subsegment
        size_t* subsegment_offsets;     // Offset of each subsegment in flattened feature array
        size_t num_subsegments;         // Number of subsegments
        size_t total_frames;            // Total frames across all subsegments
        size_t feature_dim;             // Feature dimension (80 for mel bins)
    } FbankFeatures;

    // Opaque pointer
    typedef struct FbankExtractorWrapper* FbankExtractorHandle;

    // Create/destroy extractor
    FbankExtractorHandle create_fbank_extractor();
    void destroy_fbank_extractor(FbankExtractorHandle handle);

    // Extract features
    FbankFeatures extract_fbank_features(FbankExtractorHandle handle,
                                        const char* wav_path,
                                        float* subsegments_array,
                                        size_t num_subsegments);

    // Free features
    void free_fbank_features(FbankFeatures* features);

#ifdef __cplusplus
}
#endif