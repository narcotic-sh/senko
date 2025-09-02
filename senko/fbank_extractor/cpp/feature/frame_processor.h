// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#pragma once

#include <vector>
#include "frame_extraction_options.h"
#include <random>

class FramePreprocessor {
public:
    FramePreprocessor() {}

    explicit FramePreprocessor(const FrameExtractionOptions &frame_opts);

    void dither(std::vector<float> &wav_data);

    void remove_dc_offset(std::vector<float> &wav_data);

    void pre_emphasis(std::vector<float> &wav_data);

    void windows_function(std::vector<float> &wav_data);

    void frame_pre_process(std::vector<float> &wav_data);

private:
    FrameExtractionOptions opts_;
    std::default_random_engine generator_;
    std::normal_distribution<float> distribution_;
};