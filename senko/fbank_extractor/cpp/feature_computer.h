// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#pragma once

#include <vector>
#include <span>
#include "feature/fbank_computer.h"
#include "feature/fbank_options.h"

class FeatureComputer {
public:
    FeatureComputer();
    Feature compute_feature(std::span<float> wav_span);

private:
    FbankOptions opts;
    FbankComputer fbank_computer;
};