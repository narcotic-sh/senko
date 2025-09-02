// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#pragma once

#include <vector>
#include <span>

#include "fbank_options.h"
#include "frame_processor.h"
#include "melbank_processor.h"

typedef std::vector<std::vector<float>> Feature;
typedef std::vector<float> Wave;

class FbankComputer {
public:
    FbankComputer() {};

    explicit FbankComputer(const FbankOptions &opts);

    Feature compute_feature(std::span<float> float_wav_data);

private:
    FbankOptions opts_;
    FramePreprocessor frame_preprocessor_;
    MelBankProcessor mel_bank_processor_;
    float log_energy_floor_;
    std::vector<int> bit_rev_index_;
    std::vector<float> sin_tbl_;
};