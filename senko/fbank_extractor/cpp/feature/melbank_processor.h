// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#pragma once

#include <vector>
#include <cmath>
#include "melbank_options.h"

class MelBankProcessor {
public:
    MelBankProcessor() {}

    explicit MelBankProcessor(const MelBanksOptions &mel_opts);

    void init_mel_bins(float sample_frequency, int window_length_padded);

    std::vector<std::pair<int, std::vector<float>>> get_mel_bins() {
        return mel_bins_;
    }

    inline float inverse_mel_scale(float mel_freq) {
        return 700.0f * (std::exp(mel_freq / 1127.0f) - 1.0f);
    }

    inline float mel_scale(float freq) {
        return 1127.0f * std::log(1.0f + freq / 700.0f);
    }
private:
    MelBanksOptions opts_;
    std::vector<float> center_frequency_;
    std::vector<std::pair<int, std::vector<float>>> mel_bins_;
};
