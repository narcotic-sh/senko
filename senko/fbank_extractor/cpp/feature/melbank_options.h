// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#pragma once

#include <string>
#include <sstream>

struct MelBanksOptions {
    int num_bins;
    float low_freq;
    float high_freq;
    float vtln_low;
    float vtln_high;

    explicit MelBanksOptions(int num_bins = 25) :
            num_bins(num_bins),
            low_freq(20),
            high_freq(0),
            vtln_low(100),
            vtln_high(-500) {}

    std::string show() const {
        std::ostringstream oss;
        oss << "MelBanksOptions [ "<< "num_bins: " << num_bins << "\t"
            << "low_freq: " << low_freq << "\t"
            << "high_freq: " << high_freq << "\t"
            << "vtln_low: " << vtln_low << "\t"
            << "vtln_high: " << vtln_high << " ]";
        return oss.str();
    }
};