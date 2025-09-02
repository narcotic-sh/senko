// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#pragma once

#include <vector>
#include "frame_extraction_options.h"
#include "melbank_options.h"

struct FbankOptions {
    FrameExtractionOptions frame_opts;
    MelBanksOptions mel_opts;
    bool use_energy;
    float energy_floor;
    bool raw_energy;
    bool use_log_fbank;
    bool use_power;

    explicit FbankOptions() :
            mel_opts(80),
            use_energy(false),
            energy_floor(0.0),
            raw_energy(true),
            use_log_fbank(true),
            use_power(true) {}

    inline int compute_window_shift() { return frame_opts.compute_window_shift(); }

    inline int compute_window_size() { return frame_opts.compute_window_size(); }

    inline int paddle_window_size() { return frame_opts.padded_window_size(); }

    inline int get_fbank_num_bins() { return mel_opts.num_bins; }

    std::string show() const {
        std::string frame_str = frame_opts.show();
        std::string mel_str = mel_opts.show();
        std::ostringstream oss;
        oss << "FbankOptions [ " << frame_str << "\n" << mel_str << "\n"
            << "use_energy: " << (use_energy ? "true" : "false") << "\t"
            << "energy_floor: " << energy_floor << "\t"
            << "raw_energy: " << (raw_energy ? "true" : "false") << "\t"
            << "use_log_fbank: " << (use_log_fbank ? "true" : "false") << "\t"
            << "use_power: " << (use_power ? "true" : "false") << "]";

        return oss.str();
    }
};