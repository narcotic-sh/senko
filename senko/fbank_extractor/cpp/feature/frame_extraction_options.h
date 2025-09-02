// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#pragma once

#include <string>
#include <sstream>

struct FrameExtractionOptions {
    float sample_freq;
    float frame_shift_ms; // frame shift in million seconds
    float frame_length_ms; // frame length in million seconds
    float dither;
    float pre_emphasis_coefficient;
    bool remove_dc_offset;
    std::string window_type;
    bool round_to_power_of_two;
    float blackman_coefficient;
    bool snip_edges;
    bool allow_down_sample;
    bool allow_up_sample;
    int max_feature_vectors;

    explicit FrameExtractionOptions() :
            sample_freq(16000),
            frame_shift_ms(10.0),
            frame_length_ms(25.0),
            dither(1.0),
            pre_emphasis_coefficient(0.97),
            remove_dc_offset(true),
            window_type("povey"),
            round_to_power_of_two(true),
            blackman_coefficient(0.42),
            snip_edges(true),
            allow_down_sample(false),
            allow_up_sample(false),
            max_feature_vectors(-1) {}

    int compute_window_shift();
    int compute_window_size();
    int padded_window_size();

    // show all the parameters
    std::string show() const {
        std::ostringstream oss;
        oss << "FrameExtractionOptions [ " << "sample_freq: " << sample_freq << "\t"
            << "frame_shift_ms: " << frame_shift_ms << "\t"
            << "frame_length_ms: " << frame_length_ms << "\t"
            << "dither: " << dither << "\t"
            << "pre_emphasis_coefficient: " << pre_emphasis_coefficient << "\t"
            << "remove_dc_offset: " << (remove_dc_offset ? "true" : "false") << "\t"
            << "window_type: " << window_type << "\t"
            << "round_to_power_of_two: " << (round_to_power_of_two ? "true" : "false") << "\t"
            << "blackman_coefficient: " << blackman_coefficient << "\t"
            << "snip_edges: " << (snip_edges ? "true" : "false") << "\t"
            << "allow_down_sample: " << (allow_down_sample ? "true" : "false") << "\t"
            << "allow_up_sample: " << (allow_up_sample ? "true" : "false") << "\t"
            << "max_feature_vectors: " << max_feature_vectors << " ]";
        return oss.str();
    }
};