// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#include "frame_processor.h"
#include <cassert>
#include <iostream>

FramePreprocessor::FramePreprocessor(
        const FrameExtractionOptions &frame_opts) : opts_(frame_opts),
                                                                generator_(0),
                                                                distribution_(0, 1.0) {
}

void FramePreprocessor::dither(std::vector<float> &wav_data) {
    if (opts_.dither == 0.0) return;
    for (size_t i = 0; i < wav_data.size(); i++) {
        wav_data[i] += opts_.dither * distribution_(generator_);
    }
}

void FramePreprocessor::remove_dc_offset(std::vector<float> &wav_data) {
    if (!opts_.remove_dc_offset) return;
    float mean = 0.0;
    for (size_t j = 0; j < wav_data.size(); ++j) mean += wav_data[j];
    mean /= wav_data.size();
    for (size_t j = 0; j < wav_data.size(); ++j) wav_data[j] -= mean;
}


void FramePreprocessor::pre_emphasis(std::vector<float> &wav_data) {
    float pre_emphasis_coefficient = opts_.pre_emphasis_coefficient;
    if (pre_emphasis_coefficient == 0.0) return;
    for (size_t i = wav_data.size() - 1; i > 0; i--) {
        wav_data[i] -= pre_emphasis_coefficient * wav_data[i - 1];
    }
    wav_data[0] -= pre_emphasis_coefficient * wav_data[0];
}

void FramePreprocessor::windows_function(std::vector<float> &wav_data) {
    std::vector<float> window;
    int frame_length = opts_.compute_window_size();
    assert(wav_data.size() == frame_length);
    window.resize(frame_length);
    double a = 2 * M_PI / (frame_length - 1);
    for (size_t i = 0; i < frame_length; i++) {
        double i_fl = static_cast<double>(i);
        if (opts_.window_type == "hanning") {
            window[i] = 0.5 - 0.5 * cos(a * i_fl);
        } else if (opts_.window_type == "sine") {
            // when you are checking ws wikipedia, please
            // note that 0.5 * a = M_PI/(frame_length-1)
            window[i] = sin(0.5 * a * i_fl);
        } else if (opts_.window_type == "hamming") {
            window[i] = 0.54 - 0.46 * cos(a * i_fl);
        } else if (opts_.window_type == "povey") {  // like hamming but goes to zero at edges.
            window[i] = pow(0.5 - 0.5 * cos(a * i_fl), 0.85);
        } else if (opts_.window_type == "rectangular") {
            window[i] = 1.0;
        } else if (opts_.window_type == "blackman") {
            window[i] = opts_.blackman_coefficient - 0.5 * cos(a * i_fl) +
                        (0.5 - opts_.blackman_coefficient) * cos(2 * a * i_fl);
        } else {
            std::cerr << "Unknown window type " << opts_.window_type << std::endl;
        }
    }
    for (size_t i = 0; i < wav_data.size(); i++) {
        wav_data[i] *= window[i];
    }
}

void FramePreprocessor::frame_pre_process(std::vector<float> &wav_data) {
    dither(wav_data);
    remove_dc_offset(wav_data);
    pre_emphasis(wav_data);
    windows_function(wav_data);
}