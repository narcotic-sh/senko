// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#include "frame_extraction_options.h"
#include "fbank_utils.h"

int FrameExtractionOptions::compute_window_shift() {
    return static_cast<int>(sample_freq * 0.001 * frame_shift_ms);
}

int FrameExtractionOptions::compute_window_size() {
    return static_cast<int>(sample_freq * 0.001 * frame_length_ms);
}

int FrameExtractionOptions::padded_window_size() {
    int window_size = compute_window_size();
    if(round_to_power_of_two) {
        return round_up_to_nearest_power_of_two(window_size);
    }
    else {
        return window_size;
    }
}