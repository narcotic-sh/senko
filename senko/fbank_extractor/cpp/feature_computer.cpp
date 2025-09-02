// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#include "feature_computer.h"

FeatureComputer::FeatureComputer()
{
    opts.frame_opts.sample_freq = 16000;
    opts.frame_opts.frame_shift_ms = 10.0;
    opts.frame_opts.frame_length_ms = 25.0;
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.pre_emphasis_coefficient = 0.97;
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.window_type = "povey";
    opts.frame_opts.round_to_power_of_two = true;
    opts.frame_opts.blackman_coefficient = 0.42;
    opts.frame_opts.snip_edges = true;
    opts.mel_opts.num_bins = 80;
    opts.mel_opts.low_freq = 20;
    opts.mel_opts.high_freq = 0;
    opts.mel_opts.vtln_low = 100;
    opts.mel_opts.vtln_high = -500;
    opts.use_energy = false;
    opts.energy_floor = 1.0;
    opts.raw_energy = true;
    opts.use_log_fbank = true;
    opts.use_power = true;

    fbank_computer = FbankComputer(opts);
}

Feature FeatureComputer::compute_feature(std::span<float> wav_span) {
    // Pad waveform if shorter than window size
    const size_t min_length = 400;
    std::vector<float> padded_wav;
    if (wav_span.size() < min_length) {
        padded_wav.resize(min_length, 0.0f);
        std::copy(wav_span.begin(), wav_span.end(), padded_wav.begin());
        wav_span = std::span<float>(padded_wav);
    }

    auto feat = fbank_computer.compute_feature(wav_span);

    // Mean normalization
    std::vector<float> column_means(feat[0].size(), 0.0f);
    for (const auto& row : feat) {
        for (size_t i = 0; i < row.size(); ++i) {
            column_means[i] += row[i];
        }
    }
    for (auto& mean : column_means) {
        mean /= feat.size();
    }
    for (auto& row : feat) {
        for (size_t i = 0; i < row.size(); ++i) {
            row[i] -= column_means[i];
        }
    }
    return feat;
}