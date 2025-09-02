// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#include <iostream>
#include "fbank_computer.h"
#include "fbank_utils.h"
#include <span>

FbankComputer::FbankComputer(const FbankOptions &opts) : opts_(opts),
                                                         frame_preprocessor_(opts.frame_opts),
                                                         log_energy_floor_(0.0),
                                                         mel_bank_processor_(opts.mel_opts) {
    int frame_length = opts_.frame_opts.compute_window_size();
    const int fft_n = round_up_to_nearest_power_of_two(frame_length);
    init_sin_tbl(sin_tbl_, fft_n);
    init_bit_reverse_index(bit_rev_index_, fft_n);

    int padded_window_length = opts_.frame_opts.padded_window_size();
    mel_bank_processor_.init_mel_bins(opts.frame_opts.sample_freq, padded_window_length);
}

Feature FbankComputer::compute_feature(std::span<float> float_wav_data) {
    int frame_length = opts_.compute_window_size();
    int frame_shift = opts_.compute_window_shift();
    int fft_n = round_up_to_nearest_power_of_two(frame_length);
    int num_samples = float_wav_data.size();
    int num_frames = 1 + ((num_samples - frame_length) / frame_shift);
    Feature feature;
    feature.resize(num_frames);

    float epsilon = std::numeric_limits<float>::epsilon();
    int fbank_num_bins = opts_.get_fbank_num_bins();

    std::vector<std::pair<int, std::vector<float>>> mel_bins = mel_bank_processor_.get_mel_bins();

    // std::cout << "frame_length: " << frame_length << " "
    //           << "frame_shift: " << frame_shift << " "
    //           << "fft_n: " << fft_n << " "
    //           << "num_frames: " << num_frames << " "
    //           << "mel_bins: " << mel_bins.size() << " "
    //           << "fbank_num_bins " << fbank_num_bins << " "
    //           << "epsilon: " << epsilon
    //           << std::endl;

    for (int i = 0; i < num_frames; i++) {
        std::vector<float> cur_wav_data(float_wav_data.data() + i * frame_shift,
                                        float_wav_data.data() + i * frame_shift + frame_length);
        // Contain dither,
        frame_preprocessor_.frame_pre_process(cur_wav_data);

        // build FFT
        std::vector<std::complex<float>> cur_window_data(fft_n);
        for (int j = 0; j < fft_n; j++) {
            if (j < frame_length) {
                cur_window_data[j] = std::complex<float>(cur_wav_data[j], 0.0);
            } else {
                cur_window_data[j] = std::complex<float>(0.0, 0.0);
            }
        }
        custom_fft(bit_rev_index_, sin_tbl_, cur_window_data);
        std::vector<float> power(fft_n / 2);
        for (int j = 0; j < fft_n / 2; j++) {
            power[j] = cur_window_data[j].real() * cur_window_data[j].real() +
                       cur_window_data[j].imag() * cur_window_data[j].imag();
        }
        if (!opts_.use_power) {
            for (int j = 0; j < fft_n / 2; j++) {
                power[j] = powf(power[j], 0.5);
            }
        }
        // mel filter
        feature[i].resize(opts_.get_fbank_num_bins());
        for (int j = 0; j < fbank_num_bins; j++) {
            float mel_energy = 0.0;
            int start_index = mel_bins[j].first;
            for (int k = 0; k < mel_bins[j].second.size(); k++) {
                mel_energy += mel_bins[j].second[k] * power[k + start_index];
            }
            if (opts_.use_log_fbank) {
                if (mel_energy < epsilon) mel_energy = epsilon;
                mel_energy = logf(mel_energy);
            }
            feature[i][j] = mel_energy;
        }
    }
    return feature;
}