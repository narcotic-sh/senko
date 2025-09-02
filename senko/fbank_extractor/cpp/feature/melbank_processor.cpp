// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#include "melbank_processor.h"

#include <iostream>

MelBankProcessor::MelBankProcessor(const MelBanksOptions &mel_opts): opts_(mel_opts) {
    if (opts_.num_bins < 3) {
        std::cerr << "Mel Banks do not have enough " << opts_.num_bins << " mel bins" << std::endl;
    }
}

void MelBankProcessor::init_mel_bins(float sample_frequency, int window_padded_length) {
    // std::cout << "Initializing mel bins..." << std::endl;
    // std::cout << "Sample frequency: " << sample_frequency << std::endl;
    // std::cout << "Window padded length: " << window_padded_length << std::endl;

    int num_fft_bins = window_padded_length / 2;
    int num_bins = opts_.num_bins;
    float nyquist = 0.5 * sample_frequency;
    float low_frequency = opts_.low_freq, high_frequency;
    if (opts_.high_freq > 0.0) high_frequency = opts_.high_freq;
    else high_frequency = nyquist + opts_.high_freq;

    // std::cout << "num_fft_bins: " << num_fft_bins << std::endl;
    // std::cout << "num_bins: " << num_bins << std::endl;
    // std::cout << "nyquist: " << nyquist << std::endl;
    // std::cout << "low_frequency: " << low_frequency << std::endl;
    // std::cout << "high_frequency: " << high_frequency << std::endl;

    if (low_frequency < 0.0 || low_frequency >= nyquist ||
        high_frequency <= 0.0 || high_frequency > nyquist || high_frequency <= low_frequency) {
        std::cerr << "Bad values in options: low-frequency " << low_frequency
                  << " and high-frequency " << high_frequency << " vs nyquist " << nyquist;
    }

    float fft_bin_width = sample_frequency / window_padded_length;
    float mel_low_frequency = mel_scale(low_frequency);
    float mel_high_frequency = mel_scale(high_frequency);
    float mel_frequency_delta = (mel_high_frequency - mel_low_frequency) / (num_bins + 1);

    mel_bins_.resize(num_bins);
    center_frequency_.resize(num_bins);
    for (size_t index = 0; index < num_bins; index++) {
        float left_mel = mel_low_frequency + index * mel_frequency_delta;
        float middle_mel = mel_low_frequency + (index + 1) * mel_frequency_delta;
        float right_mel = mel_low_frequency + (index + 2) * mel_frequency_delta;
        center_frequency_[index] = inverse_mel_scale(middle_mel);

        std::vector<float> cur_mel_bin(num_fft_bins);
        int first_index = -1, last_index = -1;
        for (int i = 0; i < num_fft_bins; i++) {
            float frequency = (fft_bin_width * i);
            float mel = mel_scale(frequency);
            if (mel > left_mel && mel < right_mel) {
                float weight = 0.0;
                if (mel <= middle_mel) {
                    weight = (mel - left_mel) / (middle_mel - left_mel);
                } else {
                    weight = (right_mel - mel) / (right_mel - middle_mel);
                }
                cur_mel_bin[i] = weight;
                if (first_index == -1) first_index = i;
                last_index = i;
            }
        }
        mel_bins_[index].first = first_index;
        mel_bins_[index].second.resize(last_index + 1 - first_index);
        mel_bins_[index].second.assign(cur_mel_bin.begin() + first_index,
                                       cur_mel_bin.begin() + last_index + 1);
    }

    // std::cout << "Mel bins initialized. Total bins: " << mel_bins_.size() << std::endl;
}

void subtract_feature_mean(std::vector<std::vector<float>>& feature) {
    if (feature.empty() || feature[0].empty()) return;
    size_t feat_dim = feature[0].size();
    std::vector<float> means(feat_dim, 0.0f);

    for (const auto& feature_vector : feature) {
        for (size_t i = 0; i < feat_dim; ++i) {
            means[i] += feature_vector[i];
        }
    }
    for (float& mean : means) {
        mean /= (float)feature.size();
    }

    // subtract feature mean
    for (auto& feature_vector : feature) {
        for (size_t i = 0; i < feat_dim; ++i) {
            feature_vector[i] -= means[i];
        }
    }
}
