// Code adapted from https://github.com/modelscope/3D-Speaker/tree/main/runtime/onnxruntime/feature
// which itself seems to be adapted from Kaldi (https://github.dev/kaldi-asr/kaldi)

#pragma once

#include <vector>
#include <cmath>
#include <complex>

int round_up_to_nearest_power_of_two(int n);

void init_bit_reverse_index(std::vector<int> &bit_rev_index, int n);

void init_sin_tbl(std::vector<float> &sin_tbl, int n);

void custom_fft(const std::vector<int> &bit_rev_index,
                const std::vector<float> &sin_tbl,
                std::vector<std::complex<float>> &data);
