#include <math.h>
#include <stdint.h>
#include <string.h>

#include <emscripten/emscripten.h>

namespace {

constexpr int kSampleRate = 16000;
constexpr int kFrameLength = 400;
constexpr int kFrameShift = 160;
constexpr int kFftSize = 512;
constexpr int kFftBins = kFftSize / 2;
constexpr int kMelBins = 80;
// Senko first casts absolute subsegment timestamps to float32. At hour-scale
// offsets, subtracting two nominally 1.5-second endpoints can therefore yield
// 24,001 samples. Size the input for every sample count that still fits
// native Senko's fixed 150-frame CAM++ window instead of assuming exactly
// 24,000 samples.
constexpr int kMaxFrames = 150;
constexpr int kMaxSamples =
    kFrameLength + (kMaxFrames - 1) * kFrameShift + (kFrameShift - 1);
constexpr float kPreemphasis = 0.97f;
constexpr float kEpsilon = 1.1920928955078125e-7f;
constexpr double kPi = 3.14159265358979323846;

// This module intentionally has one fixed arena. There is no allocator and the
// WebAssembly memory cannot grow. One worker owns one instance and reuses these
// buffers for every embedding window.
alignas(16) float input_samples[kMaxSamples];
alignas(16) float output_features[kMaxFrames * kMelBins];
alignas(16) float raw_features[kMaxFrames * kMelBins];
alignas(16) float real_values[kFftSize];
alignas(16) float imaginary_values[kFftSize];
alignas(16) float power_values[kFftBins];
alignas(16) float column_sums[kMelBins];
alignas(16) float povey_window[kFrameLength];
alignas(16) float twiddle_real[kFftBins];
alignas(16) float twiddle_imaginary[kFftBins];
alignas(16) float mel_weights[kMelBins * kFftBins];
uint16_t bit_reverse[kFftSize];
uint16_t mel_first[kMelBins];
uint16_t mel_last[kMelBins];

bool initialized = false;
bool previous_raw_valid = false;
int previous_frame_count = 0;

inline float mel_scale(float frequency) {
  return 1127.0f * logf(1.0f + frequency / 700.0f);
}

void initialize_tables() {
  const double angular_step = (2.0 * kPi) / (kFrameLength - 1);
  for (int i = 0; i < kFrameLength; ++i) {
    povey_window[i] = static_cast<float>(
        pow(0.5 - 0.5 * cos(angular_step * static_cast<double>(i)), 0.85));
  }

  float sin_table[kFftBins];
  for (int i = 0; i < kFftBins; ++i) {
    sin_table[i] = static_cast<float>(sin((-2.0 * kPi * i) / kFftSize));
  }
  for (int k = 0; k < kFftBins; ++k) {
    float cos_value =
        sin_table[(kFftSize / 4 - k + kFftBins) % kFftBins];
    if (k >= kFftSize / 4) cos_value = -cos_value;
    twiddle_real[k] = -cos_value;
    twiddle_imaginary[k] = sin_table[k];
  }

  for (int i = 0; i < kFftSize; ++i) {
    int source = i;
    int reversed = 0;
    for (int bit = 0; bit < 9; ++bit) {
      reversed = (reversed << 1) | (source & 1);
      source >>= 1;
    }
    bit_reverse[i] = static_cast<uint16_t>(reversed);
  }

  const float fft_bin_width =
      static_cast<float>(kSampleRate) / static_cast<float>(kFftSize);
  const float low_mel = mel_scale(20.0f);
  const float high_mel = mel_scale(8000.0f);
  const float mel_delta = (high_mel - low_mel) / (kMelBins + 1);
  memset(mel_weights, 0, sizeof(mel_weights));
  for (int bin = 0; bin < kMelBins; ++bin) {
    const float left = low_mel + bin * mel_delta;
    const float middle = low_mel + (bin + 1) * mel_delta;
    const float right = low_mel + (bin + 2) * mel_delta;
    int first = -1;
    int last = -1;
    for (int fft_bin = 0; fft_bin < kFftBins; ++fft_bin) {
      const float mel = mel_scale(fft_bin_width * fft_bin);
      if (mel > left && mel < right) {
        mel_weights[bin * kFftBins + fft_bin] =
            mel <= middle ? (mel - left) / (middle - left)
                          : (right - mel) / (right - middle);
        if (first < 0) first = fft_bin;
        last = fft_bin;
      }
    }
    mel_first[bin] = static_cast<uint16_t>(first);
    mel_last[bin] = static_cast<uint16_t>(last);
  }
}

inline int frame_count_for_samples(int sample_count) {
  const int padded_count = sample_count < kFrameLength ? kFrameLength
                                                       : sample_count;
  return 1 + (padded_count - kFrameLength) / kFrameShift;
}

void prepare_frame(int sample_count, int start) {
  float mean = 0.0f;
  for (int i = 0; i < kFrameLength; ++i) {
    const float sample = start + i < sample_count ? input_samples[start + i]
                                                  : 0.0f;
    real_values[i] = sample;
    mean += sample;
  }
  mean /= static_cast<float>(kFrameLength);

  for (int i = 0; i < kFrameLength; ++i) real_values[i] -= mean;
  for (int i = kFrameLength - 1; i > 0; --i) {
    real_values[i] -= kPreemphasis * real_values[i - 1];
  }
  real_values[0] -= kPreemphasis * real_values[0];
  for (int i = 0; i < kFrameLength; ++i) {
    real_values[i] *= povey_window[i];
  }
  memset(real_values + kFrameLength, 0,
         (kFftSize - kFrameLength) * sizeof(float));
  memset(imaginary_values, 0, sizeof(imaginary_values));
}

void fft_in_place() {
  for (int i = 0; i < kFftSize; ++i) {
    const int reversed = bit_reverse[i];
    if (i < reversed) {
      const float saved_real = real_values[i];
      const float saved_imaginary = imaginary_values[i];
      real_values[i] = real_values[reversed];
      imaginary_values[i] = imaginary_values[reversed];
      real_values[reversed] = saved_real;
      imaginary_values[reversed] = saved_imaginary;
    }
  }

  for (int half_size = 1; half_size < kFftSize; half_size *= 2) {
    const int step = kFftSize / (half_size * 2);
    for (int block = 0; block < kFftSize; block += half_size * 2) {
      int twiddle = 0;
      for (int j = block; j < block + half_size; ++j, twiddle += step) {
        const int paired = j + half_size;
        const float paired_real = real_values[paired];
        const float paired_imaginary = imaginary_values[paired];
        const float rotated_real =
            twiddle_real[twiddle] * paired_real -
            twiddle_imaginary[twiddle] * paired_imaginary;
        const float rotated_imaginary =
            twiddle_real[twiddle] * paired_imaginary +
            twiddle_imaginary[twiddle] * paired_real;
        const float current_real = real_values[j];
        const float current_imaginary = imaginary_values[j];
        real_values[paired] = current_real - rotated_real;
        imaginary_values[paired] = current_imaginary - rotated_imaginary;
        real_values[j] = current_real + rotated_real;
        imaginary_values[j] = current_imaginary + rotated_imaginary;
      }
    }
  }
}

void compute_raw_frame(int frame, int sample_count) {
  prepare_frame(sample_count, frame * kFrameShift);
  fft_in_place();

  for (int i = 0; i < kFftBins; ++i) {
    const float real = real_values[i];
    const float imaginary = imaginary_values[i];
    power_values[i] = real * real + imaginary * imaginary;
  }

  float* row = raw_features + frame * kMelBins;
  for (int bin = 0; bin < kMelBins; ++bin) {
    float energy = 0.0f;
    const float* weights = mel_weights + bin * kFftBins;
    for (int i = mel_first[bin]; i <= mel_last[bin]; ++i) {
      energy += weights[i] * power_values[i];
    }
    row[bin] = logf(energy < kEpsilon ? kEpsilon : energy);
  }
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int fbank_init() {
  if (!initialized) {
    initialize_tables();
    initialized = true;
  }
  previous_raw_valid = false;
  previous_frame_count = 0;
  return 1;
}

EMSCRIPTEN_KEEPALIVE uintptr_t fbank_input_ptr() {
  return reinterpret_cast<uintptr_t>(input_samples);
}

EMSCRIPTEN_KEEPALIVE uintptr_t fbank_output_ptr() {
  return reinterpret_cast<uintptr_t>(output_features);
}

EMSCRIPTEN_KEEPALIVE int fbank_compute(int sample_count, int reuse_frame_shift) {
  if (!initialized || sample_count < 0 || sample_count > kMaxSamples) return -1;
  const int frame_count = frame_count_for_samples(sample_count);
  if (frame_count < 1 || frame_count > kMaxFrames) return -2;

  int reused_frames = 0;
  if (previous_raw_valid && reuse_frame_shift > 0 &&
      reuse_frame_shift < previous_frame_count) {
    reused_frames = previous_frame_count - reuse_frame_shift;
    if (reused_frames > frame_count) reused_frames = frame_count;
    memmove(raw_features,
            raw_features + reuse_frame_shift * kMelBins,
            reused_frames * kMelBins * sizeof(float));
  }

  for (int frame = reused_frames; frame < frame_count; ++frame) {
    compute_raw_frame(frame, sample_count);
  }

  memset(column_sums, 0, sizeof(column_sums));
  for (int frame = 0; frame < frame_count; ++frame) {
    const float* row = raw_features + frame * kMelBins;
    for (int bin = 0; bin < kMelBins; ++bin) column_sums[bin] += row[bin];
  }
  for (int bin = 0; bin < kMelBins; ++bin) {
    column_sums[bin] /= static_cast<float>(frame_count);
  }
  for (int frame = 0; frame < frame_count; ++frame) {
    const float* source = raw_features + frame * kMelBins;
    float* target = output_features + frame * kMelBins;
    for (int bin = 0; bin < kMelBins; ++bin) {
      target[bin] = source[bin] - column_sums[bin];
    }
  }

  previous_raw_valid = true;
  previous_frame_count = frame_count;
  return frame_count;
}

EMSCRIPTEN_KEEPALIVE void fbank_reset_reuse() {
  previous_raw_valid = false;
  previous_frame_count = 0;
}

EMSCRIPTEN_KEEPALIVE void fbank_dispose() {
  previous_raw_valid = false;
  previous_frame_count = 0;
  memset(input_samples, 0, sizeof(input_samples));
  memset(output_features, 0, sizeof(output_features));
  memset(raw_features, 0, sizeof(raw_features));
}

EMSCRIPTEN_KEEPALIVE int fbank_max_samples() { return kMaxSamples; }
EMSCRIPTEN_KEEPALIVE int fbank_max_frames() { return kMaxFrames; }
EMSCRIPTEN_KEEPALIVE int fbank_bins() { return kMelBins; }

}  // extern "C"
