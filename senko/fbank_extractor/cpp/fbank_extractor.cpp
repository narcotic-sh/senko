#include "fbank_extractor.h"
#include "feature_computer.h"
#include "wav.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <thread>
#include <vector>

FbankExtractor::FbankExtractor() {}

FbankResult FbankExtractor::extract_features(const std::string& wav_path, const std::vector<std::pair<float, float>>& subsegments) {

    /*═════════════╗
    ║  Load audio  ║
    ╚═════════════*/

    wav::WavReader wav_reader(wav_path);
    std::vector<float> input_wav(wav_reader.num_samples());
    std::copy(wav_reader.data(), wav_reader.data() + wav_reader.num_samples(), input_wav.begin());

    /*════════════════════════════════════════════╗
    ║  Fbank feature extraction (multi-threaded)  ║
    ╚════════════════════════════════════════════*/

    FeatureComputer fc;

    // Each feature is 80 mel bins ("height") by up to ~150 frames ("width")
    constexpr size_t mel_bins = 80;
    constexpr size_t max_frames_per_subseg = 150;
    constexpr size_t sample_rate = 16000;

    // Allocate a single large buffer for all subsegments
    std::vector<float> big_features(subsegments.size() * max_frames_per_subseg * mel_bins, 0.f);

    // For each subsegment, track (offset_in_big_features, frames_produced)
    std::vector<std::pair<size_t, size_t>> feature_indices(subsegments.size());

    // Track frame counts for each subsegment (this is what we'll return)
    std::vector<size_t> frames_per_subsegment(subsegments.size());
    std::vector<size_t> subsegment_offsets(subsegments.size());

    // Multi-threading setup
    const unsigned int feat_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> feat_workers;
    feat_workers.reserve(feat_threads);

    // Use an atomic offset so each thread knows where to write in big_features
    std::atomic_size_t global_offset{0};

    // Thread-local buffer for padding short segments
    thread_local static std::vector<float> padded_buffer;

    auto feat_worker = [&](size_t first, size_t last) {
        for (size_t i = first; i < last; ++i) {
            const auto& sub = subsegments[i];
            size_t sample_start = static_cast<size_t>(sub.first * sample_rate);
            size_t sample_len = static_cast<size_t>((sub.second - sub.first) * sample_rate);

            if (sample_len == 0) sample_len = 1;
            if (sample_start + sample_len > input_wav.size()) {
                sample_len = input_wav.size() - sample_start;
            }

            const size_t min_len = 400;  // Ensure a minimum of 400 samples
            std::span<float> wav_span;

            if (sample_len < min_len) {
                if (padded_buffer.size() < min_len) padded_buffer.resize(min_len, 0.f);
                // Copy what we have into padded_buffer
                std::copy(input_wav.data() + sample_start,
                         input_wav.data() + sample_start + sample_len,
                         padded_buffer.begin());
                // Remaining samples in padded_buffer stay zero
                wav_span = std::span<float>(padded_buffer.data(), min_len);
            } else {
                // No need to pad
                wav_span = std::span<float>(input_wav.data() + sample_start, sample_len);
            }

            // Compute FBank features => 2D: (frames x mel_bins)
            auto feat2d = fc.compute_feature(wav_span);
            const size_t frames = feat2d.size(); // #frames generated

            // Store frame count for this subsegment
            frames_per_subsegment[i] = frames;

            // Claim a chunk in big_features for these features
            const size_t my_off = global_offset.fetch_add(frames * mel_bins);
            feature_indices[i] = {my_off, frames};
            subsegment_offsets[i] = my_off;

            // Flatten-copy (frame by frame) into big_features
            size_t write_ptr = my_off;
            for (const auto& frame : feat2d) {
                // Each frame => mel_bins floats
                std::copy(frame.begin(), frame.end(),
                         big_features.begin() + static_cast<long>(write_ptr));
                write_ptr += mel_bins;
            }
        }
    };

    // Distribute subsegments across threads
    const size_t sub_per_thread = (subsegments.size() + feat_threads - 1) / feat_threads;
    size_t idx0 = 0;
    for (unsigned int t = 0; t < feat_threads; ++t) {
        const size_t idx1 = std::min(idx0 + sub_per_thread, subsegments.size());
        feat_workers.emplace_back(feat_worker, idx0, idx1);
        idx0 = idx1;
    }

    for (auto& th : feat_workers) th.join();

    // Shrink big_features to actual usage
    const size_t used_size = global_offset.load();
    big_features.resize(used_size);
    big_features.shrink_to_fit();

    return {big_features, frames_per_subsegment, subsegment_offsets};
}