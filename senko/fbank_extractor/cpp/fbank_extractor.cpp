#include "fbank_extractor.h"
#include "feature_computer.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#include <sys/stat.h>
#define open _open
#define close _close
#ifndef O_RDONLY
#define O_RDONLY _O_RDONLY
#endif
#ifndef _O_BINARY
#define _O_BINARY 0x8000
#endif
static long long pread_win(int fd, void* buf, size_t count, long long offset) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    OVERLAPPED ov = {0};
    ov.Offset = (DWORD)(offset & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)(offset >> 32);
    DWORD read = 0;
    if (!ReadFile(h, buf, (DWORD)count, &read, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        return -1;
    }
    return (long long)read;
}
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

struct WavInfo {
    int sample_rate = 0;
    int channels = 0;
    int bits_per_sample = 0;
    int audio_format = 0;
    int64_t data_offset = 0;
    int64_t data_size = 0;
};

static uint16_t read_u16_le(const uint8_t* data, size_t offset) {
    return static_cast<uint16_t>(data[offset]) |
           static_cast<uint16_t>(data[offset + 1]) << 8;
}

static uint32_t read_u32_le(const uint8_t* data, size_t offset) {
    return static_cast<uint32_t>(data[offset]) |
           static_cast<uint32_t>(data[offset + 1]) << 8 |
           static_cast<uint32_t>(data[offset + 2]) << 16 |
           static_cast<uint32_t>(data[offset + 3]) << 24;
}

static bool read_fully(int fd, void* buffer, size_t count, int64_t offset) {
    uint8_t* dst = static_cast<uint8_t*>(buffer);
    size_t total = 0;
    while (total < count) {
#ifdef _WIN32
        long long n = pread_win(fd, dst + total, count - total, offset + static_cast<long long>(total));
#else
        ssize_t n = pread(fd, dst + total, count - total, offset + (off_t)total);
#endif
        if (n <= 0) {
            return false;
        }
        total += static_cast<size_t>(n);
    }
    return true;
}

static bool parse_wav_header(int fd, WavInfo& info) {
#ifdef _WIN32
    struct _stat64 st;
    if (_fstat64(fd, &st) != 0) {
#else
    struct stat st;
    if (fstat(fd, &st) != 0) {
#endif
        return false;
    }
    const int64_t file_size = static_cast<int64_t>(st.st_size);
    if (file_size < 12) {
        return false;
    }

    uint8_t header[12];
    if (!read_fully(fd, header, sizeof(header), 0)) {
        return false;
    }

    if (std::string(reinterpret_cast<char*>(header), 4) != "RIFF" ||
        std::string(reinterpret_cast<char*>(header + 8), 4) != "WAVE") {
        return false;
    }

    bool fmt_found = false;
    bool data_found = false;

    int64_t offset = 12;
    while (offset + 8 <= file_size) {
        uint8_t chunk_header[8];
        if (!read_fully(fd, chunk_header, sizeof(chunk_header), offset)) {
            return false;
        }

        const std::string chunk_id(reinterpret_cast<char*>(chunk_header), 4);
        const uint32_t chunk_size = read_u32_le(chunk_header, 4);
        const int64_t chunk_data_offset = offset + 8;

        if (chunk_id == "fmt ") {
            const uint32_t fmt_size = std::min<uint32_t>(chunk_size, 16);
            if (fmt_size < 16) {
                return false;
            }
            uint8_t fmt[16];
            if (!read_fully(fd, fmt, fmt_size, chunk_data_offset)) {
                return false;
            }
            info.audio_format = static_cast<int>(read_u16_le(fmt, 0));
            info.channels = static_cast<int>(read_u16_le(fmt, 2));
            info.sample_rate = static_cast<int>(read_u32_le(fmt, 4));
            info.bits_per_sample = static_cast<int>(read_u16_le(fmt, 14));
            fmt_found = true;
        } else if (chunk_id == "data") {
            info.data_offset = chunk_data_offset;
            info.data_size = chunk_size == 0 ? file_size - info.data_offset : static_cast<int64_t>(chunk_size);
            data_found = true;
        }

        const int64_t padded_size = chunk_size + (chunk_size % 2);
        offset += 8 + padded_size;

        if (fmt_found && data_found) {
            break;
        }
    }

    if (!fmt_found || !data_found) {
        return false;
    }

    if (info.data_offset + info.data_size > file_size) {
        return false;
    }

    return true;
}

class WavStreamReader {
public:
    explicit WavStreamReader(const std::string& path) {
#ifdef _WIN32
        fd_ = open(path.c_str(), O_RDONLY | _O_BINARY);
#else
        fd_ = open(path.c_str(), O_RDONLY);
#endif
        if (fd_ < 0) {
            return;
        }
        if (!parse_wav_header(fd_, info_)) {
            close(fd_);
            fd_ = -1;
            return;
        }
        bytes_per_sample_ = info_.bits_per_sample / 8;
        if (bytes_per_sample_ <= 0 || info_.channels <= 0) {
            close(fd_);
            fd_ = -1;
            return;
        }
        bytes_per_frame_ = bytes_per_sample_ * info_.channels;
        total_frames_ = static_cast<size_t>(info_.data_size / bytes_per_frame_);
    }

    ~WavStreamReader() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    bool valid() const { return fd_ >= 0; }
    size_t num_samples() const { return total_frames_; }

    bool read_samples(size_t start_frame, size_t frame_count, std::vector<float>& out) const {
        if (!valid()) {
            return false;
        }
        if (frame_count == 0 || start_frame >= total_frames_) {
            out.clear();
            return true;
        }

        const size_t available = total_frames_ - start_frame;
        const size_t to_read = std::min(frame_count, available);
        out.resize(to_read);

#ifdef _WIN32
        const int64_t byte_offset = static_cast<int64_t>(info_.data_offset) +
            static_cast<int64_t>(start_frame * bytes_per_frame_);
#else
        const off_t byte_offset = static_cast<off_t>(info_.data_offset) +
            static_cast<off_t>(start_frame * bytes_per_frame_);
#endif
        const size_t byte_count = to_read * bytes_per_frame_;
        const float scale = 1.0f / 32768.0f;

        switch (info_.bits_per_sample) {
            case 8: {
                std::vector<int8_t> interleaved(to_read * info_.channels);
                if (!read_fully(fd_, interleaved.data(), byte_count, byte_offset)) {
                    return false;
                }
                for (size_t i = 0; i < to_read; ++i) {
                    out[i] = static_cast<float>(interleaved[i * info_.channels]) * scale;
                }
                break;
            }
            case 16: {
                std::vector<int16_t> interleaved(to_read * info_.channels);
                if (!read_fully(fd_, interleaved.data(), byte_count, byte_offset)) {
                    return false;
                }
                for (size_t i = 0; i < to_read; ++i) {
                    out[i] = static_cast<float>(interleaved[i * info_.channels]) * scale;
                }
                break;
            }
            case 32: {
                if (info_.audio_format == 1) {
                    std::vector<int32_t> interleaved(to_read * info_.channels);
                    if (!read_fully(fd_, interleaved.data(), byte_count, byte_offset)) {
                        return false;
                    }
                    for (size_t i = 0; i < to_read; ++i) {
                        out[i] = static_cast<float>(interleaved[i * info_.channels]) * scale;
                    }
                } else if (info_.audio_format == 3) {
                    std::vector<float> interleaved(to_read * info_.channels);
                    if (!read_fully(fd_, interleaved.data(), byte_count, byte_offset)) {
                        return false;
                    }
                    for (size_t i = 0; i < to_read; ++i) {
                        out[i] = interleaved[i * info_.channels];
                    }
                } else {
                    return false;
                }
                break;
            }
            default:
                return false;
        }

        return true;
    }

private:
    int fd_ = -1;
    WavInfo info_;
    size_t bytes_per_sample_ = 0;
    size_t bytes_per_frame_ = 0;
    size_t total_frames_ = 0;
};

class MemoryAudioReader {
public:
    MemoryAudioReader(const float* samples, size_t num_samples) : samples_(samples), num_samples_(num_samples) {}

    size_t num_samples() const { return num_samples_; }

    bool read_samples(size_t start_frame, size_t frame_count, std::vector<float>& out) const {
        if (frame_count == 0 || start_frame >= num_samples_) {
            out.clear();
            return true;
        }

        const size_t available = num_samples_ - start_frame;
        const size_t to_read = std::min(frame_count, available);
        out.resize(to_read);
        std::copy(samples_ + start_frame,
                  samples_ + start_frame + to_read,
                  out.begin());
        return true;
    }

private:
    const float* samples_ = nullptr;
    size_t num_samples_ = 0;
};

template <typename AudioReader>
static FbankResult extract_features_impl(const AudioReader& audio_reader,
                                         const std::vector<std::pair<float, float>>& subsegments) {
    const size_t total_samples = audio_reader.num_samples();

    FeatureComputer fc;
    constexpr size_t mel_bins = 80;
    constexpr size_t max_frames_per_subseg = 150;
    constexpr size_t sample_rate = 16000;

    std::vector<float> big_features(subsegments.size() * max_frames_per_subseg * mel_bins, 0.f);
    std::vector<size_t> frames_per_subsegment(subsegments.size());
    std::vector<size_t> subsegment_offsets(subsegments.size());

    const unsigned int feat_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> feat_workers;
    feat_workers.reserve(feat_threads);

    std::atomic_size_t global_offset{0};
    thread_local static std::vector<float> segment_buffer;
    thread_local static std::vector<float> padded_buffer;

    auto feat_worker = [&](size_t first, size_t last) {
        for (size_t i = first; i < last; ++i) {
            const auto& sub = subsegments[i];
            size_t sample_start = static_cast<size_t>(sub.first * sample_rate);
            size_t sample_len = static_cast<size_t>((sub.second - sub.first) * sample_rate);

            if (sample_len == 0) sample_len = 1;
            if (sample_start >= total_samples) {
                sample_len = 0;
            } else if (sample_start + sample_len > total_samples) {
                sample_len = total_samples - sample_start;
            }

            const size_t min_len = 400;
            std::span<float> wav_span;

            if (sample_len < min_len) {
                if (padded_buffer.size() < min_len) padded_buffer.resize(min_len, 0.f);
                std::fill(padded_buffer.begin(), padded_buffer.begin() + min_len, 0.f);
                if (sample_len > 0 && audio_reader.read_samples(sample_start, sample_len, segment_buffer)) {
                    std::copy(segment_buffer.begin(),
                              segment_buffer.begin() + static_cast<long long>(sample_len),
                              padded_buffer.begin());
                }
                wav_span = std::span<float>(padded_buffer.data(), min_len);
            } else {
                if (audio_reader.read_samples(sample_start, sample_len, segment_buffer)) {
                    wav_span = std::span<float>(segment_buffer.data(), sample_len);
                } else {
                    if (padded_buffer.size() < min_len) padded_buffer.resize(min_len, 0.f);
                    std::fill(padded_buffer.begin(), padded_buffer.begin() + min_len, 0.f);
                    wav_span = std::span<float>(padded_buffer.data(), min_len);
                }
            }

            auto feat2d = fc.compute_feature(wav_span);
            const size_t frames = feat2d.size();
            frames_per_subsegment[i] = frames;

            const size_t my_off = global_offset.fetch_add(frames * mel_bins);
            subsegment_offsets[i] = my_off;

            size_t write_ptr = my_off;
            for (const auto& frame : feat2d) {
                std::copy(frame.begin(), frame.end(), big_features.begin() + static_cast<long>(write_ptr));
                write_ptr += mel_bins;
            }
        }
    };

    const size_t sub_per_thread = (subsegments.size() + feat_threads - 1) / feat_threads;
    size_t idx0 = 0;
    for (unsigned int t = 0; t < feat_threads; ++t) {
        const size_t idx1 = std::min(idx0 + sub_per_thread, subsegments.size());
        feat_workers.emplace_back(feat_worker, idx0, idx1);
        idx0 = idx1;
    }

    for (auto& th : feat_workers) {
        th.join();
    }

    const size_t used_size = global_offset.load();
    big_features.resize(used_size);
    big_features.shrink_to_fit();

    return {big_features, frames_per_subsegment, subsegment_offsets};
}

}  // namespace

FbankExtractor::FbankExtractor() {}

FbankResult FbankExtractor::extract_features(const std::string& wav_path, const std::vector<std::pair<float, float>>& subsegments) {
    WavStreamReader wav_reader(wav_path);
    if (!wav_reader.valid()) {
        return {{}, {}, {}};
    }
    return extract_features_impl(wav_reader, subsegments);
}

FbankResult FbankExtractor::extract_features_from_memory(const float* samples, size_t num_samples, const std::vector<std::pair<float, float>>& subsegments) {
    MemoryAudioReader audio_reader(samples, num_samples);
    return extract_features_impl(audio_reader, subsegments);
}
