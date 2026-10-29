#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>
#include <string>

#define WHISPER_ASSERT GGML_ASSERT

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

#define COMMON_SAMPLE_RATE 16000

namespace whisper_preprocessor {

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

bool preprocess_audio(
        const float * samples,
        size_t n_samples,
        const whisper_filters & filters,
        std::vector<whisper_mel> & output);

} // namespace whisper_preprocessor

// MiMo-V2.5 / V2.6 audio tokenizer front-end: 24 kHz, HTK log-mel (natural log, magnitude),
// one mel per clip of up to MAX_SECONDS
namespace mimo_audio_preprocessor {

constexpr int SAMPLE_RATE = 24000;
constexpr int N_FFT       = 960;
constexpr int WIN_LEN     = 960;
constexpr int HOP_LEN     = 240;
constexpr int MAX_SECONDS = 300; // audio_tokenizer max_audio_seconds

bool preprocess_audio(
        const float * samples,
        size_t n_samples,
        int n_mel,
        std::vector<whisper_preprocessor::whisper_mel> & output);

} // namespace mimo_audio_preprocessor

// Gemma 4 unified (encoder-free) audio: 16 kHz raw waveform cut into frames of frame_size samples,
// the last one zero-padded; each frame is one token. Output is "mel-major" like the whisper mel
// (data[f * n_len + t]) with n_mel = frame_size, so it follows the [n_len, n_mel] input convention.
namespace gemma4ua_preprocessor {

bool preprocess_audio(
        const float * samples,
        size_t n_samples,
        int frame_size,
        std::vector<whisper_preprocessor::whisper_mel> & output);

} // namespace gemma4ua_preprocessor

namespace whisper_precalc_filters {

whisper_preprocessor::whisper_filters get_128_bins();

} // namespace whisper_precalc_filters
