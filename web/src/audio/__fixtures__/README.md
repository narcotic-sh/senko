# Native FBank fixture

`native-fbank-1p5s.f32` is the complete 148×80 float32 output of Senko's
production C++ `FeatureComputer` for the deterministic PCM16 sequence in
`generate_native_fbank_fixture.cc`. The fixture is little-endian, row-major,
and already includes per-window CMN.

Regenerate it from the repository root with the same optimized floating-point
mode used by the native library:

```sh
c++ -std=c++20 -O3 -ffast-math \
  -Isenko/fbank_extractor/cpp \
  -Isenko/fbank_extractor/cpp/feature \
  web/src/audio/__fixtures__/generate_native_fbank_fixture.cc \
  senko/fbank_extractor/cpp/feature_computer.cpp \
  senko/fbank_extractor/cpp/feature/fbank_computer.cpp \
  senko/fbank_extractor/cpp/feature/frame_extraction_options.cpp \
  senko/fbank_extractor/cpp/feature/frame_processor.cpp \
  senko/fbank_extractor/cpp/feature/fbank_utils.cpp \
  senko/fbank_extractor/cpp/feature/melbank_processor.cpp \
  -o build/generate_native_fbank_fixture

build/generate_native_fbank_fixture \
  web/src/audio/__fixtures__/native-fbank-1p5s.f32
```
