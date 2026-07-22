#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
web_dir="$(cd "$script_dir/../.." && pwd)"
emcc_bin="${EMCC:-/opt/homebrew/bin/emcc}"
output="$web_dir/src/audio/wasm/senko-fbank.wasm"

mkdir -p "$(dirname "$output")"

"$emcc_bin" \
  "$script_dir/senko_fbank.cpp" \
  -std=c++20 \
  -O3 \
  -msimd128 \
  --no-entry \
  -s STANDALONE_WASM=1 \
  -s FILESYSTEM=0 \
  -s ALLOW_MEMORY_GROWTH=0 \
  -s INITIAL_MEMORY=524288 \
  -s STACK_SIZE=32768 \
  -s EXPORTED_FUNCTIONS='["_fbank_init","_fbank_input_ptr","_fbank_output_ptr","_fbank_compute","_fbank_reset_reuse","_fbank_dispose","_fbank_max_samples","_fbank_max_frames","_fbank_bins"]' \
  -Wl,--strip-all \
  -o "$output"

chmod 0644 "$output"
echo "Built $output ($(wc -c < "$output" | tr -d ' ') bytes)"
