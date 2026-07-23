#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
emcc_bin="${EMCC:-/opt/homebrew/bin/emcc}"
output="${1:-$script_dir/umap_layout_threaded.wasm}"

"$emcc_bin" \
  "$script_dir/umap_layout_threaded.cpp" \
  -std=c++20 \
  -O3 \
  -Wall \
  -Wextra \
  -Werror \
  -msimd128 \
  -matomics \
  -mbulk-memory \
  --no-entry \
  -s STANDALONE_WASM=1 \
  -s FILESYSTEM=0 \
  -s IMPORTED_MEMORY=1 \
  -s SHARED_MEMORY=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MEMORY_GROWTH_GEOMETRIC_STEP=0 \
  -s INITIAL_MEMORY=16777216 \
  -s MAXIMUM_MEMORY=2147483648 \
  -s STACK_SIZE=65536 \
  -s EXPORTED_FUNCTIONS='["_umap_layout_threaded_plan_offset","_umap_layout_threaded_stack_top","_umap_layout_threaded_run"]' \
  -Wl,--export=__stack_pointer \
  -Wl,--strip-all \
  -o "$output"

chmod 0644 "$output"
echo "Built $output ($(wc -c < "$output" | tr -d ' ') bytes)"
