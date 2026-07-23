#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
web_dir="$(cd "$script_dir/../.." && pwd)"
emcc_bin="${EMCC:-/opt/homebrew/bin/emcc}"
output="$web_dir/src/clustering/wasm/senko-clustering.wasm"

mkdir -p "$(dirname "$output")"

"$emcc_bin" \
  "$script_dir/senko_clustering.cpp" \
  "$script_dir/hdbscan.cpp" \
  "$script_dir/umap_neighbors.cpp" \
  -std=c++20 \
  -O3 \
  -msimd128 \
  --no-entry \
  -s STANDALONE_WASM=1 \
  -s FILESYSTEM=0 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MEMORY_GROWTH_GEOMETRIC_STEP=0 \
  -s INITIAL_MEMORY=11534336 \
  -s MAXIMUM_MEMORY=2147483648 \
  -s STACK_SIZE=65536 \
  -s EXPORTED_FUNCTIONS='["_cluster_reset","_cluster_reserve","_cluster_alloc","_cluster_heap_base","_cluster_heap_capacity","_cluster_heap_used","_cluster_normalize_rows","_cluster_approximate_cosine_knn","_cluster_refine_euclidean_knn","_cluster_exact_euclidean_knn","_cluster_umap_cosine_knn_workspace_bytes","_cluster_umap_cosine_knn","_cluster_hdbscan_workspace_bytes","_cluster_hdbscan_f64_semantics","_cluster_hdbscan_f64_diagnostics"]' \
  -Wl,--strip-all \
  -o "$output"

chmod 0644 "$output"
echo "Built $output ($(wc -c < "$output" | tr -d ' ') bytes)"
