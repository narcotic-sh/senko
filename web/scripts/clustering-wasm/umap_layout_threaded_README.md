# Native-parity threaded UMAP layout

This directory contains a validated standalone implementation of UMAP
0.5.12's parallel Euclidean layout optimizer. It is the production layout
backend for long recordings. The ordinary clustering module owns a growable,
non-shared arena; WebAssembly memories are declared shared at compile time, so
the threaded optimizer remains a separate module.

## What is preserved

`umap_layout_threaded.cpp` follows the captured native layout call:

- float32 coordinates and squared-distance reduction, with the same
  16-accumulator reduction shape;
- float64 positive/negative sample clocks and negative-sample scheduling;
- float32 alpha, `a`, `b`, gamma, gradient coefficients, clipping, and
  coordinate updates in the hot optimizer;
- UMAP's per-head tau RNG state and signed modulo behavior;
- row-major CSR edge order, `move_other=true`, and one worker at a time owning
  every edge and RNG update for a claimed head row;
- the UMAP 0.5.12 alpha quirk: epochs zero and one both use alpha 1;
- worker-zero initialization of all clocks/RNG state, followed by a barrier;
- one barrier after every epoch, matching the implicit barrier around Numba's
  parallel single-epoch call.

The embedding and per-head RNG state intentionally use ordinary shared-memory
loads/stores. Conflicting writes retain native Hogwild lost-update behavior;
only barrier, status, and cancellation words are atomic. Consequently the
parallel result is stochastic, just like unseeded native Senko.

## Execution architecture

The 6,771-byte standalone Wasm has exactly one import, `env.memory`. It does
not use Emscripten's pthread runtime or generated JavaScript glue. The build
uses:

```text
-O3 -msimd128 -matomics -mbulk-memory
--no-entry
-s STANDALONE_WASM=1
-s IMPORTED_MEMORY=1
-s SHARED_MEMORY=1
-s ALLOW_MEMORY_GROWTH=1
-s INITIAL_MEMORY=16777216
-s MAXIMUM_MEMORY=2147483648
-Wl,--export=__stack_pointer
```

The flexible import type lets the coordinator allocate an exact run-sized
memory, but each actual job supplies `initial === maximum`. Fixed job memory
avoids stale typed-array views and makes peak memory explicit.

The pipeline worker compiles the module once and keeps a small nested
module-worker pool alive. Every leaf instantiates the same cloned
`WebAssembly.Module` against the same memory. Wasm mutable globals are
per-instance, so the coordinator assigns each exported `__stack_pointer` a
disjoint 64 KiB stack. The module's math tables use Emscripten's atomic
once-initializer and are safe when several instances start concurrently.

The real kernel executes all 500 epochs in one call. Each epoch uses an atomic
cursor to distribute 16-row chunks dynamically. Keeping chunks row-aligned
preserves each head's edge and RNG order while allowing faster CPU cores to
claim more work instead of waiting behind a fixed shard. Workers then block
with `memory.atomic.wait32`; the final arrival resets the row cursor, advances
the generation, and calls `memory.atomic.notify`. There is no per-epoch
JavaScript messaging. The parent pipeline worker stays responsive because only
nested leaves enter the blocking call.

Cancellation sets the shared cancellation word and notifies the generation.
The kernel checks cancellation every 1,024 edge slots and at every barrier;
one-second bounded waits also protect against a lost notify. On a worker error,
the coordinator cancels the run and terminates every leaf. Disposal always
terminates the nested pool.

The generated Wasm was inspected to retain SIMD and ordinary `f32` scatter
stores. Recheck the disassembly and parity gates when changing Emscripten:
Hogwild is intentionally a data race at the C++ source level, so the pinned
compiler/output is part of this low-level implementation contract.

## Memory plan

`umap_layout_threaded_plan_offset` is the authoritative sizing API. The host
first instantiates a 16 MiB planner memory, reads every section offset, then
allocates a fixed memory of `kTotalBytes`. `RunHeader` documents the complete
128-byte JS/Wasm ABI.

For the one-hour fixture (5,713 vertices, 351,946 edges, 60 dimensions), all
worker counts fit the 16 MiB minimum. The layout memory includes its output,
CSR inputs, float64 clocks, RNG state, synchronization header, and disjoint
stacks.

At the captured long shape (43,804 vertices and 2,836,232 retained edges), ABI
v2 uses 91,553,792 bytes with four workers or 91,815,936 bytes with eight. The
preceding COO plan used 125,435,904 and 125,698,048 bytes respectively. CSR
row offsets and exact on-demand derivation of
`epochs_per_negative_sample` therefore save 32.31 MiB without changing the
one-hour 16 MiB allocation. Memory remains duration-linear and far below a
dense graph.

## Reproduction

Build and run one Node worker-thread diagnostic:

```sh
cd web
bash scripts/clustering-wasm/umap_layout_threaded_build.sh
node scripts/clustering-wasm/umap_layout_threaded_node.mjs \
  --workers=8 \
  --output=../.research/umap-layout-threaded-node-8.f32
```

Gate a saved projection through the exact HDBSCAN implementation:

```sh
SENKO_UMAP_THREADED_OUTPUT="$PWD/../.research/umap-layout-threaded-node-8.f32" \
  pnpm vitest run \
  scripts/clustering-wasm/umap_layout_threaded_output_parity.test.ts \
  --reporter=verbose
```

For the real Chrome matrix, start the cross-origin-isolated Vite server and run
the isolated-profile CDP harness:

```sh
pnpm dev
node scripts/clustering-wasm/umap_layout_threaded_chrome.mjs
```

The runner uses the same isolated Chrome launch policy as Senko's established
browser acceptance runner and writes the full ignored report to
`.research/umap-layout-threaded-chrome-result.json`.

## M3 Chrome 150 validation

Every run completed initialization plus 500 epoch barriers, returned success,
used exactly 16 MiB of shared layout memory, produced seven clusters, and
preserved the exact no-noise mask.

| Workers | Layout time | Pair-distance error vs seeded native | Seeded HDB/common ARI |
| ---: | ---: | ---: | ---: |
| 1 cold | 20,507 ms | 0.05253 | 0.998989 |
| 1 warm | 16,349 ms | 0.05253 | 0.998989 |
| 2 warm | 8,429 ms | 0.05749 | 0.998998 |
| 4 warm | 4,685 ms | 0.22852 | **1.000000, exact** |
| 8 warm, trial 1 | 3,934 ms | 0.07368 | 0.999399 |
| 8 warm, trial 2 | 4,066 ms | 0.06788 | 0.999328 |
| 8 warm, trial 3 | 4,215 ms | 0.13294 | 0.998998 |

The 8-worker median is 4,066 ms, about 4.0x faster than warm serial Wasm and
close to native unseeded parallel UMAP's measured 3.53 seconds. Against the
actual unseeded offline Senko partition, the three 8-worker trials scored ARI
0.999530, 0.999600, and 0.999930. HDBSCAN added 274–287 ms per trial.

ABI v2 was revalidated with a byte-identical one-worker projection hash
(`b3b7a57f…927c2a`) and warmed eight-worker times of 3,871, 3,810, and
4,255 ms (3,871 ms median). All three runs retained seven clusters with no
noise; seeded downstream ARI was 0.99960, 1.0, and 0.99919.

Pair-distance error is expected to be non-monotonic because it measures one
stochastic Hogwild trajectory against a deterministic seeded trajectory. The
downstream partition metrics are the correctness gate that matters to Senko.

## Dynamic row scheduling promotion

The production scheduler now distributes row-aligned 16-row chunks rather
than assigning permanent equal-edge shards. On an already warm target M3,
three adjacent fresh-process Node trials moved the median from 6,335 ms for
the static binary to 5,835 ms for the dynamic binary, a 7.9% reduction. The
single-worker projection retained the exact predecessor hash
`b3b7a57f…927c2a`. Three stochastic eight-worker outputs retained seven
clusters and no noise with seeded downstream ARI from 0.99839 to 0.99899.

In the first production-browser acceptance containing both dynamic scheduling
and the independently exact PyNNDescent pair deduplication, layout measured
3,380.05 ms and whole clustering measured 5,513.93 ms. The adjacent
pre-change run measured 4,412.30 ms and 7,953.42 ms respectively. Whole worker
wall moved from 17,276.48 to 14,588.16 ms despite the thermally variable
conditions, and the promoted result passed the offline reference gate with
seven speakers, 132 segments, 99.9927% speech IoU, and 99.9771% mapped-speaker
agreement at 10 ms.

## Float32 gradient promotion

The final hot arithmetic path casts the validated `a`, `b`, and gamma header
parameters to float32 once, then evaluates the UMAP gradient coefficients,
clipping, alpha schedule, and coordinate updates in float32. Sample clocks,
negative-sample counts, RNG, CSR traversal, epoch barriers, and the
16-accumulator distance reduction are unchanged. This uses ordinary
`-O3 -msimd128`; it does not require fast-math, relaxed SIMD, or another
optional browser feature.

The deterministic one-worker projection now has SHA-256
`2e53fcb45290909cd7382f9ba7175f2b1ad30b9df73848adf4906b28b4d41c29`.
It retains seven clusters, no noise, ARI 0.999068 against the seeded native
partition, and slightly lower pair-distance error against native than the
preceding float64-gradient path. The standalone module shrank from 13,640 to
6,771 bytes because it no longer links the float64 power implementation.

Across two isolated Chrome matrices, six eight-worker trials measured
1,649–1,904 ms with medians of 1,693 and 1,748 ms. Every trial retained seven
clusters and no noise. Seeded-native ARI ranged from 0.998839 to 0.999590 and
unseeded-offline ARI from 0.998519 to 0.999120. Shared memory remains exactly
16 MiB for the one-hour fixture.

Algebraically reusing one power call was measured and rejected: it provided no
speedup in either float64 or float32 and changed the float32 trajectory.
Changing only the power overload while retaining float64 gradient arithmetic
was substantially slower. Neither experiment remains in production.
