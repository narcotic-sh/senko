# Long-recording clustering acceptance

This protocol validates clustering beyond the one-hour reference without
checking a giant audio fixture into the repository. It complements, rather
than replaces, the cooled `test_audio.wav` acceptance in this directory.

## Required shapes

The native 1.5-second window and 0.6-second repeated-addition policy produces
exactly **47,999 embedding rows** for one continuous eight-hour VAD region:
47,998 ordinary windows followed by one EOF-adjusted tail. The focused
`IncrementalVadSubsegmentReducer` test constructs only that time region; it
does not allocate or decode eight hours of PCM.

Use a deterministic speaker-like `Float32Array(47_999 * 192)` for the numeric
scalability run. Generate it from a fixed PRNG seed and a small fixed set of
speaker centroids so the fixture occupies about 35.2 MiB and need not be
stored in git.

## Correctness gates

1. Keep the current unordered-pair mode for every input whose exact scratch
   requirement fits the ordinary arena. Its 5,713-row neighbor heap and final
   labels must remain byte-identical to the checked-in reference.
2. Exercise the scalable mode immediately on both sides of its selection
   boundary. Compare all neighbor indices, distances, and `isNew` flags with
   the no-deduplication reference, not only final labels.
3. Preserve candidate traversal order, four PRNG draws per row, bilateral heap
   updates, change counting, and the convergence stop. A probabilistic visited
   set is not acceptable because a false positive changes the graph.
4. Run the complete 47,999-row UMAP and sparse hierarchy. Require finite
   outputs, one label per row, stable labels across repeated browser runs, and
   successful Senko minor-cluster reassignment and centroid merging.
5. Do not split an eight-hour recording into independently clustered chunks;
   speaker identity and density relationships are global.

Offline UMAP is intentionally unseeded, so cross-runtime acceptance should use
the existing segment/cluster agreement criteria rather than byte equality with
one native stochastic run.

## Memory gates

At 47,999 rows, the triangular pair bitset alone would occupy 143,991,004
bytes. The scalable refinement path must not allocate it. With 192 input
dimensions, 64 seed neighbors, and 20 retained neighbors, embeddings, seed
indices, output heaps, snapshots, and one count-sized stamp array require about
62.8 MB (59.9 MiB) before WebAssembly page rounding.

Record and assert all of the following:

- scalable refinement completes without a fixed-arena capacity exception;
- `peakArenaUsedBytes` stays within the exact linear-memory preflight plus one
  WebAssembly page of alignment tolerance;
- the 47,999-row clustering heap remains far below the approximately 197 MiB
  required by the triangular-bitset implementation;
- `PipelineResult.memory.wasmHeapBytes` equals the retained FBank heap plus the
  clustering heap size observed **after** clustering growth;
- a second run in the same worker does not grow the heap again for the same
  shape and does not materially increase post-GC page memory.

The input `Blob` remains external and must be reported separately from known
CPU allocations, as in the ordinary benchmark protocol.

## Performance and regression gates

- Re-run the isolated, cooled three-trial `test_audio.wav` acceptance. The
  5,713-row fast path and current M3 production kernels must remain selected;
  treat a repeatable regression outside normal run-to-run noise as a failure.
- Report eight-hour synthetic seed, refinement, layout, exact 40-neighbor
  graph, hierarchy, and total clustering times separately. This is a capacity
  and scaling measurement; it does not inherit the one-hour under-30-second
  end-to-end target.
- Record which refinement mode ran, row count, WebAssembly heap bytes, arena
  high-water mark, returned-JS high-water mark, and label hash in every result
  so a silent return to quadratic retained state is visible.

## Current synthetic acceptance

The deterministic 47,999 by 192 full-clustering test passed on the target M3
in 31.85 seconds. UMAP took 18.30 seconds (4.46-second seed, 8.65-second
refinement, 0.17-second fuzzy graph, and 5.02-second layout); exact 40-NN,
hierarchy, and post-processing completed in the remaining 13.55 seconds. The
row-stamp path used 63,383,408 arena bytes, the page-rounded arena was
63,438,848 bytes, the complete WASM heap was 63,569,920 bytes, and the final
9-cluster label hash was `0a3d1ee4`.
