# Native clustering parity fixtures

This directory generates stage-by-stage numeric references for porting Senko's
offline UMAP/HDBSCAN clustering to the browser. It does not modify the offline
pipeline. The generator loads the saved CAM++ embeddings and the checked-in
offline clustering configuration, then calls the same native libraries and
the same `CommonClustering` post-processing methods.

The tool emits two kinds of reference:

- `unseeded/` is one observed run of ordinary offline Senko behavior. UMAP has
  no `random_state`, uses its parallel optimizer, and is intentionally
  stochastic.
- `seed-42/` is a diagnostic oracle. This is the same configuration with
  `random_state=42`; UMAP therefore uses its reproducible single-thread
  optimizer. The seed is for differential tests only and must not become a
  production Senko default.

## Generate and validate

From the repository root:

```sh
uv run --project web/scripts/clustering-parity --python 3.13 \
  web/scripts/clustering-parity/generate_native_fixture.py \
  --mode both
```

Generated data is intentionally kept under the ignored research directory:

```text
.research/native-reference/clustering-parity/test-audio/
  unseeded/
  seed-42/
```

To replace existing fixtures, pass `--overwrite`. To validate hashes, byte
lengths, shapes, and recorded invariants without rerunning clustering:

```sh
uv run --project web/scripts/clustering-parity --python 3.13 \
  web/scripts/clustering-parity/generate_native_fixture.py \
  --validate-only \
  .research/native-reference/clustering-parity/test-audio/seed-42
```

The script defaults to the one-hour
`.research/native-reference/test-audio-reference.npz` embedding capture.
`--source`, `--source-member`, `--config`, and `--output-root` can select a
different saved native reference without changing Senko.

## Fixture contract

Every run has a `manifest.json` with source hashes, offline source/config
hashes, package versions, exact parameters, stochastic mode, timing,
population summaries, validation results, and a hash/shape/dtype entry for
every binary artifact.

The main artifacts are:

- UMAP projection, fuzzy-graph CSR arrays, sigmas/rhos, and (when retained by
  `umap-learn`) the native k-NN arrays.
- HDBSCAN Euclidean core distances and its internal squared-distance values.
- The sorted approximate Borůvka KD-tree MST, single-linkage tree, condensed
  tree, probabilities, stability scores, and raw noise-preserving labels.
- Labels after Senko's minor-cluster policy, after its repeated centroid merge
  policy, and after the consecutive-label normalization in
  `Diarizer._perform_clustering`.

Numeric files are headerless little-endian arrays. Shapes and column meanings
are defined by their manifest entries. HDBSCAN's packed tree matrices use
Float64; integer-valued node identifiers remain exactly representable for
the supported fixture sizes.
