import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import { WasmClusteringKernels } from "../../src/clustering/wasm-kernels";
import {
  loadFixtureArtifact,
  loadParityManifest,
} from "./parity-fixture";

const parityIt =
  process.env.SENKO_RUN_UMAP_SPECTRAL_PARITY === "1" ? it : it.skip;
const longParityIt =
  process.env.SENKO_RUN_UMAP_SPECTRAL_LONG_PARITY === "1" ? it : it.skip;
// Deterministic output after explicitly SIMD-vectorizing the three f64
// reductions. The reduction grouping changes only last-bit numerics; the
// long-fixture eigenspace was separately gated against the scalar predecessor.
const SIMD_REDUCTION_LONG_VECTOR_SHA256 =
  "9b7d7ed290d2c497fcb6ad52a7d33913075a85cdb33c35524cc39e3a96ba08e5";

describe("native UMAP spectral-initialization parity", () => {
  parityIt(
    "matches the seeded one-hour normalized-Laplacian eigenspace",
    async () => {
      const fixtureDirectory = new URL(
        "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
        import.meta.url,
      );
      const manifest = await loadParityManifest(fixtureDirectory);
      const sourceOffsets = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapGraphIndptr",
      )) as Int32Array;
      const sourceColumns = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapGraphIndices",
      )) as Int32Array;
      const sourceValues = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapGraphData",
      )) as Float32Array;
      const reference = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapSpectralEmbedding",
      )) as Float64Array;
      const graph = compactNonzeroCsr(
        sourceOffsets,
        sourceColumns,
        sourceValues,
      );
      expect(graph.values).toHaveLength(351_946);

      const wasmBytes = await readFile(
        new URL(
          "../../src/clustering/wasm/senko-clustering.wasm",
          import.meta.url,
        ),
      );
      const kernels = await WasmClusteringKernels.fromBytes(wasmBytes);
      try {
        const startedAt = performance.now();
        const candidate = kernels.initializeNativeUmapSpectral(
          graph,
          5_713,
          60,
        );
        const elapsedMs = performance.now() - startedAt;
        const pairDistanceError = sampledPairDistanceRelativeError(
          reference,
          candidate.values,
          5_713,
          60,
          8_192,
        );
        console.info(
          JSON.stringify({
            elapsedMs,
            pairDistanceError,
            stats: candidate.stats,
            memory: kernels.memoryStats,
          }),
        );

        expect(candidate.stats.requestedEigenpairs).toBe(61);
        expect(candidate.stats.convergedEigenpairs).toBe(61);
        expect(candidate.stats.maximumResidual).toBeLessThan(1e-4);
        expect(candidate.stats.peakWorkingBytes).toBeLessThan(
          9 * 1024 * 1024,
        );
        expect(pairDistanceError).toBeLessThan(2e-4);
      } finally {
        kernels.dispose();
      }
    },
    30_000,
  );

  longParityIt(
    "keeps the eight-hour spectral solve inside the reusable WASM arena",
    async () => {
      const fixtureDirectory = new URL(
        "../../../.research/native-reference/clustering-parity/test-audio-long/seed-42/",
        import.meta.url,
      );
      const manifest = await loadParityManifest(fixtureDirectory);
      const sourceOffsets = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapGraphIndptr",
      )) as Int32Array;
      const sourceColumns = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapGraphIndices",
      )) as Int32Array;
      const sourceValues = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapGraphData",
      )) as Float32Array;
      const graph = compactNonzeroCsr(
        sourceOffsets,
        sourceColumns,
        sourceValues,
      );
      expect(sourceOffsets).toHaveLength(43_805);
      expect(graph.values).toHaveLength(2_836_232);

      const wasmBytes = await readFile(
        new URL(
          "../../src/clustering/wasm/senko-clustering.wasm",
          import.meta.url,
        ),
      );
      const kernels = await WasmClusteringKernels.fromBytes(wasmBytes);
      try {
        // Model the ordinary long-recording order: k-NN grows the reusable
        // arena first, then spectral must fit inside it without malloc/sbrk
        // pages above that high-water mark.
        const internal = kernels as unknown as {
          readonly exports: {
            readonly cluster_umap_cosine_knn_workspace_bytes: (
              count: number,
              dimension: number,
              neighborCount: number,
              seed: number,
            ) => number;
            readonly cluster_reserve: (bytes: number) => number;
          };
        };
        const knnWorkspace =
          internal.exports.cluster_umap_cosine_knn_workspace_bytes(
            43_804,
            192,
            40,
            42,
          );
        const knnInputsAndOutputs =
          43_804 * 192 * Float32Array.BYTES_PER_ELEMENT +
          43_804 *
            40 *
            (Int32Array.BYTES_PER_ELEMENT + Float32Array.BYTES_PER_ELEMENT);
        const knnArenaBytes =
          Math.ceil(knnInputsAndOutputs / 16) * 16 + knnWorkspace;
        expect(knnArenaBytes).toBe(92_208_192);
        expect(internal.exports.cluster_reserve(knnArenaBytes)).toBe(1);
        const heapAfterKnnReserve = kernels.memoryStats.heapBytes;
        expect(heapAfterKnnReserve).toBe(92_340_224);

        const startedAt = performance.now();
        const candidate = kernels.initializeNativeUmapSpectral(
          graph,
          43_804,
          60,
        );
        const elapsedMs = performance.now() - startedAt;
        console.info(
          JSON.stringify({
            elapsedMs,
            stats: candidate.stats,
            memory: kernels.memoryStats,
          }),
        );

        expect(candidate.stats.requestedEigenpairs).toBe(61);
        expect(candidate.stats.convergedEigenpairs).toBe(61);
        expect(candidate.stats.maximumResidual).toBeLessThan(1e-4);
        expect(candidate.stats.peakWorkingBytes).toBe(42_492_944);
        expect(sha256(candidate.values)).toBe(
          SIMD_REDUCTION_LONG_VECTOR_SHA256,
        );
        expect(kernels.memoryStats.peakArenaUsedBytes).toBeLessThan(
          84 * 1024 * 1024,
        );
        expect(kernels.memoryStats.arenaCapacityBytes).toBe(92_209_152);
        expect(kernels.memoryStats.heapBytes).toBe(heapAfterKnnReserve);
      } finally {
        kernels.dispose();
      }
    },
    30_000,
  );
});

function sha256(values: ArrayBufferView): string {
  return createHash("sha256")
    .update(
      new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
    )
    .digest("hex");
}

function compactNonzeroCsr(
  sourceOffsets: Int32Array,
  sourceColumns: Int32Array,
  sourceValues: Float32Array,
): {
  readonly rowOffsets: Int32Array;
  readonly columnIndices: Int32Array;
  readonly values: Float32Array;
} {
  const count = sourceOffsets.length - 1;
  const rowOffsets = new Int32Array(count + 1);
  const columnIndices = new Int32Array(sourceColumns.length);
  const values = new Float32Array(sourceValues.length);
  let write = 0;
  for (let row = 0; row < count; row += 1) {
    for (
      let edge = sourceOffsets[row]!;
      edge < sourceOffsets[row + 1]!;
      edge += 1
    ) {
      if (sourceValues[edge] === 0) continue;
      columnIndices[write] = sourceColumns[edge]!;
      values[write] = sourceValues[edge]!;
      write += 1;
    }
    rowOffsets[row + 1] = write;
  }
  return {
    rowOffsets,
    columnIndices: columnIndices.slice(0, write),
    values: values.slice(0, write),
  };
}

function sampledPairDistanceRelativeError(
  reference: Float64Array,
  candidate: Float64Array,
  count: number,
  dimension: number,
  sampleCount: number,
): number {
  expect(reference).toHaveLength(count * dimension);
  expect(candidate).toHaveLength(count * dimension);
  let state = 0x243f6a88;
  let numerator = 0;
  let denominator = 0;
  for (let sample = 0; sample < sampleCount; sample += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    const left = state % count;
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    let right = state % count;
    if (right === left) right = (right + 1) % count;
    let referenceSquared = 0;
    let candidateSquared = 0;
    for (let column = 0; column < dimension; column += 1) {
      const referenceDifference =
        reference[left * dimension + column]! -
        reference[right * dimension + column]!;
      const candidateDifference =
        candidate[left * dimension + column]! -
        candidate[right * dimension + column]!;
      referenceSquared += referenceDifference * referenceDifference;
      candidateSquared += candidateDifference * candidateDifference;
    }
    const referenceDistance = Math.sqrt(referenceSquared);
    const difference = Math.sqrt(candidateSquared) - referenceDistance;
    numerator += difference * difference;
    denominator += referenceDistance * referenceDistance;
  }
  return Math.sqrt(numerator / denominator);
}
