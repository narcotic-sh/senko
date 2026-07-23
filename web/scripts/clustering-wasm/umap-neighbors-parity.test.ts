import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import { WasmClusteringKernels } from "../../src/clustering/wasm-kernels";
import {
  loadFixtureArtifact,
  loadParityManifest,
} from "./parity-fixture";

const parityIt =
  process.env.SENKO_RUN_UMAP_NEIGHBOR_PARITY === "1" ? it : it.skip;

describe("native UMAP cosine-neighbor parity", () => {
  parityIt(
    "reproduces the seeded one-hour PyNNDescent graph",
    async () => {
      const fixtureDirectory = new URL(
        "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
        import.meta.url,
      );
      const manifest = await loadParityManifest(fixtureDirectory);
      const referenceIndices = await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapKnnIndices",
      );
      const referenceDistances = await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapKnnDistances",
      );
      expect(referenceIndices).toBeInstanceOf(Int32Array);
      expect(referenceDistances).toBeInstanceOf(Float32Array);

      const embeddingBytes = await readFile(
        new URL(
          "../../../.research/native-reference/embeddings.f32",
          import.meta.url,
        ),
      );
      expect(embeddingBytes.byteLength).toBe(5_713 * 192 * 4);
      expect(createHash("sha256").update(embeddingBytes).digest("hex")).toBe(
        "1d316df53ec713ab866c942b73380bf2ef7be46b4145e88d1aac0be57e67fe1b",
      );
      const embeddings = new Float32Array(
        Uint8Array.from(embeddingBytes).buffer,
      );
      const wasmBytes = await readFile(
        new URL(
          "../../src/clustering/wasm/senko-clustering.wasm",
          import.meta.url,
        ),
      );
      const kernels = await WasmClusteringKernels.fromBytes(wasmBytes);

      try {
        const startedAt = performance.now();
        const candidate = kernels.buildNativeUmapCosineKnn(
          embeddings,
          5_713,
          192,
          40,
          42,
        );
        const elapsedMs = performance.now() - startedAt;
        const metrics = compareNeighborGraphs(
          referenceIndices as Int32Array,
          referenceDistances as Float32Array,
          candidate.indices,
          candidate.distances,
          5_713,
          40,
        );
        console.info(
          JSON.stringify({
            elapsedMs,
            metrics,
            memory: kernels.memoryStats,
          }),
        );

        expect(metrics.missingNeighborCount).toBe(0);
        expect(metrics.exactIndexFraction).toBeGreaterThan(0.9998);
        expect(metrics.meanRecallAtK).toBeGreaterThan(0.9999);
        expect(metrics.maximumSharedDistanceError).toBeLessThan(1e-5);
        expect(kernels.memoryStats.peakArenaUsedBytes).toBeLessThan(
          13 * 1024 * 1024,
        );
      } finally {
        kernels.dispose();
      }
    },
    120_000,
  );
});

function compareNeighborGraphs(
  referenceIndices: Int32Array,
  referenceDistances: Float32Array,
  candidateIndices: Int32Array,
  candidateDistances: Float32Array,
  count: number,
  neighborCount: number,
): {
  readonly exactIndexFraction: number;
  readonly meanRecallAtK: number;
  readonly missingNeighborCount: number;
  readonly maximumSharedDistanceError: number;
} {
  const expectedLength = count * neighborCount;
  expect(referenceIndices).toHaveLength(expectedLength);
  expect(referenceDistances).toHaveLength(expectedLength);
  expect(candidateIndices).toHaveLength(expectedLength);
  expect(candidateDistances).toHaveLength(expectedLength);

  let exactIndexCount = 0;
  let sharedNeighborCount = 0;
  let missingNeighborCount = 0;
  let maximumSharedDistanceError = 0;
  for (let row = 0; row < count; row += 1) {
    const offset = row * neighborCount;
    const referenceRanks = new Map<number, number>();
    for (let rank = 0; rank < neighborCount; rank += 1) {
      referenceRanks.set(referenceIndices[offset + rank]!, rank);
    }
    for (let rank = 0; rank < neighborCount; rank += 1) {
      const position = offset + rank;
      const candidateIndex = candidateIndices[position]!;
      exactIndexCount += candidateIndex === referenceIndices[position] ? 1 : 0;
      if (candidateIndex < 0) {
        missingNeighborCount += 1;
        continue;
      }
      const referenceRank = referenceRanks.get(candidateIndex);
      if (referenceRank === undefined) {
        continue;
      }
      sharedNeighborCount += 1;
      maximumSharedDistanceError = Math.max(
        maximumSharedDistanceError,
        Math.abs(
          candidateDistances[position]! -
            referenceDistances[offset + referenceRank]!,
        ),
      );
    }
  }

  return {
    exactIndexFraction: exactIndexCount / expectedLength,
    meanRecallAtK: sharedNeighborCount / expectedLength,
    missingNeighborCount,
    maximumSharedDistanceError,
  };
}
