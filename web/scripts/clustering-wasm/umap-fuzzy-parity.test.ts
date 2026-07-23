import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  WasmClusteringKernels,
  type NativeUmapKnnGraph,
} from "../../src/clustering/wasm-kernels";
import {
  loadFixtureArtifact,
  loadParityManifest,
} from "./parity-fixture";

const parityIt =
  process.env.SENKO_RUN_UMAP_FUZZY_PARITY === "1" ? it : it.skip;

describe("native UMAP fuzzy-graph parity", () => {
  parityIt(
    "reproduces native smooth-kNN and fuzzy-union CSR semantics",
    async () => {
      const fixtureDirectory = new URL(
        "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
        import.meta.url,
      );
      const manifest = await loadParityManifest(fixtureDirectory);
      const knnIndices = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapKnnIndices",
      )) as Int32Array;
      const knnDistances = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapKnnDistances",
      )) as Float32Array;
      const referenceSigmas = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapSigmas",
      )) as Float32Array;
      const referenceRhos = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapRhos",
      )) as Float32Array;
      const referenceOffsets = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapGraphIndptr",
      )) as Int32Array;
      const referenceColumns = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapGraphIndices",
      )) as Int32Array;
      const referenceValues = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapGraphData",
      )) as Float32Array;
      const wasmBytes = await readFile(
        new URL(
          "../../src/clustering/wasm/senko-clustering.wasm",
          import.meta.url,
        ),
      );
      const kernels = await WasmClusteringKernels.fromBytes(wasmBytes);

      try {
        const knn: NativeUmapKnnGraph = {
          indices: knnIndices,
          distances: knnDistances,
          neighborCount: 40,
        };
        const startedAt = performance.now();
        const candidate = kernels.buildNativeUmapFuzzyGraph(knn, 5_713);
        const elapsedMs = performance.now() - startedAt;

        expect(candidate.rowOffsets).toEqual(referenceOffsets);
        expect(candidate.columnIndices).toEqual(referenceColumns);
        const sigmaError = maximumAbsoluteError(
          candidate.sigmas,
          referenceSigmas,
        );
        const rhoError = maximumAbsoluteError(candidate.rhos, referenceRhos);
        let maximumRetainedWeightError = 0;
        let prunedReferenceCount = 0;
        let maximumCandidateForPrunedReference = 0;
        for (let edge = 0; edge < referenceValues.length; edge += 1) {
          const reference = referenceValues[edge]!;
          const actual = candidate.values[edge]!;
          if (reference === 0) {
            prunedReferenceCount += 1;
            maximumCandidateForPrunedReference = Math.max(
              maximumCandidateForPrunedReference,
              actual,
            );
          } else {
            maximumRetainedWeightError = Math.max(
              maximumRetainedWeightError,
              Math.abs(actual - reference),
            );
          }
        }
        console.info(
          JSON.stringify({
            elapsedMs,
            edgeCount: candidate.values.length,
            sigmaError,
            rhoError,
            maximumRetainedWeightError,
            prunedReferenceCount,
            maximumCandidateForPrunedReference,
            memory: kernels.memoryStats,
          }),
        );

        expect(candidate.values).toHaveLength(referenceValues.length);
        expect(sigmaError).toBeLessThan(5e-7);
        expect(rhoError).toBe(0);
        expect(maximumRetainedWeightError).toBeLessThan(2e-6);
        expect(prunedReferenceCount).toBeGreaterThan(0);
        expect(maximumCandidateForPrunedReference).toBeLessThan(1 / 500);
        expect(kernels.memoryStats.peakArenaUsedBytes).toBeLessThan(
          10 * 1024 * 1024,
        );
      } finally {
        kernels.dispose();
      }
    },
    30_000,
  );
});

function maximumAbsoluteError(
  left: Float32Array,
  right: Float32Array,
): number {
  expect(left).toHaveLength(right.length);
  let maximum = 0;
  for (let index = 0; index < left.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(left[index]! - right[index]!));
  }
  return maximum;
}
