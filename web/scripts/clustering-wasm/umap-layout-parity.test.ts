import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  WasmClusteringKernels,
  type NativeUmapLayoutInitialization,
} from "../../src/clustering/wasm-kernels";
import {
  compareLabelPartitions,
} from "./parity-diagnostics";
import {
  loadFixtureArtifact,
  loadParityManifest,
} from "./parity-fixture";

const parityIt =
  process.env.SENKO_RUN_UMAP_LAYOUT_PARITY === "1" ? it : it.skip;

describe("native UMAP serial-layout parity", () => {
  parityIt(
    "preserves the native one-hour density partition after 500 epochs",
    async () => {
      const fixtureDirectory = new URL(
        "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
        import.meta.url,
      );
      const manifest = await loadParityManifest(fixtureDirectory);
      const initialization: NativeUmapLayoutInitialization = {
        embedding: (await loadFixtureArtifact(
          fixtureDirectory,
          manifest,
          "umapLayoutInitialEmbedding",
        )) as Float32Array,
        rngState: (await loadFixtureArtifact(
          fixtureDirectory,
          manifest,
          "umapLayoutRngState",
        )) as BigInt64Array,
      };
      const head = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapLayoutHead",
      )) as Int32Array;
      const tail = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapLayoutTail",
      )) as Int32Array;
      const epochsPerSample = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapLayoutEpochsPerSample",
      )) as Float64Array;
      const referenceProjection = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapProjection",
      )) as Float32Array;
      const referenceLabels = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "hdbscanRawLabels",
      )) as Int32Array;
      const wasmBytes = await readFile(
        new URL(
          "../../src/clustering/wasm/senko-clustering.wasm",
          import.meta.url,
        ),
      );
      const kernels = await WasmClusteringKernels.fromBytes(wasmBytes);
      try {
        const startedAt = performance.now();
        const projection = kernels.optimizeNativeUmapLayoutSerial(
          initialization,
          5_713,
          60,
          head,
          tail,
          epochsPerSample,
          500,
          1.932808397545408,
          0.7904949735905139,
        );
        const layoutElapsedMs = performance.now() - startedAt;
        const pairDistanceError = sampledPairDistanceRelativeError(
          referenceProjection,
          projection,
          5_713,
          60,
          8_192,
        );
        const labels = kernels.clusterHdbscanF64Semantics(
          projection,
          5_713,
          60,
          20,
          10,
        );
        const partition = compareLabelPartitions(referenceLabels, labels);
        console.info(
          JSON.stringify({
            layoutElapsedMs,
            pairDistanceError,
            partition,
            memory: kernels.memoryStats,
          }),
        );

        expect(pairDistanceError).toBeLessThan(0.06);
        expect(partition.adjustedRandIndex).toBeGreaterThan(0.998);
        expect(partition.exactNoiseMask).toBe(true);
        expect(partition.reference.clusterCount).toBe(
          partition.candidate.clusterCount,
        );
      } finally {
        kernels.dispose();
      }
    },
    60_000,
  );
});

function sampledPairDistanceRelativeError(
  reference: Float32Array,
  candidate: Float32Array,
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
