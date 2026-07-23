import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { describe, it } from "vitest";

import { WasmClusteringKernels } from "../../src/clustering/wasm-kernels";
import { compareLabelPartitions } from "./parity-diagnostics";
import {
  loadFixtureArtifact,
  loadParityManifest,
} from "./parity-fixture";

const parityIt =
  process.env.SENKO_UMAP_THREADED_OUTPUT === undefined ? it.skip : it;

describe("threaded UMAP saved-output diagnostics", () => {
  parityIt("reports native geometry and downstream partitions", async () => {
    const fixtureDirectory = new URL(
      "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
      import.meta.url,
    );
    const manifest = await loadParityManifest(fixtureDirectory);
    const referenceProjection = (await loadFixtureArtifact(
      fixtureDirectory,
      manifest,
      "umapProjection",
    )) as Float32Array;
    const referenceRawLabels = (await loadFixtureArtifact(
      fixtureDirectory,
      manifest,
      "hdbscanRawLabels",
    )) as Int32Array;
    const referenceCommonLabels = (await loadFixtureArtifact(
      fixtureDirectory,
      manifest,
      "commonLabels",
    )) as Int32Array;
    const candidateBytes = await readFile(
      resolve(process.env.SENKO_UMAP_THREADED_OUTPUT!),
    );
    const candidateProjection = new Float32Array(
      candidateBytes.buffer.slice(
        candidateBytes.byteOffset,
        candidateBytes.byteOffset + candidateBytes.byteLength,
      ),
    );
    if (candidateProjection.length !== 5_713 * 60) {
      throw new RangeError(
        `Unexpected candidate length ${candidateProjection.length}`,
      );
    }

    const wasmBytes = await readFile(
      new URL(
        "../../src/clustering/wasm/senko-clustering.wasm",
        import.meta.url,
      ),
    );
    const kernels = await WasmClusteringKernels.fromBytes(wasmBytes);
    try {
      const startedAt = performance.now();
      const candidateLabels = kernels.clusterHdbscanF64Semantics(
        candidateProjection,
        5_713,
        60,
        20,
        10,
      );
      console.info(
        JSON.stringify({
          hdbscanElapsedMs: performance.now() - startedAt,
          pairDistanceError: sampledPairDistanceRelativeError(
            referenceProjection,
            candidateProjection,
            5_713,
            60,
            8_192,
          ),
          rawPartition: compareLabelPartitions(
            referenceRawLabels,
            candidateLabels,
          ),
          commonPartition: compareLabelPartitions(
            referenceCommonLabels,
            candidateLabels,
          ),
          hdbscanMemory: kernels.memoryStats,
        }),
      );
    } finally {
      kernels.dispose();
    }
  });
});

function sampledPairDistanceRelativeError(
  reference: Float32Array,
  candidate: Float32Array,
  count: number,
  dimension: number,
  sampleCount: number,
): number {
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
