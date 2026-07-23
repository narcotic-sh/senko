import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  clusterEmbeddingsNativeSerial,
  prepareNativeLayoutGraph,
} from "../../src/clustering/native-umap";
import { WasmClusteringKernels } from "../../src/clustering/wasm-kernels";
import { compareLabelPartitions } from "./parity-diagnostics";
import {
  loadFixtureArtifact,
  loadParityManifest,
} from "./parity-fixture";

const parityIt =
  process.env.SENKO_RUN_NATIVE_UMAP_PARITY === "1" ? it : it.skip;

describe("assembled native UMAP/HDBSCAN parity path", () => {
  parityIt(
    "reproduces native graph pruning and sample clocks exactly",
    async () => {
      const fixtureDirectory = fixtureUrl();
      const manifest = await loadParityManifest(fixtureDirectory);
      const graph = prepareNativeLayoutGraph(
        {
          rowOffsets: (await loadFixtureArtifact(
            fixtureDirectory,
            manifest,
            "umapGraphIndptr",
          )) as Int32Array,
          columnIndices: (await loadFixtureArtifact(
            fixtureDirectory,
            manifest,
            "umapGraphIndices",
          )) as Int32Array,
          values: (await loadFixtureArtifact(
            fixtureDirectory,
            manifest,
            "umapGraphData",
          )) as Float32Array,
        },
        5_713,
        500,
      );
      const referenceHead = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapLayoutHead",
      )) as Int32Array;
      const referenceTail = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapLayoutTail",
      )) as Int32Array;
      const referenceEpochs = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapLayoutEpochsPerSample",
      )) as Float64Array;

      const candidateHead = new Int32Array(referenceHead.length);
      for (let row = 0; row + 1 < graph.rowOffsets.length; row += 1) {
        candidateHead.fill(
          row,
          graph.rowOffsets[row]!,
          graph.rowOffsets[row + 1]!,
        );
      }
      expect(candidateHead).toEqual(referenceHead);
      expect(graph.columnIndices).toEqual(referenceTail);
      expect(graph.epochsPerSample).toEqual(referenceEpochs);
    },
    30_000,
  );

  parityIt(
    "keeps the native common-clustering partition end to end",
    async () => {
      const fixtureDirectory = fixtureUrl();
      const manifest = await loadParityManifest(fixtureDirectory);
      const embeddingBytes = await readFile(
        new URL(
          "../../../.research/native-reference/embeddings.f32",
          import.meta.url,
        ),
      );
      const embeddings = new Float32Array(
        Uint8Array.from(embeddingBytes).buffer,
      );
      const referenceLabels = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "normalizedLabels",
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
        const candidate = clusterEmbeddingsNativeSerial(
          embeddings,
          5_713,
          192,
          42,
          kernels,
        );
        const elapsedMs = performance.now() - startedAt;
        const partition = compareLabelPartitions(
          referenceLabels,
          candidate.labels,
        );
        console.info(
          JSON.stringify({
            elapsedMs,
            stats: candidate.stats,
            partition,
            memory: kernels.memoryStats,
          }),
        );
        expect(partition.adjustedRandIndex).toBeGreaterThan(0.98);
        expect(partition.exactNoiseMask).toBe(true);
        expect(partition.reference.clusterCount).toBe(
          partition.candidate.clusterCount,
        );
      } finally {
        kernels.dispose();
      }
    },
    90_000,
  );
});

function fixtureUrl(): URL {
  return new URL(
    "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
    import.meta.url,
  );
}
