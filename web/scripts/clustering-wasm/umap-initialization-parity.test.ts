import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import { WasmClusteringKernels } from "../../src/clustering/wasm-kernels";
import {
  loadFixtureArtifact,
  loadParityManifest,
} from "./parity-fixture";

const parityIt =
  process.env.SENKO_RUN_UMAP_INITIALIZATION_PARITY === "1" ? it : it.skip;

describe("native UMAP layout-initialization parity", () => {
  parityIt(
    "exactly reproduces the seeded one-hour Float32 coordinates and RNG",
    async () => {
      const fixtureDirectory = new URL(
        "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
        import.meta.url,
      );
      const manifest = await loadParityManifest(fixtureDirectory);
      const spectral = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapSpectralEmbedding",
      )) as Float64Array;
      const referenceEmbedding = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapLayoutInitialEmbedding",
      )) as Float32Array;
      const referenceRng = (await loadFixtureArtifact(
        fixtureDirectory,
        manifest,
        "umapLayoutRngState",
      )) as BigInt64Array;
      const wasmBytes = await readFile(
        new URL(
          "../../src/clustering/wasm/senko-clustering.wasm",
          import.meta.url,
        ),
      );
      const kernels = await WasmClusteringKernels.fromBytes(wasmBytes);
      try {
        const startedAt = performance.now();
        const candidate = kernels.initializeNativeUmapLayout(
          spectral,
          5_713,
          60,
          42,
          true,
        );
        const elapsedMs = performance.now() - startedAt;
        console.info(
          JSON.stringify({
            elapsedMs,
            memory: kernels.memoryStats,
          }),
        );
        expect(candidate.embedding).toEqual(referenceEmbedding);
        expect(candidate.rngState).toEqual(referenceRng);
      } finally {
        kernels.dispose();
      }
    },
    30_000,
  );
});
