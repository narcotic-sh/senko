import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  buildApproximateCosineKnn,
  buildExactEuclideanKnn,
  normalizeRows,
} from "./knn";
import { clusterEmbeddings } from "./cluster";
import { resolveClusteringOptions } from "./types";
import { WasmClusteringKernels } from "./wasm-kernels";

describe("WasmClusteringKernels", () => {
  it("matches normalization and both k-NN reference implementations", async () => {
    const kernels = await createKernels();
    const count = 96;
    const dim = 24;
    const embeddings = deterministicMatrix(count, dim);
    const options = resolveClusteringOptions({
      neighborCount: 20,
      hashTableCount: 4,
      hashBits: 6,
      bucketSampleLimit: 24,
      temporalNeighborRadius: 6,
    });
    try {
      const normalized = kernels.normalizeRows(embeddings, count, dim);
      expect(normalized).toEqual(normalizeRows(embeddings, count, dim));

      const approximate = kernels.buildApproximateCosineKnn(
        normalized,
        count,
        dim,
        options,
      );
      const expectedApproximate = buildApproximateCosineKnn(
        normalized,
        count,
        dim,
        options,
      );
      expect(approximate.indices).toEqual(expectedApproximate.indices);
      expect(approximate.similarities).toEqual(expectedApproximate.similarities);

      const fusedApproximate =
        kernels.buildNormalizedApproximateCosineKnn(
          embeddings,
          count,
          dim,
          options,
        );
      expect(fusedApproximate.indices).toEqual(expectedApproximate.indices);
      expect(fusedApproximate.similarities).toEqual(
        expectedApproximate.similarities,
      );

      const exact = kernels.buildExactEuclideanKnn(
        embeddings,
        count,
        dim,
        options.neighborCount,
      );
      const expectedExact = buildExactEuclideanKnn(
        embeddings,
        count,
        dim,
        options.neighborCount,
      );
      expect(exact.indices).toEqual(expectedExact.indices);
      expect(exact.similarities).toEqual(expectedExact.similarities);

    } finally {
      kernels.dispose();
    }
  });

  it("uses one fixed eleven-megabyte heap and rejects use after disposal", async () => {
    const kernels = await createKernels();
    kernels.normalizeRows(deterministicMatrix(32, 16), 32, 16);
    expect(kernels.memoryStats.heapBytes).toBe(11 * 1024 * 1024);
    expect(kernels.memoryStats.arenaCapacityBytes).toBe(10 * 1024 * 1024);
    expect(kernels.memoryStats.peakArenaUsedBytes).toBe(32 * 16 * 4);
    expect(kernels.memoryStats.peakReturnedJsBytes).toBe(32 * 16 * 4);
    kernels.dispose();
    expect(() =>
      kernels.normalizeRows(new Float32Array(16), 1, 16),
    ).toThrow(/disposed/);
  });

  it("preflights refinement against the fixed arena before copying inputs", async () => {
    const kernels = await createKernels();
    const count = 6_200;
    const dim = 192;
    const seedNeighborCount = 64;
    const neighborCount = 20;
    try {
      expect(() =>
        kernels.refineEuclideanNeighbors(
          new Float32Array(count * dim),
          count,
          dim,
          neighborCount,
          new Int32Array(count * seedNeighborCount),
          seedNeighborCount,
          0x6d2b79f5,
        ),
      ).toThrow(
        /needs 10486928 scratch bytes for 6200 rows; fixed arena capacity is 10485760/,
      );
      expect(kernels.memoryStats.peakArenaUsedBytes).toBe(0);
    } finally {
      kernels.dispose();
    }
  });

  it("preserves complete deterministic clustering labels", async () => {
    const kernels = await createKernels();
    const count = 144;
    const dim = 32;
    const embeddings = deterministicSpeakerMatrix(3, count / 3, dim);
    try {
      const options = { umapRandomSeed: 0x6d2b79f5 } as const;
      const expected = clusterEmbeddings(embeddings, count, dim, options);
      const actual = clusterEmbeddings(
        embeddings,
        count,
        dim,
        options,
        kernels,
      );
      expect(actual).toEqual(expected);
    } finally {
      kernels.dispose();
    }
  });
});

async function createKernels(): Promise<WasmClusteringKernels> {
  const bytes = await readFile(
    new URL("./wasm/senko-clustering.wasm", import.meta.url),
  );
  return WasmClusteringKernels.fromBytes(Uint8Array.from(bytes).buffer);
}

function deterministicMatrix(count: number, dim: number): Float32Array {
  const result = new Float32Array(count * dim);
  let state = 0x12345678;
  for (let i = 0; i < result.length; i += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    result[i] = ((state >>> 8) / 0x01000000) * 2 - 1;
  }
  return result;
}

function deterministicSpeakerMatrix(
  speakerCount: number,
  perSpeaker: number,
  dim: number,
): Float32Array {
  const result = new Float32Array(speakerCount * perSpeaker * dim);
  let state = 0x42d00d;
  for (let speaker = 0; speaker < speakerCount; speaker += 1) {
    for (let sample = 0; sample < perSpeaker; sample += 1) {
      const offset = (speaker * perSpeaker + sample) * dim;
      for (let column = 0; column < dim; column += 1) {
        state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
        const noise = ((state >>> 8) / 0x01000000 - 0.5) * 0.08;
        result[offset + column] =
          (column === speaker || column === speaker + speakerCount ? 1 : 0) +
          noise;
      }
    }
  }
  return result;
}
