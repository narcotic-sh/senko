import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  buildApproximateCosineKnn,
  buildExactEuclideanKnn,
  normalizeRows,
} from "./knn";
import { clusterEmbeddings } from "./cluster";
import {
  resolveClusteringOptions,
  type UmapProjectionStats,
} from "./types";
import { refineEuclideanNeighborsReference } from "./umap";
import { WasmClusteringKernels } from "./wasm-kernels";

const scaleIt =
  process.env.SENKO_RUN_CLUSTERING_SCALE_TEST === "1" ? it : it.skip;
const fullScaleIt =
  process.env.SENKO_RUN_CLUSTERING_FULL_SCALE_TEST === "1" ? it : it.skip;

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

  it("matches native HDBSCAN semantics on a two-cluster noise fixture", async () => {
    const kernels = await createKernels();
    const projection = new Float32Array([
      -0.15, 0.08,
      -0.08, -0.04,
      0.02, 0.12,
      0.09, -0.1,
      0.16, 0.03,
      -0.02, -0.16,
      9.85, 10.08,
      9.92, 9.96,
      10.02, 10.12,
      10.09, 9.9,
      10.16, 10.03,
      9.98, 9.84,
      50, -40,
    ]);
    try {
      const labels = kernels.clusterHdbscanF64Semantics(
        projection,
        13,
        2,
        2,
        3,
      );
      expect(labels[12]).toBe(-1);
      expect(labels[0]).not.toBe(-1);
      expect(labels[6]).not.toBe(-1);
      expect(labels[0]).not.toBe(labels[6]);
      expect([...labels.subarray(0, 6)]).toEqual(
        Array.from({ length: 6 }, () => labels[0]),
      );
      expect([...labels.subarray(6, 12)]).toEqual(
        Array.from({ length: 6 }, () => labels[6]),
      );
    } finally {
      kernels.dispose();
    }
  });

  it("starts with the compact eleven-megabyte heap and rejects use after disposal", async () => {
    const kernels = await createKernels();
    kernels.normalizeRows(deterministicMatrix(32, 16), 32, 16);
    expect(kernels.memoryStats.heapBytes).toBe(11 * 1024 * 1024);
    expect(kernels.memoryStats.arenaCapacityBytes).toBe(10 * 1024 * 1024);
    expect(kernels.memoryStats.peakArenaUsedBytes).toBe(32 * 16 * 4);
    expect(kernels.memoryStats.peakReturnedJsBytes).toBe(32 * 16 * 4);
    const finalMemoryStats = kernels.memoryStats;
    kernels.dispose();
    expect(kernels.memoryStats).toEqual(finalMemoryStats);
    expect(() =>
      kernels.normalizeRows(new Float32Array(16), 1, 16),
    ).toThrow(/disposed/);
  });

  it("grows the reusable arena by exact WebAssembly pages before making views", async () => {
    const kernels = await createKernels();
    try {
      const count = 3_000_000;
      const result = kernels.normalizeRows(new Float32Array(count), count, 1);
      const expectedCapacity =
        Math.ceil(result.byteLength / (64 * 1024)) * 64 * 1024;
      expect(result).toHaveLength(count);
      expect(kernels.memoryStats.arenaCapacityBytes).toBe(expectedCapacity);
      expect(kernels.memoryStats.heapBytes).toBeGreaterThan(11 * 1024 * 1024);
      expect(kernels.memoryStats.heapBytes % (64 * 1024)).toBe(0);
      expect(kernels.memoryStats.peakArenaUsedBytes).toBe(result.byteLength);

      kernels.normalizeRows(deterministicMatrix(32, 16), 32, 16);
      expect(kernels.memoryStats.arenaCapacityBytes).toBe(expectedCapacity);
      expect(kernels.memoryStats.peakArenaUsedBytes).toBe(result.byteLength);
    } finally {
      kernels.dispose();
    }
  });

  it("uses linear row stamps for a 47,999-row refinement", async () => {
    const kernels = await createKernels();
    const count = 47_999;
    const embeddings = new Float32Array(count);
    const seedIndices = new Int32Array(count);
    for (let row = 0; row < count; row += 1) {
      embeddings[row] = (row % 997) / 997;
      seedIndices[row] = (row + 1) % count;
    }
    try {
      const refined = kernels.refineEuclideanNeighbors(
        embeddings,
        count,
        1,
        2,
        seedIndices,
        1,
        0x6d2b79f5,
      );
      expect(refined.indices).toHaveLength(count * 2);
      expect(refined.distances).toHaveLength(count * 2);
      expect(refined.isNew).toHaveLength(count * 2);
      expect(kernels.memoryStats.lastRefinementMode).toBe("row-stamps");
      expect(kernels.memoryStats.peakArenaUsedBytes).toBe(1_919_980);
      expect(kernels.memoryStats.arenaCapacityBytes).toBe(10 * 1024 * 1024);
    } finally {
      kernels.dispose();
    }
  });

  it("matches the no-dedup predecessor across the dense/stamp boundary", async () => {
    const kernels = await createKernels();
    try {
      for (const [count, expectedMode] of [
        [12_668, "dense-pair-bitset"],
        [12_669, "row-stamps"],
      ] as const) {
        const embeddings = new Float32Array(count);
        const seedIndices = new Int32Array(count);
        for (let row = 0; row < count; row += 1) {
          embeddings[row] = Math.fround(
            Math.sin(row * 0.017) + (row % 11) * 0.003,
          );
          seedIndices[row] = (row + 1) % count;
        }
        const expected = refineEuclideanNeighborsReference(
          embeddings,
          count,
          1,
          2,
          seedIndices,
          1,
          0x6d2b79f5,
        );
        const actual = kernels.refineEuclideanNeighbors(
          embeddings,
          count,
          1,
          2,
          seedIndices,
          1,
          0x6d2b79f5,
        );

        expect(kernels.memoryStats.lastRefinementMode).toBe(expectedMode);
        expect(actual.indices).toEqual(expected.indices);
        expect(actual.distances).toEqual(expected.distances);
        expect(actual.isNew).toEqual(expected.isNew);
      }
    } finally {
      kernels.dispose();
    }
  });

  scaleIt(
    "scales native-shape refinement to 47,999 rows without repeat growth",
    async () => {
      const kernels = await createKernels();
      const count = 47_999;
      const dim = 192;
      const seedNeighborCount = 64;
      const neighborCount = 20;
      const embeddings = new Float32Array(count * dim);
      const seedIndices = new Int32Array(count * seedNeighborCount);
      try {
        const first = kernels.refineEuclideanNeighbors(
          embeddings,
          count,
          dim,
          neighborCount,
          seedIndices,
          seedNeighborCount,
          0x6d2b79f5,
        );
        expectValidRefinement(first, count, neighborCount);
        expect(kernels.memoryStats.lastRefinementMode).toBe("row-stamps");
        expect(kernels.memoryStats.peakArenaUsedBytes).toBe(62_782_700);
        expect(kernels.memoryStats.arenaCapacityBytes).toBe(62_783_488);
        const heapBytes = kernels.memoryStats.heapBytes;

        const second = kernels.refineEuclideanNeighbors(
          embeddings,
          count,
          dim,
          neighborCount,
          seedIndices,
          seedNeighborCount,
          0x6d2b79f5,
        );
        expectValidRefinement(second, count, neighborCount);
        expect(second.indices).toEqual(first.indices);
        expect(second.distances).toEqual(first.distances);
        expect(kernels.memoryStats.arenaCapacityBytes).toBe(62_783_488);
        expect(kernels.memoryStats.heapBytes).toBe(heapBytes);
      } finally {
        kernels.dispose();
      }
    },
    120_000,
  );

  fullScaleIt(
    "completes full clustering for 47,999 native-shape embeddings",
    async () => {
      const kernels = await createKernels();
      const count = 47_999;
      const dim = 192;
      const embeddings = deterministicLongSpeakerMatrix(count, dim, 8);
      let umap: UmapProjectionStats | undefined;
      try {
        const started = performance.now();
        const labels = clusterEmbeddings(
          embeddings,
          count,
          dim,
          {
            onUmapStats(stats) {
              umap = stats;
            },
          },
          kernels,
        );
        const totalMs = performance.now() - started;
        const clusterCount = new Set(labels).size;
        const labelHash = hashLabels(labels);

        expect(labels).toHaveLength(count);
        expect([...labels].every((label) => label >= 0)).toBe(true);
        expect(clusterCount).toBe(9);
        expect(labelHash).toBe("0a3d1ee4");
        expect(umap?.count).toBe(count);
        expect(umap?.outputDimension).toBe(10);
        expect(kernels.memoryStats.lastRefinementMode).toBe("row-stamps");
        expect(kernels.memoryStats.peakArenaUsedBytes).toBeLessThan(
          70 * 1024 * 1024,
        );
        console.info(
          JSON.stringify({
            test: "full-47,999-row-clustering",
            totalMs,
            clusterCount,
            labelHash,
            umap,
            memory: kernels.memoryStats,
          }),
        );
      } finally {
        kernels.dispose();
      }
    },
    240_000,
  );

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
      expect(kernels.memoryStats.lastRefinementMode).toBe(
        "dense-pair-bitset",
      );
    } finally {
      kernels.dispose();
    }
  });
});

function expectValidRefinement(
  refinement: {
    readonly indices: Int32Array;
    readonly distances: Float32Array;
    readonly isNew: Uint8Array;
  },
  count: number,
  neighborCount: number,
): void {
  expect(refinement.indices).toHaveLength(count * neighborCount);
  expect(refinement.distances).toHaveLength(count * neighborCount);
  expect(refinement.isNew).toHaveLength(count * neighborCount);
  for (let row = 0; row < count; row += 997) {
    const offset = row * neighborCount;
    expect(refinement.indices[offset]).toBeGreaterThanOrEqual(0);
    expect(refinement.indices[offset]).toBeLessThan(count);
    expect(refinement.distances[offset]).toBe(0);
  }
}

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

function deterministicLongSpeakerMatrix(
  count: number,
  dim: number,
  speakerCount: number,
): Float32Array {
  const result = new Float32Array(count * dim);
  let state = 0x8badf00d;
  for (let row = 0; row < count; row += 1) {
    const speaker = Math.floor(row / 75) % speakerCount;
    const offset = row * dim;
    for (let column = 0; column < dim; column += 1) {
      state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
      const noise = ((state >>> 8) / 0x01000000 - 0.5) * 0.06;
      result[offset + column] =
        (column === speaker || column === speaker + speakerCount ? 1 : 0) +
        noise;
    }
  }
  return result;
}

function hashLabels(labels: Int32Array): string {
  let hash = 0x811c9dc5;
  for (const label of labels) {
    hash = Math.imul(hash ^ label, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}
