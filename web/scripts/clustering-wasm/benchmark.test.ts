import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";

import { describe, expect, it } from "vitest";

import { clusterSparseGraph } from "../../src/clustering/hierarchy";
import {
  buildApproximateCosineKnn,
  buildExactEuclideanKnn,
  normalizeRows,
} from "../../src/clustering/knn";
import type { ClusteringNumericKernels } from "../../src/clustering/numeric-kernels";
import {
  mergeSimilarCentroids,
  normalizeLabels,
  reassignMinorClusters,
} from "../../src/clustering/postprocess";
import { resolveClusteringOptions } from "../../src/clustering/types";
import type { UmapProjectionStats } from "../../src/clustering/types";
import { projectWithUmap } from "../../src/clustering/umap";
import { WasmClusteringKernels } from "../../src/clustering/wasm-kernels";

const COUNT = 5_713;
const DIMENSION = 192;
const enabled = process.env.SENKO_RUN_CLUSTERING_BENCHMARK === "1";

describe.skipIf(!enabled)("clustering WASM benchmark", () => {
  it("profiles the native-reference fixture and preserves exact labels", async () => {
    const embeddings = readFixture();
    const wasmBytes = readFileSync(
      new URL("../../src/clustering/wasm/senko-clustering.wasm", import.meta.url),
    );
    const kernels = await WasmClusteringKernels.fromBytes(
      Uint8Array.from(wasmBytes).buffer,
    );
    try {
      const warmupStarted = performance.now();
      kernels.warmup();
      console.info(
        JSON.stringify({
          backend: "wasm",
          warmupMs: performance.now() - warmupStarted,
        }),
      );
      const seedParity = benchmarkSeedParity(embeddings, kernels);
      console.info(JSON.stringify(seedParity.reference));
      console.info(JSON.stringify(seedParity.candidate));
      expect(seedParity.candidate.exactIndices).toBe(true);
      expect(seedParity.candidate.exactSimilarities).toBe(true);
      expect(seedParity.candidate.memory.heapBytes).toBe(9 * 1024 * 1024);
      expect(seedParity.candidate.memory.peakArenaUsedBytes).toBeLessThanOrEqual(
        seedParity.candidate.memory.arenaCapacityBytes,
      );

      const reference = runPipeline(embeddings);
      console.info(JSON.stringify({ backend: "typescript", ...reference.stats }));

      for (let iteration = 0; iteration < 3; iteration += 1) {
        const candidate = runPipeline(embeddings, kernels);
        const exactLabels = equalInt32(candidate.labels, reference.labels);
        const ari = adjustedRandIndex(candidate.labels, reference.labels);
        console.info(
          JSON.stringify({
            backend: "wasm",
            iteration,
            ...candidate.stats,
            exactLabels,
            adjustedRandIndex: ari,
            memory: kernels.memoryStats,
          }),
        );
        expect(exactLabels).toBe(true);
        expect(ari).toBe(1);
      }
    } finally {
      kernels.dispose();
    }
  }, 60_000);
});

interface BenchmarkStats {
  readonly totalMs: number;
  readonly exactKnnMs: number;
  readonly hierarchyMs: number;
  readonly postprocessMs: number;
  readonly clusterCount: number;
  readonly umap: UmapProjectionStats;
}

function benchmarkSeedParity(
  embeddings: Float32Array,
  kernels: WasmClusteringKernels,
): {
  readonly reference: {
    readonly backend: "typescript-seed";
    readonly totalMs: number;
  };
  readonly candidate: {
    readonly backend: "wasm-fused-seed";
    readonly totalMs: number;
    readonly exactIndices: boolean;
    readonly exactSimilarities: boolean;
    readonly memory: ReturnType<typeof readKernelMemoryStats>;
  };
} {
  const options = resolveClusteringOptions({ neighborCount: 64 });
  const referenceStarted = performance.now();
  const normalized = normalizeRows(embeddings, COUNT, DIMENSION);
  const reference = buildApproximateCosineKnn(
    normalized,
    COUNT,
    DIMENSION,
    options,
  );
  const referenceMs = performance.now() - referenceStarted;

  const candidateStarted = performance.now();
  const candidate = kernels.buildNormalizedApproximateCosineKnn(
    embeddings,
    COUNT,
    DIMENSION,
    options,
  );
  const candidateMs = performance.now() - candidateStarted;
  return {
    reference: {
      backend: "typescript-seed",
      totalMs: referenceMs,
    },
    candidate: {
      backend: "wasm-fused-seed",
      totalMs: candidateMs,
      exactIndices: equalInt32(candidate.indices, reference.indices),
      exactSimilarities: equalFloat32(
        candidate.similarities,
        reference.similarities,
      ),
      memory: readKernelMemoryStats(kernels),
    },
  };
}

function readKernelMemoryStats(kernels: WasmClusteringKernels) {
  return kernels.memoryStats;
}

function runPipeline(
  embeddings: Float32Array,
  kernels?: ClusteringNumericKernels,
): { readonly labels: Int32Array; readonly stats: BenchmarkStats } {
  let umap: UmapProjectionStats | undefined;
  const options = resolveClusteringOptions({
    onUmapStats(value) {
      umap = value;
    },
  });
  const started = performance.now();
  const projection = projectWithUmap(
    embeddings,
    COUNT,
    DIMENSION,
    options,
    kernels,
  );
  const exactStarted = performance.now();
  const graph = buildExactEuclideanKnn(
    projection.values,
    COUNT,
    projection.dimension,
    options.neighborCount,
    kernels,
  );
  const exactKnnMs = performance.now() - exactStarted;
  const hierarchyStarted = performance.now();
  const labels = clusterSparseGraph(
    graph,
    COUNT,
    options.minSamples,
    options.minClusterSize,
  );
  const hierarchyMs = performance.now() - hierarchyStarted;
  const postprocessStarted = performance.now();
  reassignMinorClusters(
    labels,
    embeddings,
    COUNT,
    DIMENSION,
    options.minClusterSize,
  );
  if (options.mergeThreshold !== null) {
    mergeSimilarCentroids(
      labels,
      embeddings,
      COUNT,
      DIMENSION,
      options.mergeThreshold,
    );
  }
  normalizeLabels(labels);
  const postprocessMs = performance.now() - postprocessStarted;
  if (umap === undefined) {
    throw new Error("UMAP statistics listener was not called");
  }
  return {
    labels,
    stats: {
      totalMs: performance.now() - started,
      exactKnnMs,
      hierarchyMs,
      postprocessMs,
      clusterCount: new Set(labels).size,
      umap,
    },
  };
}

function readFixture(): Float32Array {
  const bytes = readFileSync(
    new URL("../../../.research/native-reference/embeddings.f32", import.meta.url),
  );
  if (bytes.byteLength !== COUNT * DIMENSION * Float32Array.BYTES_PER_ELEMENT) {
    throw new Error(`Unexpected embeddings fixture size ${bytes.byteLength}`);
  }
  const result = new Float32Array(COUNT * DIMENSION);
  new Uint8Array(result.buffer).set(bytes);
  return result;
}

function equalInt32(left: Int32Array, right: Int32Array): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function equalFloat32(left: Float32Array, right: Float32Array): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (!Object.is(left[index], right[index])) return false;
  }
  return true;
}

function adjustedRandIndex(left: Int32Array, right: Int32Array): number {
  if (left.length !== right.length) {
    throw new RangeError("ARI label arrays must have equal lengths");
  }
  const leftCounts = new Map<number, number>();
  const rightCounts = new Map<number, number>();
  const intersections = new Map<string, number>();
  for (let index = 0; index < left.length; index += 1) {
    const leftLabel = left[index]!;
    const rightLabel = right[index]!;
    leftCounts.set(leftLabel, (leftCounts.get(leftLabel) ?? 0) + 1);
    rightCounts.set(rightLabel, (rightCounts.get(rightLabel) ?? 0) + 1);
    const key = `${leftLabel}:${rightLabel}`;
    intersections.set(key, (intersections.get(key) ?? 0) + 1);
  }
  const pairs = (count: number): number => (count * (count - 1)) / 2;
  let intersectionPairs = 0;
  let leftPairs = 0;
  let rightPairs = 0;
  for (const count of intersections.values()) intersectionPairs += pairs(count);
  for (const count of leftCounts.values()) leftPairs += pairs(count);
  for (const count of rightCounts.values()) rightPairs += pairs(count);
  const totalPairs = pairs(left.length);
  const expected = (leftPairs * rightPairs) / totalPairs;
  const maximum = (leftPairs + rightPairs) / 2;
  return maximum === expected
    ? 1
    : (intersectionPairs - expected) / (maximum - expected);
}
