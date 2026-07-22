import { describe, expect, it } from "vitest";

import { clusterEmbeddings } from "./cluster";
import {
  mergeSimilarCentroids,
  normalizeLabels,
  reassignMinorClusters,
} from "./postprocess";
import { buildExactEuclideanKnn, normalizeRows } from "./knn";
import type { UmapProjectionStats } from "./types";

describe("clusterEmbeddings", () => {
  it("regresses default UMAP plus sparse hierarchy on a compact speaker corpus", () => {
    const dim = 32;
    const perSpeaker = 48;
    const embeddings = makeSpeakerEmbeddings(3, perSpeaker, dim, 0.055);

    const labels = clusterEmbeddings(embeddings, 3 * perSpeaker, dim);

    expect(new Set(labels).size).toBe(3);
    expect(purity(labels, perSpeaker, 3)).toBe(1);
  });

  it("is deterministic", () => {
    const dim = 24;
    const embeddings = makeSpeakerEmbeddings(4, 36, dim, 0.07);

    const first = clusterEmbeddings(embeddings, 144, dim);
    const second = clusterEmbeddings(embeddings, 144, dim);

    expect([...second]).toEqual([...first]);
  });

  it("reports deterministic typed-array UMAP allocation statistics", () => {
    const dim = 24;
    const count = 72;
    const embeddings = makeSpeakerEmbeddings(3, count / 3, dim, 0.05);
    let first: UmapProjectionStats | undefined;
    let second: UmapProjectionStats | undefined;

    clusterEmbeddings(embeddings, count, dim, {
      onUmapStats(stats) {
        first = stats;
      },
    });
    clusterEmbeddings(embeddings, count, dim, {
      onUmapStats(stats) {
        second = stats;
      },
    });

    expect(first).toBeDefined();
    expect(second).toBeDefined();
    expect(first!.outputBytes).toBe(count * 10 * Float32Array.BYTES_PER_ELEMENT);
    expect(first!.peakTemporaryBytes).toBeGreaterThan(first!.outputBytes);
    expect(first!.peakWorkingBytes).toBe(second!.peakWorkingBytes);
    expect(first!.graphEdgeCount).toBe(second!.graphEdgeCount);
  });

  it("separates interleaved speakers that share a common embedding direction", () => {
    const speakerCount = 3;
    const perSpeaker = 42;
    const dim = 32;
    const embeddings = makeInterleavedCorrelatedEmbeddings(
      speakerCount,
      perSpeaker,
      dim,
    );

    const labels = clusterEmbeddings(embeddings, speakerCount * perSpeaker, dim);

    expect(new Set(labels).size).toBe(speakerCount);
    for (let speaker = 0; speaker < speakerCount; speaker += 1) {
      const speakerLabels = new Set<number>();
      for (let sample = 0; sample < perSpeaker; sample += 1) {
        speakerLabels.add(labels[sample * speakerCount + speaker]!);
      }
      expect(speakerLabels.size).toBe(1);
    }
  });

  it("handles short recordings as one cluster", () => {
    const embeddings = makeSpeakerEmbeddings(2, 4, 8, 0.01);
    expect([...clusterEmbeddings(embeddings, 8, 8)]).toEqual(new Array(8).fill(0));
  });

  it("validates shape and finite input values", () => {
    expect(() => clusterEmbeddings(new Float32Array(3), 2, 2)).toThrow(
      /does not match/,
    );
    expect(() =>
      clusterEmbeddings(new Float32Array([0, Number.NaN]), 1, 2),
    ).toThrow(/non-finite/);
  });

  it("returns an empty typed array", () => {
    expect(clusterEmbeddings(new Float32Array(), 0, 192)).toEqual(
      new Int32Array(),
    );
  });
});

describe("Senko post-processing", () => {
  it("reassigns minor clusters to the nearest major centroid", () => {
    const dim = 2;
    const embeddings = new Float32Array([
      1, 0,
      0.99, 0.01,
      0.98, -0.02,
      0, 1,
      0.01, 0.99,
      -0.02, 0.98,
      0.95, 0.05,
    ]);
    const normalized = normalizeRows(embeddings, 7, dim);
    const labels = new Int32Array([10, 10, 10, 20, 20, 20, -1]);

    reassignMinorClusters(labels, normalized, 7, dim, 3);

    expect([...labels]).toEqual([10, 10, 10, 20, 20, 20, 10]);
  });

  it("preserves a sizable HDBSCAN -1 population like offline Senko", () => {
    const dim = 2;
    const embeddings = new Float32Array([
      2, 0,
      1.9, 0.1,
      2.1, -0.1,
      0, 2,
      0.1, 1.9,
      -0.1, 2.1,
      1.8, 0.2,
      0.2, 1.8,
      1.7, 0.1,
      0.1, 1.7,
    ]);
    const labels = new Int32Array([10, 10, 10, 20, 20, 20, -1, -1, -1, -1]);

    reassignMinorClusters(labels, embeddings, 10, dim, 3);

    expect(new Set(labels)).toEqual(new Set([-1, 10, 20]));
    expect([...labels.slice(6)]).toEqual([-1, -1, -1, -1]);

    // CommonClustering later normalizes sorted labels. The retained -1 group
    // is therefore a speaker, not noise in pipeline metrics.
    normalizeLabels(labels);
    expect(new Set(labels)).toEqual(new Set([0, 1, 2]));
    expect([...labels].every((label) => label >= 0)).toBe(true);
  });

  it("repeatedly merges centroids at the configured cosine threshold", () => {
    const dim = 3;
    const embeddings = normalizeRows(
      new Float32Array([
        1, 0, 0,
        0.99, 0.02, 0,
        0.98, 0.1, 0,
        0.97, 0.11, 0,
        0, 0, 1,
        0, 0.02, 0.99,
      ]),
      6,
      dim,
    );
    const labels = new Int32Array([5, 5, 9, 9, 12, 12]);

    mergeSimilarCentroids(labels, embeddings, 6, dim, 0.875);
    normalizeLabels(labels);

    expect([...labels]).toEqual([0, 0, 0, 0, 1, 1]);
  });
});

describe("Euclidean projection graph", () => {
  it("keeps exact nearest neighbors without a dense distance matrix", () => {
    const graph = buildExactEuclideanKnn(
      new Float32Array([
        0, 0,
        1, 0,
        3, 0,
      ]),
      3,
      2,
      1,
    );

    expect([...graph.indices]).toEqual([1, 0, 1]);
    expect([...graph.similarities]).toEqual([0, 0, -1]);
  });
});

function makeSpeakerEmbeddings(
  speakerCount: number,
  perSpeaker: number,
  dim: number,
  noise: number,
): Float32Array {
  const result = new Float32Array(speakerCount * perSpeaker * dim);
  let randomState = 0x12345678;
  for (let speaker = 0; speaker < speakerCount; speaker += 1) {
    for (let sample = 0; sample < perSpeaker; sample += 1) {
      const offset = (speaker * perSpeaker + sample) * dim;
      for (let column = 0; column < dim; column += 1) {
        randomState = xorshift32(randomState);
        const uniform = randomState / 0xffff_ffff;
        const center = column === speaker ? 1 : 0;
        result[offset + column] = center + (uniform * 2 - 1) * noise;
      }
    }
  }
  return result;
}

function makeInterleavedCorrelatedEmbeddings(
  speakerCount: number,
  perSpeaker: number,
  dim: number,
): Float32Array {
  const result = new Float32Array(speakerCount * perSpeaker * dim);
  let randomState = 0x4f1bbcdc;
  for (let sample = 0; sample < perSpeaker; sample += 1) {
    for (let speaker = 0; speaker < speakerCount; speaker += 1) {
      const row = sample * speakerCount + speaker;
      for (let column = 0; column < dim; column += 1) {
        randomState = xorshift32(randomState);
        const noise = ((randomState / 0xffff_ffff) * 2 - 1) * 0.025;
        const common = column === 0 ? 0.8 : 0;
        const speakerDirection = column === speaker + 1 ? 0.6 : 0;
        result[row * dim + column] = common + speakerDirection + noise;
      }
    }
  }
  return result;
}

function purity(labels: Int32Array, perSpeaker: number, speakerCount: number): number {
  let correct = 0;
  for (let speaker = 0; speaker < speakerCount; speaker += 1) {
    const counts = new Map<number, number>();
    for (let i = 0; i < perSpeaker; i += 1) {
      const label = labels[speaker * perSpeaker + i]!;
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    correct += Math.max(...counts.values());
  }
  return correct / labels.length;
}

function xorshift32(value: number): number {
  let result = value | 0;
  result ^= result << 13;
  result ^= result >>> 17;
  result ^= result << 5;
  return result >>> 0;
}
