import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { postprocessClustering } from "../pipeline/postprocess";
import {
  buildPrunedCosineLaplacian,
  clusterEmbeddingsSpectral,
  type SpectralClusteringStats,
} from "./spectral";

const nativeFixturePath = resolve(
  import.meta.dirname,
  "../../../.research/native-reference/test-audio-short-embeddings.f32",
);

describe("short-recording spectral clustering", () => {
  it("matches Senko's cosine pruning and symmetrization on a tiny matrix", () => {
    const graph = buildPrunedCosineLaplacian(
      new Float32Array([
        1, 0,
        0.9, 0.1,
        0, 1,
        -1, 0,
      ]),
      4,
      2,
      0.5,
      1,
    );

    expect(graph.retainedPerRow).toBe(2);
    expect(graph.rowOffsets[4]).toBe(graph.undirectedEdgeCount * 2);
    for (let row = 0; row < 4; row += 1) {
      let degree = 0;
      for (
        let edge = graph.rowOffsets[row]!;
        edge < graph.rowOffsets[row + 1]!;
        edge += 1
      ) {
        const column = graph.columns[edge]!;
        const reverseStart = graph.rowOffsets[column]!;
        const reverseEnd = graph.rowOffsets[column + 1]!;
        let reverseWeight: number | undefined;
        for (let reverse = reverseStart; reverse < reverseEnd; reverse += 1) {
          if (graph.columns[reverse] === row) {
            reverseWeight = graph.weights[reverse]!;
            break;
          }
        }
        expect(reverseWeight).toBe(graph.weights[edge]);
        degree += Math.abs(graph.weights[edge]!);
      }
      expect(graph.degrees[row]).toBeCloseTo(degree, 6);
    }
  });

  it("uses Senko's under-ten shortcut", () => {
    const labels = clusterEmbeddingsSpectral(new Float32Array(9 * 3), 9, 3);
    expect([...labels]).toEqual(new Array(9).fill(0));
  });

  it("separates compact synthetic speaker groups", () => {
    const speakerCount = 3;
    const perSpeaker = 40;
    const dim = 24;
    const embeddings = new Float32Array(speakerCount * perSpeaker * dim);
    let state = 0x243f6a88;
    for (let speaker = 0; speaker < speakerCount; speaker += 1) {
      for (let sample = 0; sample < perSpeaker; sample += 1) {
        const row = speaker * perSpeaker + sample;
        for (let column = 0; column < dim; column += 1) {
          state = xorshift32(state);
          embeddings[row * dim + column] =
            (column === speaker ? 1 : 0) +
            ((state / 0xffff_ffff) * 2 - 1) * 0.035;
        }
      }
    }

    const labels = clusterEmbeddingsSpectral(
      embeddings,
      speakerCount * perSpeaker,
      dim,
    );

    expect(clusterSizes(labels)).toEqual([40, 40, 40]);
  });

  it.runIf(existsSync(nativeFixturePath))(
    "matches the native short-audio speaker partition",
    () => {
    const count = 806;
    const dim = 192;
    const embeddings = readFloat32Fixture(
      "test-audio-short-embeddings.f32",
    );
    const expected = readInt32Fixture(
      "test-audio-short-spectral-labels.i32",
    );
    let stats: SpectralClusteringStats | undefined;

    const labels = clusterEmbeddingsSpectral(embeddings, count, dim, {
      onStats(value) {
        stats = value;
      },
    });

    expect(samePartition(labels, expected)).toBe(true);
    expect([...clusterSizes(labels)]).toEqual([296, 209, 209, 92]);
    expect(stats?.speakerCountBeforePostprocess).toBe(4);
    expect(stats?.retainedPerRow).toBe(10);
    expect(stats?.peakWorkingBytes).toBeLessThan(8 * 1024 * 1024);
    expect(stats?.avoidedDenseMatrixBytes).toBe(5_197_088);
    const boundaries = readFloat64Fixture(
      "test-audio-short-subsegments.f64",
    );
    const subsegments = Array.from({ length: count }, (_, index) => ({
      index,
      start: boundaries[index * 2]!,
      end: boundaries[index * 2 + 1]!,
    }));
    const postprocessed = postprocessClustering(
      embeddings,
      labels,
      subsegments,
    );
    const native = JSON.parse(
      readFixture("test-audio-short-reference.json").toString("utf8"),
    ) as NativeShortReference;
    expect(postprocessed.rawSpeakerCount).toBe(native.raw_speakers_detected);
    expect(postprocessed.speakerCount).toBe(native.merged_speakers_detected);
    expect(postprocessed.rawSegments).toHaveLength(native.raw_segments.length);
    expect(postprocessed.mergedSegments).toHaveLength(
      native.merged_segments.length,
    );
    for (let index = 0; index < native.raw_segments.length; index += 1) {
      const actual = postprocessed.rawSegments[index]!;
      const expectedSegment = native.raw_segments[index]!;
      expect(actual.speaker).toBe(expectedSegment.speaker);
      expect(actual.startSeconds).toBeCloseTo(expectedSegment.start, 10);
      expect(actual.endSeconds).toBeCloseTo(expectedSegment.end, 10);
    }
    for (let index = 0; index < native.merged_segments.length; index += 1) {
      const actual = postprocessed.mergedSegments[index]!;
      const expectedSegment = native.merged_segments[index]!;
      expect(actual.speaker).toBe(expectedSegment.speaker);
      expect(actual.startSeconds).toBeCloseTo(expectedSegment.start, 10);
      expect(actual.endSeconds).toBeCloseTo(expectedSegment.end, 10);
    }
    if (process.env.SENKO_REPORT_SPECTRAL_STATS === "1") {
      console.info(JSON.stringify(stats));
    }
    },
    30_000,
  );
});

function readFloat32Fixture(name: string): Float32Array {
  const bytes = readFixture(name);
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
}

function readInt32Fixture(name: string): Int32Array {
  const bytes = readFixture(name);
  return new Int32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
}

function readFloat64Fixture(name: string): Float64Array {
  const bytes = readFixture(name);
  return new Float64Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
}

function readFixture(name: string): Buffer {
  return readFileSync(
    resolve(import.meta.dirname, `../../../.research/native-reference/${name}`),
  );
}

function samePartition(left: Int32Array, right: Int32Array): boolean {
  if (left.length !== right.length) {
    return false;
  }
  const forward = new Map<number, number>();
  const reverse = new Map<number, number>();
  for (let index = 0; index < left.length; index += 1) {
    const leftLabel = left[index]!;
    const rightLabel = right[index]!;
    if (
      (forward.has(leftLabel) && forward.get(leftLabel) !== rightLabel) ||
      (reverse.has(rightLabel) && reverse.get(rightLabel) !== leftLabel)
    ) {
      return false;
    }
    forward.set(leftLabel, rightLabel);
    reverse.set(rightLabel, leftLabel);
  }
  return true;
}

function clusterSizes(labels: Int32Array): number[] {
  const sizes = new Map<number, number>();
  for (const label of labels) {
    sizes.set(label, (sizes.get(label) ?? 0) + 1);
  }
  return [...sizes.values()].sort((left, right) => right - left);
}

function xorshift32(value: number): number {
  let result = value | 0;
  result ^= result << 13;
  result ^= result >>> 17;
  result ^= result << 5;
  return result >>> 0;
}

interface NativeShortReference {
  readonly raw_speakers_detected: number;
  readonly merged_speakers_detected: number;
  readonly raw_segments: readonly NativeSegment[];
  readonly merged_segments: readonly NativeSegment[];
}

interface NativeSegment {
  readonly speaker: string;
  readonly start: number;
  readonly end: number;
}
