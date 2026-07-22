import { describe, expect, it } from "vitest";

import type { DiarizationSegment } from "../runtime/types";
import type { Subsegment } from "./types";
import {
  mergeDiarizationSegments,
  normalizeClusterLabels,
  postprocessClustering,
} from "./postprocess";

function subsegment(index: number, start: number, end: number): Subsegment {
  return { index, start, end };
}

describe("cluster label normalization", () => {
  it("sorts arbitrary labels, including HDBSCAN's noise label", () => {
    expect([...normalizeClusterLabels([7, -1, 7, 2])]).toEqual([2, 0, 2, 1]);
  });

  it("rejects labels that cannot represent cluster integers", () => {
    expect(() => normalizeClusterLabels([0, 1.5])).toThrow(/safe integer/);
    expect(() => normalizeClusterLabels([0, Number.NaN])).toThrow(/safe integer/);
  });
});

describe("segment merging", () => {
  it("merges same-speaker gaps at the inclusive four-second boundary", () => {
    const input: DiarizationSegment[] = [
      { startSeconds: 0, endSeconds: 2, speaker: "SPEAKER_01" },
      { startSeconds: 6, endSeconds: 8, speaker: "SPEAKER_01" },
      { startSeconds: 12.001, endSeconds: 14, speaker: "SPEAKER_01" },
    ];

    expect(mergeDiarizationSegments(input)).toEqual([
      { startSeconds: 0, endSeconds: 8, speaker: "SPEAKER_01" },
      { startSeconds: 12.001, endSeconds: 14, speaker: "SPEAKER_01" },
    ]);
    expect(input[0]?.endSeconds).toBe(2);
  });

  it("removes durations at the inclusive 0.78-second boundary", () => {
    expect(
      mergeDiarizationSegments([
        { startSeconds: 0, endSeconds: 0.78, speaker: "SPEAKER_01" },
        { startSeconds: 2, endSeconds: 2.780001, speaker: "SPEAKER_02" },
      ]),
    ).toEqual([{ startSeconds: 2, endSeconds: 2.780001, speaker: "SPEAKER_02" }]);
  });

  it("joins matching speakers around a discarded short segment", () => {
    expect(
      mergeDiarizationSegments([
        { startSeconds: 0, endSeconds: 2, speaker: "SPEAKER_01" },
        { startSeconds: 2, endSeconds: 2.7, speaker: "SPEAKER_02" },
        { startSeconds: 2.7, endSeconds: 5, speaker: "SPEAKER_01" },
      ]),
    ).toEqual([{ startSeconds: 0, endSeconds: 5, speaker: "SPEAKER_01" }]);
  });

  it("does not bridge a short segment when its neighbors differ", () => {
    expect(
      mergeDiarizationSegments([
        { startSeconds: 0, endSeconds: 2, speaker: "SPEAKER_01" },
        { startSeconds: 2, endSeconds: 2.5, speaker: "SPEAKER_02" },
        { startSeconds: 2.5, endSeconds: 5, speaker: "SPEAKER_03" },
      ]),
    ).toEqual([
      { startSeconds: 0, endSeconds: 2, speaker: "SPEAKER_01" },
      { startSeconds: 2.5, endSeconds: 5, speaker: "SPEAKER_03" },
    ]);
  });

  it("assigns rather than maximizes the end of an overlapping same-speaker turn", () => {
    expect(
      mergeDiarizationSegments([
        { startSeconds: 0, endSeconds: 10, speaker: "SPEAKER_01" },
        { startSeconds: 9, endSeconds: 9.5, speaker: "SPEAKER_01" },
      ]),
    ).toEqual([{ startSeconds: 0, endSeconds: 9.5, speaker: "SPEAKER_01" }]);
  });

  it("bridges matching neighbors around a short turn without applying the gap limit again", () => {
    expect(
      mergeDiarizationSegments([
        { startSeconds: 0, endSeconds: 2, speaker: "SPEAKER_01" },
        { startSeconds: 10, endSeconds: 10.5, speaker: "SPEAKER_02" },
        { startSeconds: 20, endSeconds: 22, speaker: "SPEAKER_01" },
      ]),
    ).toEqual([{ startSeconds: 0, endSeconds: 22, speaker: "SPEAKER_01" }]);
  });

  it("removes short first and last turns while rechecking the shifted index", () => {
    expect(
      mergeDiarizationSegments([
        { startSeconds: 0, endSeconds: 0.2, speaker: "SPEAKER_01" },
        { startSeconds: 1, endSeconds: 1.3, speaker: "SPEAKER_02" },
        { startSeconds: 2, endSeconds: 4, speaker: "SPEAKER_03" },
        { startSeconds: 5, endSeconds: 5.5, speaker: "SPEAKER_04" },
      ]),
    ).toEqual([{ startSeconds: 2, endSeconds: 4, speaker: "SPEAKER_03" }]);
  });
});

describe("clustering postprocessing", () => {
  it("ports centroid, overlap, merge-alias, and speaking-time rank behavior", () => {
    const result = postprocessClustering(
      new Float32Array([
        1, 1,
        10, 0,
        3, 3,
        0, 8,
      ]),
      [7, -1, 7, 2],
      [
        subsegment(0, 0, 1.5),
        subsegment(1, 0.6, 2.1),
        subsegment(2, 1.2, 2.7),
        subsegment(3, 1.8, 3.3),
      ],
    );

    expect([...result.normalizedLabels]).toEqual([2, 0, 2, 1]);
    expect(result.rawSegments).toEqual([
      { startSeconds: 0, endSeconds: 2.25, speaker: "SPEAKER_01" },
      { startSeconds: 1.05, endSeconds: 1.65, speaker: "SPEAKER_03" },
      { startSeconds: 1.65, endSeconds: 2.25, speaker: "SPEAKER_01" },
      { startSeconds: 2.25, endSeconds: 3.3, speaker: "SPEAKER_02" },
    ]);
    expect(result.mergedSegments).toEqual([
      { startSeconds: 0, endSeconds: 2.25, speaker: "SPEAKER_01" },
      { startSeconds: 2.25, endSeconds: 3.3, speaker: "SPEAKER_02" },
    ]);
    expect([...result.centroids.SPEAKER_01!]).toEqual([2, 2]);
    expect([...result.centroids.SPEAKER_02!]).toEqual([0, 8]);
    expect([...result.centroids.SPEAKER_03!]).toEqual([10, 0]);
    expect(result.rawSpeakerCount).toBe(3);
    expect(result.speakerCount).toBe(2);
  });

  it("coalesces overlapping adjacent subsegments for one speaker", () => {
    const result = postprocessClustering(
      new Float32Array([1, 2]),
      [42, 42],
      [subsegment(0, -0.5, 1.5), subsegment(1, 0.6, 2.1)],
    );

    expect(result.rawSegments).toEqual([
      { startSeconds: 0, endSeconds: 2.1, speaker: "SPEAKER_01" },
    ]);
    expect(result.mergedSegments).toEqual(result.rawSegments);
    expect([...result.centroids.SPEAKER_01!]).toEqual([1.5]);
  });

  it("uses stable first-occurrence order to break speaking-time ties", () => {
    const result = postprocessClustering(
      new Float32Array([9, 2]),
      [9, 2],
      [subsegment(0, 0, 1), subsegment(1, 1, 2)],
    );

    expect(result.rawSegments.map((segment) => segment.speaker)).toEqual([
      "SPEAKER_01",
      "SPEAKER_02",
    ]);
    expect([...result.centroids.SPEAKER_01!]).toEqual([9]);
    expect([...result.centroids.SPEAKER_02!]).toEqual([2]);
  });

  it("counts final speakers after filtering while preserving raw speakers and centroids", () => {
    const result = postprocessClustering(
      new Float32Array([1, 2]),
      [0, 1],
      [subsegment(0, 0, 2), subsegment(1, 10, 10.5)],
    );

    expect(result.rawSegments).toEqual([
      { startSeconds: 0, endSeconds: 2, speaker: "SPEAKER_01" },
      { startSeconds: 10, endSeconds: 10.5, speaker: "SPEAKER_02" },
    ]);
    expect(result.mergedSegments).toEqual([
      { startSeconds: 0, endSeconds: 2, speaker: "SPEAKER_01" },
    ]);
    expect(Object.keys(result.centroids)).toEqual(["SPEAKER_01", "SPEAKER_02"]);
    expect(result.rawSpeakerCount).toBe(2);
    expect(result.speakerCount).toBe(1);
  });

  it("returns an empty result for empty clustering output", () => {
    expect(postprocessClustering(new Float32Array(), [], [])).toEqual({
      normalizedLabels: new Int32Array(),
      rawSegments: [],
      mergedSegments: [],
      centroids: {},
      rawSpeakerCount: 0,
      speakerCount: 0,
    });
  });

  it("rejects inconsistent tensor shapes", () => {
    expect(() =>
      postprocessClustering(
        new Float32Array([1, 2]),
        [0],
        [subsegment(0, 0, 1), subsegment(1, 1, 2)],
      ),
    ).toThrow(/one cluster label per subsegment/);
    expect(() =>
      postprocessClustering(
        new Float32Array([1, 2, 3]),
        [0, 1],
        [subsegment(0, 0, 1), subsegment(1, 1, 2)],
      ),
    ).toThrow(/not divisible/);
  });
});
