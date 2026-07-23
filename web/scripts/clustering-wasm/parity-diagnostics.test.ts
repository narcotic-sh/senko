import { describe, expect, it } from "vitest";

import {
  adjustedRandIndex,
  compareLabelPartitions,
  compareMstEdges,
  compareNumericArrays,
  summarizeLabels,
} from "./parity-diagnostics";

describe("clustering parity diagnostics", () => {
  it("compares equivalent raw labels independently of cluster numbering", () => {
    const reference = new Int32Array([-1, 8, 8, 3, 3, 3, -1, 12]);
    const candidate = new Int32Array([-1, 40, 40, 7, 7, 7, -1, 2]);

    const diagnostics = compareLabelPartitions(reference, candidate);

    expect(diagnostics.adjustedRandIndex).toBe(1);
    expect(diagnostics.exactPartition).toBe(true);
    expect(diagnostics.exactNoiseMask).toBe(true);
    expect(diagnostics.noiseMismatchCount).toBe(0);
    expect(diagnostics.reference).toEqual({
      sampleCount: 8,
      clusterCount: 3,
      noiseCount: 2,
      clusterSizesDescending: [3, 2, 1],
      histogram: [
        { label: -1, size: 2 },
        { label: 3, size: 3 },
        { label: 8, size: 2 },
        { label: 12, size: 1 },
      ],
    });
    expect(diagnostics.candidate.clusterSizesDescending).toEqual([3, 2, 1]);
  });

  it("detects a wrong noise mask even when ARI reports identical partitions", () => {
    const reference = new Int32Array([-1, -1, 4, 4]);
    const candidate = new Int32Array([9, 9, -1, -1]);

    const diagnostics = compareLabelPartitions(reference, candidate);

    expect(diagnostics.adjustedRandIndex).toBe(1);
    expect(diagnostics.exactPartition).toBe(false);
    expect(diagnostics.exactNoiseMask).toBe(false);
    expect(diagnostics.noiseMismatchCount).toBe(4);
    expect(diagnostics.firstNoiseMismatchIndices).toEqual([0, 1, 2, 3]);
  });

  it("reports partition drift and limits sampled noise mismatches", () => {
    const reference = new Int32Array([-1, -1, -1, 2, 2, 3, 3, 3]);
    const candidate = new Int32Array([0, 0, 0, 0, 0, -1, -1, -1]);

    const diagnostics = compareLabelPartitions(reference, candidate, 2);

    expect(diagnostics.adjustedRandIndex).toBeLessThan(1);
    expect(diagnostics.exactPartition).toBe(false);
    expect(diagnostics.exactNoiseMask).toBe(false);
    expect(diagnostics.noiseMismatchCount).toBe(6);
    expect(diagnostics.firstNoiseMismatchIndices).toEqual([0, 1]);
    expect(diagnostics.reference.clusterCount).toBe(2);
    expect(diagnostics.candidate.clusterCount).toBe(1);
  });

  it("matches sklearn ARI edge-case semantics", () => {
    expect(adjustedRandIndex(new Int32Array(), new Int32Array())).toBe(1);
    expect(adjustedRandIndex(new Int32Array([5]), new Int32Array([-1]))).toBe(1);
    expect(
      adjustedRandIndex(
        new Int32Array([0, 0, 1, 1]),
        new Int32Array([0, 1, 0, 1]),
      ),
    ).toBeCloseTo(-0.5, 15);
  });

  it("summarizes an all-noise result without inventing a cluster", () => {
    expect(summarizeLabels(new Int32Array([-1, -1, -1]))).toEqual({
      sampleCount: 3,
      clusterCount: 0,
      noiseCount: 3,
      clusterSizesDescending: [],
      histogram: [{ label: -1, size: 3 }],
    });
  });

  it("compares Float64 intermediates with combined absolute/relative tolerance", () => {
    const diagnostics = compareNumericArrays(
      new Float64Array([0, 1, 1_000_000, Number.POSITIVE_INFINITY]),
      new Float64Array([5e-10, 1 + 5e-7, 1_000_000.5, Number.NEGATIVE_INFINITY]),
      { absolute: 1e-9, relative: 1e-6 },
    );

    expect(diagnostics.mismatchCount).toBe(1);
    expect(diagnostics.nonFiniteMismatchCount).toBe(1);
    expect(diagnostics.firstMismatches).toHaveLength(1);
    expect(diagnostics.firstMismatches[0]!.index).toBe(3);
    expect(diagnostics.maxAbsoluteError).toBe(Number.POSITIVE_INFINITY);
  });

  it("compares MST endpoints independently of edge order and orientation", () => {
    const reference = [
      { from: 0, to: 3, weight: 0.25 },
      { from: 3, to: 2, weight: 0.5 },
      { from: 2, to: 1, weight: 0.75 },
    ];
    const candidate = [
      { from: 1, to: 2, weight: 0.750_000_000_01 },
      { from: 3, to: 0, weight: 0.25 },
      { from: 2, to: 3, weight: 0.5 },
    ];

    const diagnostics = compareMstEdges(reference, candidate, {
      absolute: 1e-9,
      relative: 0,
    });

    expect(diagnostics.exactEndpoints).toBe(true);
    expect(diagnostics.missingEndpointPairs).toEqual([]);
    expect(diagnostics.unexpectedEndpointPairs).toEqual([]);
    expect(diagnostics.weights.mismatchCount).toBe(0);
  });

  it("separates MST topology drift from weight drift", () => {
    const diagnostics = compareMstEdges(
      [
        { from: 0, to: 1, weight: 0.2 },
        { from: 1, to: 2, weight: 0.4 },
      ],
      [
        { from: 0, to: 1, weight: 0.3 },
        { from: 0, to: 2, weight: 0.4 },
      ],
      { absolute: 0, relative: 0 },
    );

    expect(diagnostics.exactEndpoints).toBe(false);
    expect(diagnostics.missingEndpointPairs).toEqual(["1:2"]);
    expect(diagnostics.unexpectedEndpointPairs).toEqual(["0:2"]);
    expect(diagnostics.weights.mismatchCount).toBe(1);
  });

  it("rejects malformed diagnostics inputs", () => {
    expect(() =>
      compareLabelPartitions(new Int32Array([0]), new Int32Array()),
    ).toThrow(/equal lengths/);
    expect(() => summarizeLabels(new Int32Array([-2]))).toThrow(
      /greater than or equal to -1/,
    );
    expect(() =>
      compareNumericArrays(new Float64Array([1]), new Float64Array([1]), {
        absolute: -1,
        relative: 0,
      }),
    ).toThrow(/non-negative/);
    expect(() =>
      compareMstEdges(
        [
          { from: 0, to: 1, weight: 1 },
          { from: 1, to: 0, weight: 1 },
        ],
        [],
        { absolute: 0, relative: 0 },
      ),
    ).toThrow(/duplicate edge/);
  });
});
