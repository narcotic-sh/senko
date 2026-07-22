import { describe, expect, it } from "vitest";

import {
  buildSortedMutualReachabilityEdges,
  clusterSparseGraph,
} from "./hierarchy";
import type { KnnGraph } from "./knn";

describe("sparse hierarchy", () => {
  it("radix-sorts mutual-reachability edges in exact comparator order", () => {
    const count = 37;
    const neighborCount = 7;
    const indices = new Int32Array(count * neighborCount);
    const similarities = new Float32Array(indices.length);
    const coreDistances = new Float64Array(count);

    for (let row = 0; row < count; row += 1) {
      coreDistances[row] = ((row * 7) % 9) / 20;
      const offset = row * neighborCount;
      for (let rank = 0; rank < neighborCount; rank += 1) {
        const edge = offset + rank;
        indices[edge] =
          rank === 0
            ? row
            : rank === 1 && row % 5 === 0
              ? -1
              : (row * 11 + rank * rank + (rank % 2) * 3) % count;
        // Repeated values exercise endpoint tie-breaking and stable duplicate
        // ordering; negative affinities exercise weights greater than one.
        similarities[edge] = ((row * 3 + rank * 5) % 13) / 8 - 0.25;
      }
    }
    const graph: KnnGraph = { indices, similarities, neighborCount };

    const expectedWeights = new Float64Array(indices.length);
    const expectedOrder: number[] = [];
    for (let from = 0; from < count; from += 1) {
      const offset = from * neighborCount;
      for (let rank = 0; rank < neighborCount; rank += 1) {
        const edge = offset + rank;
        const to = indices[edge]!;
        if (to < 0 || to === from) continue;
        const distance = Math.max(0, 1 - similarities[edge]!);
        expectedWeights[edge] = Math.max(
          coreDistances[from]!,
          coreDistances[to]!,
          distance,
        );
        expectedOrder.push(edge);
      }
    }
    expectedOrder.sort((left, right) => {
      const weightDifference =
        expectedWeights[left]! - expectedWeights[right]!;
      if (weightDifference !== 0) return weightDifference;
      const fromDifference =
        Math.floor(left / neighborCount) - Math.floor(right / neighborCount);
      return fromDifference !== 0
        ? fromDifference
        : indices[left]! - indices[right]!;
    });

    const actual = buildSortedMutualReachabilityEdges(
      graph,
      count,
      coreDistances,
    );
    expect([...actual.order.subarray(0, actual.edgeCount)]).toEqual(
      expectedOrder,
    );
    for (const edge of expectedOrder) {
      expect(Object.is(actual.weights[edge], expectedWeights[edge])).toBe(true);
    }
  });

  it("keeps disconnected components as separate density clusters", () => {
    const graph: KnnGraph = {
      indices: new Int32Array([
        1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4,
      ]),
      similarities: new Float32Array([
        0.95, 0.9, 0.95, 0.9, 0.9, 0.9, 0.95, 0.9, 0.95, 0.9, 0.9, 0.9,
      ]),
      neighborCount: 2,
    };

    expect([...clusterSparseGraph(graph, 6, 2, 2)]).toEqual([
      0, 0, 0, 1, 1, 1,
    ]);
  });
});
