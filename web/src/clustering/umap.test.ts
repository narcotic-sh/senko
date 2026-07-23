import { describe, expect, it } from "vitest";

import { radixSortFuzzyEdgePairs } from "./umap";

describe("UMAP fuzzy-graph edge ordering", () => {
  it("matches packed-key ordering exactly at ordinary row counts", () => {
    const count = 257;
    const tails = new Uint32Array([9, 0, 256, 9, 4, 9, 0, 256, 4]);
    const heads = new Uint32Array([3, 8, 0, 2, 12, 2, 7, 1, 12]);
    const values = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
    const order = [...tails.keys()];
    order.sort((left, right) => {
      const leftKey = tails[left]! * count + heads[left]!;
      const rightKey = tails[right]! * count + heads[right]!;
      return leftKey - rightKey || left - right;
    });
    const expectedTails = order.map((index) => tails[index]);
    const expectedHeads = order.map((index) => heads[index]);
    const expectedValues = order.map((index) => values[index]);

    const sorted = radixSortFuzzyEdgePairs(
      heads.slice(),
      tails.slice(),
      values.slice(),
      tails.length,
    );

    expect([...sorted.tails]).toEqual(expectedTails);
    expect([...sorted.heads]).toEqual(expectedHeads);
    expect([...sorted.values]).toEqual(expectedValues);
  });

  it("keeps distinct endpoints beyond the packed Uint32 boundary", () => {
    const safeCount = 65_536;
    const wideCount = safeCount + 1;
    expect(((safeCount - 1) * safeCount + safeCount - 1) >>> 0).toBe(
      0xffff_ffff,
    );

    // These two valid edges collide when `tail * count + head` is narrowed to
    // Uint32 at 65,537 rows. Endpoint-pair sorting keeps them distinct.
    const lowTailKey = (0 * wideCount + 65_536) >>> 0;
    const highTailKey = (65_536 * wideCount + 0) >>> 0;
    expect(highTailKey).toBe(lowTailKey);

    const sorted = radixSortFuzzyEdgePairs(
      new Uint32Array([0, 65_536, 1, 65_535]),
      new Uint32Array([65_536, 0, 65_536, 0]),
      new Float32Array([0.2, 0.1, 0.4, 0.3]),
      4,
    );
    expect([...sorted.tails]).toEqual([0, 0, 65_536, 65_536]);
    expect([...sorted.heads]).toEqual([65_535, 65_536, 0, 1]);
    expect([...sorted.values]).toEqual([
      new Float32Array([0.3])[0],
      new Float32Array([0.1])[0],
      new Float32Array([0.2])[0],
      new Float32Array([0.4])[0],
    ]);
  });
});
