import { describe, expect, it } from "vitest";

import {
  classify,
  summarize,
} from "./dual-device-concurrency-diagnostic";

describe("dual-device concurrency diagnostic statistics", () => {
  it("summarizes an even timing sample with interpolated quantiles", () => {
    const result = summarize([4, 1, 3, 2]);

    expect(result).toMatchObject({
      count: 4,
      minMs: 1,
      medianMs: 2.5,
      meanMs: 2.5,
      p90Ms: 3.7,
      maxMs: 4,
    });
    expect(result.coefficientOfVariation).toBeCloseTo(Math.sqrt(1.25) / 2.5);
  });

  it("keeps the value-of-information verdict thresholds explicit", () => {
    expect(classify(1.4, 0.3, false)).toBe("parity-failed");
    expect(classify(1.25, 0.2, true)).toBe("strong");
    expect(classify(1.1, 0.09, true)).toBe("material");
    expect(classify(1.09, 0.2, true)).toBe("immaterial");
  });
});
