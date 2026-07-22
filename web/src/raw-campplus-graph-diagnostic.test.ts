import { describe, expect, it } from "vitest";

import { DENSE_BOTTLENECK_VARIANTS } from "./pipeline/campplus-webgpu/dense-cam";
import { FCM_VARIANTS } from "./pipeline/campplus-webgpu/fcm";
import { CAMPPLUS_RAW_NUMERIC_VARIANTS } from "./pipeline/campplus-webgpu/graph";
import { PACKED_BCT_CONV_VARIANTS } from "./pipeline/campplus-webgpu/packed-bct-conv";
import { POINTWISE_TRANSIT_VARIANTS } from "./pipeline/campplus-webgpu/pointwise-transit";
import {
  deterministicFeatures,
  parseBatchSize,
  parseDenseBottleneckVariant,
  parseFcmVariant,
  parseNumericVariant,
  parsePointwiseTransitVariant,
  parseTdnnVariant,
  repeatReferenceRows,
} from "./raw-campplus-graph-diagnostic";

describe("raw CAM++ graph diagnostic FCM selection", () => {
  it("accepts diagnostic B64 while keeping B32 as the diagnostic default", () => {
    expect(parseBatchSize(null)).toBe(32);
    expect(parseBatchSize("64")).toBe(64);
    expect(() => parseBatchSize("128")).toThrow("4, 8, 16, 32, or 64");
  });

  it("repeats the checked B32 input and oracle rows for B64 parity", () => {
    const features = deterministicFeatures(64);
    const rowFeatureCount = 150 * 80;
    expect(
      features.slice(32 * rowFeatureCount, 33 * rowFeatureCount),
    ).toEqual(features.slice(0, rowFeatureCount));

    const source = new Float32Array(32 * 192);
    for (let row = 0; row < 32; row += 1) source[row * 192] = row + 0.5;
    const expanded = repeatReferenceRows(source, 64);
    expect(expanded).toHaveLength(64 * 192);
    expect(expanded[32 * 192]).toBe(0.5);
    expect(expanded[63 * 192]).toBe(31.5);
  });

  it("defaults to the measured production variant", () => {
    expect(parseFcmVariant(null)).toBe("tile4-fold");
  });

  it("defaults to FP16 production and retains one explicit FP32 baseline", () => {
    expect(parseNumericVariant(null)).toBe("production");
    expect(CAMPPLUS_RAW_NUMERIC_VARIANTS).toEqual([
      "production",
      "float32-baseline",
    ]);
    for (const variant of CAMPPLUS_RAW_NUMERIC_VARIANTS) {
      expect(parseNumericVariant(variant)).toBe(variant);
    }
    expect(() => parseNumericVariant("fp16")).toThrow(
      CAMPPLUS_RAW_NUMERIC_VARIANTS.join(", "),
    );
  });

  it.each(FCM_VARIANTS)("accepts %s", (variant) => {
    expect(parseFcmVariant(variant)).toBe(variant);
  });

  it("reports every stable name for an invalid query value", () => {
    expect(() => parseFcmVariant("tile8-fold")).toThrow(
      FCM_VARIANTS.join(", "),
    );
  });

  it("defaults to the production dense bottleneck", () => {
    expect(parseDenseBottleneckVariant(null)).toBe("direct-tile4-wg128");
  });

  it.each(DENSE_BOTTLENECK_VARIANTS)("accepts dense bottleneck %s", (variant) => {
    expect(parseDenseBottleneckVariant(variant)).toBe(variant);
  });

  it("reports every dense bottleneck name for an invalid query value", () => {
    expect(() => parseDenseBottleneckVariant("direct-tile8-wg128")).toThrow(
      DENSE_BOTTLENECK_VARIANTS.join(", "),
    );
  });

  it("defaults to the production TDNN variant", () => {
    expect(parseTdnnVariant(null)).toBe("direct-tile8-wg96");
  });

  it.each(PACKED_BCT_CONV_VARIANTS)("accepts TDNN variant %s", (variant) => {
    expect(parseTdnnVariant(variant)).toBe(variant);
  });

  it("reports every TDNN name for an invalid query value", () => {
    expect(() => parseTdnnVariant("direct-tile16-wg96")).toThrow(
      PACKED_BCT_CONV_VARIANTS.join(", "),
    );
  });

  it("defaults to the production pointwise transit", () => {
    expect(parsePointwiseTransitVariant(null)).toBe("chunk512");
  });

  it.each(POINTWISE_TRANSIT_VARIANTS)("accepts pointwise transit %s", (variant) => {
    expect(parsePointwiseTransitVariant(variant)).toBe(variant);
  });

  it("reports every pointwise transit name for an invalid query value", () => {
    expect(() => parsePointwiseTransitVariant("chunk256")).toThrow(
      POINTWISE_TRANSIT_VARIANTS.join(", "),
    );
  });
});
