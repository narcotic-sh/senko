import { describe, expect, it } from "vitest";

import { DENSE_BOTTLENECK_VARIANTS } from "./pipeline/campplus-webgpu/dense-cam";
import { FCM_VARIANTS } from "./pipeline/campplus-webgpu/fcm";
import { PACKED_BCT_CONV_VARIANTS } from "./pipeline/campplus-webgpu/packed-bct-conv";
import { POINTWISE_TRANSIT_VARIANTS } from "./pipeline/campplus-webgpu/pointwise-transit";
import {
  parseDenseBottleneckVariant,
  parseFcmVariant,
  parsePointwiseTransitVariant,
  parseTdnnVariant,
} from "./raw-campplus-graph-diagnostic";

describe("raw CAM++ graph diagnostic FCM selection", () => {
  it("defaults to the measured production variant", () => {
    expect(parseFcmVariant(null)).toBe("tile4-fold");
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
