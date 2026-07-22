import { describe, expect, it } from "vitest";

import { FCM_VARIANTS } from "./pipeline/campplus-webgpu/fcm";
import { parseFcmVariant } from "./raw-campplus-graph-diagnostic";

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
});
