import { describe, expect, it } from "vitest";

import type { CampPlusArenaSlice } from "./arena";
import {
  DEFAULT_FCM_VARIANT,
  DEFAULT_FCM_ACCUMULATION,
  FCM_CONV_WGSL,
  FCM_DISPATCH_GPU_BUFFER_BYTES,
  FCM_FIRST_WGSL,
  FCM_VARIANTS,
  LEGACY_FCM_VARIANT,
  fcmConvWgsl,
  fcmDispatchWorkgroups,
  fcmFirstWgsl,
  fcmVariantConfiguration,
  isFcmVariant,
  validateFcmDimensions,
  type FcmConvDescriptor,
  type FcmResidual,
} from "./fcm";

const CONVOLUTION = { weight: "weight", bias: "bias" } as const;
const INPUT: CampPlusArenaSlice = {
  label: "input",
  byteOffset: 0,
  byteLength: 32 * 80 * 150 * 2,
};
const OUTPUT: CampPlusArenaSlice = {
  label: "output",
  byteOffset: 32 * 80 * 150 * 2,
  byteLength: 32 * 80 * 150 * 2,
};

describe("FCM WebGPU variants", () => {
  it("selects the measured tile4 kernel as the production default", () => {
    expect(DEFAULT_FCM_VARIANT).toBe("tile4-fold");
    expect(DEFAULT_FCM_ACCUMULATION).toBe("float16");
    expect(fcmDispatchWorkgroups(DEFAULT_FCM_VARIANT, 16, 80)).toEqual([
      2, 1280, 1,
    ]);
    expect(fcmFirstWgsl(DEFAULT_FCM_VARIANT)).toContain(
      "let first_output_group = group.x * 4u",
    );
    expect(fcmFirstWgsl(DEFAULT_FCM_VARIANT)).toContain(
      "var accumulator_3 = biases[",
    );
    expect(fcmConvWgsl(DEFAULT_FCM_VARIANT)).toContain(
      "array<vec4<f16>, 1280>",
    );
  });

  it("retains the original shader and split-tail geometry as a diagnostic baseline", () => {
    expect(LEGACY_FCM_VARIANT).toBe("tile1-split");
    expect(fcmFirstWgsl(LEGACY_FCM_VARIANT, "float32")).toBe(FCM_FIRST_WGSL);
    expect(fcmConvWgsl(LEGACY_FCM_VARIANT, "float32")).toBe(FCM_CONV_WGSL);
    expect(fcmDispatchWorkgroups(LEGACY_FCM_VARIANT, 16, 80)).toEqual([
      8, 1280, 2,
    ]);
  });

  it("pins the four stable diagnostic names and their B16 geometry", () => {
    expect(FCM_VARIANTS).toEqual([
      "tile1-split",
      "tile1-fold",
      "tile2-fold",
      "tile4-fold",
    ]);
    for (const variant of FCM_VARIANTS) expect(isFcmVariant(variant)).toBe(true);
    expect(isFcmVariant("tile8-fold")).toBe(false);

    expect(fcmDispatchWorkgroups("tile1-fold", 16, 80)).toEqual([
      8, 1280, 1,
    ]);
    expect(fcmDispatchWorkgroups("tile2-fold", 16, 40)).toEqual([
      4, 640, 1,
    ]);
    expect(fcmDispatchWorkgroups("tile4-fold", 16, 10)).toEqual([
      2, 160, 1,
    ]);

    const outputFrequencies = [80, 40, 40, 40, 40, 20, 20, 20, 20, 10];
    const totals = Object.fromEntries(
      FCM_VARIANTS.map((variant) => [
        variant,
        outputFrequencies.reduce((sum, frequency) => {
          const [x, y, z] = fcmDispatchWorkgroups(variant, 16, frequency);
          return sum + x * y * z;
        }, 0),
      ]),
    );
    expect(totals).toEqual({
      "tile1-split": 84_480,
      "tile1-fold": 42_240,
      "tile2-fold": 21_120,
      "tile4-fold": 10_560,
    });
  });

  it("does not change explicit GPU-buffer accounting", () => {
    expect(FCM_DISPATCH_GPU_BUFFER_BYTES).toBe(64);
    expect(10 * FCM_DISPATCH_GPU_BUFFER_BYTES).toBe(640);
    expect(
      FCM_VARIANTS.map((variant) => ({
        variant,
        gpuBufferBytes: 10 * FCM_DISPATCH_GPU_BUFFER_BYTES,
      })),
    ).toEqual(
      FCM_VARIANTS.map((variant) => ({ variant, gpuBufferBytes: 640 })),
    );
  });

  it("keeps tile4 workgroup storage below the existing 16 KiB requirement", () => {
    expect(fcmVariantConfiguration("tile1-split")).toEqual({
      outputTile: 1,
      foldTimeTail: false,
      firstWorkgroupStorageBytes: 72,
      convWorkgroupStorageBytes: 2560,
    });
    expect(fcmVariantConfiguration("tile4-fold")).toEqual({
      outputTile: 4,
      foldTimeTail: true,
      firstWorkgroupStorageBytes: 288,
      convWorkgroupStorageBytes: 10_240,
    });
  });

  it("materializes explicit tile accumulators and one folded time loop", () => {
    const first = fcmFirstWgsl("tile4-fold", "float32");
    const conv = fcmConvWgsl("tile4-fold", "float32");
    expect(first).toContain("array<vec4<f16>, 36>");
    expect(first).toContain("let first_output_group = group.x * 4u");
    expect(first).toContain("var accumulator_3 = vec4<f32>");
    expect(first).toContain("var output_time = local_id.x");
    expect(first).toContain("output_time += 128u");
    expect(first).not.toContain("group.z * 128u");

    expect(conv).toContain("array<vec4<f16>, 1280>");
    expect(conv).toContain("var main_3 = vec4<f32>");
    expect(conv).toContain("var shortcut_3 = vec4<f32>");
    expect(conv).toContain("let input_0 = f32(arena[input_index])");
    expect(conv).toContain(
      "main_3 = fma(vec4<f32>(input_0), vec4<f32>(weight_cache[960u + weight_index]), main_3)",
    );
    expect(conv).toContain(
      "shortcut_3 = fma(vec4<f32>(input_0), vec4<f32>(weight_cache[1248u + input_channel]), shortcut_3)",
    );
    expect(conv).not.toContain("group.z * 128u");
  });

  it("retains an FP32 baseline for the production FP16 geometry", () => {
    const baselineFirst = fcmFirstWgsl("tile4-fold", "float32");
    const baselineConv = fcmConvWgsl("tile4-fold", "float32");
    const fullFirst = fcmFirstWgsl("tile4-fold", "float16");
    const fullConv = fcmConvWgsl("tile4-fold", "float16");

    expect(baselineFirst).toContain("var accumulator_3 = vec4<f32>");
    expect(baselineConv).toContain("var main_3 = vec4<f32>");
    expect(baselineConv).toContain("var shortcut_3 = vec4<f32>");

    expect(fullFirst).toContain("var accumulator_3 = biases[");
    expect(fullFirst).toContain("vec4<f16>(f16(value))");
    expect(fullConv).toContain("var main_3 = biases[");
    expect(fullConv).toContain("var shortcut_3 = shortcut_biases[");
    expect(fullConv).not.toContain("main_3_partial");

    for (const code of [baselineFirst, baselineConv, fullFirst, fullConv]) {
      expect(code).toContain("let first_output_group = group.x * 4u");
      expect(code).toContain("@workgroup_size(128)");
    }
  });

  it("preserves main stride, identity residual, and learned shortcut paths", () => {
    for (const variant of ["tile1-fold", "tile2-fold", "tile4-fold"] as const) {
      const code = fcmConvWgsl(variant, "float32");
      expect(code).toContain(
        "output_freq * parameters.stride_freq + kernel_freq",
      );
      expect(code).toContain("parameters.residual_mode == 1u");
      expect(code).toContain("parameters.residual_mode == 2u");
      expect(code).toContain(
        "output_freq * parameters.residual_stride_freq",
      );
      expect(code).toContain("let main_rounded_0 = vec4<f16>(main_0)");
      expect(code).toContain(
        "result_0 = vec4<f16>(main_rounded_0 + vec4<f16>(shortcut_0))",
      );
    }
  });
});

describe("FCM dimension validation", () => {
  it.each([
    ["none stride 1", 40, 40, 1, { kind: "none" }],
    ["none stride 2", 40, 20, 2, { kind: "none" }],
    ["identity stride 1", 40, 40, 1, { kind: "identity", input: INPUT }],
    [
      "learned shortcut stride 2",
      20,
      20,
      1,
      {
        kind: "learned",
        input: INPUT,
        inputFreq: 40,
        strideFreq: 2,
        convolution: CONVOLUTION,
      },
    ],
  ] as const)("accepts %s", (_label, inputFreq, outputFreq, strideFreq, residual) => {
    expect(() =>
      validateFcmDimensions(
        descriptor(inputFreq, outputFreq, strideFreq, residual),
      ),
    ).not.toThrow();
  });

  it("rejects an identity residual across a strided main convolution", () => {
    expect(() =>
      validateFcmDimensions(
        descriptor(40, 20, 2, { kind: "identity", input: INPUT }),
      ),
    ).toThrow("identity residual shape mismatch");
  });

  it("rejects a learned shortcut whose stride does not produce the output frequency", () => {
    expect(() =>
      validateFcmDimensions(
        descriptor(20, 20, 1, {
          kind: "learned",
          input: INPUT,
          inputFreq: 40,
          strideFreq: 1,
          convolution: CONVOLUTION,
        }),
      ),
    ).toThrow("learned residual shape mismatch");
  });
});

function descriptor(
  inputFreq: number,
  outputFreq: number,
  strideFreq: number,
  residual: FcmResidual,
): FcmConvDescriptor {
  return {
    label: "test-fcm",
    convolution: CONVOLUTION,
    input: INPUT,
    inputFreq,
    output: OUTPUT,
    outputFreq,
    strideFreq,
    batchSize: 1,
    residual,
    outputRelu: true,
  };
}
