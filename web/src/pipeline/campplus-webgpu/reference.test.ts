import { describe, expect, it } from "vitest";

import {
  evaluatePackedBctConvReference,
  float16BitsToFloat32,
  float32ToFloat16Bits,
  packAffineReference,
  packBiasReference,
  packOikWeightReference,
  roundToFloat16,
  type PackedBctReferenceParameters,
} from "./reference";

describe("packed BCT convolution reference", () => {
  it("matches an independent logical OIK convolution with affine and FP16 boundaries", () => {
    const parameters: PackedBctReferenceParameters = {
      batchSize: 2,
      inputChannels: 5,
      outputChannels: 6,
      inputFrames: 9,
      outputFrames: 5,
      kernelElements: 3,
      stride: 2,
      dilation: 1,
      padLeft: 1,
      outputRelu: true,
    };
    const inputValues = deterministicValues(
      parameters.batchSize * parameters.inputChannels * parameters.inputFrames,
      0.7,
    );
    const logicalWeight = deterministicValues(
      parameters.outputChannels * parameters.inputChannels * parameters.kernelElements,
      0.17,
    );
    const bias = deterministicValues(parameters.outputChannels, 0.08);
    const scale = new Float32Array([0.8, -1.2, 0.5, 1.6, -0.3]);
    const shift = new Float32Array([0.1, 0.2, -0.05, 0.4, 0.3]);
    const input = Uint16Array.from(inputValues, float32ToFloat16Bits);
    const packedWeight = packOikWeightReference(
      logicalWeight,
      parameters.outputChannels,
      parameters.inputChannels,
      parameters.kernelElements,
    );
    const packedBias = packBiasReference(bias);
    const packedAffine = packAffineReference(scale, shift);
    const actual = evaluatePackedBctConvReference(
      input,
      packedWeight,
      packedBias,
      parameters,
      packedAffine,
    );
    const expected = directLogicalConvolution(
      input,
      logicalWeight,
      bias,
      parameters,
      scale,
      shift,
    );
    expect(Array.from(actual)).toEqual(Array.from(expected));
  });

  it("rounds representative normal, subnormal, and special values", () => {
    for (const value of [0, -0, 1, -2.5, 65_504, 2 ** -14, 2 ** -24, Infinity, -Infinity]) {
      const roundTrip = float16BitsToFloat32(float32ToFloat16Bits(value));
      expect(Object.is(roundTrip, value) || roundTrip === value).toBe(true);
    }
    expect(Number.isNaN(float16BitsToFloat32(float32ToFloat16Bits(Number.NaN)))).toBe(true);
    expect(roundToFloat16(1.000_6)).toBe(1.000_976_562_5);
  });
});

function directLogicalConvolution(
  input: Uint16Array,
  logicalWeight: Float32Array,
  logicalBias: Float32Array,
  parameters: PackedBctReferenceParameters,
  scale: Float32Array,
  shift: Float32Array,
): Uint16Array {
  const output = new Uint16Array(
    parameters.batchSize * parameters.outputChannels * parameters.outputFrames,
  );
  for (let batch = 0; batch < parameters.batchSize; batch += 1) {
    for (let outputChannel = 0; outputChannel < parameters.outputChannels; outputChannel += 1) {
      for (let frame = 0; frame < parameters.outputFrames; frame += 1) {
        let sum = roundToFloat16(logicalBias[outputChannel]!);
        for (let kernel = 0; kernel < parameters.kernelElements; kernel += 1) {
          const sourceFrame = frame * parameters.stride + kernel - parameters.padLeft;
          if (sourceFrame < 0 || sourceFrame >= parameters.inputFrames) continue;
          for (let inputChannel = 0; inputChannel < parameters.inputChannels; inputChannel += 1) {
            const inputIndex =
              (batch * parameters.inputChannels + inputChannel) * parameters.inputFrames +
              sourceFrame;
            const normalized = Math.max(
              roundToFloat16(
                float16BitsToFloat32(input[inputIndex]!) * scale[inputChannel]! +
                  shift[inputChannel]!,
              ),
              0,
            );
            const weightIndex =
              (outputChannel * parameters.inputChannels + inputChannel) *
                parameters.kernelElements +
              kernel;
            sum += normalized * roundToFloat16(logicalWeight[weightIndex]!);
          }
        }
        if (parameters.outputRelu) sum = Math.max(roundToFloat16(sum), 0);
        const outputIndex =
          (batch * parameters.outputChannels + outputChannel) * parameters.outputFrames + frame;
        output[outputIndex] = float32ToFloat16Bits(sum);
      }
    }
  }
  return output;
}

function deterministicValues(length: number, scale: number): Float32Array {
  const result = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    result[index] = (Math.sin(index * 1.37 + 0.41) + Math.cos(index * 0.23)) * scale;
  }
  return result;
}
