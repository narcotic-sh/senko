import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import { SENKO_FBANK_BINS } from "./fbank";
import { WasmSenkoFbank } from "./wasm-fbank";

describe("WasmSenkoFbank", () => {
  it("matches the complete native C++ fixture", async () => {
    const fbank = await createFbank();
    try {
      const samples = pcmToFloat(deterministicPcm(24_000));
      const actual = fbank.compute(samples);
      const expected = await readFloat32(
        new URL("./__fixtures__/native-fbank-1p5s.f32", import.meta.url),
      );

      expect(actual.frameCount).toBe(148);
      expect(actual.binCount).toBe(SENKO_FBANK_BINS);
      const errors = errorStats(actual.data, expected);
      expect(errors.maxAbsoluteError).toBeLessThan(5e-4);
      expect(errors.rootMeanSquaredError).toBeLessThan(5e-5);
    } finally {
      fbank.dispose();
    }
  });

  it("reuses overlapping raw frames without changing normalized output", async () => {
    const pcm = pcmToFloat(deterministicPcm(33_600));
    const reused = await createFbank();
    const independent = await createFbank();
    try {
      reused.compute(pcm.subarray(0, 24_000));
      const reusedOutput = reused.compute(pcm.subarray(9_600, 33_600), {
        reusableFrameShift: 60,
      });
      const expected = independent.compute(pcm.subarray(9_600, 33_600));
      expect(errorStats(reusedOutput.data, expected.data).maxAbsoluteError).toBe(
        0,
      );
    } finally {
      reused.dispose();
      independent.dispose();
    }
  });

  it("bounds float32 timestamp drift within its fixed half-megabyte heap", async () => {
    const fbank = await createFbank();
    expect(fbank.memoryStats).toEqual({
      heapBytes: 512 * 1024,
      inputCapacitySamples: 24_399,
      outputCapacityFrames: 150,
    });
    expect(fbank.compute(new Float32Array(24_001)).frameCount).toBe(148);
    expect(fbank.compute(new Float32Array(24_399)).frameCount).toBe(150);
    expect(() => fbank.compute(new Float32Array(24_400))).toThrow(
      /fixed WASM capacity is 24399/,
    );
    expect(fbank.memoryStats.heapBytes).toBe(512 * 1024);
    fbank.dispose();
    expect(() => fbank.compute(new Float32Array(400))).toThrow(/disposed/);
  });
});

async function createFbank(): Promise<WasmSenkoFbank> {
  const bytes = await readFile(
    new URL("./wasm/senko-fbank.wasm", import.meta.url),
  );
  return WasmSenkoFbank.fromBytes(Uint8Array.from(bytes).buffer);
}

function deterministicPcm(length: number): Int16Array {
  const samples = new Int16Array(length);
  let state = 0x12345678;
  for (let i = 0; i < length; i += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    samples[i] = (((state >>> 16) - 32_768) >> 2) as number;
  }
  return samples;
}

function pcmToFloat(pcm: Int16Array): Float32Array {
  const samples = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i += 1) {
    samples[i] = pcm[i]! / 32_768;
  }
  return samples;
}

async function readFloat32(url: URL): Promise<Float32Array> {
  const bytes = await readFile(url);
  const result = new Float32Array(bytes.byteLength / 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < result.length; i += 1) {
    result[i] = view.getFloat32(i * 4, true);
  }
  return result;
}

function errorStats(
  actual: Float32Array,
  expected: Float32Array,
): { maxAbsoluteError: number; rootMeanSquaredError: number } {
  expect(actual.length).toBe(expected.length);
  let maxAbsoluteError = 0;
  let squaredError = 0;
  for (let i = 0; i < expected.length; i += 1) {
    const error = actual[i]! - expected[i]!;
    maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(error));
    squaredError += error * error;
  }
  return {
    maxAbsoluteError,
    rootMeanSquaredError: Math.sqrt(squaredError / expected.length),
  };
}
