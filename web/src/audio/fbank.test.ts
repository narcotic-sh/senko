import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  frameCountForSamples,
  SENKO_FBANK_BINS,
  secondsToFbankWindow,
  SenkoFbank,
  StreamingFbankExtractor,
  type FbankWindowResult,
  type FbankComputeHint,
  type FbankMatrix,
} from "./fbank";
import { Pcm16WavReader } from "./wav";
import { CountingMemorySource, makePcm16Wav } from "./test-helpers";

describe("SenkoFbank", () => {
  it("matches the complete native C++ fixture", async () => {
    const pcm = deterministicPcm(24_000);
    const reader = await Pcm16WavReader.open(
      new CountingMemorySource(makePcm16Wav(pcm)),
    );
    const samples = await reader.readSamples(0, pcm.length);
    const actual = new SenkoFbank().compute(samples);
    const expected = await readFloat32Fixture(
      new URL("./__fixtures__/native-fbank-1p5s.f32", import.meta.url),
    );

    expect(actual.frameCount).toBe(148);
    expect(actual.binCount).toBe(SENKO_FBANK_BINS);
    expect(actual.data.length).toBe(expected.length);

    let maxAbsoluteError = 0;
    let squaredError = 0;
    for (let i = 0; i < expected.length; i += 1) {
      const error = actual.data[i]! - expected[i]!;
      maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(error));
      squaredError += error * error;
    }
    const rootMeanSquaredError = Math.sqrt(squaredError / expected.length);
    expect(maxAbsoluteError).toBeLessThan(5e-4);
    expect(rootMeanSquaredError).toBeLessThan(5e-5);
  });

  it("applies mean normalization independently to each window", () => {
    const pcm = deterministicPcm(24_000);
    const samples = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i += 1) {
      samples[i] = pcm[i]! / 32_768;
    }
    const matrix = new SenkoFbank().compute(samples);

    for (let bin = 0; bin < matrix.binCount; bin += 1) {
      let mean = 0;
      for (let frame = 0; frame < matrix.frameCount; frame += 1) {
        mean += matrix.data[frame * matrix.binCount + bin]!;
      }
      // Native float32 accumulation leaves a few parts per million of residue.
      expect(Math.abs(mean / matrix.frameCount)).toBeLessThan(3e-6);
    }
  });

  it("uses Kaldi snip-edges frame counts and pads short windows", () => {
    expect(frameCountForSamples(0)).toBe(1);
    expect(frameCountForSamples(399)).toBe(1);
    expect(frameCountForSamples(400)).toBe(1);
    expect(frameCountForSamples(559)).toBe(1);
    expect(frameCountForSamples(560)).toBe(2);
    expect(frameCountForSamples(24_000)).toBe(148);

    const short = new SenkoFbank().compute(new Float32Array([0.25]));
    expect(short.frameCount).toBe(1);
    expect(short.data.every(Number.isFinite)).toBe(true);
  });
});

describe("StreamingFbankExtractor", () => {
  it("reuses decoded PCM overlap and yields one window at a time", async () => {
    const pcm = deterministicPcm(43_200);
    const source = new CountingMemorySource(makePcm16Wav(pcm));
    const reader = await Pcm16WavReader.open(source);
    const bytesBeforeWindows = source.bytesRead;
    const extractor = new StreamingFbankExtractor(reader);
    const requests = [
      { startSample: 0, sampleCount: 24_000, id: 0 },
      { startSample: 9_600, sampleCount: 24_000, id: 1 },
      { startSample: 19_200, sampleCount: 24_000, id: 2 },
    ];
    const results: FbankWindowResult[] = [];
    for await (const result of extractor.extract(requests)) results.push(result);

    expect(results.map((result) => result.features.frameCount)).toEqual([
      148, 148, 148,
    ]);
    expect(extractor.stats).toMatchObject({
      requestedSamples: 72_000,
      decodedSamples: 43_200,
      reusedSamples: 28_800,
      peakCachedSamples: 24_000,
      windows: 3,
    });
    expect(source.bytesRead - bytesBeforeWindows).toBe(43_200 * 2);
  });

  it("converts seconds with native Senko's truncation semantics", () => {
    expect(secondsToFbankWindow(1.25, 2.75, "window")).toEqual({
      startSample: 20_000,
      sampleCount: 24_000,
      id: "window",
    });
    expect(secondsToFbankWindow(0.5062500000000001, 2.00625)).toEqual({
      startSample: 8_100,
      sampleCount: 23_999,
    });
    expect(secondsToFbankWindow(-1, 0.5)).toEqual({
      startSample: 0,
      sampleCount: 24_000,
    });
  });

  it("signals frame-aligned overlap to reusable backends", async () => {
    const pcm = deterministicPcm(48_000);
    const reader = await Pcm16WavReader.open(
      new CountingMemorySource(makePcm16Wav(pcm)),
    );
    const hints: Array<FbankComputeHint | undefined> = [];
    const backend = {
      compute(
        _samples: Float32Array,
        hint?: FbankComputeHint,
      ): FbankMatrix {
        hints.push(hint);
        return {
          data: new Float32Array(148 * SENKO_FBANK_BINS),
          frameCount: 148,
          binCount: SENKO_FBANK_BINS,
        };
      },
    };
    const extractor = new StreamingFbankExtractor(reader, backend);
    for await (const _result of extractor.extract([
      { startSample: 0, sampleCount: 24_000 },
      { startSample: 9_600, sampleCount: 24_000 },
      { startSample: 19_201, sampleCount: 24_000 },
    ])) {
      // Consume the stream.
    }
    expect(hints).toEqual([undefined, { reusableFrameShift: 60 }, undefined]);
  });
});

function deterministicPcm(length: number): Int16Array {
  const samples = new Int16Array(length);
  let state = 0x12345678;
  for (let i = 0; i < length; i += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    samples[i] = (((state >>> 16) - 32_768) >> 2) as number;
  }
  return samples;
}

async function readFloat32Fixture(url: URL): Promise<Float32Array> {
  const bytes = await readFile(url);
  expect(bytes.byteLength % 4).toBe(0);
  const values = new Float32Array(bytes.byteLength / 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < values.length; i += 1) {
    values[i] = view.getFloat32(i * 4, true);
  }
  return values;
}
