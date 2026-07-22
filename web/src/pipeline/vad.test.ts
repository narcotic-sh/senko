import { describe, expect, it } from "vitest";

import {
  createVadChunks,
  decodeVadLogits,
  mergeVadSegments,
  runVad,
  VAD_CHUNK_SAMPLES,
  VAD_OUTPUT_CLASSES,
  VAD_OUTPUT_FRAMES,
} from "./vad";

function logitsForClasses(classes: readonly number[]): Float32Array {
  const logits = new Float32Array(classes.length * VAD_OUTPUT_CLASSES);
  logits.fill(-10);
  classes.forEach((classIndex, frame) => {
    logits[frame * VAD_OUTPUT_CLASSES + classIndex] = 10;
  });
  return logits;
}

describe("VAD chunking", () => {
  it("creates non-overlapping ten-second chunks", () => {
    expect(createVadChunks(VAD_CHUNK_SAMPLES * 2 + 17)).toEqual([
      { sampleOffset: 0, sampleCount: VAD_CHUNK_SAMPLES, timeOffset: 0 },
      {
        sampleOffset: VAD_CHUNK_SAMPLES,
        sampleCount: VAD_CHUNK_SAMPLES,
        timeOffset: 10,
      },
      { sampleOffset: VAD_CHUNK_SAMPLES * 2, sampleCount: 17, timeOffset: 20 },
    ]);
  });
});

describe("VAD orchestration", () => {
  it("coalesces each contiguous model batch into one PCM range", async () => {
    const sampleCount = VAD_CHUNK_SAMPLES * 2 + 17;
    const reads: Array<{ offset: number; count: number }> = [];
    const inputs: Float32Array[] = [];
    const source = {
      sampleRate: 16_000,
      sampleCount,
      readInto(
        offset: number,
        count: number,
        destination: Float32Array,
        destinationOffset = 0,
      ) {
        reads.push({ offset, count });
        destination.fill(0.25, destinationOffset, destinationOffset + count);
      },
    };
    const backend = {
      batchSize: 2,
      chunkSamples: VAD_CHUNK_SAMPLES,
      outputFrames: VAD_OUTPUT_FRAMES,
      outputClasses: VAD_OUTPUT_CLASSES,
      async run(input: Float32Array) {
        inputs.push(input.slice());
        return new Float32Array(2 * VAD_OUTPUT_FRAMES * VAD_OUTPUT_CLASSES);
      },
    };

    expect(await runVad(source, backend)).toEqual([]);
    expect(reads).toEqual([
      { offset: 0, count: VAD_CHUNK_SAMPLES * 2 },
      { offset: VAD_CHUNK_SAMPLES * 2, count: 17 },
    ]);
    expect(inputs).toHaveLength(2);
    expect(inputs[1]!.subarray(0, 17).every((value) => value === 0.25)).toBe(true);
    expect(inputs[1]!.subarray(17).every((value) => value === 0)).toBe(true);
  });

  it("keeps native timestamps from the zero-padded final model chunk", async () => {
    const source = {
      sampleRate: 16_000,
      sampleCount: 1,
      readInto(
        _offset: number,
        count: number,
        destination: Float32Array,
        destinationOffset = 0,
      ) {
        destination.fill(0, destinationOffset, destinationOffset + count);
      },
    };
    const backend = {
      batchSize: 1,
      chunkSamples: VAD_CHUNK_SAMPLES,
      outputFrames: VAD_OUTPUT_FRAMES,
      outputClasses: VAD_OUTPUT_CLASSES,
      async run() {
        return logitsForClasses(new Array<number>(VAD_OUTPUT_FRAMES).fill(1));
      },
    };

    // Native Senko does not clip the final segment to the one-sample file. It
    // reports all active frames emitted for the zero-padded 10-second chunk.
    expect(await runVad(source, backend)).toEqual([
      { start: 0, end: VAD_OUTPUT_FRAMES * 0.016875 },
    ]);
  });
});

describe("VAD decoding", () => {
  it("treats every non-empty powerset class as speech", () => {
    const classes = new Array<number>(VAD_OUTPUT_FRAMES).fill(0);
    classes.fill(4, 10, 20);
    const result = decodeVadLogits(logitsForClasses(classes), createVadChunks(160_000), {
      minSpeechSeconds: 0,
    });
    expect(result).toEqual([{ start: 0.16875, end: 0.3375 }]);
  });

  it("merges short gaps and removes short speech islands", () => {
    expect(
      mergeVadSegments([
        { start: 0, end: 0.2 },
        { start: 0.25, end: 0.5 },
        { start: 1, end: 1.1 },
      ]),
    ).toEqual([{ start: 0, end: 0.5 }]);
  });

  it("rejects truncated tensors", () => {
    expect(() =>
      decodeVadLogits(
        new Float32Array(VAD_OUTPUT_FRAMES * VAD_OUTPUT_CLASSES - 1),
        createVadChunks(160_000),
      ),
    ).toThrow(/expected at least/);
  });
});
