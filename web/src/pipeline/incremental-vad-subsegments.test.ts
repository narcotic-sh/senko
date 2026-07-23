import { describe, expect, it } from "vitest";

import { createSubsegments } from "./subsegments";
import type { Subsegment, TimeSegment, VadChunk } from "./types";
import {
  createVadChunks,
  decodeVadLogits,
  mergeVadSegments,
  VAD_CHUNK_SAMPLES,
  VAD_FRAME_STEP_SECONDS,
  VAD_OUTPUT_CLASSES,
  VAD_OUTPUT_FRAMES,
} from "./vad";
import {
  IncrementalVadSubsegmentReducer,
  type IncrementalVadSubsegmentOptions,
} from "./incremental-vad-subsegments";

interface OracleResult {
  readonly vadSegments: readonly TimeSegment[];
  readonly subsegments: readonly Subsegment[];
}

describe("IncrementalVadSubsegmentReducer", () => {
  it("matches the batch oracle for overlaps and the exact merge/minimum boundaries", () => {
    const exactMergeGap = [
      { start: 0, end: 0.1 },
      // JavaScript evaluates 0.2 - 0.1 to the same double as 0.1.
      { start: 0.2, end: 0.5 },
    ];
    expect(exactMergeGap[1]!.start - exactMergeGap[0]!.end).toBe(0.1);

    const raw = [
      ...exactMergeGap,
      // Strictly beyond the merge threshold, so this starts a new candidate.
      { start: 0.600_000_000_001, end: 0.9 },
      // Contained overlap must not shorten the current candidate.
      { start: 0.7, end: 0.8 },
      // Exactly 250 ms is retained.
      { start: 2, end: 2.25 },
      // Just under 250 ms is discarded.
      { start: 3, end: 3.249_999_999 },
    ] satisfies TimeSegment[];

    const reducer = new IncrementalVadSubsegmentReducer();
    const emission = reducer.consumeBatch(raw, 4);
    const result = reducer.finish();
    const expected = oracle(raw);

    expect(emission.finalizedVadSegments).toEqual(expected.vadSegments);
    expect(result.finalizedVadSegments).toEqual([]);
    expectExactResult(result, expected);
  });

  it("holds the B8 tail at 79.9 seconds and finalizes only past the strict watermark", () => {
    const held = new IncrementalVadSubsegmentReducer();
    const heldEmission = held.consumeBatch(
      [{ start: 79.5, end: 79.9 }],
      80,
    );
    expect(heldEmission.finalizedVadSegments).toEqual([]);
    expect(held.finish().vadSegments).toEqual([{ start: 79.5, end: 79.9 }]);

    const safe = new IncrementalVadSubsegmentReducer();
    const safeEmission = safe.consumeBatch(
      [{ start: 79.5, end: 79.899_999_999 }],
      80,
    );
    expect(safeEmission.finalizedVadSegments).toEqual([
      { start: 79.5, end: 79.899_999_999 },
    ]);
    expect(safe.finish().finalizedVadSegments).toEqual([]);
  });

  it.each([
    { previousEndFrame: 587, lastMergingNextFrame: 0 },
    { previousEndFrame: 588, lastMergingNextFrame: 1 },
    { previousEndFrame: 589, lastMergingNextFrame: 2 },
  ])(
    "bridges frame $previousEndFrame only through next frame $lastMergingNextFrame",
    ({ previousEndFrame, lastMergingNextFrame }) => {
      const merging = runB8BoundaryCase(
        previousEndFrame,
        lastMergingNextFrame,
      );
      expect(merging.firstEmission.finalizedVadSegments).toEqual([]);
      expect(merging.result.vadSegments).toHaveLength(1);
      expectExactResult(merging.result, merging.expected);

      const split = runB8BoundaryCase(
        previousEndFrame,
        lastMergingNextFrame + 1,
      );
      expect(split.firstEmission.finalizedVadSegments).toEqual([]);
      expect(split.result.vadSegments).toHaveLength(2);
      expectExactResult(split.result, split.expected);
    },
  );

  it("emits immutable full windows during continuous speech and delays the tail until EOF", () => {
    const chunks = createVadChunks(17 * VAD_CHUNK_SAMPLES);
    const reducer = new IncrementalVadSubsegmentReducer();
    const allRaw: TimeSegment[] = [];
    const emitted: Subsegment[] = [];
    const finalized: TimeSegment[] = [];

    for (let start = 0; start < chunks.length; start += 8) {
      const batch = chunks.slice(start, start + 8);
      const raw = decodeAllSpeech(batch);
      allRaw.push(...raw);
      const nextChunkTime = batch[0]!.timeOffset + batch.length * 10;
      const emission = reducer.consumeBatch(raw, nextChunkTime);
      emitted.push(...emission.emittedSubsegments);
      finalized.push(...emission.finalizedVadSegments);
      if (start === 0) {
        expect(emission.finalizedVadSegments).toEqual([]);
        expect(emission.emittedSubsegments.length).toBeGreaterThan(0);
      }
    }

    const expected = oracle(allRaw);
    expect(finalized).toEqual([]);
    expect(emitted).toEqual(
      expected.subsegments.slice(0, emitted.length),
    );
    expect(emitted.at(-1)!.end).toBeLessThan(expected.vadSegments[0]!.end);

    const result = reducer.finish();
    emitted.push(...result.emittedSubsegments);
    finalized.push(...result.finalizedVadSegments);
    expect(emitted).toEqual(result.subsegments);
    expect(finalized).toEqual(result.vadSegments);
    expect(result.emittedSubsegments).toEqual([
      expected.subsegments.at(-1),
    ]);
    expectExactResult(result, expected);
  });

  it("filters short speech only after possible gap merging", () => {
    const raw = [
      { start: 0, end: 0.15 },
      { start: 0.2, end: 0.35 },
      { start: 1, end: 1.2 },
    ] satisfies TimeSegment[];
    const reducer = new IncrementalVadSubsegmentReducer();
    const emission = reducer.consumeBatch(raw, 2);
    const result = reducer.finish();

    expect(emission.finalizedVadSegments).toEqual([
      { start: 0, end: 0.35 },
    ]);
    expect(result.vadSegments).toEqual([{ start: 0, end: 0.35 }]);
    expect(result.subsegments).toEqual([
      { index: 0, start: -1.15, end: 0.35 },
    ]);
    expectExactResult(result, oracle(raw));
  });

  it("does not leak early windows when custom windows are shorter than min speech", () => {
    const options = {
      durationSeconds: 0.1,
      shiftSeconds: 0.04,
    } satisfies IncrementalVadSubsegmentOptions;
    const reducer = new IncrementalVadSubsegmentReducer(options);
    const first = reducer.consumeBatch([{ start: 0, end: 0.2 }], 0.25);
    expect(first.emittedSubsegments).toEqual([]);
    const second = reducer.consumeBatch([], 0.31);
    expect(second.finalizedVadSegments).toEqual([]);
    expect(second.emittedSubsegments).toEqual([]);
    expectExactResult(reducer.finish(), oracle([{ start: 0, end: 0.2 }], options));
  });

  it("preserves repeated-addition floating-point drift over an hour", () => {
    const raw = Array.from({ length: 370 }, (_, index) => ({
      start: index * 10,
      end: index * 10 + VAD_OUTPUT_FRAMES * VAD_FRAME_STEP_SECONDS,
    }));
    const reducer = new IncrementalVadSubsegmentReducer();
    const emitted: Subsegment[] = [];
    const finalized: TimeSegment[] = [];

    for (let start = 0; start < raw.length; start += 8) {
      const batch = raw.slice(start, start + 8);
      const emission = reducer.consumeBatch(
        batch,
        Math.min(start + batch.length, raw.length) * 10,
      );
      emitted.push(...emission.emittedSubsegments);
      finalized.push(...emission.finalizedVadSegments);
    }
    const result = reducer.finish();
    emitted.push(...result.emittedSubsegments);
    finalized.push(...result.finalizedVadSegments);

    const expected = oracle(raw);
    expect(emitted).toEqual(result.subsegments);
    expect(finalized).toEqual(result.vadSegments);
    expect(result.subsegments.length).toBeGreaterThan(6_000);
    expectExactResult(result, expected);
  });

  it("produces 47,999 windows for eight continuous hours without audio", () => {
    const durationSeconds = 8 * 60 * 60;
    const reducer = new IncrementalVadSubsegmentReducer();

    const stable = reducer.consumeBatch(
      [{ start: 0, end: durationSeconds }],
      durationSeconds,
    );
    const result = reducer.finish();

    // The strict native loop emits 47,998 ordinary 1.5-second windows through
    // repeated 0.6-second addition, then EOF contributes one adjusted tail.
    expect(stable.emittedSubsegments).toHaveLength(47_998);
    expect(result.emittedSubsegments).toEqual([
      { index: 47_998, start: durationSeconds - 1.5, end: durationSeconds },
    ]);
    expect(result.subsegments).toHaveLength(47_999);
    expect(result.subsegments[0]).toEqual({ index: 0, start: 0, end: 1.5 });
    expect(result.subsegments.at(-1)).toEqual({
      index: 47_998,
      start: durationSeconds - 1.5,
      end: durationSeconds,
    });
    expect(result.vadSegments).toEqual([{ start: 0, end: durationSeconds }]);
  });

  it("preserves the untrimmed zero-padded final VAD chunk and ignores padded B8 rows", () => {
    const chunks = createVadChunks(1);
    const logits = new Float32Array(
      8 * VAD_OUTPUT_FRAMES * VAD_OUTPUT_CLASSES,
    );
    setActiveRange(logits, 0, 0, VAD_OUTPUT_FRAMES);
    // These rows are backend padding and must never reach the reducer.
    for (let row = 1; row < 8; row += 1) {
      setActiveRange(logits, row, 0, VAD_OUTPUT_FRAMES);
    }
    const actualValueCount =
      chunks.length * VAD_OUTPUT_FRAMES * VAD_OUTPUT_CLASSES;
    const raw = decodeRaw(
      logits.subarray(0, actualValueCount),
      chunks,
    );
    const reducer = new IncrementalVadSubsegmentReducer();
    const emission = reducer.consumeBatch(raw, 10);
    expect(emission.finalizedVadSegments).toEqual([]);
    const result = reducer.finish();
    const expectedEnd = VAD_OUTPUT_FRAMES * VAD_FRAME_STEP_SECONDS;

    expect(result.vadSegments).toEqual([{ start: 0, end: expectedEnd }]);
    expect(result.vadSegments[0]!.end).toBeGreaterThan(1 / 16_000);
    expectExactResult(result, oracle(raw));
  });

  it("rejects non-chronological batches, invalid watermarks, and use after EOF", () => {
    const reducer = new IncrementalVadSubsegmentReducer();
    expect(() => reducer.consumeBatch([{ start: 1, end: 2 }], 1.5)).toThrow(
      /watermarks/,
    );
    expect(() =>
      reducer.consumeBatch(
        [
          { start: 0.5, end: 0.75 },
          { start: 0.25, end: 0.4 },
        ],
        1,
      ),
    ).toThrow(/chronological/);

    const completed = new IncrementalVadSubsegmentReducer();
    completed.finish();
    expect(() => completed.consumeBatch([], 10)).toThrow(/already finished/);
    expect(() => completed.finish()).toThrow(/already finished/);
  });
});

function runB8BoundaryCase(
  previousEndFrame: number,
  nextStartFrame: number,
) {
  const chunks = createVadChunks(10 * VAD_CHUNK_SAMPLES);
  const firstChunks = chunks.slice(0, 8);
  const firstLogits = new Float32Array(
    firstChunks.length * VAD_OUTPUT_FRAMES * VAD_OUTPUT_CLASSES,
  );
  setActiveRange(firstLogits, 7, 560, previousEndFrame);
  const firstRaw = decodeRaw(firstLogits, firstChunks);

  const secondChunks = chunks.slice(8, 9);
  const secondLogits = new Float32Array(
    VAD_OUTPUT_FRAMES * VAD_OUTPUT_CLASSES,
  );
  setActiveRange(secondLogits, 0, nextStartFrame, 40);
  const secondRaw = decodeRaw(secondLogits, secondChunks);

  const reducer = new IncrementalVadSubsegmentReducer();
  const firstEmission = reducer.consumeBatch(firstRaw, 80);
  reducer.consumeBatch(secondRaw, 90);
  const result = reducer.finish();
  const expected = oracle([...firstRaw, ...secondRaw]);
  return { firstEmission, result, expected };
}

function decodeAllSpeech(chunks: readonly VadChunk[]): TimeSegment[] {
  const logits = new Float32Array(
    chunks.length * VAD_OUTPUT_FRAMES * VAD_OUTPUT_CLASSES,
  );
  for (let row = 0; row < chunks.length; row += 1) {
    setActiveRange(logits, row, 0, VAD_OUTPUT_FRAMES);
  }
  return decodeRaw(logits, chunks);
}

function decodeRaw(
  logits: Float32Array,
  chunks: readonly VadChunk[],
): TimeSegment[] {
  return decodeVadLogits(logits, chunks, {
    minSpeechSeconds: 0,
    mergeGapSeconds: 0,
  });
}

function setActiveRange(
  logits: Float32Array,
  row: number,
  startFrame: number,
  endFrame: number,
): void {
  for (let frame = startFrame; frame < endFrame; frame += 1) {
    const base =
      (row * VAD_OUTPUT_FRAMES + frame) * VAD_OUTPUT_CLASSES;
    logits[base + 1] = 1;
  }
}

function oracle(
  rawSegments: readonly TimeSegment[],
  options: IncrementalVadSubsegmentOptions = {},
): OracleResult {
  const vadSegments = mergeVadSegments(rawSegments, {
    ...(options.mergeGapSeconds === undefined
      ? {}
      : { mergeGapSeconds: options.mergeGapSeconds }),
    ...(options.minSpeechSeconds === undefined
      ? {}
      : { minSpeechSeconds: options.minSpeechSeconds }),
  });
  return {
    vadSegments,
    subsegments: createSubsegments(vadSegments, {
      ...(options.durationSeconds === undefined
        ? {}
        : { durationSeconds: options.durationSeconds }),
      ...(options.shiftSeconds === undefined
        ? {}
        : { shiftSeconds: options.shiftSeconds }),
    }),
  };
}

function expectExactResult(
  actual: OracleResult,
  expected: OracleResult,
): void {
  expect(actual.vadSegments).toEqual(expected.vadSegments);
  expect(actual.subsegments).toEqual(expected.subsegments);
  expect(segmentBytes(actual.vadSegments)).toEqual(
    segmentBytes(expected.vadSegments),
  );
  expect(segmentBytes(actual.subsegments)).toEqual(
    segmentBytes(expected.subsegments),
  );
  expect(actual.subsegments.map(({ index }) => index)).toEqual(
    expected.subsegments.map(({ index }) => index),
  );
}

function segmentBytes(
  segments: readonly TimeSegment[],
): Uint8Array<ArrayBuffer> {
  const values = new Float64Array(segments.length * 2);
  for (let index = 0; index < segments.length; index += 1) {
    values[index * 2] = segments[index]!.start;
    values[index * 2 + 1] = segments[index]!.end;
  }
  return new Uint8Array(values.buffer);
}
