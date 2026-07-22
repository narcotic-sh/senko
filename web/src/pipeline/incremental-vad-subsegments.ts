import {
  DEFAULT_VAD_MERGE_GAP_SECONDS,
  DEFAULT_VAD_MIN_SPEECH_SECONDS,
} from "./vad";
import { DEFAULT_SUBSEGMENT_DURATION_SECONDS } from "./subsegments";
import type { Subsegment, TimeSegment } from "./types";

export interface IncrementalVadSubsegmentOptions {
  readonly mergeGapSeconds?: number;
  readonly minSpeechSeconds?: number;
  readonly durationSeconds?: number;
  readonly shiftSeconds?: number;
}

/** Newly irrevocable output produced while advancing the VAD watermark. */
export interface IncrementalVadSubsegmentEmission {
  readonly finalizedVadSegments: readonly TimeSegment[];
  readonly emittedSubsegments: readonly Subsegment[];
}

/** Complete output after the final zero-padded VAD chunk has been decoded. */
export interface IncrementalVadSubsegmentResult
  extends IncrementalVadSubsegmentEmission {
  readonly vadSegments: readonly TimeSegment[];
  readonly subsegments: readonly Subsegment[];
}

interface PendingVadSegment extends TimeSegment {
  nextWindowStart: number;
}

/**
 * Incrementally reproduces `mergeVadSegments` followed by `createSubsegments`.
 *
 * Each batch must contain the unfiltered, zero-gap decoded segments for a
 * chronological group of VAD chunks. `nextUnprocessedChunkTime` is the start
 * time of the first chunk whose logits are not in that batch. At most one
 * merge candidate remains unresolved across calls.
 *
 * Full embedding windows are emitted as soon as their strict right boundary
 * is known to lie inside the candidate. The adjusted final window is held
 * until the candidate is irrevocable, because its start depends on the final
 * VAD end time.
 */
export class IncrementalVadSubsegmentReducer {
  private readonly mergeGapSeconds: number;
  private readonly minSpeechSeconds: number;
  private readonly durationSeconds: number;
  private readonly shiftSeconds: number;
  private readonly allVadSegments: TimeSegment[] = [];
  private readonly allSubsegments: Subsegment[] = [];

  private pending: PendingVadSegment | undefined;
  private nextUnprocessedChunkTime = 0;
  private lastRawStart = Number.NEGATIVE_INFINITY;
  private finished = false;

  constructor(options: IncrementalVadSubsegmentOptions = {}) {
    this.mergeGapSeconds =
      options.mergeGapSeconds ?? DEFAULT_VAD_MERGE_GAP_SECONDS;
    this.minSpeechSeconds =
      options.minSpeechSeconds ?? DEFAULT_VAD_MIN_SPEECH_SECONDS;
    this.durationSeconds =
      options.durationSeconds ?? DEFAULT_SUBSEGMENT_DURATION_SECONDS;
    this.shiftSeconds =
      options.shiftSeconds ?? this.durationSeconds / 2.5;

    requireFiniteNonNegative(this.mergeGapSeconds, "VAD merge gap");
    requireFiniteNonNegative(this.minSpeechSeconds, "VAD minimum speech duration");
    requireFinitePositive(this.durationSeconds, "subsegment duration");
    requireFinitePositive(this.shiftSeconds, "subsegment shift");
  }

  consumeBatch(
    rawSegments: readonly TimeSegment[],
    nextUnprocessedChunkTime: number,
  ): IncrementalVadSubsegmentEmission {
    this.assertOpen();
    if (
      !Number.isFinite(nextUnprocessedChunkTime) ||
      nextUnprocessedChunkTime < this.nextUnprocessedChunkTime
    ) {
      throw new RangeError(
        "The next unprocessed VAD chunk time must be finite and monotonic",
      );
    }

    const finalizedVadSegments: TimeSegment[] = [];
    const emittedSubsegments: Subsegment[] = [];
    for (const segment of rawSegments) {
      this.validateRawSegment(segment, nextUnprocessedChunkTime);
      this.consumeRawSegment(
        segment,
        finalizedVadSegments,
        emittedSubsegments,
      );
      this.lastRawStart = segment.start;
    }

    this.nextUnprocessedChunkTime = nextUnprocessedChunkTime;
    if (
      this.pending !== undefined &&
      nextUnprocessedChunkTime - this.pending.end > this.mergeGapSeconds
    ) {
      this.finalizePending(finalizedVadSegments, emittedSubsegments);
    }
    return { finalizedVadSegments, emittedSubsegments };
  }

  finish(): IncrementalVadSubsegmentResult {
    this.assertOpen();
    const finalizedVadSegments: TimeSegment[] = [];
    const emittedSubsegments: Subsegment[] = [];
    if (this.pending !== undefined) {
      this.finalizePending(finalizedVadSegments, emittedSubsegments);
    }
    this.finished = true;
    return {
      finalizedVadSegments,
      emittedSubsegments,
      vadSegments: this.allVadSegments.slice(),
      subsegments: this.allSubsegments.slice(),
    };
  }

  private consumeRawSegment(
    segment: TimeSegment,
    finalizedVadSegments: TimeSegment[],
    emittedSubsegments: Subsegment[],
  ): void {
    const pending = this.pending;
    if (pending === undefined) {
      this.pending = {
        start: segment.start,
        end: segment.end,
        nextWindowStart: segment.start,
      };
    } else if (segment.start - pending.end <= this.mergeGapSeconds) {
      pending.end = Math.max(pending.end, segment.end);
    } else {
      this.finalizePending(finalizedVadSegments, emittedSubsegments);
      this.pending = {
        start: segment.start,
        end: segment.end,
        nextWindowStart: segment.start,
      };
    }
    this.emitStableWindows(emittedSubsegments);
  }

  private emitStableWindows(emittedSubsegments: Subsegment[]): void {
    const pending = this.pending;
    if (
      pending === undefined ||
      pending.end - pending.start < this.minSpeechSeconds
    ) {
      return;
    }

    while (pending.nextWindowStart + this.durationSeconds < pending.end) {
      const start = pending.nextWindowStart;
      this.emitSubsegment(
        start,
        start + this.durationSeconds,
        emittedSubsegments,
      );
      // Match native Senko and createSubsegments: repeated addition is part of
      // the long-file floating-point contract.
      pending.nextWindowStart += this.shiftSeconds;
    }
  }

  private finalizePending(
    finalizedVadSegments: TimeSegment[],
    emittedSubsegments: Subsegment[],
  ): void {
    const pending = this.pending;
    if (pending === undefined) return;

    const finalized = {
      start: Math.max(0, pending.start),
      end: pending.end,
    };
    if (finalized.end - finalized.start >= this.minSpeechSeconds) {
      this.emitStableWindows(emittedSubsegments);
      if (pending.nextWindowStart < finalized.end) {
        const tailStart = Math.min(
          finalized.end - this.durationSeconds,
          pending.nextWindowStart,
        );
        this.emitSubsegment(tailStart, finalized.end, emittedSubsegments);
      }
      this.allVadSegments.push(finalized);
      finalizedVadSegments.push(finalized);
    }
    this.pending = undefined;
  }

  private emitSubsegment(
    start: number,
    end: number,
    emittedSubsegments: Subsegment[],
  ): void {
    const subsegment = {
      index: this.allSubsegments.length,
      start,
      end,
    };
    this.allSubsegments.push(subsegment);
    emittedSubsegments.push(subsegment);
  }

  private validateRawSegment(
    segment: TimeSegment,
    nextUnprocessedChunkTime: number,
  ): void {
    if (
      !Number.isFinite(segment.start) ||
      !Number.isFinite(segment.end) ||
      segment.start < 0 ||
      !(segment.end > segment.start)
    ) {
      throw new RangeError("Decoded VAD segments must be finite, positive intervals");
    }
    if (segment.start < this.lastRawStart) {
      throw new RangeError("Decoded VAD segments must be chronological");
    }
    if (
      segment.start < this.nextUnprocessedChunkTime ||
      segment.start >= nextUnprocessedChunkTime ||
      segment.end > nextUnprocessedChunkTime
    ) {
      throw new RangeError(
        "Decoded VAD segments must lie between consecutive chunk watermarks",
      );
    }
  }

  private assertOpen(): void {
    if (this.finished) {
      throw new Error("The incremental VAD reducer has already finished");
    }
  }
}

function requireFiniteNonNegative(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${label} must be finite and non-negative`);
  }
}

function requireFinitePositive(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${label} must be finite and positive`);
  }
}
