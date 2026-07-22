import type { Subsegment, TimeSegment } from "./types";

export const DEFAULT_SUBSEGMENT_DURATION_SECONDS = 1.5;

export interface SubsegmentOptions {
  durationSeconds?: number;
  shiftSeconds?: number;
}

/** Mirrors Senko's default 1.5-second / 0.6-second embedding window policy. */
export function createSubsegments(
  vad: readonly TimeSegment[],
  options: SubsegmentOptions = {},
): Subsegment[] {
  const duration =
    options.durationSeconds ?? DEFAULT_SUBSEGMENT_DURATION_SECONDS;
  const shift = options.shiftSeconds ?? duration / 2.5;
  if (!(duration > 0) || !(shift > 0)) {
    throw new RangeError("Subsegment duration and shift must be positive");
  }

  const output: Subsegment[] = [];
  for (const segment of vad) {
    if (!(segment.end > segment.start)) continue;
    let start = segment.start;
    while (start + duration < segment.end) {
      output.push({ index: output.length, start, end: start + duration });
      start += shift;
    }
    if (start < segment.end) {
      start = Math.min(segment.end - duration, start);
      output.push({ index: output.length, start, end: segment.end });
    }
  }
  return output;
}
