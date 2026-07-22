import type { DiarizationSegment } from "../runtime/types";
import type { Subsegment } from "./types";

const MERGE_GAP_SECONDS = 4;
const MIN_MERGED_SEGMENT_SECONDS = 0.78;

interface MutableDiarizationSegment {
  startSeconds: number;
  endSeconds: number;
  speaker: string;
}

export interface ClusteringPostprocessResult {
  /** Labels normalized by sorted original label, before speaking-time renaming. */
  readonly normalizedLabels: Int32Array;
  /** Senko's raw segments, including mutations caused by its merge pass. */
  readonly rawSegments: readonly DiarizationSegment[];
  readonly mergedSegments: readonly DiarizationSegment[];
  readonly centroids: Readonly<Record<string, Float32Array>>;
  /** Unique speakers represented by raw segments and centroids. */
  readonly rawSpeakerCount: number;
  /** Unique speakers that survive Senko's <= 0.78-second merge filter. */
  readonly speakerCount: number;
}

function speakerId(index: number): string {
  return `SPEAKER_${String(index + 1).padStart(2, "0")}`;
}

function sortedUniqueLabels(labels: ArrayLike<number>): number[] {
  const unique = new Set<number>();
  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index];
    if (label === undefined || !Number.isSafeInteger(label)) {
      throw new TypeError(`Cluster label at index ${index} is not a safe integer`);
    }
    unique.add(label);
  }
  return [...unique].sort((left, right) => left - right);
}

/** Mirrors NumPy's sorted `unique`-based normalization used by Senko. */
export function normalizeClusterLabels(labels: ArrayLike<number>): Int32Array {
  const unique = sortedUniqueLabels(labels);
  const normalizedByLabel = new Map<number, number>();
  unique.forEach((label, index) => normalizedByLabel.set(label, index));

  const normalized = new Int32Array(labels.length);
  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index];
    if (label === undefined) {
      throw new TypeError(`Missing cluster label at index ${index}`);
    }
    const normalizedLabel = normalizedByLabel.get(label);
    if (normalizedLabel === undefined) {
      throw new Error(`Failed to normalize cluster label at index ${index}`);
    }
    normalized[index] = normalizedLabel;
  }
  return normalized;
}

function mergeMutableSegments(
  segments: readonly MutableDiarizationSegment[],
): MutableDiarizationSegment[] {
  const merged: MutableDiarizationSegment[] = [];
  let current: MutableDiarizationSegment | undefined;

  // Step 1: merge same-speaker segments separated by at most four seconds.
  for (const segment of segments) {
    if (current === undefined) {
      current = segment;
    } else if (
      current.speaker === segment.speaker &&
      segment.startSeconds - current.endSeconds <= MERGE_GAP_SECONDS
    ) {
      // Assignment, rather than max(), deliberately matches the native code.
      current.endSeconds = segment.endSeconds;
    } else {
      merged.push(current);
      current = segment;
    }
  }
  if (current !== undefined) merged.push(current);

  // Step 2: discard segments of 0.78 seconds or less. A short segment between
  // matching speakers causes those surrounding segments to be joined first.
  let index = 0;
  while (index < merged.length) {
    const segment = merged[index];
    if (segment === undefined) break;

    if (segment.endSeconds - segment.startSeconds <= MIN_MERGED_SEGMENT_SECONDS) {
      if (index > 0 && index < merged.length - 1) {
        const previous = merged[index - 1];
        const next = merged[index + 1];
        if (previous !== undefined && next !== undefined && previous.speaker === next.speaker) {
          previous.endSeconds = next.endSeconds;
          merged.splice(index + 1, 1);
        }
      }
      merged.splice(index, 1);
    } else {
      index += 1;
    }
  }

  return merged;
}

/**
 * Pure public form of Senko's `_merge_segments`: inputs are copied, while all
 * ordering, mutation, gap, and duration semantics of the native pass remain.
 */
export function mergeDiarizationSegments(
  segments: readonly DiarizationSegment[],
): DiarizationSegment[] {
  const copies: MutableDiarizationSegment[] = segments.map((segment) => ({
    startSeconds: segment.startSeconds,
    endSeconds: segment.endSeconds,
    speaker: segment.speaker,
  }));
  return mergeMutableSegments(copies);
}

function calculateCentroids(
  embeddings: Float32Array,
  embeddingDimension: number,
  normalizedLabels: Int32Array,
  speakerCount: number,
): Record<string, Float32Array> {
  const centroids = Array.from(
    { length: speakerCount },
    () => new Float32Array(embeddingDimension),
  );
  const counts = new Uint32Array(speakerCount);

  for (let row = 0; row < normalizedLabels.length; row += 1) {
    const label = normalizedLabels[row];
    if (label === undefined) throw new Error(`Missing normalized label at index ${row}`);
    const centroid = centroids[label];
    if (centroid === undefined) throw new Error(`Normalized label ${label} is out of range`);
    counts[label] = (counts[label] ?? 0) + 1;

    const offset = row * embeddingDimension;
    for (let column = 0; column < embeddingDimension; column += 1) {
      const value = embeddings[offset + column];
      if (value === undefined) throw new Error("Embedding tensor ended unexpectedly");
      centroid[column] = (centroid[column] ?? 0) + value;
    }
  }

  const result: Record<string, Float32Array> = {};
  centroids.forEach((centroid, label) => {
    const count = counts[label];
    if (count === undefined || count === 0) {
      throw new Error(`Cluster ${label} contains no embeddings`);
    }
    for (let column = 0; column < centroid.length; column += 1) {
      centroid[column] = (centroid[column] ?? 0) / count;
    }
    result[speakerId(label)] = centroid;
  });
  return result;
}

function createRawSegments(
  subsegments: readonly Subsegment[],
  normalizedLabels: Int32Array,
): MutableDiarizationSegment[] {
  const raw: MutableDiarizationSegment[] = [];

  for (let index = 0; index < subsegments.length; index += 1) {
    const subsegment = subsegments[index];
    const normalizedLabel = normalizedLabels[index];
    if (subsegment === undefined || normalizedLabel === undefined) {
      throw new Error(`Missing subsegment data at index ${index}`);
    }

    let startSeconds = Math.max(0, subsegment.start);
    const endSeconds = Math.max(0, subsegment.end);
    const speaker = speakerId(normalizedLabel);
    const previous = raw[raw.length - 1];

    if (previous === undefined) {
      raw.push({ startSeconds, endSeconds, speaker });
    } else if (speaker === previous.speaker) {
      if (startSeconds > previous.endSeconds) {
        raw.push({ startSeconds, endSeconds, speaker });
      } else {
        previous.endSeconds = endSeconds;
      }
    } else {
      if (startSeconds < previous.endSeconds) {
        const midpoint = Math.max(0, (previous.endSeconds + startSeconds) / 2);
        previous.endSeconds = midpoint;
        startSeconds = midpoint;
      }
      raw.push({ startSeconds, endSeconds, speaker });
    }
  }

  return raw;
}

/**
 * Converts clustering output to Senko-compatible diarization segments.
 * Embeddings are a row-major `[subsegment, embeddingDimension]` tensor; the
 * dimension is inferred from `embeddings.length / labels.length`.
 */
export function postprocessClustering(
  embeddings: Float32Array,
  labels: ArrayLike<number>,
  subsegments: readonly Subsegment[],
): ClusteringPostprocessResult {
  if (labels.length !== subsegments.length) {
    throw new RangeError(
      `Expected one cluster label per subsegment; got ${labels.length} labels and ${subsegments.length} subsegments`,
    );
  }
  if (labels.length === 0) {
    if (embeddings.length !== 0) {
      throw new RangeError("Embeddings must be empty when there are no cluster labels");
    }
    return {
      normalizedLabels: new Int32Array(),
      rawSegments: [],
      mergedSegments: [],
      centroids: {},
      rawSpeakerCount: 0,
      speakerCount: 0,
    };
  }
  if (embeddings.length === 0 || embeddings.length % labels.length !== 0) {
    throw new RangeError(
      `Embedding tensor length ${embeddings.length} is not divisible by row count ${labels.length}`,
    );
  }

  const normalizedLabels = normalizeClusterLabels(labels);
  const uniqueLabels = sortedUniqueLabels(labels);
  const embeddingDimension = embeddings.length / labels.length;
  const centroids = calculateCentroids(
    embeddings,
    embeddingDimension,
    normalizedLabels,
    uniqueLabels.length,
  );
  const rawSegments = createRawSegments(subsegments, normalizedLabels);

  // Native Senko passes its raw objects directly into `_merge_segments`.
  // Preserve that aliasing here because those mutations affect speaking-time
  // ranking and therefore the final speaker IDs.
  const mergedSegments = mergeMutableSegments(rawSegments);

  const speakingTimeBySpeaker = new Map<string, number>();
  for (const segment of rawSegments) {
    const duration = segment.endSeconds - segment.startSeconds;
    speakingTimeBySpeaker.set(
      segment.speaker,
      (speakingTimeBySpeaker.get(segment.speaker) ?? 0) + duration,
    );
  }

  // ECMAScript sorting is stable, matching Python's tie behavior: speakers
  // with equal duration remain in their first-occurrence order.
  const rankedSpeakers = [...speakingTimeBySpeaker.entries()].sort(
    (left, right) => right[1] - left[1],
  );
  const speakerMapping = new Map<string, string>();
  rankedSpeakers.forEach(([oldSpeaker], index) => {
    speakerMapping.set(oldSpeaker, speakerId(index));
  });

  const updated = new Set<MutableDiarizationSegment>();
  for (const segment of [...rawSegments, ...mergedSegments]) {
    if (updated.has(segment)) continue;
    const renamed = speakerMapping.get(segment.speaker);
    if (renamed === undefined) throw new Error(`No rank exists for ${segment.speaker}`);
    segment.speaker = renamed;
    updated.add(segment);
  }

  const rankedCentroids: Record<string, Float32Array> = {};
  for (const [oldSpeaker, newSpeaker] of speakerMapping) {
    const centroid = centroids[oldSpeaker];
    if (centroid !== undefined) rankedCentroids[newSpeaker] = centroid;
  }

  return {
    normalizedLabels,
    rawSegments,
    mergedSegments,
    centroids: rankedCentroids,
    rawSpeakerCount: new Set(rawSegments.map((segment) => segment.speaker)).size,
    // Native `merged_speakers_detected` is counted from the segments left after
    // `_merge_segments`, not from labels/centroids. A speaker whose only turns
    // are filtered therefore remains in raw output and centroids but is absent
    // from the final count.
    speakerCount: new Set(mergedSegments.map((segment) => segment.speaker)).size,
  };
}
