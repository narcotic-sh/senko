import type { MonoPcmSource, TimeSegment, VadBatchBackend, VadChunk } from "./types";

export const VAD_SAMPLE_RATE = 16_000;
export const VAD_CHUNK_SAMPLES = 160_000;
export const VAD_OUTPUT_FRAMES = 589;
export const VAD_OUTPUT_CLASSES = 7;
export const VAD_FRAME_STEP_SECONDS = 0.016875;
export const DEFAULT_VAD_MIN_SPEECH_SECONDS = 0.25;
export const DEFAULT_VAD_MERGE_GAP_SECONDS = 0.1;

export interface VadDecodeOptions {
  frameStepSeconds?: number;
  minSpeechSeconds?: number;
  mergeGapSeconds?: number;
}

export interface VadBatchResult {
  /** Unfiltered, zero-gap regions decoded from only this model batch. */
  readonly rawSegments: readonly TimeSegment[];
  /**
   * Nominal start of the next 10-second chunk. For the final batch this is the
   * synthetic boundary after the last real chunk, which lets an incremental
   * reducer apply the ordinary merge watermark before its explicit EOF flush.
   */
  readonly nextUnprocessedChunkTime: number;
  readonly completedChunks: number;
  readonly totalChunks: number;
}

export function createVadChunks(sampleCount: number): VadChunk[] {
  if (!Number.isSafeInteger(sampleCount) || sampleCount < 0) {
    throw new RangeError(`Invalid sample count: ${sampleCount}`);
  }

  const chunks: VadChunk[] = [];
  for (let sampleOffset = 0; sampleOffset < sampleCount; sampleOffset += VAD_CHUNK_SAMPLES) {
    const chunkSamples = Math.min(VAD_CHUNK_SAMPLES, sampleCount - sampleOffset);
    chunks.push({
      sampleOffset,
      sampleCount: chunkSamples,
      timeOffset: sampleOffset / VAD_SAMPLE_RATE,
    });
  }
  return chunks;
}

/** Decode raw seven-class pyannote powerset logits using Senko's macOS policy. */
export function decodeVadLogits(
  logits: Float32Array,
  chunks: readonly VadChunk[],
  options: VadDecodeOptions = {},
): TimeSegment[] {
  const expected = chunks.length * VAD_OUTPUT_FRAMES * VAD_OUTPUT_CLASSES;
  if (logits.length < expected) {
    throw new RangeError(`VAD output has ${logits.length} values; expected at least ${expected}`);
  }

  const frameStep = options.frameStepSeconds ?? VAD_FRAME_STEP_SECONDS;
  const raw: TimeSegment[] = [];

  for (let batchIndex = 0; batchIndex < chunks.length; batchIndex += 1) {
    const chunk = chunks[batchIndex]!;
    let inSpeech = false;
    let speechStart = 0;

    for (let frameIndex = 0; frameIndex < VAD_OUTPUT_FRAMES; frameIndex += 1) {
      const base =
        (batchIndex * VAD_OUTPUT_FRAMES + frameIndex) * VAD_OUTPUT_CLASSES;
      let bestClass = 0;
      let bestLogit = logits[base]!;
      for (let classIndex = 1; classIndex < VAD_OUTPUT_CLASSES; classIndex += 1) {
        const value = logits[base + classIndex]!;
        if (value > bestLogit) {
          bestLogit = value;
          bestClass = classIndex;
        }
      }

      // Class zero is the empty powerset. Every other class contains at least
      // one local speaker, so no softmax or explicit powerset materialization is
      // needed for VAD.
      const active = bestClass !== 0;
      const time = chunk.timeOffset + frameIndex * frameStep;
      if (active && !inSpeech) {
        inSpeech = true;
        speechStart = time;
      } else if (!active && inSpeech) {
        raw.push({ start: speechStart, end: time });
        inSpeech = false;
      }
    }

    if (inSpeech) {
      raw.push({
        start: speechStart,
        end: chunk.timeOffset + VAD_OUTPUT_FRAMES * frameStep,
      });
    }
  }

  return mergeVadSegments(raw, options);
}

export function mergeVadSegments(
  segments: readonly TimeSegment[],
  options: VadDecodeOptions = {},
): TimeSegment[] {
  if (segments.length === 0) return [];

  const mergeGap =
    options.mergeGapSeconds ?? DEFAULT_VAD_MERGE_GAP_SECONDS;
  const minSpeech =
    options.minSpeechSeconds ?? DEFAULT_VAD_MIN_SPEECH_SECONDS;
  const sorted = [...segments].sort((left, right) => left.start - right.start);
  const merged: TimeSegment[] = [];
  let current = { ...sorted[0]! };

  for (let index = 1; index < sorted.length; index += 1) {
    const next = sorted[index]!;
    if (next.start - current.end <= mergeGap) {
      current.end = Math.max(current.end, next.end);
    } else {
      merged.push(current);
      current = { ...next };
    }
  }
  merged.push(current);

  return merged
    .map((segment) => ({
      start: Math.max(0, segment.start),
      end: segment.end,
    }))
    .filter((segment) => segment.end - segment.start >= minSpeech);
}

export async function runVad(
  source: MonoPcmSource | Float32Array,
  backend: VadBatchBackend,
  onProgress?: (completed: number, total: number) => void,
): Promise<TimeSegment[]> {
  const rawSegments: TimeSegment[] = [];
  for await (const batch of runVadBatches(source, backend)) {
    rawSegments.push(...batch.rawSegments);
    onProgress?.(batch.completedChunks, batch.totalChunks);
  }
  return mergeVadSegments(rawSegments);
}

/**
 * Run fixed-size VAD batches while retaining only one reusable input tensor.
 * Consumers can feed each chronological result into an incremental merge and
 * subsegment reducer without duplicating runVad's audio and padding policy.
 */
export async function* runVadBatches(
  source: MonoPcmSource | Float32Array,
  backend: VadBatchBackend,
): AsyncGenerator<VadBatchResult, void, void> {
  if (backend.chunkSamples !== VAD_CHUNK_SAMPLES) {
    throw new Error(`Unsupported VAD chunk size: ${backend.chunkSamples}`);
  }
  if (
    backend.outputFrames !== VAD_OUTPUT_FRAMES ||
    backend.outputClasses !== VAD_OUTPUT_CLASSES
  ) {
    throw new Error(
      `Unsupported VAD output shape: ${backend.outputFrames}x${backend.outputClasses}`,
    );
  }

  const pcm: MonoPcmSource =
    source instanceof Float32Array
      ? {
          sampleRate: VAD_SAMPLE_RATE,
          sampleCount: source.length,
          readInto(sampleOffset, sampleCount, destination, destinationOffset = 0) {
            destination.set(
              source.subarray(sampleOffset, sampleOffset + sampleCount),
              destinationOffset,
            );
          },
        }
      : source;
  if (pcm.sampleRate !== VAD_SAMPLE_RATE) {
    throw new Error(`Expected ${VAD_SAMPLE_RATE} Hz audio, received ${pcm.sampleRate} Hz`);
  }

  const chunks = createVadChunks(pcm.sampleCount);
  // Reuse one fixed batch for the entire recording. Allocating 5.12 MiB for
  // every B8 call made hour-long files accumulate tens of MiB of dead V8
  // backing stores between garbage collections even though only one batch is
  // logically live.
  const input = new Float32Array(backend.batchSize * VAD_CHUNK_SAMPLES);

  for (let start = 0; start < chunks.length; start += backend.batchSize) {
    const actualChunks = chunks.slice(start, start + backend.batchSize);
    input.fill(0);
    if (actualChunks.length > 0) {
      const first = actualChunks[0]!;
      const sampleCount = actualChunks.reduce(
        (sum, chunk) => sum + chunk.sampleCount,
        0,
      );
      await pcm.readInto(
        first.sampleOffset,
        sampleCount,
        input,
        0,
      );
    }

    const logits = await backend.run(input);
    const actualValueCount =
      actualChunks.length * VAD_OUTPUT_FRAMES * VAD_OUTPUT_CLASSES;
    const last = actualChunks.at(-1)!;
    yield {
      rawSegments: decodeVadLogits(
        logits.subarray(0, actualValueCount),
        actualChunks,
        {
          minSpeechSeconds: 0,
          mergeGapSeconds: 0,
        },
      ),
      nextUnprocessedChunkTime:
        (last.sampleOffset + VAD_CHUNK_SAMPLES) / VAD_SAMPLE_RATE,
      completedChunks: Math.min(
        start + actualChunks.length,
        chunks.length,
      ),
      totalChunks: chunks.length,
    };
  }
}
