export interface TimeSegment {
  start: number;
  end: number;
}

export interface VadChunk {
  sampleOffset: number;
  sampleCount: number;
  timeOffset: number;
}

export interface Subsegment extends TimeSegment {
  index: number;
}

export interface StageMeasurement {
  stage: string;
  elapsedMs: number;
}

export interface PipelineProgress {
  stage: string;
  completed: number;
  total: number;
}

export type ProgressListener = (progress: PipelineProgress) => void;

/** A random-access mono PCM source. Implementations may stream from a Blob. */
export interface MonoPcmSource {
  readonly sampleRate: number;
  readonly sampleCount: number;
  readInto(
    sampleOffset: number,
    sampleCount: number,
    destination: Float32Array,
    destinationOffset?: number,
  ): Promise<void> | void;
}

export interface VadBatchBackend {
  readonly batchSize: number;
  readonly chunkSamples: number;
  readonly outputFrames: number;
  readonly outputClasses: number;
  run(audio: Float32Array): Promise<Float32Array>;
}

export interface EmbeddingBatchBackend {
  readonly batchSize: number;
  readonly frames: number;
  readonly featureDim: number;
  readonly embeddingDim: number;
  /** Maximum safe concurrent submissions. Omitted backends are treated as serial. */
  readonly maxInFlightRuns?: number;
  run(features: Float32Array): Promise<Float32Array>;
}
