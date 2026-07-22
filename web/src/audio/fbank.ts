import { Pcm16WavReader } from "./wav";

export const SENKO_FBANK_SAMPLE_RATE = 16_000;
export const SENKO_FBANK_FRAME_LENGTH = 400;
export const SENKO_FBANK_FRAME_SHIFT = 160;
export const SENKO_FBANK_FFT_SIZE = 512;
export const SENKO_FBANK_BINS = 80;

const FFT_BINS = SENKO_FBANK_FFT_SIZE / 2;
const PREEMPHASIS = Math.fround(0.97);
const LOW_FREQUENCY = 20;
const HIGH_FREQUENCY = 8_000;
const FLOAT32_EPSILON = 1.1920928955078125e-7;

export interface FbankMatrix {
  /** Row-major `[frameCount, binCount]` data. */
  readonly data: Float32Array;
  readonly frameCount: number;
  readonly binCount: 80;
}

export interface FbankWindowRequest {
  readonly startSample: number;
  readonly sampleCount: number;
  readonly id?: string | number;
}

export interface FbankWindowResult {
  readonly request: FbankWindowRequest;
  readonly actualSampleCount: number;
  readonly features: FbankMatrix;
}

export interface FbankComputeHint {
  /** Number of 10 ms frames by which this window advanced from the last one. */
  readonly reusableFrameShift: number;
}

export interface FbankComputer {
  compute(samples: Float32Array, hint?: FbankComputeHint): FbankMatrix;
  dispose?(): void;
}

export interface FbankStreamStats {
  requestedSamples: number;
  decodedSamples: number;
  reusedSamples: number;
  peakCachedSamples: number;
  windows: number;
}

interface MelFilter {
  readonly start: number;
  readonly weights: Float32Array;
}

/**
 * Senko's fixed Kaldi-style filter bank reference implementation.
 *
 * Scratch arrays are reused between calls. Create one instance per worker and
 * do not invoke the same instance concurrently.
 */
export class SenkoFbank {
  private readonly poveyWindow = createPoveyWindow();
  private readonly bitReverse = createBitReverseIndices();
  private readonly twiddleReal = new Float32Array(FFT_BINS);
  private readonly twiddleImag = new Float32Array(FFT_BINS);
  private readonly melFilters = createMelFilters();

  private readonly real = new Float32Array(SENKO_FBANK_FFT_SIZE);
  private readonly imaginary = new Float32Array(SENKO_FBANK_FFT_SIZE);
  private readonly power = new Float32Array(FFT_BINS);
  private readonly meanSums = new Float32Array(SENKO_FBANK_BINS);

  constructor() {
    this.initializeTwiddles();
  }

  compute(samples: Float32Array): FbankMatrix {
    const frameCount = frameCountForSamples(samples.length);
    const output = new Float32Array(frameCount * SENKO_FBANK_BINS);
    this.meanSums.fill(0);

    for (let frame = 0; frame < frameCount; frame += 1) {
      this.prepareFrame(samples, frame * SENKO_FBANK_FRAME_SHIFT);
      this.fftInPlace();
      this.computePower();

      const outputOffset = frame * SENKO_FBANK_BINS;
      for (let bin = 0; bin < SENKO_FBANK_BINS; bin += 1) {
        const filter = this.melFilters[bin]!;
        let energy = Math.fround(0);
        for (let i = 0; i < filter.weights.length; i += 1) {
          energy = Math.fround(
            energy +
              Math.fround(
                filter.weights[i]! * this.power[filter.start + i]!,
              ),
          );
        }
        const logEnergy = Math.fround(
          Math.log(Math.max(energy, FLOAT32_EPSILON)),
        );
        output[outputOffset + bin] = logEnergy;
        this.meanSums[bin] = Math.fround(this.meanSums[bin]! + logEnergy);
      }
    }

    // Per-window cepstral mean normalization, matching FeatureComputer.cpp.
    for (let bin = 0; bin < SENKO_FBANK_BINS; bin += 1) {
      this.meanSums[bin] = Math.fround(this.meanSums[bin]! / frameCount);
    }
    for (let frame = 0; frame < frameCount; frame += 1) {
      const offset = frame * SENKO_FBANK_BINS;
      for (let bin = 0; bin < SENKO_FBANK_BINS; bin += 1) {
        output[offset + bin] = Math.fround(
          output[offset + bin]! - this.meanSums[bin]!,
        );
      }
    }

    return { data: output, frameCount, binCount: SENKO_FBANK_BINS };
  }

  private prepareFrame(samples: Float32Array, start: number): void {
    let mean = Math.fround(0);
    for (let i = 0; i < SENKO_FBANK_FRAME_LENGTH; i += 1) {
      const sample = start + i < samples.length ? samples[start + i]! : 0;
      this.real[i] = sample;
      mean = Math.fround(mean + sample);
    }
    mean = Math.fround(mean / SENKO_FBANK_FRAME_LENGTH);

    for (let i = 0; i < SENKO_FBANK_FRAME_LENGTH; i += 1) {
      this.real[i] = Math.fround(this.real[i]! - mean);
    }
    for (let i = SENKO_FBANK_FRAME_LENGTH - 1; i > 0; i -= 1) {
      this.real[i] = Math.fround(
        this.real[i]! - Math.fround(PREEMPHASIS * this.real[i - 1]!),
      );
    }
    this.real[0] = Math.fround(
      this.real[0]! - Math.fround(PREEMPHASIS * this.real[0]!),
    );

    for (let i = 0; i < SENKO_FBANK_FRAME_LENGTH; i += 1) {
      this.real[i] = Math.fround(this.real[i]! * this.poveyWindow[i]!);
    }
    this.real.fill(0, SENKO_FBANK_FRAME_LENGTH);
    this.imaginary.fill(0);
  }

  private fftInPlace(): void {
    for (let i = 0; i < SENKO_FBANK_FFT_SIZE; i += 1) {
      const reversed = this.bitReverse[i]!;
      if (i < reversed) {
        const real = this.real[i]!;
        const imaginary = this.imaginary[i]!;
        this.real[i] = this.real[reversed]!;
        this.imaginary[i] = this.imaginary[reversed]!;
        this.real[reversed] = real;
        this.imaginary[reversed] = imaginary;
      }
    }

    for (let halfSize = 1; halfSize < SENKO_FBANK_FFT_SIZE; halfSize *= 2) {
      const step = SENKO_FBANK_FFT_SIZE / (halfSize * 2);
      for (
        let block = 0;
        block < SENKO_FBANK_FFT_SIZE;
        block += halfSize * 2
      ) {
        let twiddle = 0;
        for (let j = block; j < block + halfSize; j += 1) {
          const paired = j + halfSize;
          const wr = this.twiddleReal[twiddle]!;
          const wi = this.twiddleImag[twiddle]!;
          const pairedReal = this.real[paired]!;
          const pairedImaginary = this.imaginary[paired]!;
          const rotatedReal = Math.fround(
            Math.fround(wr * pairedReal) - Math.fround(wi * pairedImaginary),
          );
          const rotatedImaginary = Math.fround(
            Math.fround(wr * pairedImaginary) + Math.fround(wi * pairedReal),
          );
          const currentReal = this.real[j]!;
          const currentImaginary = this.imaginary[j]!;
          this.real[paired] = Math.fround(currentReal - rotatedReal);
          this.imaginary[paired] = Math.fround(
            currentImaginary - rotatedImaginary,
          );
          this.real[j] = Math.fround(currentReal + rotatedReal);
          this.imaginary[j] = Math.fround(
            currentImaginary + rotatedImaginary,
          );
          twiddle += step;
        }
      }
    }
  }

  private computePower(): void {
    for (let i = 0; i < FFT_BINS; i += 1) {
      const real = this.real[i]!;
      const imaginary = this.imaginary[i]!;
      this.power[i] = Math.fround(
        Math.fround(real * real) + Math.fround(imaginary * imaginary),
      );
    }
  }

  private initializeTwiddles(): void {
    // Reproduce fbank_utils.cpp's float sin table and derived cosine values.
    const sinTable = new Float32Array(FFT_BINS);
    for (let i = 0; i < FFT_BINS; i += 1) {
      sinTable[i] = Math.sin((-2 * Math.PI * i) / SENKO_FBANK_FFT_SIZE);
    }
    for (let k = 0; k < FFT_BINS; k += 1) {
      let cosValue = sinTable[
        (SENKO_FBANK_FFT_SIZE / 4 - k + FFT_BINS) % FFT_BINS
      ]!;
      if (k >= SENKO_FBANK_FFT_SIZE / 4) {
        cosValue = -cosValue;
      }
      this.twiddleReal[k] = -cosValue;
      this.twiddleImag[k] = sinTable[k]!;
    }
  }
}

/**
 * Streams requested windows in order. Only one decoded PCM window and one
 * FBank result are live at a time; consumers can batch/upload yielded results.
 */
export class StreamingFbankExtractor {
  readonly stats: FbankStreamStats = {
    requestedSamples: 0,
    decodedSamples: 0,
    reusedSamples: 0,
    peakCachedSamples: 0,
    windows: 0,
  };

  private readonly cache: OverlapSampleCache;
  private previousWindow:
    | { readonly startSample: number; readonly sampleCount: number }
    | undefined;

  constructor(
    reader: Pcm16WavReader,
    private readonly fbank: FbankComputer = new SenkoFbank(),
  ) {
    this.cache = new OverlapSampleCache(reader, this.stats);
  }

  async *extract(
    requests: Iterable<FbankWindowRequest> | AsyncIterable<FbankWindowRequest>,
  ): AsyncGenerator<FbankWindowResult> {
    for await (const request of requests) {
      validateWindowRequest(request);
      const actualSampleCount =
        request.startSample >= this.cache.sampleCount
          ? 0
          : Math.min(
              request.sampleCount,
              this.cache.sampleCount - request.startSample,
            );
      const samples = await this.cache.get(
        request.startSample,
        actualSampleCount,
      );
      const reusableFrameShift = this.reusableFrameShift(
        request.startSample,
        actualSampleCount,
      );
      const features = this.fbank.compute(
        samples,
        reusableFrameShift === 0 ? undefined : { reusableFrameShift },
      );
      this.previousWindow = {
        startSample: request.startSample,
        sampleCount: actualSampleCount,
      };
      this.stats.windows += 1;
      yield { request, actualSampleCount, features };
    }
  }

  dispose(): void {
    this.previousWindow = undefined;
    this.cache.dispose();
    this.fbank.dispose?.();
  }

  private reusableFrameShift(
    startSample: number,
    sampleCount: number,
  ): number {
    const previous = this.previousWindow;
    if (previous === undefined || previous.sampleCount !== sampleCount) return 0;
    const sampleShift = startSample - previous.startSample;
    if (
      sampleShift <= 0 ||
      sampleShift >= sampleCount ||
      sampleShift % SENKO_FBANK_FRAME_SHIFT !== 0
    ) {
      return 0;
    }
    return sampleShift / SENKO_FBANK_FRAME_SHIFT;
  }
}

export function secondsToFbankWindow(
  startSeconds: number,
  endSeconds: number,
  id?: string | number,
): FbankWindowRequest {
  if (
    !Number.isFinite(startSeconds) ||
    !Number.isFinite(endSeconds) ||
    endSeconds < startSeconds
  ) {
    throw new RangeError("window seconds must be finite and non-decreasing");
  }
  // Python first flattens the subsegments into a float32 NumPy array. Its C++
  // extractor then performs both the subtraction and multiplication as float
  // operations before truncating to size_t. Preserve that sequence instead of
  // calculating boundaries from JavaScript doubles.
  const nativeStart = Math.fround(startSeconds);
  const nativeEnd = Math.fround(endSeconds);
  // Senko's native arm64 build converts a negative float offset to size_t 0.
  // Negative starts can arise for a sub-1.5-second VAD island near time zero.
  const startSample = Math.max(
    0,
    Math.trunc(Math.fround(nativeStart * SENKO_FBANK_SAMPLE_RATE)),
  );
  const duration = Math.fround(nativeEnd - nativeStart);
  const request = {
    startSample,
    // Native Senko truncates duration separately from the start position.
    sampleCount: Math.max(
      1,
      Math.trunc(Math.fround(duration * SENKO_FBANK_SAMPLE_RATE)),
    ),
  };
  return id === undefined ? request : { ...request, id };
}

export function frameCountForSamples(sampleCount: number): number {
  if (!Number.isSafeInteger(sampleCount) || sampleCount < 0) {
    throw new RangeError("sampleCount must be a non-negative safe integer");
  }
  const paddedCount = Math.max(sampleCount, SENKO_FBANK_FRAME_LENGTH);
  return (
    1 +
    Math.floor(
      (paddedCount - SENKO_FBANK_FRAME_LENGTH) / SENKO_FBANK_FRAME_SHIFT,
    )
  );
}

class OverlapSampleCache {
  private startSample = 0;
  private samples = new Float32Array(0);
  private validLength = 0;

  constructor(
    private readonly reader: Pcm16WavReader,
    private readonly stats: FbankStreamStats,
  ) {}

  get sampleCount(): number {
    return this.reader.sampleCount;
  }

  async get(startSample: number, sampleCount: number): Promise<Float32Array> {
    this.stats.requestedSamples += sampleCount;
    if (sampleCount === 0) {
      this.startSample = startSample;
      this.validLength = 0;
      return this.samples.subarray(0, 0);
    }

    const endSample = startSample + sampleCount;
    const cacheEnd = this.startSample + this.validLength;
    const overlapStart = Math.max(startSample, this.startSample);
    const overlapEnd = Math.min(endSample, cacheEnd);
    const hasOverlap = overlapEnd > overlapStart;
    const reused = hasOverlap ? overlapEnd - overlapStart : 0;

    if (this.samples.length < sampleCount) {
      const next = new Float32Array(sampleCount);
      if (hasOverlap) {
        const sourceOffset = overlapStart - this.startSample;
        const targetOffset = overlapStart - startSample;
        next.set(
          this.samples.subarray(sourceOffset, sourceOffset + reused),
          targetOffset,
        );
      }
      this.samples = next;
    } else if (hasOverlap) {
      const sourceOffset = overlapStart - this.startSample;
      const targetOffset = overlapStart - startSample;
      this.samples.copyWithin(
        targetOffset,
        sourceOffset,
        sourceOffset + reused,
      );
    }

    let decoded = 0;
    const prefixCount = hasOverlap ? overlapStart - startSample : 0;
    if (prefixCount > 0) {
      decoded += await this.reader.readSamplesInto(
        startSample,
        this.samples,
        0,
        prefixCount,
      );
    }
    const suffixStart = hasOverlap ? overlapEnd : startSample;
    const suffixCount = endSample - suffixStart;
    if (suffixCount > 0) {
      decoded += await this.reader.readSamplesInto(
        suffixStart,
        this.samples,
        suffixStart - startSample,
        suffixCount,
      );
    }

    this.stats.decodedSamples += decoded;
    this.stats.reusedSamples += reused;
    this.stats.peakCachedSamples = Math.max(
      this.stats.peakCachedSamples,
      sampleCount,
    );
    this.startSample = startSample;
    this.validLength = sampleCount;
    return this.samples.subarray(0, sampleCount);
  }

  dispose(): void {
    this.startSample = 0;
    this.validLength = 0;
    this.samples = new Float32Array(0);
  }
}

function createPoveyWindow(): Float32Array {
  const window = new Float32Array(SENKO_FBANK_FRAME_LENGTH);
  const angularStep =
    (2 * Math.PI) / (SENKO_FBANK_FRAME_LENGTH - 1);
  for (let i = 0; i < window.length; i += 1) {
    window[i] = Math.pow(
      0.5 - 0.5 * Math.cos(angularStep * i),
      0.85,
    );
  }
  return window;
}

function createBitReverseIndices(): Uint16Array {
  const result = new Uint16Array(SENKO_FBANK_FFT_SIZE);
  const bits = Math.log2(SENKO_FBANK_FFT_SIZE);
  for (let i = 0; i < result.length; i += 1) {
    let source = i;
    let reversed = 0;
    for (let bit = 0; bit < bits; bit += 1) {
      reversed = (reversed << 1) | (source & 1);
      source >>= 1;
    }
    result[i] = reversed;
  }
  return result;
}

function createMelFilters(): readonly MelFilter[] {
  const fftBinWidth = Math.fround(
    SENKO_FBANK_SAMPLE_RATE / SENKO_FBANK_FFT_SIZE,
  );
  const lowMel = melScale(LOW_FREQUENCY);
  const highMel = melScale(HIGH_FREQUENCY);
  const melDelta = Math.fround(
    Math.fround(highMel - lowMel) / (SENKO_FBANK_BINS + 1),
  );
  const filters: MelFilter[] = [];

  for (let bin = 0; bin < SENKO_FBANK_BINS; bin += 1) {
    const left = Math.fround(lowMel + Math.fround(bin * melDelta));
    const middle = Math.fround(
      lowMel + Math.fround((bin + 1) * melDelta),
    );
    const right = Math.fround(
      lowMel + Math.fround((bin + 2) * melDelta),
    );
    let first = -1;
    let last = -1;
    const dense = new Float32Array(FFT_BINS);

    for (let fftBin = 0; fftBin < FFT_BINS; fftBin += 1) {
      const frequency = Math.fround(fftBinWidth * fftBin);
      const mel = melScale(frequency);
      if (mel > left && mel < right) {
        dense[fftBin] =
          mel <= middle
            ? Math.fround(
                Math.fround(mel - left) / Math.fround(middle - left),
              )
            : Math.fround(
                Math.fround(right - mel) / Math.fround(right - middle),
              );
        if (first < 0) first = fftBin;
        last = fftBin;
      }
    }

    if (first < 0 || last < first) {
      throw new Error(`empty mel filter ${bin}`);
    }
    filters.push({ start: first, weights: dense.slice(first, last + 1) });
  }
  return filters;
}

function melScale(frequency: number): number {
  return Math.fround(
    Math.fround(1127) *
      Math.fround(Math.log(Math.fround(1 + Math.fround(frequency / 700)))),
  );
}

function validateWindowRequest(request: FbankWindowRequest): void {
  if (!Number.isSafeInteger(request.startSample) || request.startSample < 0) {
    throw new RangeError("window startSample must be a non-negative safe integer");
  }
  if (!Number.isSafeInteger(request.sampleCount) || request.sampleCount < 1) {
    throw new RangeError("window sampleCount must be a positive safe integer");
  }
}
