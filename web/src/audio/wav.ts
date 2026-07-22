export interface RandomAccessByteSource {
  readonly size: number;
  read(offset: number, length: number): Promise<ArrayBuffer>;
}

export interface ReusableChunkByteSource extends RandomAccessByteSource {
  /** Exact persistent backing-store capacity used by `consume`. */
  readonly reusableReadBufferBytes: number;
  /**
   * Consume an arbitrary byte range through a reusable buffer.
   *
   * The callback is synchronous: its view is transferred/detached before the
   * next callback. Callers must finish reading it before returning.
   */
  consume(
    offset: number,
    length: number,
    consumer: (bytes: Uint8Array<ArrayBuffer>) => void,
  ): Promise<void>;
}

export const WAV_REUSABLE_READ_BUFFER_BYTES = 320_000;

export class BlobByteSource implements ReusableChunkByteSource {
  readonly size: number;
  readonly reusableReadBufferBytes = WAV_REUSABLE_READ_BUFFER_BYTES;
  private reusable = new Uint8Array(WAV_REUSABLE_READ_BUFFER_BYTES);
  private consumeTail: Promise<void> = Promise.resolve();

  constructor(private readonly blob: Blob) {
    this.size = blob.size;
  }

  async read(offset: number, length: number): Promise<ArrayBuffer> {
    validateRange(offset, length, this.size, "byte");
    return this.blob.slice(offset, offset + length).arrayBuffer();
  }

  consume(
    offset: number,
    length: number,
    consumer: (bytes: Uint8Array<ArrayBuffer>) => void,
  ): Promise<void> {
    validateRange(offset, length, this.size, "byte");
    const operation = this.consumeTail.then(() =>
      this.consumeUnlocked(offset, length, consumer),
    );
    // Preserve random-access semantics for concurrent callers while keeping
    // exactly one reusable backing store. A rejected read must not poison the
    // serialization chain for later independent requests.
    this.consumeTail = operation.catch(() => undefined);
    return operation;
  }

  private async consumeUnlocked(
    offset: number,
    length: number,
    consumer: (bytes: Uint8Array<ArrayBuffer>) => void,
  ): Promise<void> {
    if (length === 0) return;
    const stream = this.blob.slice(offset, offset + length).stream();
    const reader = stream.getReader({ mode: "byob" });
    let consumed = 0;
    try {
      while (consumed < length) {
        const result = await reader.read(this.reusable);
        const bytes = result.value;
        if (result.done || bytes === undefined || bytes.byteLength === 0) {
          throw new WavFormatError(
            `short reusable Blob read: expected ${length} bytes, got ${consumed}`,
          );
        }
        if (
          bytes.byteOffset !== 0 ||
          bytes.buffer.byteLength !== this.reusableReadBufferBytes ||
          bytes.byteLength > length - consumed
        ) {
          throw new WavFormatError("Chrome changed its reusable Blob BYOB contract");
        }
        // Chrome transfers the supplied backing store: `this.reusable` is now
        // detached and `bytes.buffer` is the same 320 KiB allocation under a
        // new ArrayBuffer wrapper. Decode synchronously before recycling it.
        consumer(bytes);
        consumed += bytes.byteLength;
        this.reusable = new Uint8Array(bytes.buffer);
      }
      await reader.cancel();
    } finally {
      reader.releaseLock();
      if (this.reusable.byteLength === 0) {
        // Only an exceptional read can strand the transferred allocation.
        this.reusable = new Uint8Array(WAV_REUSABLE_READ_BUFFER_BYTES);
      }
    }
  }
}

export class WavFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WavFormatError";
  }
}

export interface Pcm16WavInfo {
  readonly sampleRate: 16_000;
  readonly channels: 1;
  readonly bitsPerSample: 16;
  readonly sampleCount: number;
  readonly durationSeconds: number;
  readonly dataOffset: number;
  readonly dataByteLength: number;
}

const REQUIRED_SAMPLE_RATE = 16_000;
const REQUIRED_CHANNELS = 1;
const REQUIRED_BITS_PER_SAMPLE = 16;
const BYTES_PER_SAMPLE = 2;
// VAD consumes 10-second windows. Reading one complete window at a time avoids
// ~900 short-lived Blob ArrayBuffers for an hour-long recording while keeping
// the largest temporary PCM16 byte view at only 320 KiB.
const DEFAULT_READ_SAMPLES = 160_000;

interface ParsedFormat {
  audioFormat: number;
  channels: number;
  sampleRate: number;
  byteRate: number;
  blockAlign: number;
  bitsPerSample: number;
}

/**
 * A random-access reader for Senko's supported WAV format.
 *
 * Only requested PCM ranges are decoded. A browser `File`/`Blob` therefore
 * never becomes a full-file `Float32Array`, which is important for hour-long
 * recordings.
 */
export class Pcm16WavReader {
  readonly info: Pcm16WavInfo;

  private constructor(
    private readonly source: RandomAccessByteSource,
    info: Pcm16WavInfo,
  ) {
    this.info = info;
  }

  static async open(
    input: Blob | RandomAccessByteSource,
  ): Promise<Pcm16WavReader> {
    const source = isRandomAccessByteSource(input)
      ? input
      : new BlobByteSource(input);
    const info = await parsePcm16Wav(source);
    return new Pcm16WavReader(source, info);
  }

  get sampleRate(): 16_000 {
    return this.info.sampleRate;
  }

  get sampleCount(): number {
    return this.info.sampleCount;
  }

  /** Persistent byte scratch owned by a Blob source; zero for generic sources. */
  get reusableReadBufferBytes(): number {
    return isReusableChunkByteSource(this.source)
      ? this.source.reusableReadBufferBytes
      : 0;
  }

  async readSamples(startSample: number, sampleCount: number): Promise<Float32Array> {
    validateSampleRequest(startSample, sampleCount);
    const available = availableSamples(
      startSample,
      sampleCount,
      this.info.sampleCount,
    );
    const output = new Float32Array(available);
    await this.readSamplesInto(startSample, output);
    return output;
  }

  /**
   * Decode at most `sampleCount` samples into `target` and return the number
   * written. Reads are clipped at EOF, matching Senko's native extractor.
   */
  async readSamplesInto(
    startSample: number,
    target: Float32Array,
    targetOffset = 0,
    sampleCount = target.length - targetOffset,
  ): Promise<number> {
    validateSampleRequest(startSample, sampleCount);
    if (!Number.isInteger(targetOffset) || targetOffset < 0) {
      throw new RangeError("targetOffset must be a non-negative integer");
    }
    if (targetOffset + sampleCount > target.length) {
      throw new RangeError("target range exceeds the destination array");
    }

    const count = availableSamples(
      startSample,
      sampleCount,
      this.info.sampleCount,
    );
    if (count === 0) return 0;
    if (isReusableChunkByteSource(this.source)) {
      await decodeReusablePcm16Range(
        this.source,
        this.info.dataOffset + startSample * BYTES_PER_SAMPLE,
        count,
        target,
        targetOffset,
      );
      return count;
    }
    let decoded = 0;

    while (decoded < count) {
      const chunkSamples = Math.min(DEFAULT_READ_SAMPLES, count - decoded);
      const byteOffset =
        this.info.dataOffset + (startSample + decoded) * BYTES_PER_SAMPLE;
      const byteLength = chunkSamples * BYTES_PER_SAMPLE;
      const bytes = await this.source.read(byteOffset, byteLength);
      if (bytes.byteLength !== byteLength) {
        throw new WavFormatError(
          `short WAV read: expected ${byteLength} bytes, got ${bytes.byteLength}`,
        );
      }

      const view = new DataView(bytes);
      for (let i = 0; i < chunkSamples; i += 1) {
        target[targetOffset + decoded + i] =
          view.getInt16(i * BYTES_PER_SAMPLE, true) / 32_768;
      }
      decoded += chunkSamples;
    }

    return decoded;
  }
}

async function decodeReusablePcm16Range(
  source: ReusableChunkByteSource,
  byteOffset: number,
  sampleCount: number,
  target: Float32Array,
  targetOffset: number,
): Promise<void> {
  let decoded = 0;
  let lowByte: number | undefined;
  await source.consume(
    byteOffset,
    sampleCount * BYTES_PER_SAMPLE,
    (bytes) => {
      let byteIndex = 0;
      if (lowByte !== undefined) {
        const unsigned = lowByte | (bytes[0]! << 8);
        target[targetOffset + decoded] =
          (unsigned >= 0x8000 ? unsigned - 0x1_0000 : unsigned) / 32_768;
        decoded += 1;
        lowByte = undefined;
        byteIndex = 1;
      }
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (; byteIndex + 1 < bytes.byteLength; byteIndex += BYTES_PER_SAMPLE) {
        target[targetOffset + decoded] = view.getInt16(byteIndex, true) / 32_768;
        decoded += 1;
      }
      if (byteIndex < bytes.byteLength) lowByte = bytes[byteIndex];
    },
  );
  if (lowByte !== undefined || decoded !== sampleCount) {
    throw new WavFormatError(
      `reusable WAV read decoded ${decoded}/${sampleCount} samples`,
    );
  }
}

async function parsePcm16Wav(
  source: RandomAccessByteSource,
): Promise<Pcm16WavInfo> {
  if (!Number.isSafeInteger(source.size) || source.size < 12) {
    throw new WavFormatError("WAV is too small to contain a RIFF header");
  }

  const riff = await readExactly(source, 0, 12);
  if (fourCc(riff, 0) !== "RIFF" || fourCc(riff, 8) !== "WAVE") {
    throw new WavFormatError("expected a RIFF/WAVE file");
  }

  let format: ParsedFormat | undefined;
  let dataOffset: number | undefined;
  let dataByteLength: number | undefined;
  let offset = 12;
  let chunkCount = 0;

  while (offset + 8 <= source.size) {
    if (chunkCount > 1_000_000) {
      throw new WavFormatError("unreasonable number of WAV chunks");
    }
    chunkCount += 1;

    const chunkHeader = await readExactly(source, offset, 8);
    const chunkId = fourCc(chunkHeader, 0);
    const declaredSize = chunkHeader.getUint32(4, true);
    const chunkDataOffset = offset + 8;
    const remaining = source.size - chunkDataOffset;

    if (declaredSize > remaining && !(chunkId === "data" && declaredSize === 0)) {
      throw new WavFormatError(`WAV chunk ${chunkId} extends past EOF`);
    }

    if (chunkId === "fmt ") {
      if (declaredSize < 16) {
        throw new WavFormatError("WAV fmt chunk is shorter than 16 bytes");
      }
      const fmt = await readExactly(source, chunkDataOffset, 16);
      format = {
        audioFormat: fmt.getUint16(0, true),
        channels: fmt.getUint16(2, true),
        sampleRate: fmt.getUint32(4, true),
        byteRate: fmt.getUint32(8, true),
        blockAlign: fmt.getUint16(12, true),
        bitsPerSample: fmt.getUint16(14, true),
      };
    } else if (chunkId === "data") {
      dataOffset = chunkDataOffset;
      dataByteLength = declaredSize === 0 ? remaining : declaredSize;
    }

    if (format !== undefined && dataOffset !== undefined) {
      break;
    }

    const paddedSize = declaredSize + (declaredSize & 1);
    const nextOffset = chunkDataOffset + paddedSize;
    if (!Number.isSafeInteger(nextOffset) || nextOffset <= offset) {
      throw new WavFormatError("invalid WAV chunk size");
    }
    offset = nextOffset;
  }

  if (format === undefined || dataOffset === undefined || dataByteLength === undefined) {
    throw new WavFormatError("WAV must contain fmt and data chunks");
  }

  validateSenkoFormat(format);
  if (dataByteLength % BYTES_PER_SAMPLE !== 0) {
    throw new WavFormatError("PCM16 data chunk has an odd byte length");
  }

  const sampleCount = dataByteLength / BYTES_PER_SAMPLE;
  return {
    sampleRate: REQUIRED_SAMPLE_RATE,
    channels: REQUIRED_CHANNELS,
    bitsPerSample: REQUIRED_BITS_PER_SAMPLE,
    sampleCount,
    durationSeconds: sampleCount / REQUIRED_SAMPLE_RATE,
    dataOffset,
    dataByteLength,
  };
}

function validateSenkoFormat(format: ParsedFormat): void {
  if (format.audioFormat !== 1) {
    throw new WavFormatError(
      `unsupported WAV encoding ${format.audioFormat}; expected integer PCM`,
    );
  }
  if (format.channels !== REQUIRED_CHANNELS) {
    throw new WavFormatError(
      `unsupported channel count ${format.channels}; expected mono`,
    );
  }
  if (format.sampleRate !== REQUIRED_SAMPLE_RATE) {
    throw new WavFormatError(
      `unsupported sample rate ${format.sampleRate}; expected 16000 Hz`,
    );
  }
  if (format.bitsPerSample !== REQUIRED_BITS_PER_SAMPLE) {
    throw new WavFormatError(
      `unsupported bit depth ${format.bitsPerSample}; expected PCM16`,
    );
  }
  if (format.blockAlign !== BYTES_PER_SAMPLE) {
    throw new WavFormatError(
      `invalid PCM16 block alignment ${format.blockAlign}`,
    );
  }
  if (format.byteRate !== REQUIRED_SAMPLE_RATE * BYTES_PER_SAMPLE) {
    throw new WavFormatError(`invalid PCM16 byte rate ${format.byteRate}`);
  }
}

async function readExactly(
  source: RandomAccessByteSource,
  offset: number,
  length: number,
): Promise<DataView> {
  validateRange(offset, length, source.size, "byte");
  const bytes = await source.read(offset, length);
  if (bytes.byteLength !== length) {
    throw new WavFormatError(
      `short WAV read: expected ${length} bytes, got ${bytes.byteLength}`,
    );
  }
  return new DataView(bytes);
}

function fourCc(view: DataView, offset: number): string {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

function isRandomAccessByteSource(
  value: Blob | RandomAccessByteSource,
): value is RandomAccessByteSource {
  return "read" in value && typeof value.read === "function";
}

function isReusableChunkByteSource(
  value: RandomAccessByteSource,
): value is ReusableChunkByteSource {
  return (
    "consume" in value &&
    typeof value.consume === "function" &&
    "reusableReadBufferBytes" in value &&
    value.reusableReadBufferBytes === WAV_REUSABLE_READ_BUFFER_BYTES
  );
}

function validateRange(
  offset: number,
  length: number,
  size: number,
  unit: string,
): void {
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new RangeError(`${unit} offset must be a non-negative safe integer`);
  }
  if (!Number.isSafeInteger(length) || length < 0) {
    throw new RangeError(`${unit} length must be a non-negative safe integer`);
  }
  if (offset + length > size) {
    throw new RangeError(`${unit} range exceeds source size`);
  }
}

function validateSampleRequest(startSample: number, sampleCount: number): void {
  if (!Number.isSafeInteger(startSample) || startSample < 0) {
    throw new RangeError("startSample must be a non-negative safe integer");
  }
  if (!Number.isSafeInteger(sampleCount) || sampleCount < 0) {
    throw new RangeError("sampleCount must be a non-negative safe integer");
  }
}

function availableSamples(
  startSample: number,
  requested: number,
  total: number,
): number {
  return startSample >= total ? 0 : Math.min(requested, total - startSample);
}
