import type { RandomAccessByteSource } from "./wav";

export class CountingMemorySource implements RandomAccessByteSource {
  readonly size: number;
  bytesRead = 0;
  largestRead = 0;

  constructor(private readonly bytes: Uint8Array) {
    this.size = bytes.byteLength;
  }

  async read(offset: number, length: number): Promise<ArrayBuffer> {
    this.bytesRead += length;
    this.largestRead = Math.max(this.largestRead, length);
    const copy = this.bytes.slice(offset, offset + length);
    return copy.buffer as ArrayBuffer;
  }
}

interface WavOptions {
  sampleRate?: number;
  channels?: number;
  junk?: boolean;
}

export function makePcm16Wav(
  samples: Int16Array,
  options: WavOptions = {},
): Uint8Array {
  const sampleRate = options.sampleRate ?? 16_000;
  const channels = options.channels ?? 1;
  const junkLength = options.junk ? 3 : 0;
  const junkChunkLength = options.junk ? 8 + junkLength + 1 : 0;
  const fmtChunkLength = 8 + 16;
  const dataLength = samples.byteLength;
  const totalLength = 12 + junkChunkLength + fmtChunkLength + 8 + dataLength;
  const bytes = new Uint8Array(totalLength);
  const view = new DataView(bytes.buffer);

  writeFourCc(bytes, 0, "RIFF");
  view.setUint32(4, totalLength - 8, true);
  writeFourCc(bytes, 8, "WAVE");

  let offset = 12;
  if (options.junk) {
    writeFourCc(bytes, offset, "JUNK");
    view.setUint32(offset + 4, junkLength, true);
    bytes.set([11, 22, 33], offset + 8);
    offset += junkChunkLength;
  }

  writeFourCc(bytes, offset, "fmt ");
  view.setUint32(offset + 4, 16, true);
  view.setUint16(offset + 8, 1, true);
  view.setUint16(offset + 10, channels, true);
  view.setUint32(offset + 12, sampleRate, true);
  view.setUint32(offset + 16, sampleRate * channels * 2, true);
  view.setUint16(offset + 20, channels * 2, true);
  view.setUint16(offset + 22, 16, true);
  offset += fmtChunkLength;

  writeFourCc(bytes, offset, "data");
  view.setUint32(offset + 4, dataLength, true);
  offset += 8;
  for (let i = 0; i < samples.length; i += 1) {
    view.setInt16(offset + i * 2, samples[i]!, true);
  }
  return bytes;
}

function writeFourCc(bytes: Uint8Array, offset: number, value: string): void {
  for (let i = 0; i < 4; i += 1) {
    bytes[offset + i] = value.charCodeAt(i);
  }
}
