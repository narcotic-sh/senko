import { describe, expect, it } from "vitest";

import {
  BlobByteSource,
  Pcm16WavReader,
  WAV_REUSABLE_READ_BUFFER_BYTES,
  WavFormatError,
} from "./wav";
import { CountingMemorySource, makePcm16Wav } from "./test-helpers";

describe("Pcm16WavReader", () => {
  it("parses chunked mono PCM16 WAV and decodes only requested samples", async () => {
    const pcm = new Int16Array([-32768, -1, 0, 1, 32767]);
    const source = new CountingMemorySource(makePcm16Wav(pcm, { junk: true }));
    const reader = await Pcm16WavReader.open(source);

    expect(reader.info).toMatchObject({
      sampleRate: 16_000,
      channels: 1,
      bitsPerSample: 16,
      sampleCount: pcm.length,
    });

    const bytesBeforePcm = source.bytesRead;
    const samples = await reader.readSamples(1, 3);
    expect(Array.from(samples)).toEqual([-1 / 32_768, 0, 1 / 32_768]);
    expect(source.bytesRead - bytesBeforePcm).toBe(3 * 2);
    expect(source.largestRead).toBeLessThan(source.size);
  });

  it("clips range reads at EOF", async () => {
    const source = new CountingMemorySource(
      makePcm16Wav(new Int16Array([100, 200, 300])),
    );
    const reader = await Pcm16WavReader.open(source);
    const target = new Float32Array(5);

    const written = await reader.readSamplesInto(2, target);
    expect(written).toBe(1);
    expect(target[0]).toBeCloseTo(300 / 32_768, 8);
    expect(target.slice(1)).toEqual(new Float32Array(4));
  });

  it("decodes Blob PCM through one reusable BYOB backing store", async () => {
    const pcm = Int16Array.from(
      { length: 200_003 },
      (_, index) => ((index * 997) % 65_536) - 32_768,
    );
    const reader = await Pcm16WavReader.open(
      new Blob([toArrayBuffer(makePcm16Wav(pcm))]),
    );
    expect(reader.reusableReadBufferBytes).toBe(WAV_REUSABLE_READ_BUFFER_BYTES);

    const target = new Float32Array(pcm.length);
    expect(await reader.readSamplesInto(0, target)).toBe(pcm.length);
    for (const index of [0, 1, 159_999, 160_000, pcm.length - 1]) {
      expect(target[index]).toBe(pcm[index]! / 32_768);
    }
  });

  it("keeps arbitrary and concurrent Blob ranges exact", async () => {
    const bytes = Uint8Array.from({ length: 251 }, (_, index) => index);
    const source = new BlobByteSource(new Blob([bytes]));
    const consume = async (offset: number, length: number): Promise<number[]> => {
      const result: number[] = [];
      await source.consume(offset, length, (chunk) => result.push(...chunk));
      return result;
    };

    const [middle, prefix] = await Promise.all([consume(73, 101), consume(0, 37)]);
    expect(middle).toEqual(Array.from(bytes.subarray(73, 174)));
    expect(prefix).toEqual(Array.from(bytes.subarray(0, 37)));
    expect(new Uint8Array(await source.read(200, 17))).toEqual(
      bytes.subarray(200, 217),
    );
  });

  it("rejects formats that the Senko models do not accept", async () => {
    const wrongRate = makePcm16Wav(new Int16Array(4), { sampleRate: 8_000 });
    await expect(Pcm16WavReader.open(new Blob([toArrayBuffer(wrongRate)]))).rejects.toThrow(
      /expected 16000 Hz/,
    );

    const stereo = makePcm16Wav(new Int16Array(4), { channels: 2 });
    await expect(
      Pcm16WavReader.open(new Blob([toArrayBuffer(stereo)])),
    ).rejects.toBeInstanceOf(WavFormatError);
  });
});

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
}
