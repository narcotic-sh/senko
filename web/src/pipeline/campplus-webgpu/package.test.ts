import { beforeAll, describe, expect, it } from "vitest";

import { parseCampPlusMetadata } from "./metadata";
import { parseCampPlusBinaryHeader, uploadAndValidateBinary } from "./package";
import { makeSyntheticCampPlusFixture } from "./test-fixture";

interface FakeGpuBuffer {
  readonly size: number;
  readonly bytes: Uint8Array;
  destroyed: boolean;
  destroy(): void;
}

beforeAll(() => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 0x80, COPY_DST: 0x08 },
  });
});

describe("CAM++ streaming package upload", () => {
  it("parses the fixed little-endian binary header", () => {
    const { binary } = makeSyntheticCampPlusFixture();
    expect(parseCampPlusBinaryHeader(binary.subarray(0, 256))).toMatchObject({
      formatVersion: 1,
      totalBytes: 1024,
      sourceBatch: 32,
      frames: 150,
      featureDim: 80,
      embeddingDim: 192,
    });
  });

  it("uploads arbitrary response chunks with at most a three-byte carry", async () => {
    const { binary, metadata: rawMetadata } = makeSyntheticCampPlusFixture();
    const metadata = parseCampPlusMetadata(rawMetadata);
    const { device, buffers, writeSizes } = fakeDevice();
    const response = chunkedResponse(binary, [1, 2, 3, 257, 5, 128, 7, 509]);
    const result = await uploadAndValidateBinary(device, response, metadata);
    expect(result).toBe(buffers[0]);
    expect(buffers[0]!.bytes).toEqual(binary);
    expect(buffers[0]!.destroyed).toBe(false);
    expect(writeSizes.every((size) => size % 4 === 0)).toBe(true);
  });

  it("destroys the GPU allocation when a streamed checksum fails", async () => {
    const { binary, metadata: rawMetadata } = makeSyntheticCampPlusFixture();
    const metadata = parseCampPlusMetadata(rawMetadata);
    const corrupted = binary.slice();
    corrupted[900] = corrupted[900]! ^ 1;
    const { device, buffers } = fakeDevice();
    await expect(
      uploadAndValidateBinary(device, chunkedResponse(corrupted, [11, 13, 17]), metadata),
    ).rejects.toThrow(/SHA-256/);
    expect(buffers[0]!.destroyed).toBe(true);
  });
});

function fakeDevice(): {
  readonly device: GPUDevice;
  readonly buffers: FakeGpuBuffer[];
  readonly writeSizes: number[];
} {
  const buffers: FakeGpuBuffer[] = [];
  const writeSizes: number[] = [];
  const device = {
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
      const buffer: FakeGpuBuffer = {
        size: Number(descriptor.size),
        bytes: new Uint8Array(Number(descriptor.size)),
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      buffers.push(buffer);
      return buffer as unknown as GPUBuffer;
    },
    queue: {
      writeBuffer(target: GPUBuffer, offset: number, source: GPUAllowSharedBufferSource): void {
        const fake = target as unknown as FakeGpuBuffer;
        const bytes = ArrayBuffer.isView(source)
          ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength)
          : new Uint8Array(source);
        fake.bytes.set(bytes, offset);
        writeSizes.push(bytes.byteLength);
      },
    },
  };
  return { device: device as unknown as GPUDevice, buffers, writeSizes };
}

function chunkedResponse(bytes: Uint8Array, sizes: readonly number[]): Response {
  let offset = 0;
  let sizeIndex = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (offset >= bytes.length) {
        controller.close();
        return;
      }
      const length = Math.min(sizes[sizeIndex % sizes.length]!, bytes.length - offset);
      controller.enqueue(bytes.slice(offset, offset + length));
      offset += length;
      sizeIndex += 1;
    },
  });
  return new Response(stream, { headers: { "content-length": String(bytes.byteLength) } });
}
