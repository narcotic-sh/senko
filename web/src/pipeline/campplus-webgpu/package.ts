/// <reference types="@webgpu/types" />

import {
  parseCampPlusMetadata,
  type CampPlusPackageMetadata,
  type CampPlusPackedSection,
} from "./metadata";
import { bytesToHex, IncrementalSha256 } from "./sha256";

const MAGIC = "SNKCAMW1";
const HEADER_BYTES = 256;

export interface CampPlusPackageLoadProgress {
  readonly stage: "metadata" | "weights" | "validation";
  readonly loadedBytes: number;
  readonly totalBytes: number;
}

export interface CampPlusPackageLoadOptions {
  readonly fetch?: typeof fetch;
  readonly onProgress?: (progress: CampPlusPackageLoadProgress) => void;
}

export interface CampPlusBinaryHeader {
  readonly formatVersion: number;
  readonly headerBytes: number;
  readonly sectionAlignment: number;
  readonly sectionCount: number;
  readonly totalBytes: number;
  readonly sourceSha256: string;
  readonly payloadSha256: string;
  readonly sourceBatch: number;
  readonly frames: number;
  readonly featureDim: number;
  readonly embeddingDim: number;
}

export class CampPlusGpuPackage {
  readonly sections: ReadonlyMap<string, CampPlusPackedSection>;
  private destroyed = false;

  private constructor(
    readonly metadata: CampPlusPackageMetadata,
    readonly weightsBuffer: GPUBuffer,
    readonly binaryUrl: string,
  ) {
    this.sections = new Map(metadata.sections.map((section) => [section.id, section]));
  }

  static async load(
    device: GPUDevice,
    metadataUrl: string,
    options: CampPlusPackageLoadOptions = {},
  ): Promise<CampPlusGpuPackage> {
    const fetchAsset = options.fetch ?? fetch;
    options.onProgress?.({ stage: "metadata", loadedBytes: 0, totalBytes: 0 });
    const metadataResponse = await fetchAsset(metadataUrl);
    if (!metadataResponse.ok) {
      throw new Error(
        `CAM++ metadata request failed (${metadataResponse.status} ${metadataResponse.statusText})`,
      );
    }
    const metadata = parseCampPlusMetadata(await metadataResponse.json());
    requireDeviceSupport(device, metadata);
    const binaryUrl = resolveAssetUrl(metadata.binary.file, metadataUrl);
    const binaryResponse = await fetchAsset(binaryUrl);
    if (!binaryResponse.ok) {
      throw new Error(
        `CAM++ binary request failed (${binaryResponse.status} ${binaryResponse.statusText})`,
      );
    }
    const weightsBuffer = await uploadAndValidateBinary(
      device,
      binaryResponse,
      metadata,
      options.onProgress,
    );
    return new CampPlusGpuPackage(metadata, weightsBuffer, binaryUrl);
  }

  section(id: string): CampPlusPackedSection {
    if (this.destroyed) throw new Error("CAM++ package has been destroyed");
    const section = this.sections.get(id);
    if (section === undefined) throw new Error(`Unknown CAM++ package section: ${id}`);
    return section;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.weightsBuffer.destroy();
  }
}

export async function uploadAndValidateBinary(
  device: GPUDevice,
  response: Response,
  metadata: CampPlusPackageMetadata,
  onProgress?: (progress: CampPlusPackageLoadProgress) => void,
): Promise<GPUBuffer> {
  const declaredLength = response.headers.get("content-length");
  if (declaredLength !== null && Number(declaredLength) !== metadata.binary.byteLength) {
    throw new Error("CAM++ response Content-Length does not match metadata");
  }
  if (response.body === null) {
    throw new Error("CAM++ binary response is not streamable");
  }

  const weights = device.createBuffer({
    label: "senko-campplus-packed-weights",
    size: metadata.binary.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const reader = response.body.getReader();
  const headerBytes = new Uint8Array(HEADER_BYTES);
  const wholeHash = new IncrementalSha256();
  const payloadHash = new IncrementalSha256();
  let totalRead = 0;
  let uploadedBytes = 0;
  let carry = new Uint8Array(0);
  let lastReported = 0;
  try {
    onProgress?.({ stage: "weights", loadedBytes: 0, totalBytes: metadata.binary.byteLength });
    while (true) {
      const item = await reader.read();
      if (item.done) break;
      const chunk = item.value;
      if (totalRead + chunk.byteLength > metadata.binary.byteLength) {
        throw new Error("CAM++ binary is longer than declared metadata");
      }
      wholeHash.update(chunk);
      const payloadStart = Math.max(0, HEADER_BYTES - totalRead);
      if (payloadStart < chunk.byteLength) payloadHash.update(chunk.subarray(payloadStart));
      if (totalRead < HEADER_BYTES) {
        const copied = Math.min(chunk.byteLength, HEADER_BYTES - totalRead);
        headerBytes.set(chunk.subarray(0, copied), totalRead);
      }

      let chunkOffset = 0;
      if (carry.byteLength > 0) {
        const needed = 4 - carry.byteLength;
        if (chunk.byteLength < needed) {
          const combined = new Uint8Array(carry.byteLength + chunk.byteLength);
          combined.set(carry);
          combined.set(chunk, carry.byteLength);
          carry = combined;
          totalRead += chunk.byteLength;
          continue;
        }
        const word = new Uint8Array(4);
        word.set(carry);
        word.set(chunk.subarray(0, needed), carry.byteLength);
        device.queue.writeBuffer(weights, uploadedBytes, word);
        uploadedBytes += 4;
        chunkOffset = needed;
        carry = new Uint8Array(0);
      }

      const alignedLength = (chunk.byteLength - chunkOffset) & ~3;
      if (alignedLength > 0) {
        device.queue.writeBuffer(
          weights,
          uploadedBytes,
          chunk.subarray(chunkOffset, chunkOffset + alignedLength),
        );
        uploadedBytes += alignedLength;
        chunkOffset += alignedLength;
      }
      if (chunkOffset < chunk.byteLength) carry = chunk.slice(chunkOffset);
      totalRead += chunk.byteLength;
      if (totalRead - lastReported >= 1024 * 1024 || totalRead === metadata.binary.byteLength) {
        lastReported = totalRead;
        onProgress?.({
          stage: "weights",
          loadedBytes: totalRead,
          totalBytes: metadata.binary.byteLength,
        });
      }
    }

    if (totalRead !== metadata.binary.byteLength || uploadedBytes !== totalRead || carry.byteLength > 0) {
      throw new Error("CAM++ binary ended before its declared aligned length");
    }
    onProgress?.({
      stage: "validation",
      loadedBytes: totalRead,
      totalBytes: metadata.binary.byteLength,
    });
    const header = parseCampPlusBinaryHeader(headerBytes);
    validateHeader(header, metadata);
    const actualPayloadHash = payloadHash.digestHex();
    const actualWholeHash = wholeHash.digestHex();
    if (
      actualPayloadHash !== metadata.binary.payloadSha256 ||
      actualWholeHash !== metadata.binary.sha256
    ) {
      throw new Error("CAM++ binary SHA-256 validation failed");
    }
    return weights;
  } catch (error) {
    void reader.cancel(error).catch(() => undefined);
    weights.destroy();
    throw error;
  } finally {
    reader.releaseLock();
  }
}

export function parseCampPlusBinaryHeader(bytes: Uint8Array): CampPlusBinaryHeader {
  if (bytes.byteLength < HEADER_BYTES) throw new Error("Truncated CAM++ package header");
  const magic = new TextDecoder("ascii", { fatal: true }).decode(bytes.subarray(0, 8));
  if (magic !== MAGIC) throw new Error("Invalid CAM++ package magic");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const totalBytesBig = view.getBigUint64(24, true);
  if (totalBytesBig > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error("CAM++ package length is outside JavaScript's exact integer range");
  }
  return {
    formatVersion: view.getUint32(8, true),
    headerBytes: view.getUint32(12, true),
    sectionAlignment: view.getUint32(16, true),
    sectionCount: view.getUint32(20, true),
    totalBytes: Number(totalBytesBig),
    sourceSha256: bytesToHex(bytes.subarray(32, 64)),
    payloadSha256: bytesToHex(bytes.subarray(64, 96)),
    sourceBatch: view.getUint32(96, true),
    frames: view.getUint32(100, true),
    featureDim: view.getUint32(104, true),
    embeddingDim: view.getUint32(108, true),
  };
}

function validateHeader(
  header: CampPlusBinaryHeader,
  metadata: CampPlusPackageMetadata,
): void {
  const input = metadata.contract.inputShape;
  if (
    header.formatVersion !== 1 ||
    header.headerBytes !== metadata.binary.headerBytes ||
    header.sectionAlignment !== metadata.binary.sectionAlignment ||
    header.sectionCount !== metadata.binary.sectionCount ||
    header.totalBytes !== metadata.binary.byteLength ||
    header.sourceSha256 !== metadata.source.sha256 ||
    header.payloadSha256 !== metadata.binary.payloadSha256 ||
    header.sourceBatch !== input[0] ||
    header.frames !== input[1] ||
    header.featureDim !== input[2] ||
    header.embeddingDim !== metadata.contract.outputShape[1]
  ) {
    throw new Error("CAM++ binary header does not match metadata");
  }
}

function requireDeviceSupport(device: GPUDevice, metadata: CampPlusPackageMetadata): void {
  for (const feature of metadata.contract.requiredWebGpuFeatures) {
    if (!device.features.has(feature)) {
      throw new Error(`Raw CAM++ ${metadata.contract.internalDtype} package requires ${feature}`);
    }
  }
  if (metadata.binary.byteLength > device.limits.maxStorageBufferBindingSize) {
    throw new Error("CAM++ packed weights exceed maxStorageBufferBindingSize");
  }
  if (metadata.binary.byteLength > device.limits.maxBufferSize) {
    throw new Error("CAM++ packed weights exceed maxBufferSize");
  }
  if (metadata.memory.activationArenaBytes > device.limits.maxStorageBufferBindingSize) {
    throw new Error("CAM++ activation arena exceeds maxStorageBufferBindingSize");
  }
  if (metadata.memory.activationArenaBytes > device.limits.maxBufferSize) {
    throw new Error("CAM++ activation arena exceeds maxBufferSize");
  }
}

function resolveAssetUrl(file: string, metadataUrl: string): string {
  const base = new URL(metadataUrl, globalThis.location?.href ?? "http://localhost/");
  return new URL(file, base).href;
}
