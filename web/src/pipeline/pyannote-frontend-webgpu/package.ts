/// <reference types="@webgpu/types" />

import { IncrementalSha256 } from "../campplus-webgpu/sha256";
import {
  parsePyannoteFrontendMetadata,
  type PyannoteFrontendPackageMetadata,
  type PyannoteFrontendPackedSection,
} from "./metadata";

const MAGIC = "SNKVADF1";
const HEADER_BYTES = 256;

export interface PyannoteFrontendBinaryHeader {
  readonly formatVersion: number;
  readonly headerBytes: number;
  readonly sectionAlignment: number;
  readonly sectionCount: number;
  readonly totalBytes: number;
  readonly sourceSha256: string;
  readonly payloadSha256: string;
  readonly batch: number;
  readonly samples: number;
  readonly frames: number;
  readonly features: number;
}

export interface PyannoteFrontendPackageLoadOptions {
  readonly fetch?: typeof fetch;
}

export class PyannoteFrontendGpuPackage {
  readonly sections: ReadonlyMap<string, PyannoteFrontendPackedSection>;
  private destroyed = false;

  private constructor(
    readonly metadata: PyannoteFrontendPackageMetadata,
    readonly weightsBuffer: GPUBuffer,
  ) {
    this.sections = new Map(metadata.sections.map((section) => [section.id, section]));
  }

  static async load(
    device: GPUDevice,
    metadataUrl: string,
    options: PyannoteFrontendPackageLoadOptions = {},
  ): Promise<PyannoteFrontendGpuPackage> {
    const fetchAsset = options.fetch ?? fetch;
    const metadataResponse = await fetchAsset(metadataUrl);
    if (!metadataResponse.ok) {
      throw new Error(
        `Pyannote frontend metadata request failed (${metadataResponse.status} ${metadataResponse.statusText})`,
      );
    }
    const metadata = parsePyannoteFrontendMetadata(await metadataResponse.json());
    if (
      (metadata.contract.intermediateDtype === "float16" ||
        metadata.contract.weightDtype === "float16") &&
      !device.features.has("shader-f16")
    ) {
      throw new Error("FP16 raw pyannote frontend requires shader-f16 support");
    }
    if (
      metadata.binary.byteLength > device.limits.maxStorageBufferBindingSize ||
      metadata.memory.activationArenaBytes > device.limits.maxStorageBufferBindingSize ||
      metadata.memory.activationArenaBytes > device.limits.maxBufferSize
    ) {
      throw new Error("Pyannote frontend package exceeds this WebGPU device's buffer limits");
    }
    const base = new URL(metadataUrl, globalThis.location?.href ?? "http://localhost/");
    const binaryResponse = await fetchAsset(new URL(metadata.binary.file, base));
    if (!binaryResponse.ok) {
      throw new Error(
        `Pyannote frontend binary request failed (${binaryResponse.status} ${binaryResponse.statusText})`,
      );
    }
    const bytes = new Uint8Array(await binaryResponse.arrayBuffer());
    validateBinary(bytes, metadata);
    const weightsBuffer = device.createBuffer({
      label: "senko-pyannote-frontend-weights",
      size: bytes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(weightsBuffer, 0, bytes);
    return new PyannoteFrontendGpuPackage(metadata, weightsBuffer);
  }

  section(id: string): PyannoteFrontendPackedSection {
    if (this.destroyed) throw new Error("Pyannote frontend package has been destroyed");
    const section = this.sections.get(id);
    if (section === undefined) throw new Error(`Unknown pyannote frontend section: ${id}`);
    return section;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.weightsBuffer.destroy();
  }
}

export function parsePyannoteFrontendBinaryHeader(
  bytes: Uint8Array,
): PyannoteFrontendBinaryHeader {
  if (bytes.byteLength < HEADER_BYTES) throw new Error("Truncated pyannote frontend header");
  const magic = new TextDecoder("ascii", { fatal: true }).decode(bytes.subarray(0, 8));
  if (magic !== MAGIC) throw new Error("Invalid pyannote frontend package magic");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const totalBytes = view.getBigUint64(24, true);
  if (totalBytes > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error("Pyannote frontend package length is outside JavaScript's exact range");
  }
  return {
    formatVersion: view.getUint32(8, true),
    headerBytes: view.getUint32(12, true),
    sectionAlignment: view.getUint32(16, true),
    sectionCount: view.getUint32(20, true),
    totalBytes: Number(totalBytes),
    sourceSha256: hex(bytes.subarray(32, 64)),
    payloadSha256: hex(bytes.subarray(64, 96)),
    batch: view.getUint32(96, true),
    samples: view.getUint32(100, true),
    frames: view.getUint32(104, true),
    features: view.getUint32(108, true),
  };
}

function validateBinary(
  bytes: Uint8Array,
  metadata: PyannoteFrontendPackageMetadata,
): void {
  if (bytes.byteLength !== metadata.binary.byteLength) {
    throw new Error("Pyannote frontend binary length does not match metadata");
  }
  const header = parsePyannoteFrontendBinaryHeader(bytes.subarray(0, HEADER_BYTES));
  const input = metadata.contract.inputShape;
  const output = metadata.contract.outputShape;
  if (
    header.formatVersion !== 1 ||
    header.headerBytes !== metadata.binary.headerBytes ||
    header.sectionAlignment !== metadata.binary.sectionAlignment ||
    header.sectionCount !== metadata.binary.sectionCount ||
    header.totalBytes !== bytes.byteLength ||
    header.sourceSha256 !== metadata.source.sha256 ||
    header.payloadSha256 !== metadata.binary.payloadSha256 ||
    header.batch !== input[0] ||
    header.samples !== input[2] ||
    header.frames !== output[1] ||
    header.features !== output[2] ||
    IncrementalSha256.hex(bytes) !== metadata.binary.sha256 ||
    IncrementalSha256.hex(bytes.subarray(HEADER_BYTES)) !== metadata.binary.payloadSha256
  ) {
    throw new Error("Pyannote frontend binary header or SHA-256 does not match metadata");
  }
}

function hex(bytes: Uint8Array): string {
  let result = "";
  for (const value of bytes) result += value.toString(16).padStart(2, "0");
  return result;
}
