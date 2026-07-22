/// <reference types="@webgpu/types" />

import { IncrementalSha256 } from "../campplus-webgpu/sha256";
import {
  parsePyannoteTailMetadata,
  type PyannoteTailMetadata,
  type PyannoteTailSection,
} from "./metadata";

const MAGIC = "SNKVADT1";
const HEADER_BYTES = 256;
const UNIFORM_BYTES = 64;

export interface RawPyannoteTailGpuBytes {
  readonly weights: number;
  readonly output: number;
  readonly readback: number;
  readonly uniform: number;
  readonly total: number;
}

export class RawPyannoteTail {
  readonly outputBuffer: GPUBuffer;
  readonly readbackBuffer: GPUBuffer;
  readonly gpuBytes: RawPyannoteTailGpuBytes;
  private destroyed = false;

  private constructor(
    readonly metadata: PyannoteTailMetadata,
    private readonly weightsBuffer: GPUBuffer,
    private readonly uniformBuffer: GPUBuffer,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    outputBuffer: GPUBuffer,
    readbackBuffer: GPUBuffer,
  ) {
    this.outputBuffer = outputBuffer;
    this.readbackBuffer = readbackBuffer;
    this.gpuBytes = {
      weights: weightsBuffer.size,
      output: outputBuffer.size,
      readback: readbackBuffer.size,
      uniform: uniformBuffer.size,
      total: weightsBuffer.size + outputBuffer.size + readbackBuffer.size + uniformBuffer.size,
    };
  }

  static async create(
    device: GPUDevice,
    recurrentInput: GPUBuffer,
    metadataUrl: string,
    fetchAsset: typeof fetch = fetch,
  ): Promise<RawPyannoteTail> {
    if (!device.features.has("shader-f16")) {
      throw new Error("Raw pyannote tail requires shader-f16 support");
    }
    const metadataResponse = await fetchAsset(metadataUrl);
    if (!metadataResponse.ok) throw new Error(`Raw tail metadata HTTP ${metadataResponse.status}`);
    const metadata = parsePyannoteTailMetadata(await metadataResponse.json());
    const binaryUrl = new URL(
      metadata.binary.file,
      new URL(metadataUrl, globalThis.location?.href ?? "http://localhost/"),
    );
    const binaryResponse = await fetchAsset(binaryUrl);
    if (!binaryResponse.ok) throw new Error(`Raw tail binary HTTP ${binaryResponse.status}`);
    const bytes = new Uint8Array(await binaryResponse.arrayBuffer());
    validateBinary(bytes, metadata);
    const shader = device.createShaderModule({
      label: "senko-pyannote-raw-tail",
      code: TAIL_WGSL,
    });
    const info = await shader.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `Pyannote tail WGSL failed: ${errors.map((item) => item.message).join("; ")}`,
      );
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "senko-pyannote-raw-tail",
      layout: "auto",
      compute: { module: shader, entryPoint: "main" },
    });
    const weightsBuffer = device.createBuffer({
      label: "senko-pyannote-tail-weights",
      size: bytes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputBuffer = device.createBuffer({
      label: "senko-pyannote-tail-output",
      size: metadata.outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readbackBuffer = device.createBuffer({
      label: "senko-pyannote-tail-readback",
      size: metadata.readbackBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const uniformBuffer = device.createBuffer({
      label: "senko-pyannote-tail-parameters",
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(weightsBuffer, 0, bytes);
    const parameters = new ArrayBuffer(UNIFORM_BYTES);
    const view = new DataView(parameters);
    const offsets = [
      section(metadata, "linear:0:weight").byteOffset / 8,
      section(metadata, "linear:0:bias").byteOffset / 8,
      section(metadata, "linear:1:weight").byteOffset / 8,
      section(metadata, "linear:1:bias").byteOffset / 8,
      section(metadata, "linear:2:weight").byteOffset / 8,
      section(metadata, "linear:2:bias").byteOffset / 8,
    ];
    [metadata.batch, 589, 256, 128, 7, ...offsets].forEach((value, index) =>
      view.setUint32(index * 4, value, true),
    );
    view.setFloat32(44, 0.01, true);
    device.queue.writeBuffer(uniformBuffer, 0, parameters);
    const bindGroup = device.createBindGroup({
      label: "senko-pyannote-raw-tail-bindings",
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: recurrentInput, size: metadata.batch * 589 * 256 * 4 },
        },
        { binding: 1, resource: { buffer: weightsBuffer, size: weightsBuffer.size } },
        { binding: 2, resource: { buffer: outputBuffer, size: outputBuffer.size } },
        { binding: 3, resource: { buffer: uniformBuffer, size: uniformBuffer.size } },
      ],
    });
    return new RawPyannoteTail(
      metadata,
      weightsBuffer,
      uniformBuffer,
      pipeline,
      bindGroup,
      outputBuffer,
      readbackBuffer,
    );
  }

  encode(
    encoder: GPUCommandEncoder,
    copyToReadback = false,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    this.assertAlive();
    const pass = encoder.beginComputePass({
      label: "senko-pyannote-raw-tail",
      ...(timestampWrites === undefined ? {} : { timestampWrites }),
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(589, this.metadata.batch);
    pass.end();
    if (copyToReadback) {
      encoder.copyBufferToBuffer(
        this.outputBuffer,
        0,
        this.readbackBuffer,
        0,
        this.metadata.outputBytes,
      );
    }
  }

  async readback(): Promise<Float32Array> {
    this.assertAlive();
    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    try {
      return new Float32Array(this.readbackBuffer.getMappedRange()).slice();
    } finally {
      this.readbackBuffer.unmap();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.weightsBuffer.destroy();
    this.uniformBuffer.destroy();
    this.outputBuffer.destroy();
    this.readbackBuffer.destroy();
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Raw pyannote tail has been destroyed");
  }
}

function section(metadata: PyannoteTailMetadata, id: string): PyannoteTailSection {
  const value = metadata.sections.get(id);
  if (value === undefined) throw new Error(`Missing raw tail section ${id}`);
  return value;
}

function validateBinary(bytes: Uint8Array, metadata: PyannoteTailMetadata): void {
  if (bytes.byteLength !== metadata.binary.byteLength || bytes.byteLength < HEADER_BYTES) {
    throw new Error("Raw pyannote tail binary length mismatch");
  }
  const magic = new TextDecoder("ascii", { fatal: true }).decode(bytes.subarray(0, 8));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (
    magic !== MAGIC ||
    view.getUint32(8, true) !== 1 ||
    view.getUint32(12, true) !== HEADER_BYTES ||
    view.getUint32(20, true) !== metadata.binary.sectionCount ||
    Number(view.getBigUint64(24, true)) !== bytes.byteLength ||
    hex(bytes.subarray(32, 64)) !== metadata.sourceSha256 ||
    hex(bytes.subarray(64, 96)) !== metadata.binary.payloadSha256 ||
    view.getUint32(96, true) !== metadata.batch ||
    view.getUint32(100, true) !== 589 ||
    view.getUint32(104, true) !== 256 ||
    view.getUint32(108, true) !== 7 ||
    IncrementalSha256.hex(bytes) !== metadata.binary.sha256 ||
    IncrementalSha256.hex(bytes.subarray(HEADER_BYTES)) !== metadata.binary.payloadSha256
  ) {
    throw new Error("Raw pyannote tail header/SHA mismatch");
  }
}

function hex(bytes: Uint8Array): string {
  let result = "";
  for (const byte of bytes) result += byte.toString(16).padStart(2, "0");
  return result;
}

const TAIL_WGSL = /* wgsl */ `
enable f16;
struct FloatBuffer { values: array<f32> };
struct Half4Buffer { values: array<vec4<f16>> };
struct Parameters {
  batch: u32,
  frames: u32,
  input_features: u32,
  hidden: u32,
  classes: u32,
  weight0: u32,
  bias0: u32,
  weight1: u32,
  bias1: u32,
  weight2: u32,
  bias2: u32,
  leaky_alpha: f32,
};
@group(0) @binding(0) var<storage, read> recurrent: FloatBuffer;
@group(0) @binding(1) var<storage, read> packed: Half4Buffer;
@group(0) @binding(2) var<storage, read_write> logits: FloatBuffer;
@group(0) @binding(3) var<uniform> parameters: Parameters;
var<workgroup> input_cache: array<f32, 256>;
var<workgroup> hidden0: array<f32, 128>;
var<workgroup> hidden1: array<f32, 128>;

fn packed_value(offset: u32, input_index: u32, output_index: u32, groups: u32) -> f32 {
  return f32(packed.values[offset + input_index * groups + output_index / 4u][output_index % 4u]);
}

fn activate(value: f32) -> f32 {
  return select(parameters.leaky_alpha * value, value, value >= 0.0);
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let frame = group_id.x;
  let batch_index = group_id.y;
  let lane = local_id.x;
  let input_base = (batch_index * parameters.frames + frame) * parameters.input_features;
  input_cache[lane] = recurrent.values[input_base + lane];
  input_cache[lane + 128u] = recurrent.values[input_base + lane + 128u];
  workgroupBarrier();

  var accumulated = f32(packed.values[parameters.bias0 + lane / 4u][lane % 4u]);
  for (var input_index = 0u; input_index < 256u; input_index += 1u) {
    accumulated = fma(
      input_cache[input_index],
      packed_value(parameters.weight0, input_index, lane, 32u),
      accumulated,
    );
  }
  hidden0[lane] = activate(accumulated);
  workgroupBarrier();

  accumulated = f32(packed.values[parameters.bias1 + lane / 4u][lane % 4u]);
  for (var input_index = 0u; input_index < 128u; input_index += 1u) {
    accumulated = fma(
      hidden0[input_index],
      packed_value(parameters.weight1, input_index, lane, 32u),
      accumulated,
    );
  }
  hidden1[lane] = activate(accumulated);
  workgroupBarrier();

  if (lane < parameters.classes) {
    accumulated = f32(packed.values[parameters.bias2 + lane / 4u][lane % 4u]);
    for (var input_index = 0u; input_index < 128u; input_index += 1u) {
      accumulated = fma(
        hidden1[input_index],
        packed_value(parameters.weight2, input_index, lane, 2u),
        accumulated,
      );
    }
    logits.values[(batch_index * parameters.frames + frame) * parameters.classes + lane] = accumulated;
  }
}
`;
