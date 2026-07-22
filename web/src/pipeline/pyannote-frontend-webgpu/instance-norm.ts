/// <reference types="@webgpu/types" />

import type { PyannoteFrontendPackedSection } from "./metadata";
import { PyannoteFrontendGpuPackage } from "./package";

const UNIFORM_BYTES = 64;

export interface BctNormDescriptor {
  readonly label: string;
  readonly input: GPUBuffer;
  readonly inputBytes: number;
  readonly statistics: GPUBuffer;
  readonly statisticsBytes: number;
  readonly affine: PyannoteFrontendPackedSection;
  readonly batch: number;
  readonly channels: number;
  readonly frames: number;
  readonly epsilon: number;
}

export class PyannoteBctNormDispatch {
  constructor(
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    readonly uniformBuffer: GPUBuffer,
    private readonly batch: number,
    private readonly channels: number,
    readonly label: string,
  ) {}

  encode(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: this.label });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(this.channels, this.batch);
    pass.end();
  }

  destroy(): void {
    this.uniformBuffer.destroy();
  }
}

export class PyannoteBctNormKernel {
  private constructor(
    private readonly device: GPUDevice,
    private readonly gpuPackage: PyannoteFrontendGpuPackage,
    private readonly pipeline: GPUComputePipeline,
  ) {}

  static async create(
    device: GPUDevice,
    gpuPackage: PyannoteFrontendGpuPackage,
  ): Promise<PyannoteBctNormKernel> {
    const precision = gpuPackage.metadata.contract.intermediateDtype;
    const module = device.createShaderModule({
      label: `senko-pyannote-${precision}-bct-instance-norm`,
      code: bctNormWgsl(precision),
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `Pyannote BCT InstanceNorm WGSL failed: ${errors.map((item) => item.message).join("; ")}`,
      );
    }
    const pipeline = await device.createComputePipelineAsync({
      label: `senko-pyannote-${precision}-bct-instance-norm`,
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    return new PyannoteBctNormKernel(device, gpuPackage, pipeline);
  }

  createDispatch(descriptor: BctNormDescriptor): PyannoteBctNormDispatch {
    if (
      descriptor.affine.kind !== "instance_norm_affine" ||
      descriptor.affine.layout !== "C4_GAMMA_BETA" ||
      descriptor.affine.dtype !== "float32" ||
      descriptor.affine.logicalShape[0] !== descriptor.channels
    ) {
      throw new Error(`${descriptor.label} affine does not match its channel count`);
    }
    const parameters = new ArrayBuffer(UNIFORM_BYTES);
    const view = new DataView(parameters);
    view.setUint32(0, descriptor.batch, true);
    view.setUint32(4, descriptor.channels, true);
    view.setUint32(8, descriptor.frames, true);
    view.setFloat32(16, descriptor.epsilon, true);
    const uniformBuffer = this.device.createBuffer({
      label: `${descriptor.label}-parameters`,
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, parameters);
    const bindGroup = this.device.createBindGroup({
      label: `${descriptor.label}-bindings`,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: descriptor.input, size: descriptor.inputBytes },
        },
        {
          binding: 1,
          resource: {
            buffer: this.gpuPackage.weightsBuffer,
            offset: descriptor.affine.byteOffset,
            size: descriptor.affine.byteLength,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: descriptor.statistics,
            size: descriptor.statisticsBytes,
          },
        },
        { binding: 3, resource: { buffer: uniformBuffer, size: UNIFORM_BYTES } },
      ],
    });
    return new PyannoteBctNormDispatch(
      this.pipeline,
      bindGroup,
      uniformBuffer,
      descriptor.batch,
      descriptor.channels,
      descriptor.label,
    );
  }
}

export interface F32BtfNormDescriptor {
  readonly label: string;
  readonly values: GPUBuffer;
  readonly valueBytes: number;
  readonly affine: PyannoteFrontendPackedSection;
  readonly batch: number;
  readonly channels: number;
  readonly frames: number;
  readonly epsilon: number;
  readonly leakyAlpha: number;
}

export class PyannoteF32BtfNormDispatch {
  constructor(
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    readonly uniformBuffer: GPUBuffer,
    private readonly batch: number,
    private readonly channels: number,
    readonly label: string,
  ) {}

  encode(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: this.label });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(this.channels, this.batch);
    pass.end();
  }

  destroy(): void {
    this.uniformBuffer.destroy();
  }
}

export class PyannoteF32BtfNormKernel {
  private constructor(
    private readonly device: GPUDevice,
    private readonly gpuPackage: PyannoteFrontendGpuPackage,
    private readonly pipeline: GPUComputePipeline,
  ) {}

  static async create(
    device: GPUDevice,
    gpuPackage: PyannoteFrontendGpuPackage,
  ): Promise<PyannoteF32BtfNormKernel> {
    const module = device.createShaderModule({
      label: "senko-pyannote-f32-btf-instance-norm-leaky",
      code: F32_BTF_NORM_WGSL,
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `Pyannote final InstanceNorm WGSL failed: ${errors.map((item) => item.message).join("; ")}`,
      );
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "senko-pyannote-f32-btf-instance-norm-leaky",
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    return new PyannoteF32BtfNormKernel(device, gpuPackage, pipeline);
  }

  createDispatch(descriptor: F32BtfNormDescriptor): PyannoteF32BtfNormDispatch {
    if (
      descriptor.affine.kind !== "instance_norm_affine" ||
      descriptor.affine.layout !== "C4_GAMMA_BETA" ||
      descriptor.affine.dtype !== "float32" ||
      descriptor.affine.logicalShape[0] !== descriptor.channels
    ) {
      throw new Error(`${descriptor.label} affine does not match its channel count`);
    }
    const parameters = new ArrayBuffer(UNIFORM_BYTES);
    const view = new DataView(parameters);
    view.setUint32(0, descriptor.batch, true);
    view.setUint32(4, descriptor.channels, true);
    view.setUint32(8, descriptor.frames, true);
    view.setFloat32(16, descriptor.epsilon, true);
    view.setFloat32(20, descriptor.leakyAlpha, true);
    const uniformBuffer = this.device.createBuffer({
      label: `${descriptor.label}-parameters`,
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, parameters);
    const bindGroup = this.device.createBindGroup({
      label: `${descriptor.label}-bindings`,
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: descriptor.values, size: descriptor.valueBytes } },
        {
          binding: 1,
          resource: {
            buffer: this.gpuPackage.weightsBuffer,
            offset: descriptor.affine.byteOffset,
            size: descriptor.affine.byteLength,
          },
        },
        { binding: 2, resource: { buffer: uniformBuffer, size: UNIFORM_BYTES } },
      ],
    });
    return new PyannoteF32BtfNormDispatch(
      this.pipeline,
      bindGroup,
      uniformBuffer,
      descriptor.batch,
      descriptor.channels,
      descriptor.label,
    );
  }
}

export function bctNormWgsl(
  precision: "float16" | "float32",
): string {
  const halfPrecision = precision === "float16";
  return /* wgsl */ `
${halfPrecision ? "enable f16;" : ""}
struct InputBuffer {
  values: array<${halfPrecision ? "f16" : "f32"}>
};
struct FloatBuffer { values: array<f32> };
struct Parameters {
  batch: u32,
  channels: u32,
  frames: u32,
  reserved: u32,
  epsilon: f32,
};
@group(0) @binding(0) var<storage, read> input_values: InputBuffer;
@group(0) @binding(1) var<storage, read> affine: FloatBuffer;
@group(0) @binding(2) var<storage, read_write> statistics: FloatBuffer;
@group(0) @binding(3) var<uniform> parameters: Parameters;
var<workgroup> partial: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let channel = group_id.x;
  let batch_index = group_id.y;
  let lane = local_id.x;
  let base = (batch_index * parameters.channels + channel) * parameters.frames;
  var sum = 0.0;
  var frame = lane;
  loop {
    if (frame >= parameters.frames) { break; }
    sum += f32(input_values.values[base + frame]);
    frame += 128u;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var width = 64u;
  loop {
    if (lane < width) { partial[lane] += partial[lane + width]; }
    workgroupBarrier();
    if (width == 1u) { break; }
    width /= 2u;
  }
  let mean = partial[0] / f32(parameters.frames);
  workgroupBarrier();
  var variance_sum = 0.0;
  frame = lane;
  loop {
    if (frame >= parameters.frames) { break; }
    let centered = f32(input_values.values[base + frame]) - mean;
    variance_sum += centered * centered;
    frame += 128u;
  }
  partial[lane] = variance_sum;
  workgroupBarrier();
  width = 64u;
  loop {
    if (lane < width) { partial[lane] += partial[lane + width]; }
    workgroupBarrier();
    if (width == 1u) { break; }
    width /= 2u;
  }
  if (lane == 0u) {
    let affine_group = channel / 4u;
    let affine_lane = channel % 4u;
    let gamma = affine.values[affine_group * 8u + affine_lane];
    let beta = affine.values[affine_group * 8u + 4u + affine_lane];
    let variance = partial[0] / f32(parameters.frames);
    let scale = gamma * inverseSqrt(variance + parameters.epsilon);
    let statistics_index = (batch_index * parameters.channels + channel) * 2u;
    statistics.values[statistics_index] = scale;
    statistics.values[statistics_index + 1u] = beta - mean * scale;
  }
}
`;
}

/** Legacy names retained for diagnostic imports; package metadata selects precision. */
export { PyannoteBctNormKernel as PyannoteF16BctNormKernel };
export type PyannoteF16BctNormDispatch = PyannoteBctNormDispatch;
export type F16BctNormDescriptor = BctNormDescriptor;

const F32_BTF_NORM_WGSL = /* wgsl */ `
struct FloatBuffer { values: array<f32> };
struct Parameters {
  batch: u32,
  channels: u32,
  frames: u32,
  reserved: u32,
  epsilon: f32,
  leaky_alpha: f32,
};
@group(0) @binding(0) var<storage, read_write> values: FloatBuffer;
@group(0) @binding(1) var<storage, read> affine: FloatBuffer;
@group(0) @binding(2) var<uniform> parameters: Parameters;
var<workgroup> partial: array<f32, 128>;

fn offset(batch_index: u32, frame: u32, channel: u32) -> u32 {
  return (batch_index * parameters.frames + frame) * parameters.channels + channel;
}

@compute @workgroup_size(128)
fn main(
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let channel = group_id.x;
  let batch_index = group_id.y;
  let lane = local_id.x;
  var sum = 0.0;
  var frame = lane;
  loop {
    if (frame >= parameters.frames) { break; }
    sum += values.values[offset(batch_index, frame, channel)];
    frame += 128u;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var width = 64u;
  loop {
    if (lane < width) { partial[lane] += partial[lane + width]; }
    workgroupBarrier();
    if (width == 1u) { break; }
    width /= 2u;
  }
  let mean = partial[0] / f32(parameters.frames);
  workgroupBarrier();
  var variance_sum = 0.0;
  frame = lane;
  loop {
    if (frame >= parameters.frames) { break; }
    let centered = values.values[offset(batch_index, frame, channel)] - mean;
    variance_sum += centered * centered;
    frame += 128u;
  }
  partial[lane] = variance_sum;
  workgroupBarrier();
  width = 64u;
  loop {
    if (lane < width) { partial[lane] += partial[lane + width]; }
    workgroupBarrier();
    if (width == 1u) { break; }
    width /= 2u;
  }
  let affine_group = channel / 4u;
  let affine_lane = channel % 4u;
  let gamma = affine.values[affine_group * 8u + affine_lane];
  let beta = affine.values[affine_group * 8u + 4u + affine_lane];
  let variance = partial[0] / f32(parameters.frames);
  let scale = gamma * inverseSqrt(variance + parameters.epsilon);
  let shift = beta - mean * scale;
  frame = lane;
  loop {
    if (frame >= parameters.frames) { break; }
    let target_index = offset(batch_index, frame, channel);
    let normalized = values.values[target_index] * scale + shift;
    values.values[target_index] = select(
      parameters.leaky_alpha * normalized,
      normalized,
      normalized >= 0.0,
    );
    frame += 128u;
  }
}
`;
