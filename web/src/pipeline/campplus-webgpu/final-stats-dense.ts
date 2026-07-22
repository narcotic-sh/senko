/// <reference types="@webgpu/types" />

import { CampPlusActivationArena, type CampPlusArenaSlice } from "./arena";
import type { CampPlusPackedSection, PackedConvolutionRef } from "./metadata";
import { CampPlusGpuPackage } from "./package";

const WORKGROUP_SIZE = 128;
const INPUT_CHANNELS = 512;
const STAT_CHANNELS = INPUT_CHANNELS * 2;
const OUTPUT_CHANNELS = 192;
const FRAMES = 75;
const UNIFORM_BYTES = 64;

export interface FinalStatsDenseDescriptor {
  readonly label: string;
  readonly input: CampPlusArenaSlice;
  readonly inputStorageChannels: number;
  readonly batchSize: number;
  readonly dense: PackedConvolutionRef;
  readonly outputAffine: string;
  readonly output: GPUBuffer;
}

export class FinalStatsDenseDispatch {
  readonly gpuBufferBytes = UNIFORM_BYTES;
  private destroyed = false;

  constructor(
    readonly label: string,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly uniform: GPUBuffer,
    private readonly batchSize: number,
  ) {}

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    if (this.destroyed) throw new Error(`CAM++ final dispatch ${this.label} is destroyed`);
    const descriptor: GPUComputePassDescriptor =
      timestampWrites === undefined
        ? { label: this.label }
        : { label: this.label, timestampWrites };
    const pass = encoder.beginComputePass(descriptor);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(this.batchSize);
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uniform.destroy();
  }
}

/** StatsPool + 1024x192 dense + output affine/ReLU in one workgroup per item. */
export class FinalStatsDenseKernel {
  private constructor(
    private readonly device: GPUDevice,
    private readonly gpuPackage: CampPlusGpuPackage,
    private readonly arena: CampPlusActivationArena,
    private readonly pipeline: GPUComputePipeline,
    private readonly layout: GPUBindGroupLayout,
  ) {}

  static async create(
    device: GPUDevice,
    gpuPackage: CampPlusGpuPackage,
    arena: CampPlusActivationArena,
  ): Promise<FinalStatsDenseKernel> {
    if (device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE) {
      throw new Error(`CAM++ final kernel requires ${WORKGROUP_SIZE} workgroup lanes`);
    }
    const layout = device.createBindGroupLayout({
      label: "senko-campplus-final-stats-dense-bindings",
      entries: [
        storageEntry(0, "read-only-storage"),
        storageEntry(1, "read-only-storage"),
        storageEntry(2, "read-only-storage"),
        storageEntry(3, "read-only-storage"),
        storageEntry(4, "storage"),
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", minBindingSize: UNIFORM_BYTES },
        },
      ],
    });
    const module = device.createShaderModule({
      label: "senko-campplus-final-stats-dense",
      code: FINAL_STATS_DENSE_WGSL,
    });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `CAM++ final WGSL failed: ${errors.map((message) => message.message).join("; ")}`,
      );
    }
    const pipeline = await device.createComputePipelineAsync({
      label: "senko-campplus-final-stats-dense",
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "main" },
    });
    return new FinalStatsDenseKernel(device, gpuPackage, arena, pipeline, layout);
  }

  createDispatch(descriptor: FinalStatsDenseDescriptor): FinalStatsDenseDispatch {
    const weight = this.gpuPackage.section(descriptor.dense.weight);
    const bias = this.gpuPackage.section(descriptor.dense.bias);
    const affine = this.gpuPackage.section(descriptor.outputAffine);
    validateSections(weight, bias, affine);
    if (
      !Number.isSafeInteger(descriptor.batchSize) ||
      descriptor.batchSize <= 0 ||
      !Number.isSafeInteger(descriptor.inputStorageChannels) ||
      descriptor.inputStorageChannels < INPUT_CHANNELS
    ) {
      throw new RangeError(`${descriptor.label} has an invalid final tensor contract`);
    }
    const inputBytes =
      descriptor.batchSize * descriptor.inputStorageChannels * FRAMES * 2;
    if (
      descriptor.input.byteOffset % 256 !== 0 ||
      descriptor.input.byteLength < inputBytes ||
      descriptor.input.byteOffset + descriptor.input.byteLength > this.arena.byteLength
    ) {
      throw new RangeError(`${descriptor.label} final input does not fit the arena`);
    }
    const outputBytes = descriptor.batchSize * OUTPUT_CHANNELS * 4;
    if (descriptor.output.size < outputBytes) {
      throw new RangeError(`${descriptor.label} final FP32 output buffer is too small`);
    }
    const parameters = new Uint32Array([
      descriptor.input.byteOffset / 2,
      descriptor.batchSize,
      INPUT_CHANNELS,
      descriptor.inputStorageChannels,
      FRAMES,
      OUTPUT_CHANNELS,
      STAT_CHANNELS / 4,
      OUTPUT_CHANNELS / 4,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
    ]);
    const uniform = this.device.createBuffer({
      label: `${descriptor.label}-parameters`,
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniform, 0, parameters);
    try {
      const bindGroup = this.device.createBindGroup({
        label: `${descriptor.label}-bindings`,
        layout: this.layout,
        entries: [
          {
            binding: 0,
            resource: { buffer: this.arena.buffer, size: this.arena.byteLength },
          },
          sectionEntry(1, this.gpuPackage.weightsBuffer, weight),
          sectionEntry(2, this.gpuPackage.weightsBuffer, bias),
          sectionEntry(3, this.gpuPackage.weightsBuffer, affine),
          { binding: 4, resource: { buffer: descriptor.output, size: outputBytes } },
          { binding: 5, resource: { buffer: uniform, size: UNIFORM_BYTES } },
        ],
      });
      return new FinalStatsDenseDispatch(
        descriptor.label,
        this.pipeline,
        bindGroup,
        uniform,
        descriptor.batchSize,
      );
    } catch (error) {
      uniform.destroy();
      throw error;
    }
  }
}

function validateSections(
  weight: CampPlusPackedSection,
  bias: CampPlusPackedSection,
  affine: CampPlusPackedSection,
): void {
  if (
    weight.kind !== "conv_weight" ||
    weight.layout !== "K_O4_I4_I_O" ||
    weight.logicalShape[0] !== OUTPUT_CHANNELS ||
    weight.logicalShape[1] !== STAT_CHANNELS ||
    weight.logicalShape[2] !== 1 ||
    bias.kind !== "conv_bias" ||
    bias.logicalShape[0] !== OUTPUT_CHANNELS ||
    affine.kind !== "batch_norm_affine" ||
    affine.logicalShape[0] !== OUTPUT_CHANNELS
  ) {
    throw new Error("Packed CAM++ final sections do not match the static graph");
  }
}

function storageEntry(
  binding: number,
  type: GPUBufferBindingType,
): GPUBindGroupLayoutEntry {
  return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } };
}

function sectionEntry(
  binding: number,
  buffer: GPUBuffer,
  section: CampPlusPackedSection,
): GPUBindGroupEntry {
  return {
    binding,
    resource: { buffer, offset: section.byteOffset, size: section.byteLength },
  };
}

export const FINAL_STATS_DENSE_WGSL = /* wgsl */ `
enable f16;

struct Parameters {
  input_offset: u32,
  batch_size: u32,
  input_channels: u32,
  input_storage_channels: u32,
  frames: u32,
  output_channels: u32,
  statistic_groups: u32,
  output_groups: u32,
  r0: u32, r1: u32, r2: u32, r3: u32,
  r4: u32, r5: u32, r6: u32, r7: u32,
}

@group(0) @binding(0) var<storage, read> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> output_affine: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> parameters: Parameters;

var<workgroup> statistics: array<f16, 1024>;

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let batch = group.x;
  if (batch >= parameters.batch_size) { return; }

  var channel = local_id.x;
  while (channel < parameters.input_channels) {
    let channel_base =
      parameters.input_offset +
      (batch * parameters.input_storage_channels + channel) * parameters.frames;
    var sum = 0.0;
    for (var frame = 0u; frame < parameters.frames; frame += 1u) {
      sum += f32(arena[channel_base + frame]);
    }
    let mean = f16(sum / f32(parameters.frames));
    statistics[channel] = mean;

    var squared_sum = 0.0;
    for (var frame = 0u; frame < parameters.frames; frame += 1u) {
      let centered = f16(arena[channel_base + frame] - mean);
      squared_sum += f32(f16(centered * centered));
    }
    let variance = f16(squared_sum / f32(parameters.frames));
    statistics[parameters.input_channels + channel] =
      sqrt(f16(variance + f16(0.00001)));
    channel += 128u;
  }
  workgroupBarrier();

  let output_group = local_id.x;
  if (output_group >= parameters.output_groups) { return; }
  var accumulator = vec4<f32>(biases[output_group]);
  for (var input_group = 0u; input_group < parameters.statistic_groups; input_group += 1u) {
    let statistic_base = input_group * 4u;
    let weight_base =
      (output_group * parameters.statistic_groups + input_group) * 4u;
    accumulator = fma(
      vec4<f32>(f32(statistics[statistic_base])),
      vec4<f32>(weights[weight_base]),
      accumulator,
    );
    accumulator = fma(
      vec4<f32>(f32(statistics[statistic_base + 1u])),
      vec4<f32>(weights[weight_base + 1u]),
      accumulator,
    );
    accumulator = fma(
      vec4<f32>(f32(statistics[statistic_base + 2u])),
      vec4<f32>(weights[weight_base + 2u]),
      accumulator,
    );
    accumulator = fma(
      vec4<f32>(f32(statistics[statistic_base + 3u])),
      vec4<f32>(weights[weight_base + 3u]),
      accumulator,
    );
  }
  let dense = vec4<f16>(accumulator);
  let scale = output_affine[output_group * 2u];
  let shift = output_affine[output_group * 2u + 1u];
  let rounded = max(
    vec4<f16>(vec4<f32>(dense) * scale + shift),
    vec4<f16>(f16(0.0)),
  );
  let output_channel = output_group * 4u;
  for (var lane = 0u; lane < 4u; lane += 1u) {
    output[batch * parameters.output_channels + output_channel + lane] =
      f32(rounded[lane]);
  }
}
`;
