/// <reference types="@webgpu/types" />

import { CampPlusActivationArena, type CampPlusArenaSlice } from "./arena";
import type { CampPlusPackedSection, PackedConvolutionRef } from "./metadata";
import { CampPlusGpuPackage } from "./package";
import { campPlusStorageBytes, campPlusStorageWgsl } from "./storage";

const WORKGROUP_SIZE = 128;
const MAX_INPUT_CHANNELS = 1024;
const UNIFORM_BYTES = 80;
export const POINTWISE_TRANSIT_REQUIRED_WORKGROUP_STORAGE_BYTES = 16_384;
export const POINTWISE_TRANSIT_TILE4_WORKGROUP_STORAGE_BYTES = 32_768;
export const POINTWISE_TRANSIT_VARIANTS = [
  "full-cache",
  "chunk512",
] as const;
export type PointwiseTransitVariant = (typeof POINTWISE_TRANSIT_VARIANTS)[number];
export const DEFAULT_POINTWISE_TRANSIT_VARIANT: PointwiseTransitVariant = "chunk512";
export type PointwiseTransitAccumulation = "float32" | "float16";
export const DEFAULT_POINTWISE_TRANSIT_ACCUMULATION: PointwiseTransitAccumulation =
  "float16";

export function isPointwiseTransitVariant(value: string): value is PointwiseTransitVariant {
  return (POINTWISE_TRANSIT_VARIANTS as readonly string[]).includes(value);
}

export interface PointwiseTransitDescriptor {
  readonly label: string;
  readonly convolution: PackedConvolutionRef;
  readonly preactivationAffine: string;
  readonly input: CampPlusArenaSlice;
  readonly output: CampPlusArenaSlice;
  readonly batchSize: number;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly frames: number;
  readonly inputStorageChannels: number;
  readonly outputStorageChannels: number;
  readonly outputRelu: boolean;
}

export class PointwiseTransitDispatch {
  readonly gpuBufferBytes = UNIFORM_BYTES;
  private destroyed = false;

  constructor(
    readonly label: string,
    readonly outputTile: 2 | 4,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly uniform: GPUBuffer,
    private readonly workgroups: readonly [number, number, number],
  ) {}

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    if (this.destroyed) throw new Error(`CAM++ transit ${this.label} is destroyed`);
    const descriptor: GPUComputePassDescriptor =
      timestampWrites === undefined
        ? { label: this.label }
        : { label: this.label, timestampWrites };
    const pass = encoder.beginComputePass(descriptor);
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(...this.workgroups);
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.uniform.destroy();
  }
}

/** Fused BN/ReLU + 1x1 transit with one activation feeding 2 or 4 output vec4s. */
export class PointwiseTransitKernels {
  private constructor(
    private readonly device: GPUDevice,
    private readonly gpuPackage: CampPlusGpuPackage,
    private readonly arena: CampPlusActivationArena,
    readonly variant: PointwiseTransitVariant,
    readonly accumulation: PointwiseTransitAccumulation,
    private readonly outputTile: 2 | 4,
    private readonly pipeline: GPUComputePipeline,
    private readonly layout: GPUBindGroupLayout,
  ) {}

  static async create(
    device: GPUDevice,
    gpuPackage: CampPlusGpuPackage,
    arena: CampPlusActivationArena,
    variant: PointwiseTransitVariant = DEFAULT_POINTWISE_TRANSIT_VARIANT,
    accumulation: PointwiseTransitAccumulation =
      DEFAULT_POINTWISE_TRANSIT_ACCUMULATION,
  ): Promise<PointwiseTransitKernels> {
    const storageDtype = gpuPackage.metadata.contract.internalDtype;
    const storageBytes = campPlusStorageBytes(storageDtype);
    if (
      device.limits.maxComputeWorkgroupStorageSize <
      POINTWISE_TRANSIT_REQUIRED_WORKGROUP_STORAGE_BYTES
    ) {
      throw new Error(
        `CAM++ tiled transits require ${POINTWISE_TRANSIT_REQUIRED_WORKGROUP_STORAGE_BYTES} workgroup bytes`,
      );
    }
    const layout = device.createBindGroupLayout({
      label: "senko-campplus-pointwise-transit-bindings",
      entries: [
        storageEntry(0, "storage"),
        storageEntry(1, "read-only-storage"),
        storageEntry(2, "read-only-storage"),
        storageEntry(3, "read-only-storage"),
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", minBindingSize: UNIFORM_BYTES },
        },
      ],
    });
    let effectiveVariant = variant;
    let tile2Bytes =
      (effectiveVariant === "chunk512" ? 512 : 1024) * 2 * 4 * storageBytes;
    if (
      tile2Bytes > device.limits.maxComputeWorkgroupStorageSize &&
      storageDtype === "float32" &&
      effectiveVariant === "full-cache"
    ) {
      effectiveVariant = "chunk512";
      tile2Bytes = 512 * 2 * 4 * storageBytes;
    }
    if (tile2Bytes > device.limits.maxComputeWorkgroupStorageSize) {
      throw new Error(
        `CAM++ ${storageDtype} tiled transits require ${tile2Bytes} workgroup bytes`,
      );
    }
    const outputTile: 2 | 4 =
      device.limits.maxComputeWorkgroupStorageSize >= tile2Bytes * 2 ? 4 : 2;
    const pipeline = await createPipeline(
      device,
      layout,
      outputTile,
      effectiveVariant,
      accumulation,
      storageDtype,
    );
    return new PointwiseTransitKernels(
      device,
      gpuPackage,
      arena,
      effectiveVariant,
      accumulation,
      outputTile,
      pipeline,
      layout,
    );
  }

  createDispatch(descriptor: PointwiseTransitDescriptor): PointwiseTransitDispatch {
    const storageBytes = campPlusStorageBytes(
      this.gpuPackage.metadata.contract.internalDtype,
    );
    validateDescriptor(descriptor, this.arena.byteLength, storageBytes);
    const weight = this.gpuPackage.section(descriptor.convolution.weight);
    const bias = this.gpuPackage.section(descriptor.convolution.bias);
    const affine = this.gpuPackage.section(descriptor.preactivationAffine);
    validateSections(descriptor, weight, bias, affine);
    const outputGroups = descriptor.outputChannels / 4;
    const outputTile = this.outputTile;
    if (outputGroups % outputTile !== 0) {
      throw new Error(`${descriptor.label} output groups are not divisible by tile ${outputTile}`);
    }
    const parameters = new Uint32Array([
      descriptor.input.byteOffset / storageBytes,
      descriptor.output.byteOffset / storageBytes,
      descriptor.batchSize,
      descriptor.inputChannels,
      descriptor.outputChannels,
      descriptor.frames,
      descriptor.inputChannels / 4,
      outputGroups,
      descriptor.inputStorageChannels,
      descriptor.outputStorageChannels,
      descriptor.outputRelu ? 1 : 0,
      0,
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
          { binding: 4, resource: { buffer: uniform, size: UNIFORM_BYTES } },
        ],
      });
      return new PointwiseTransitDispatch(
        descriptor.label,
        outputTile,
        this.pipeline,
        bindGroup,
        uniform,
        [outputGroups / outputTile, descriptor.batchSize, 1],
      );
    } catch (error) {
      uniform.destroy();
      throw error;
    }
  }
}

function validateDescriptor(
  descriptor: PointwiseTransitDescriptor,
  arenaBytes: number,
  storageBytes: 2 | 4,
): void {
  if (
    !Number.isSafeInteger(descriptor.batchSize) ||
    descriptor.batchSize <= 0 ||
    !Number.isSafeInteger(descriptor.inputChannels) ||
    descriptor.inputChannels <= 0 ||
    descriptor.inputChannels > MAX_INPUT_CHANNELS ||
    descriptor.inputChannels % 4 !== 0 ||
    !Number.isSafeInteger(descriptor.outputChannels) ||
    descriptor.outputChannels <= 0 ||
    descriptor.outputChannels % 16 !== 0 ||
    !Number.isSafeInteger(descriptor.frames) ||
    descriptor.frames <= 0 ||
    descriptor.frames > WORKGROUP_SIZE ||
    descriptor.inputStorageChannels < descriptor.inputChannels ||
    descriptor.outputStorageChannels < descriptor.outputChannels
  ) {
    throw new RangeError(`${descriptor.label} has an invalid tiled-transit contract`);
  }
  const inputBytes =
    descriptor.batchSize * descriptor.inputStorageChannels * descriptor.frames * storageBytes;
  const outputBytes =
    descriptor.batchSize * descriptor.outputStorageChannels * descriptor.frames * storageBytes;
  validateSlice(descriptor.input, inputBytes, arenaBytes);
  validateSlice(descriptor.output, outputBytes, arenaBytes);
  if (rangesOverlap(descriptor.input, descriptor.output)) {
    throw new Error(`${descriptor.label} transit input and output overlap`);
  }
}

function validateSections(
  descriptor: PointwiseTransitDescriptor,
  weight: CampPlusPackedSection,
  bias: CampPlusPackedSection,
  affine: CampPlusPackedSection,
): void {
  if (
    weight.kind !== "conv_weight" ||
    weight.layout !== "K_O4_I4_I_O" ||
    weight.logicalShape[0] !== descriptor.outputChannels ||
    weight.logicalShape[1] !== descriptor.inputChannels ||
    weight.logicalShape[2] !== 1 ||
    bias.kind !== "conv_bias" ||
    bias.logicalShape[0] !== descriptor.outputChannels ||
    affine.kind !== "batch_norm_affine" ||
    affine.logicalShape[0] !== descriptor.inputChannels
  ) {
    throw new Error(`${descriptor.label} packed sections do not match the tiled transit`);
  }
}

function validateSlice(
  slice: CampPlusArenaSlice,
  requiredBytes: number,
  arenaBytes: number,
): void {
  if (
    slice.byteOffset % 256 !== 0 ||
    slice.byteLength < requiredBytes ||
    slice.byteOffset + slice.byteLength > arenaBytes
  ) {
    throw new RangeError(`CAM++ arena slice ${slice.label} does not fit the tiled transit`);
  }
}

function rangesOverlap(left: CampPlusArenaSlice, right: CampPlusArenaSlice): boolean {
  return (
    left.byteOffset < right.byteOffset + right.byteLength &&
    right.byteOffset < left.byteOffset + left.byteLength
  );
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

async function createPipeline(
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  outputTile: 2 | 4,
  variant: PointwiseTransitVariant,
  accumulation: PointwiseTransitAccumulation,
  storageDtype: "float16" | "float32",
): Promise<GPUComputePipeline> {
  const label = `senko-campplus-pointwise-transit-tile${outputTile}-${variant}-${accumulation}`;
  const module = device.createShaderModule({
    label,
    code: campPlusStorageWgsl(
      variant === "chunk512"
        ? pointwiseTransitChunk512Wgsl(outputTile, accumulation)
        : pointwiseTransitWgsl(outputTile, accumulation),
      storageDtype,
    ),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length > 0) {
    throw new Error(`${label} WGSL failed: ${errors.map((item) => item.message).join("; ")}`);
  }
  return device.createComputePipelineAsync({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: "main" },
  });
}

function pointwiseAccumulationSteps(
  tile: number,
  cacheBase: string,
  accumulation: PointwiseTransitAccumulation,
): string {
  const activation = (lane: number): string =>
    accumulation === "float16"
      ? `vec4<f16>(activated_${lane})`
      : `vec4<f32>(f32(activated_${lane}))`;
  const weight = (offset: string): string => {
    const source = `weight_cache[${cacheBase}${offset}]`;
    return accumulation === "float16" ? source : `vec4<f32>(${source})`;
  };
  return `      accumulator_${tile} = fma(${activation(0)}, ${weight("")}, accumulator_${tile});
      accumulator_${tile} = fma(${activation(1)}, ${weight(" + 1u")}, accumulator_${tile});
      accumulator_${tile} = fma(${activation(2)}, ${weight(" + 2u")}, accumulator_${tile});
      accumulator_${tile} = fma(${activation(3)}, ${weight(" + 3u")}, accumulator_${tile});`;
}

function pointwiseAccumulatorDeclaration(
  indentation: string,
  tile: number,
  accumulation: PointwiseTransitAccumulation,
): string {
  const bias = `biases[first_output_group + ${tile}u]`;
  const initialValue = accumulation === "float16" ? bias : `vec4<f32>(${bias})`;
  return `${indentation}var accumulator_${tile} = ${initialValue};`;
}

export function pointwiseTransitWgsl(
  outputTile: 2 | 4,
  accumulation: PointwiseTransitAccumulation = DEFAULT_POINTWISE_TRANSIT_ACCUMULATION,
): string {
  const accumulatorType = accumulation === "float16" ? "f16" : "f32";
  const finishedAccumulator =
    accumulation === "float16" ? "accumulator" : "vec4<f16>(accumulator)";
  const accumulatorDeclarations = Array.from(
    { length: outputTile },
    (_, tile) => pointwiseAccumulatorDeclaration("    ", tile, accumulation),
  ).join("\n");
  const accumulationSteps = Array.from({ length: outputTile }, (_, tile) => {
    const cacheBase = `${tile}u * parameters.input_channels + channel_base`;
    return pointwiseAccumulationSteps(tile, cacheBase, accumulation);
  }).join("\n");
  const stores = Array.from(
    { length: outputTile },
    (_, tile) =>
      `    store_output(first_output_group + ${tile}u, batch, frame, finish_output(accumulator_${tile}));`,
  ).join("\n");
  return /* wgsl */ `
enable f16;

struct Parameters {
  input_offset: u32,
  output_offset: u32,
  batch_size: u32,
  input_channels: u32,
  output_channels: u32,
  frames: u32,
  input_groups: u32,
  output_groups: u32,
  input_storage_channels: u32,
  output_storage_channels: u32,
  output_relu: u32,
  r0: u32, r1: u32, r2: u32, r3: u32, r4: u32,
  r5: u32, r6: u32, r7: u32, r8: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> affine: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;
var<workgroup> weight_cache: array<vec4<f16>, ${MAX_INPUT_CHANNELS * outputTile}>;

fn finish_output(accumulator: vec4<${accumulatorType}>) -> vec4<f16> {
  var rounded = ${finishedAccumulator};
  if (parameters.output_relu != 0u) {
    rounded = max(rounded, vec4<f16>(f16(0.0)));
  }
  return rounded;
}

fn store_output(output_group: u32, batch: u32, frame: u32, value: vec4<f16>) {
  let output_channel = output_group * 4u;
  let base =
    parameters.output_offset +
    ((batch * parameters.output_storage_channels + output_channel) * parameters.frames + frame);
  arena[base] = value[0];
  arena[base + parameters.frames] = value[1];
  arena[base + 2u * parameters.frames] = value[2];
  arena[base + 3u * parameters.frames] = value[3];
}

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let first_output_group = group.x * ${outputTile}u;
  let batch = group.y;
  let cached_vectors = parameters.input_channels * ${outputTile}u;
  var cache_index = local_id.x;
  while (cache_index < cached_vectors) {
    let tile = cache_index / parameters.input_channels;
    let input_channel = cache_index - tile * parameters.input_channels;
    let input_group = input_channel / 4u;
    let input_lane = input_channel & 3u;
    let output_group = first_output_group + tile;
    let packed_index =
      (output_group * parameters.input_groups + input_group) * 4u + input_lane;
    weight_cache[cache_index] = weights[packed_index];
    cache_index += 128u;
  }
  workgroupBarrier();

  let frame = local_id.x;
  if (frame < parameters.frames && batch < parameters.batch_size) {
${accumulatorDeclarations}
    let batch_channel_base = batch * parameters.input_storage_channels;
    for (var input_group = 0u; input_group < parameters.input_groups; input_group += 1u) {
      let channel_base = input_group * 4u;
      let input_index =
        parameters.input_offset +
        ((batch_channel_base + channel_base) * parameters.frames + frame);
      let scale = affine[input_group * 2u];
      let shift = affine[input_group * 2u + 1u];
      let activated_0 = max(f16(f32(arena[input_index]) * scale[0] + shift[0]), f16(0.0));
      let activated_1 = max(f16(f32(arena[input_index + parameters.frames]) * scale[1] + shift[1]), f16(0.0));
      let activated_2 = max(f16(f32(arena[input_index + 2u * parameters.frames]) * scale[2] + shift[2]), f16(0.0));
      let activated_3 = max(f16(f32(arena[input_index + 3u * parameters.frames]) * scale[3] + shift[3]), f16(0.0));
${accumulationSteps}
    }
${stores}
  }
}
`;
}

/**
 * Tile-4 transit with a 512-channel strip-mined weight cache. This cuts
 * workgroup storage from 32 KiB to 16 KiB while preserving input/FMA order.
 */
export function pointwiseTransitChunk512Wgsl(
  outputTile: 2 | 4,
  accumulation: PointwiseTransitAccumulation = DEFAULT_POINTWISE_TRANSIT_ACCUMULATION,
): string {
  const cacheChannels = 512;
  const accumulatorType = accumulation === "float16" ? "f16" : "f32";
  const finishedAccumulator =
    accumulation === "float16" ? "accumulator" : "vec4<f16>(accumulator)";
  const accumulatorDeclarations = Array.from(
    { length: outputTile },
    (_, tile) => pointwiseAccumulatorDeclaration("  ", tile, accumulation),
  ).join("\n");
  const accumulationSteps = Array.from({ length: outputTile }, (_, tile) => {
    const cacheBase = `${tile}u * ${cacheChannels}u + cache_channel_base`;
    return pointwiseAccumulationSteps(tile, cacheBase, accumulation);
  }).join("\n");
  const stores = Array.from(
    { length: outputTile },
    (_, tile) =>
      `    store_output(first_output_group + ${tile}u, batch, frame, finish_output(accumulator_${tile}));`,
  ).join("\n");
  return /* wgsl */ `
enable f16;

struct Parameters {
  input_offset: u32,
  output_offset: u32,
  batch_size: u32,
  input_channels: u32,
  output_channels: u32,
  frames: u32,
  input_groups: u32,
  output_groups: u32,
  input_storage_channels: u32,
  output_storage_channels: u32,
  output_relu: u32,
  r0: u32, r1: u32, r2: u32, r3: u32, r4: u32,
  r5: u32, r6: u32, r7: u32, r8: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> affine: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;
var<workgroup> weight_cache: array<vec4<f16>, ${cacheChannels * outputTile}>;

fn finish_output(accumulator: vec4<${accumulatorType}>) -> vec4<f16> {
  var rounded = ${finishedAccumulator};
  if (parameters.output_relu != 0u) {
    rounded = max(rounded, vec4<f16>(f16(0.0)));
  }
  return rounded;
}

fn store_output(output_group: u32, batch: u32, frame: u32, value: vec4<f16>) {
  let output_channel = output_group * 4u;
  let base =
    parameters.output_offset +
    ((batch * parameters.output_storage_channels + output_channel) * parameters.frames + frame);
  arena[base] = value[0];
  arena[base + parameters.frames] = value[1];
  arena[base + 2u * parameters.frames] = value[2];
  arena[base + 3u * parameters.frames] = value[3];
}

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let first_output_group = group.x * ${outputTile}u;
  let batch = group.y;
  let frame = local_id.x;
  let frame_active = frame < parameters.frames && batch < parameters.batch_size;
${accumulatorDeclarations}
  let batch_channel_base = batch * parameters.input_storage_channels;
  var chunk_start = 0u;
  loop {
    let chunk_end = min(chunk_start + ${cacheChannels}u, parameters.input_channels);
    let chunk_channels = chunk_end - chunk_start;
    var cache_index = local_id.x;
    while (cache_index < ${cacheChannels * outputTile}u) {
      let tile = cache_index / ${cacheChannels}u;
      let chunk_channel = cache_index - tile * ${cacheChannels}u;
      if (chunk_channel < chunk_channels) {
        let input_channel = chunk_start + chunk_channel;
        let input_group = input_channel / 4u;
        let input_lane = input_channel & 3u;
        let output_group = first_output_group + tile;
        let packed_index =
          (output_group * parameters.input_groups + input_group) * 4u + input_lane;
        weight_cache[cache_index] = weights[packed_index];
      }
      cache_index += 128u;
    }
    workgroupBarrier();

    if (frame_active) {
      let first_input_group = chunk_start / 4u;
      let end_input_group = chunk_end / 4u;
      for (var input_group = first_input_group; input_group < end_input_group; input_group += 1u) {
        let channel_base = input_group * 4u;
        let cache_channel_base = channel_base - chunk_start;
        let input_index =
          parameters.input_offset +
          ((batch_channel_base + channel_base) * parameters.frames + frame);
        let scale = affine[input_group * 2u];
        let shift = affine[input_group * 2u + 1u];
        let activated_0 = max(f16(f32(arena[input_index]) * scale[0] + shift[0]), f16(0.0));
        let activated_1 = max(f16(f32(arena[input_index + parameters.frames]) * scale[1] + shift[1]), f16(0.0));
        let activated_2 = max(f16(f32(arena[input_index + 2u * parameters.frames]) * scale[2] + shift[2]), f16(0.0));
        let activated_3 = max(f16(f32(arena[input_index + 3u * parameters.frames]) * scale[3] + shift[3]), f16(0.0));
${accumulationSteps}
      }
    }
    if (chunk_end == parameters.input_channels) { break; }
    workgroupBarrier();
    chunk_start = chunk_end;
  }
  if (frame_active) {
${stores}
  }
}
`;
}
