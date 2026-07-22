/// <reference types="@webgpu/types" />

import { CampPlusActivationArena, type CampPlusArenaSlice } from "./arena";
import type {
  CampPlusPackedSection,
  PackedConvolutionRef,
} from "./metadata";
import { CampPlusGpuPackage } from "./package";

const CACHED_WORKGROUP_SIZE = 128;
const MAX_CACHED_WEIGHT_VECTORS = 1600;
export const PACKED_BCT_REQUIRED_WORKGROUP_STORAGE_BYTES =
  MAX_CACHED_WEIGHT_VECTORS * 8;
const UNIFORM_BYTES = 80;

export const PACKED_BCT_CONV_VARIANTS = [
  "cached-tile1-wg128",
  "direct-tile2-wg96",
  "direct-tile4-wg96",
  "direct-tile8-wg96",
  "direct-tile4-wg128",
] as const;

export type PackedBctConvVariant = (typeof PACKED_BCT_CONV_VARIANTS)[number];
export type PackedBctConvOutputTile = 1 | 2 | 4 | 8;
export type PackedBctConvWorkgroupSize = 96 | 128;
export type PackedBctConvWeightSource = "workgroup-cache" | "direct";

export interface PackedBctConvVariantConfiguration {
  readonly outputTile: PackedBctConvOutputTile;
  readonly workgroupSize: PackedBctConvWorkgroupSize;
  readonly weightSource: PackedBctConvWeightSource;
  readonly workgroupStorageBytes: number;
}

/** Retained byte-for-byte packed-convolution baseline for diagnostic A/Bs. */
export const LEGACY_PACKED_BCT_CONV_VARIANT: PackedBctConvVariant =
  "cached-tile1-wg128";

/** Best pooled B16 graph variant measured on the target Apple M3. */
export const DEFAULT_PACKED_BCT_CONV_VARIANT: PackedBctConvVariant =
  "direct-tile8-wg96";

const PACKED_BCT_CONV_VARIANT_CONFIGURATIONS: Readonly<
  Record<PackedBctConvVariant, PackedBctConvVariantConfiguration>
> = {
  "cached-tile1-wg128": {
    outputTile: 1,
    workgroupSize: CACHED_WORKGROUP_SIZE,
    weightSource: "workgroup-cache",
    workgroupStorageBytes: PACKED_BCT_REQUIRED_WORKGROUP_STORAGE_BYTES,
  },
  "direct-tile2-wg96": {
    outputTile: 2,
    workgroupSize: 96,
    weightSource: "direct",
    workgroupStorageBytes: 0,
  },
  "direct-tile4-wg96": {
    outputTile: 4,
    workgroupSize: 96,
    weightSource: "direct",
    workgroupStorageBytes: 0,
  },
  "direct-tile8-wg96": {
    outputTile: 8,
    workgroupSize: 96,
    weightSource: "direct",
    workgroupStorageBytes: 0,
  },
  "direct-tile4-wg128": {
    outputTile: 4,
    workgroupSize: 128,
    weightSource: "direct",
    workgroupStorageBytes: 0,
  },
};

export function isPackedBctConvVariant(value: string): value is PackedBctConvVariant {
  return (PACKED_BCT_CONV_VARIANTS as readonly string[]).includes(value);
}

export function packedBctConvVariantConfiguration(
  variant: PackedBctConvVariant,
): PackedBctConvVariantConfiguration {
  return PACKED_BCT_CONV_VARIANT_CONFIGURATIONS[variant];
}

export function packedBctConvDispatchWorkgroups(
  variant: PackedBctConvVariant,
  outputGroups: number,
  batchSize: number,
  outputFrames: number,
): readonly [number, number, number] {
  const configuration = packedBctConvVariantConfiguration(variant);
  if (
    !Number.isSafeInteger(outputGroups) ||
    outputGroups <= 0 ||
    outputGroups % configuration.outputTile !== 0
  ) {
    throw new RangeError(
      `Packed convolution output groups must be a positive multiple of tile ${configuration.outputTile}`,
    );
  }
  if (!Number.isSafeInteger(batchSize) || batchSize <= 0) {
    throw new RangeError("Packed convolution batch size must be a positive integer");
  }
  if (!Number.isSafeInteger(outputFrames) || outputFrames <= 0) {
    throw new RangeError("Packed convolution output frames must be a positive integer");
  }
  return [
    outputGroups / configuration.outputTile,
    batchSize,
    ceilDiv(outputFrames, configuration.workgroupSize),
  ];
}

export interface PackedBctConvDescriptor {
  readonly label: string;
  readonly convolution: PackedConvolutionRef;
  readonly input: CampPlusArenaSlice;
  readonly output: CampPlusArenaSlice;
  readonly batchSize: number;
  readonly inputChannels: number;
  readonly inputFrames: number;
  readonly outputFrames: number;
  readonly stride: number;
  readonly dilation: number;
  readonly padLeft: number;
  readonly padRight: number;
  readonly preactivationAffine?: string;
  readonly outputRelu: boolean;
  readonly inputStorageChannels?: number;
  readonly outputStorageChannels?: number;
  readonly inputChannelOffset?: number;
  readonly outputChannelOffset?: number;
}

export class PackedBctConvDispatch {
  readonly gpuBufferBytes = UNIFORM_BYTES;
  private destroyed = false;

  constructor(
    readonly label: string,
    readonly output: CampPlusArenaSlice,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly uniformBuffer: GPUBuffer,
    private readonly workgroups: readonly [number, number, number],
  ) {}

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    if (this.destroyed) throw new Error(`CAM++ dispatch ${this.label} has been destroyed`);
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
    this.uniformBuffer.destroy();
  }
}

/** Packed FP16 BCT convolution used by TDNN, dense bottlenecks, and transits. */
export class PackedBctConvKernel {
  private constructor(
    private readonly device: GPUDevice,
    private readonly gpuPackage: CampPlusGpuPackage,
    private readonly arena: CampPlusActivationArena,
    readonly variant: PackedBctConvVariant,
    private readonly configuration: PackedBctConvVariantConfiguration,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroupLayout: GPUBindGroupLayout,
  ) {}

  static async create(
    device: GPUDevice,
    gpuPackage: CampPlusGpuPackage,
    arena: CampPlusActivationArena,
    variant: PackedBctConvVariant = DEFAULT_PACKED_BCT_CONV_VARIANT,
  ): Promise<PackedBctConvKernel> {
    const configuration = packedBctConvVariantConfiguration(variant);
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < configuration.workgroupSize ||
      device.limits.maxComputeWorkgroupSizeX < configuration.workgroupSize
    ) {
      throw new Error(
        `Raw CAM++ ${variant} requires ${configuration.workgroupSize} compute invocations on workgroup X`,
      );
    }
    if (
      device.limits.maxComputeWorkgroupStorageSize <
      configuration.workgroupStorageBytes
    ) {
      throw new Error(
        `Raw CAM++ ${variant} requires ${configuration.workgroupStorageBytes} workgroup bytes`,
      );
    }
    const label = `senko-campplus-packed-bct-conv-${variant}`;
    const module = device.createShaderModule({
      label,
      code:
        configuration.weightSource === "workgroup-cache"
          ? PACKED_BCT_CONV_WGSL
          : packedBctDirectWgsl(
              configuration.outputTile,
              configuration.workgroupSize,
            ),
    });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `CAM++ packed convolution WGSL failed: ${errors.map((item) => item.message).join("; ")}`,
      );
    }
    const bindGroupLayout = device.createBindGroupLayout({
      label: `${label}-bindings`,
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", minBindingSize: UNIFORM_BYTES },
        },
      ],
    });
    const pipeline = await device.createComputePipelineAsync({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: "main" },
    });
    return new PackedBctConvKernel(
      device,
      gpuPackage,
      arena,
      variant,
      configuration,
      pipeline,
      bindGroupLayout,
    );
  }

  createDispatch(descriptor: PackedBctConvDescriptor): PackedBctConvDispatch {
    const weight = this.gpuPackage.section(descriptor.convolution.weight);
    const bias = this.gpuPackage.section(descriptor.convolution.bias);
    validateConvolutionSections(weight, bias, descriptor);
    validateDescriptor(descriptor, weight, this.arena.byteLength);
    const affine =
      descriptor.preactivationAffine === undefined
        ? undefined
        : this.gpuPackage.section(descriptor.preactivationAffine);
    if (affine !== undefined) validateAffine(affine, descriptor.inputChannels);

    const outputChannels = weight.logicalShape[0]!;
    const inputGroups = ceilDiv(descriptor.inputChannels, 4);
    const outputGroups = ceilDiv(outputChannels, 4);
    const kernelElements = product(weight.logicalShape.slice(2));
    const paddedInputChannels = inputGroups * 4;
    const cachedVectors = kernelElements * paddedInputChannels;
    if (
      this.configuration.weightSource === "workgroup-cache" &&
      cachedVectors > MAX_CACHED_WEIGHT_VECTORS
    ) {
      throw new Error(
        `${descriptor.label} needs ${cachedVectors} cached vec4 weights; maximum is ${MAX_CACHED_WEIGHT_VECTORS}`,
      );
    }
    if (outputGroups % this.configuration.outputTile !== 0) {
      throw new Error(
        `${descriptor.label} output groups are not divisible by tile ${this.configuration.outputTile}`,
      );
    }

    const inputStorageChannels = descriptor.inputStorageChannels ?? descriptor.inputChannels;
    const outputStorageChannels = descriptor.outputStorageChannels ?? outputChannels;
    const inputChannelOffset = descriptor.inputChannelOffset ?? 0;
    const outputChannelOffset = descriptor.outputChannelOffset ?? 0;
    const parameters = new Uint32Array([
      descriptor.input.byteOffset / 2,
      descriptor.output.byteOffset / 2,
      descriptor.batchSize,
      descriptor.inputChannels,
      outputChannels,
      descriptor.inputFrames,
      descriptor.outputFrames,
      kernelElements,
      descriptor.stride,
      descriptor.dilation,
      descriptor.padLeft,
      inputGroups,
      outputGroups,
      paddedInputChannels,
      affine === undefined ? 0 : 1,
      descriptor.outputRelu ? 1 : 0,
      inputStorageChannels,
      outputStorageChannels,
      inputChannelOffset,
      outputChannelOffset,
    ]);
    const uniformBuffer = this.device.createBuffer({
      label: `${descriptor.label}-parameters`,
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, parameters);
    try {
      const bindGroup = this.device.createBindGroup({
        label: `${descriptor.label}-bindings`,
        layout: this.bindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: { buffer: this.arena.buffer, size: this.arena.byteLength },
          },
          {
            binding: 1,
            resource: {
              buffer: this.gpuPackage.weightsBuffer,
              offset: weight.byteOffset,
              size: weight.byteLength,
            },
          },
          {
            binding: 2,
            resource: {
              buffer: this.gpuPackage.weightsBuffer,
              offset: bias.byteOffset,
              size: bias.byteLength,
            },
          },
          {
            binding: 3,
            resource:
              affine === undefined
                ? { buffer: this.gpuPackage.weightsBuffer, offset: 0, size: 256 }
                : {
                    buffer: this.gpuPackage.weightsBuffer,
                    offset: affine.byteOffset,
                    size: affine.byteLength,
                  },
          },
          { binding: 4, resource: { buffer: uniformBuffer, size: UNIFORM_BYTES } },
        ],
      });
      return new PackedBctConvDispatch(
        descriptor.label,
        descriptor.output,
        this.pipeline,
        bindGroup,
        uniformBuffer,
        packedBctConvDispatchWorkgroups(
          this.variant,
          outputGroups,
          descriptor.batchSize,
          descriptor.outputFrames,
        ),
      );
    } catch (error) {
      uniformBuffer.destroy();
      throw error;
    }
  }
}

function validateConvolutionSections(
  weight: CampPlusPackedSection,
  bias: CampPlusPackedSection,
  descriptor: PackedBctConvDescriptor,
): void {
  if (weight.kind !== "conv_weight" || weight.layout !== "K_O4_I4_I_O") {
    throw new Error(`${descriptor.label} weight section is not a packed convolution`);
  }
  if (bias.kind !== "conv_bias" || bias.layout !== "O4") {
    throw new Error(`${descriptor.label} bias section is not a packed convolution bias`);
  }
  if (
    weight.logicalShape[0] !== bias.logicalShape[0] ||
    weight.logicalShape[1] !== descriptor.inputChannels
  ) {
    throw new Error(`${descriptor.label} channel contract does not match its packed sections`);
  }
}

function validateDescriptor(
  descriptor: PackedBctConvDescriptor,
  weight: CampPlusPackedSection,
  arenaBytes: number,
): void {
  const integerFields = [
    descriptor.batchSize,
    descriptor.inputChannels,
    descriptor.inputFrames,
    descriptor.outputFrames,
    descriptor.stride,
    descriptor.dilation,
  ];
  if (integerFields.some((value) => !Number.isSafeInteger(value) || value <= 0)) {
    throw new RangeError(`${descriptor.label} has non-positive convolution dimensions`);
  }
  if (
    !Number.isSafeInteger(descriptor.padLeft) ||
    !Number.isSafeInteger(descriptor.padRight) ||
    descriptor.padLeft < 0 ||
    descriptor.padRight < 0
  ) {
    throw new RangeError(`${descriptor.label} has invalid padding`);
  }
  if (weight.logicalShape.length !== 3) {
    throw new Error(`${descriptor.label} currently supports packed Conv1d weights only`);
  }
  const kernel = weight.logicalShape[2]!;
  const expectedFrames =
    Math.floor(
      (descriptor.inputFrames +
        descriptor.padLeft +
        descriptor.padRight -
        descriptor.dilation * (kernel - 1) -
        1) /
        descriptor.stride,
    ) + 1;
  if (descriptor.outputFrames !== expectedFrames) {
    throw new Error(
      `${descriptor.label} output frames ${descriptor.outputFrames} != convolution result ${expectedFrames}`,
    );
  }
  const outputChannels = weight.logicalShape[0]!;
  const inputStorageChannels = descriptor.inputStorageChannels ?? descriptor.inputChannels;
  const outputStorageChannels = descriptor.outputStorageChannels ?? outputChannels;
  const inputChannelOffset = descriptor.inputChannelOffset ?? 0;
  const outputChannelOffset = descriptor.outputChannelOffset ?? 0;
  if (
    !Number.isSafeInteger(inputStorageChannels) ||
    !Number.isSafeInteger(outputStorageChannels) ||
    !Number.isSafeInteger(inputChannelOffset) ||
    !Number.isSafeInteger(outputChannelOffset) ||
    inputChannelOffset < 0 ||
    outputChannelOffset < 0 ||
    inputChannelOffset + descriptor.inputChannels > inputStorageChannels ||
    outputChannelOffset + outputChannels > outputStorageChannels
  ) {
    throw new RangeError(`${descriptor.label} has an invalid physical channel stride`);
  }
  const inputBytes = descriptor.batchSize * inputStorageChannels * descriptor.inputFrames * 2;
  const outputBytes = descriptor.batchSize * outputStorageChannels * descriptor.outputFrames * 2;
  validateArenaRange(descriptor.input, inputBytes, arenaBytes);
  validateArenaRange(descriptor.output, outputBytes, arenaBytes);
  if (rangesOverlap(descriptor.input, descriptor.output)) {
    throw new Error(`${descriptor.label} input and output ranges overlap within one dispatch`);
  }
}

function validateAffine(section: CampPlusPackedSection, inputChannels: number): void {
  if (
    section.kind !== "batch_norm_affine" ||
    section.layout !== "C4_SCALE_SHIFT" ||
    section.logicalShape[0] !== inputChannels
  ) {
    throw new Error("CAM++ preactivation affine does not match convolution input channels");
  }
}

function validateArenaRange(
  slice: CampPlusArenaSlice,
  requiredBytes: number,
  arenaBytes: number,
): void {
  if (
    slice.byteOffset < 0 ||
    slice.byteLength < requiredBytes ||
    slice.byteOffset + slice.byteLength > arenaBytes ||
    slice.byteOffset % 256 !== 0
  ) {
    throw new RangeError(`CAM++ arena slice ${slice.label} does not fit its tensor`);
  }
}

function rangesOverlap(left: CampPlusArenaSlice, right: CampPlusArenaSlice): boolean {
  return (
    left.byteOffset < right.byteOffset + right.byteLength &&
    right.byteOffset < left.byteOffset + left.byteLength
  );
}

function ceilDiv(value: number, divisor: number): number {
  return Math.floor((value + divisor - 1) / divisor);
}

function product(values: readonly number[]): number {
  return values.reduce((result, value) => result * value, 1);
}

/**
 * Direct-weight convolution sharing each input/affine evaluation across an
 * output tile. Each accumulator still observes kernel, channel, rounding, and
 * store operations in exactly the same order as the cached tile-1 kernel.
 */
export function packedBctDirectWgsl(
  outputTile: PackedBctConvOutputTile,
  workgroupSize: PackedBctConvWorkgroupSize,
): string {
  if (![1, 2, 4, 8].includes(outputTile)) {
    throw new RangeError(`Unsupported packed convolution output tile ${outputTile}`);
  }
  if (workgroupSize !== 96 && workgroupSize !== 128) {
    throw new RangeError(`Unsupported packed convolution workgroup size ${workgroupSize}`);
  }
  const accumulatorDeclarations = Array.from(
    { length: outputTile },
    (_, tile) =>
      `  var accumulator_${tile} = vec4<f32>(biases[first_output_group + ${tile}u]);`,
  ).join("\n");
  const inputLaneSteps = Array.from({ length: 4 }, (_, lane) => {
    const accumulationSteps = Array.from({ length: outputTile }, (_, tile) => {
      return `        let weight_index_${tile}_${lane} =
          (((kernel_index * parameters.output_groups + first_output_group + ${tile}u) *
            parameters.input_groups + input_group) * 4u + ${lane}u);
        accumulator_${tile} = fma(
          vec4<f32>(input_value),
          vec4<f32>(weights[weight_index_${tile}_${lane}]),
          accumulator_${tile},
        );`;
    }).join("\n");
    return `      if (channel_base + ${lane}u < parameters.input_channels) {
        let input_channel = channel_base + ${lane}u;
        let input_index =
          parameters.input_offset +
          ((batch * parameters.input_storage_channels + parameters.input_channel_offset +
            input_channel) * parameters.input_frames + u32(source_frame));
        var input_value = f32(arena[input_index]);
        if (parameters.has_affine != 0u) {
          let scale = affine[input_group * 2u][${lane}];
          let shift = affine[input_group * 2u + 1u][${lane}];
          input_value = max(f32(f16(input_value * scale + shift)), 0.0);
        }
${accumulationSteps}
      }`;
  }).join("\n");
  const roundedDeclarations = Array.from(
    { length: outputTile },
    (_, tile) => `  var rounded_${tile} = vec4<f16>(accumulator_${tile});
  if (parameters.output_relu != 0u) {
    rounded_${tile} = max(rounded_${tile}, vec4<f16>(f16(0.0)));
  }`,
  ).join("\n");
  const outputStores = Array.from({ length: outputTile }, (_, tile) => {
    return `  let output_channel_base_${tile} = (first_output_group + ${tile}u) * 4u;
  for (var lane_${tile} = 0u; lane_${tile} < 4u; lane_${tile} += 1u) {
    let output_channel = output_channel_base_${tile} + lane_${tile};
    if (output_channel < parameters.output_channels) {
      let output_index =
        parameters.output_offset +
        ((batch * parameters.output_storage_channels + parameters.output_channel_offset +
          output_channel) * parameters.output_frames + output_frame);
      arena[output_index] = rounded_${tile}[lane_${tile}];
    }
  }`;
  }).join("\n");

  return /* wgsl */ `
enable f16;

struct Parameters {
  input_offset: u32,
  output_offset: u32,
  batch_size: u32,
  input_channels: u32,
  output_channels: u32,
  input_frames: u32,
  output_frames: u32,
  kernel_elements: u32,
  stride: u32,
  dilation: u32,
  pad_left: u32,
  input_groups: u32,
  output_groups: u32,
  padded_input_channels: u32,
  has_affine: u32,
  output_relu: u32,
  input_storage_channels: u32,
  output_storage_channels: u32,
  input_channel_offset: u32,
  output_channel_offset: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> affine: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let first_output_group = workgroup_id.x * ${outputTile}u;
  let batch = workgroup_id.y;
  let output_frame = workgroup_id.z * ${workgroupSize}u + local_id.x;
  if (output_frame >= parameters.output_frames || batch >= parameters.batch_size) {
    return;
  }

${accumulatorDeclarations}
  for (var kernel_index = 0u; kernel_index < parameters.kernel_elements; kernel_index += 1u) {
    let source_frame =
      i32(output_frame * parameters.stride + kernel_index * parameters.dilation) -
      i32(parameters.pad_left);
    if (source_frame < 0 || source_frame >= i32(parameters.input_frames)) {
      continue;
    }
    for (var input_group = 0u; input_group < parameters.input_groups; input_group += 1u) {
      let channel_base = input_group * 4u;
${inputLaneSteps}
    }
  }

${roundedDeclarations}
${outputStores}
}
`;
}

export const PACKED_BCT_CONV_WGSL = /* wgsl */ `
enable f16;

struct Parameters {
  input_offset: u32,
  output_offset: u32,
  batch_size: u32,
  input_channels: u32,
  output_channels: u32,
  input_frames: u32,
  output_frames: u32,
  kernel_elements: u32,
  stride: u32,
  dilation: u32,
  pad_left: u32,
  input_groups: u32,
  output_groups: u32,
  padded_input_channels: u32,
  has_affine: u32,
  output_relu: u32,
  input_storage_channels: u32,
  output_storage_channels: u32,
  input_channel_offset: u32,
  output_channel_offset: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> affine: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

var<workgroup> weight_cache: array<vec4<f16>, 1600>;

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let output_group = workgroup_id.x;
  let batch = workgroup_id.y;
  let cached_vectors = parameters.kernel_elements * parameters.padded_input_channels;
  var cache_index = local_id.x;
  while (cache_index < cached_vectors) {
    let kernel_index = cache_index / parameters.padded_input_channels;
    let input_scalar = cache_index - kernel_index * parameters.padded_input_channels;
    let input_group = input_scalar / 4u;
    let input_lane = input_scalar & 3u;
    let packed_index =
      (((kernel_index * parameters.output_groups + output_group) * parameters.input_groups +
        input_group) * 4u + input_lane);
    weight_cache[cache_index] = weights[packed_index];
    cache_index += 128u;
  }
  workgroupBarrier();

  let output_frame = workgroup_id.z * 128u + local_id.x;
  if (output_frame >= parameters.output_frames || batch >= parameters.batch_size) {
    return;
  }

  var accumulators = vec4<f32>(biases[output_group]);
  for (var kernel_index = 0u; kernel_index < parameters.kernel_elements; kernel_index += 1u) {
    let source_frame =
      i32(output_frame * parameters.stride + kernel_index * parameters.dilation) -
      i32(parameters.pad_left);
    if (source_frame < 0 || source_frame >= i32(parameters.input_frames)) {
      continue;
    }
    for (var input_channel = 0u; input_channel < parameters.input_channels; input_channel += 1u) {
      let input_index =
        parameters.input_offset +
        ((batch * parameters.input_storage_channels + parameters.input_channel_offset + input_channel) * parameters.input_frames +
          u32(source_frame));
      var input_value = f32(arena[input_index]);
      if (parameters.has_affine != 0u) {
        let input_group = input_channel / 4u;
        let input_lane = input_channel & 3u;
        let scale = affine[input_group * 2u][input_lane];
        let shift = affine[input_group * 2u + 1u][input_lane];
        input_value = max(f32(f16(input_value * scale + shift)), 0.0);
      }
      let cached_index = kernel_index * parameters.padded_input_channels + input_channel;
      accumulators = fma(
        vec4<f32>(input_value),
        vec4<f32>(weight_cache[cached_index]),
        accumulators,
      );
    }
  }

  var rounded = vec4<f16>(accumulators);
  if (parameters.output_relu != 0u) {
    rounded = max(rounded, vec4<f16>(f16(0.0)));
  }
  let output_channel_base = output_group * 4u;
  for (var lane = 0u; lane < 4u; lane += 1u) {
    let output_channel = output_channel_base + lane;
    if (output_channel < parameters.output_channels) {
      let output_index =
        parameters.output_offset +
        ((batch * parameters.output_storage_channels + parameters.output_channel_offset + output_channel) * parameters.output_frames +
          output_frame);
      arena[output_index] = rounded[lane];
    }
  }
}
`;
