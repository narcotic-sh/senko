/// <reference types="@webgpu/types" />

import { CampPlusActivationArena, type CampPlusArenaSlice } from "./arena";
import type { CampPlusPackedSection, PackedConvolutionRef } from "./metadata";
import { CampPlusGpuPackage } from "./package";
import { campPlusStorageBytes, campPlusStorageWgsl } from "./storage";

const CHANNELS = 32;
const OUTPUT_GROUPS = CHANNELS / 4;
const FRAMES = 150;
const WORKGROUP_SIZE = 128;
export const FCM_DISPATCH_GPU_BUFFER_BYTES = 64;

export const FCM_VARIANTS = [
  "tile1-split",
  "tile1-fold",
  "tile2-fold",
  "tile4-fold",
] as const;

export type FcmVariant = (typeof FCM_VARIANTS)[number];
export type FcmOutputTile = 1 | 2 | 4;
export type FcmAccumulation = "float32" | "float16";

/** Retained byte-for-byte baseline used by the diagnostic A/B. */
export const LEGACY_FCM_VARIANT: FcmVariant = "tile1-split";

/** Best pooled whole-graph B16 variant measured on the target Apple M3. */
export const DEFAULT_FCM_VARIANT: FcmVariant = "tile4-fold";
export const DEFAULT_FCM_ACCUMULATION: FcmAccumulation = "float16";

export interface FcmVariantConfiguration {
  readonly outputTile: FcmOutputTile;
  readonly foldTimeTail: boolean;
  readonly firstWorkgroupStorageBytes: number;
  readonly convWorkgroupStorageBytes: number;
}

const FCM_VARIANT_CONFIGURATIONS = {
  "tile1-split": { outputTile: 1, foldTimeTail: false },
  "tile1-fold": { outputTile: 1, foldTimeTail: true },
  "tile2-fold": { outputTile: 2, foldTimeTail: true },
  "tile4-fold": { outputTile: 4, foldTimeTail: true },
} as const satisfies Record<
  FcmVariant,
  Pick<FcmVariantConfiguration, "outputTile" | "foldTimeTail">
>;

export function isFcmVariant(value: string): value is FcmVariant {
  return (FCM_VARIANTS as readonly string[]).includes(value);
}

export function fcmVariantConfiguration(
  variant: FcmVariant,
): FcmVariantConfiguration {
  const configuration = FCM_VARIANT_CONFIGURATIONS[variant];
  return {
    ...configuration,
    firstWorkgroupStorageBytes: configuration.outputTile * 9 * 8,
    convWorkgroupStorageBytes: configuration.outputTile * 320 * 8,
  };
}

export function fcmDispatchWorkgroups(
  variant: FcmVariant,
  batchSize: number,
  outputFreq: number,
): readonly [number, number, number] {
  if (!Number.isSafeInteger(batchSize) || batchSize <= 0) {
    throw new RangeError("FCM batch size must be a positive integer");
  }
  if (!Number.isSafeInteger(outputFreq) || outputFreq <= 0) {
    throw new RangeError("FCM output frequency must be a positive integer");
  }
  const configuration = fcmVariantConfiguration(variant);
  return [
    OUTPUT_GROUPS / configuration.outputTile,
    batchSize * outputFreq,
    configuration.foldTimeTail ? 1 : ceilDiv(FRAMES, WORKGROUP_SIZE),
  ];
}

export type FcmResidual =
  | { readonly kind: "none" }
  | { readonly kind: "identity"; readonly input: CampPlusArenaSlice }
  | {
      readonly kind: "learned";
      readonly input: CampPlusArenaSlice;
      readonly inputFreq: number;
      readonly strideFreq: number;
      readonly convolution: PackedConvolutionRef;
    };

export interface FcmFirstConvDescriptor {
  readonly label: string;
  readonly convolution: PackedConvolutionRef;
  readonly input: GPUBuffer;
  readonly output: CampPlusArenaSlice;
  readonly batchSize: number;
}

export interface FcmConvDescriptor {
  readonly label: string;
  readonly convolution: PackedConvolutionRef;
  readonly input: CampPlusArenaSlice;
  readonly inputFreq: number;
  readonly output: CampPlusArenaSlice;
  readonly outputFreq: number;
  readonly strideFreq: number;
  readonly batchSize: number;
  readonly residual: FcmResidual;
  readonly outputRelu: boolean;
}

export class FcmDispatch {
  readonly gpuBufferBytes = FCM_DISPATCH_GPU_BUFFER_BYTES;
  private destroyed = false;

  constructor(
    readonly label: string,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    private readonly uniform: GPUBuffer,
    private readonly workgroups: readonly [number, number, number],
  ) {}

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    if (this.destroyed) throw new Error(`CAM++ FCM dispatch ${this.label} is destroyed`);
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

/** Specialized BCFT kernels for the 12-convolution FCM residual front end. */
export class FcmKernels {
  private constructor(
    private readonly device: GPUDevice,
    private readonly gpuPackage: CampPlusGpuPackage,
    private readonly arena: CampPlusActivationArena,
    private readonly firstPipeline: GPUComputePipeline,
    private readonly firstLayout: GPUBindGroupLayout,
    private readonly convPipeline: GPUComputePipeline,
    private readonly convLayout: GPUBindGroupLayout,
    readonly variant: FcmVariant,
    readonly accumulation: FcmAccumulation,
  ) {}

  static async create(
    device: GPUDevice,
    gpuPackage: CampPlusGpuPackage,
    arena: CampPlusActivationArena,
    variant: FcmVariant = DEFAULT_FCM_VARIANT,
    accumulation: FcmAccumulation = DEFAULT_FCM_ACCUMULATION,
  ): Promise<FcmKernels> {
    const storageDtype = gpuPackage.metadata.contract.internalDtype;
    const storageBytes = campPlusStorageBytes(storageDtype);
    let effectiveVariant = variant;
    let configuration = fcmVariantConfiguration(effectiveVariant);
    if (
      storageDtype === "float32" &&
      configuration.convWorkgroupStorageBytes * 2 >
        device.limits.maxComputeWorkgroupStorageSize &&
      effectiveVariant === "tile4-fold"
    ) {
      effectiveVariant = "tile2-fold";
      configuration = fcmVariantConfiguration(effectiveVariant);
    }
    const requiredWorkgroupStorageBytes =
      configuration.convWorkgroupStorageBytes * (storageBytes / 2);
    if (
      device.limits.maxComputeWorkgroupStorageSize <
      requiredWorkgroupStorageBytes
    ) {
      throw new Error(
        `CAM++ FCM ${effectiveVariant} requires ${requiredWorkgroupStorageBytes} workgroup bytes`,
      );
    }
    const firstLayout = device.createBindGroupLayout({
      label: "senko-campplus-fcm-first-bindings",
      entries: [
        readStorage(0),
        writeStorage(1),
        readStorage(2),
        readStorage(3),
        uniformEntry(4),
      ],
    });
    const convLayout = device.createBindGroupLayout({
      label: "senko-campplus-fcm-conv-bindings",
      entries: [
        writeStorage(0),
        readStorage(1),
        readStorage(2),
        readStorage(3),
        readStorage(4),
        uniformEntry(5),
      ],
    });
    const labelSuffix =
      effectiveVariant === DEFAULT_FCM_VARIANT && accumulation === DEFAULT_FCM_ACCUMULATION
        ? ""
        : `-${effectiveVariant}-${accumulation}`;
    const [firstPipeline, convPipeline] = await Promise.all([
      checkedPipeline(
        device,
        `senko-campplus-fcm-first${labelSuffix}`,
        campPlusStorageWgsl(
          fcmFirstWgsl(effectiveVariant, accumulation),
          storageDtype,
        ),
        firstLayout,
      ),
      checkedPipeline(
        device,
        `senko-campplus-fcm-conv${labelSuffix}`,
        campPlusStorageWgsl(
          fcmConvWgsl(effectiveVariant, accumulation),
          storageDtype,
        ),
        convLayout,
      ),
    ]);
    return new FcmKernels(
      device,
      gpuPackage,
      arena,
      firstPipeline,
      firstLayout,
      convPipeline,
      convLayout,
      effectiveVariant,
      accumulation,
    );
  }

  createFirstDispatch(descriptor: FcmFirstConvDescriptor): FcmDispatch {
    const weight = this.convWeight(descriptor.convolution, CHANNELS, 1, 9);
    const bias = this.convBias(descriptor.convolution, CHANNELS);
    const inputBytes = descriptor.batchSize * FRAMES * 80 * 4;
    const storageBytes = campPlusStorageBytes(
      this.gpuPackage.metadata.contract.internalDtype,
    );
    validateSlice(
      descriptor.output,
      descriptor.batchSize * CHANNELS * 80 * FRAMES * storageBytes,
      this.arena.byteLength,
    );
    if (descriptor.input.size < inputBytes) throw new Error("FCM FP32 input buffer is too small");
    const parameters = new Uint32Array([
      descriptor.output.byteOffset / storageBytes,
      descriptor.batchSize,
      80,
      FRAMES,
      OUTPUT_GROUPS,
      0,
      0,
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
    const uniform = initializedUniform(this.device, `${descriptor.label}-parameters`, parameters);
    try {
      const bindGroup = this.device.createBindGroup({
        label: `${descriptor.label}-bindings`,
        layout: this.firstLayout,
        entries: [
          { binding: 0, resource: { buffer: descriptor.input, size: inputBytes } },
          arenaEntry(1, this.arena),
          sectionEntry(2, this.gpuPackage.weightsBuffer, weight),
          sectionEntry(3, this.gpuPackage.weightsBuffer, bias),
          {
            binding: 4,
            resource: { buffer: uniform, size: FCM_DISPATCH_GPU_BUFFER_BYTES },
          },
        ],
      });
      return new FcmDispatch(
        descriptor.label,
        this.firstPipeline,
        bindGroup,
        uniform,
        fcmDispatchWorkgroups(this.variant, descriptor.batchSize, 80),
      );
    } catch (error) {
      uniform.destroy();
      throw error;
    }
  }

  createConvDispatch(descriptor: FcmConvDescriptor): FcmDispatch {
    const weight = this.convWeight(descriptor.convolution, CHANNELS, CHANNELS, 9);
    const bias = this.convBias(descriptor.convolution, CHANNELS);
    const shortcutWeight =
      descriptor.residual.kind === "learned"
        ? this.convWeight(descriptor.residual.convolution, CHANNELS, CHANNELS, 1)
        : undefined;
    const shortcutBias =
      descriptor.residual.kind === "learned"
        ? this.convBias(descriptor.residual.convolution, CHANNELS)
        : undefined;
    validateFcmDimensions(descriptor);
    const storageBytes = campPlusStorageBytes(
      this.gpuPackage.metadata.contract.internalDtype,
    );
    validateSlice(
      descriptor.input,
      descriptor.batchSize * CHANNELS * descriptor.inputFreq * FRAMES * storageBytes,
      this.arena.byteLength,
    );
    validateSlice(
      descriptor.output,
      descriptor.batchSize * CHANNELS * descriptor.outputFreq * FRAMES * storageBytes,
      this.arena.byteLength,
    );
    let residualOffset = 0;
    let residualInputFreq = descriptor.outputFreq;
    let residualStride = 1;
    let residualMode = 0;
    if (descriptor.residual.kind !== "none") {
      const inputFreq =
        descriptor.residual.kind === "learned"
          ? descriptor.residual.inputFreq
          : descriptor.outputFreq;
      validateSlice(
        descriptor.residual.input,
        descriptor.batchSize * CHANNELS * inputFreq * FRAMES * storageBytes,
        this.arena.byteLength,
      );
      residualOffset = descriptor.residual.input.byteOffset / storageBytes;
      residualInputFreq = inputFreq;
      residualStride =
        descriptor.residual.kind === "learned" ? descriptor.residual.strideFreq : 1;
      residualMode = descriptor.residual.kind === "learned" ? 2 : 1;
    }
    if (
      rangesOverlap(descriptor.input, descriptor.output) ||
      (descriptor.residual.kind !== "none" && rangesOverlap(descriptor.residual.input, descriptor.output))
    ) {
      throw new Error(`${descriptor.label} FCM input and output ranges overlap`);
    }
    const parameters = new Uint32Array([
      descriptor.input.byteOffset / storageBytes,
      residualOffset,
      descriptor.output.byteOffset / storageBytes,
      descriptor.batchSize,
      descriptor.inputFreq,
      descriptor.outputFreq,
      FRAMES,
      descriptor.strideFreq,
      residualInputFreq,
      residualStride,
      residualMode,
      descriptor.outputRelu ? 1 : 0,
      0,
      0,
      0,
      0,
    ]);
    const uniform = initializedUniform(this.device, `${descriptor.label}-parameters`, parameters);
    try {
      const bindGroup = this.device.createBindGroup({
        label: `${descriptor.label}-bindings`,
        layout: this.convLayout,
        entries: [
          arenaEntry(0, this.arena),
          sectionEntry(1, this.gpuPackage.weightsBuffer, weight),
          sectionEntry(2, this.gpuPackage.weightsBuffer, bias),
          shortcutWeight === undefined
            ? dummyEntry(3, this.gpuPackage.weightsBuffer)
            : sectionEntry(3, this.gpuPackage.weightsBuffer, shortcutWeight),
          shortcutBias === undefined
            ? dummyEntry(4, this.gpuPackage.weightsBuffer)
            : sectionEntry(4, this.gpuPackage.weightsBuffer, shortcutBias),
          {
            binding: 5,
            resource: { buffer: uniform, size: FCM_DISPATCH_GPU_BUFFER_BYTES },
          },
        ],
      });
      return new FcmDispatch(
        descriptor.label,
        this.convPipeline,
        bindGroup,
        uniform,
        fcmDispatchWorkgroups(
          this.variant,
          descriptor.batchSize,
          descriptor.outputFreq,
        ),
      );
    } catch (error) {
      uniform.destroy();
      throw error;
    }
  }

  private convWeight(
    convolution: PackedConvolutionRef,
    outputChannels: number,
    inputChannels: number,
    kernelElements: number,
  ): CampPlusPackedSection {
    const section = this.gpuPackage.section(convolution.weight);
    if (
      section.kind !== "conv_weight" ||
      section.logicalShape[0] !== outputChannels ||
      section.logicalShape[1] !== inputChannels ||
      product(section.logicalShape.slice(2)) !== kernelElements
    ) {
      throw new Error(`Unexpected FCM weight section ${section.id}`);
    }
    return section;
  }

  private convBias(
    convolution: PackedConvolutionRef,
    outputChannels: number,
  ): CampPlusPackedSection {
    const section = this.gpuPackage.section(convolution.bias);
    if (section.kind !== "conv_bias" || section.logicalShape[0] !== outputChannels) {
      throw new Error(`Unexpected FCM bias section ${section.id}`);
    }
    return section;
  }
}

export function validateFcmDimensions(descriptor: FcmConvDescriptor): void {
  const expected = Math.floor((descriptor.inputFreq + 2 - 3) / descriptor.strideFreq) + 1;
  if (descriptor.outputFreq !== expected || ![1, 2].includes(descriptor.strideFreq)) {
    throw new Error(`${descriptor.label} has invalid FCM frequency dimensions`);
  }
  if (
    descriptor.residual.kind === "identity" &&
    (descriptor.inputFreq !== descriptor.outputFreq || descriptor.strideFreq !== 1)
  ) {
    throw new Error(`${descriptor.label} identity residual shape mismatch`);
  }
  if (
    descriptor.residual.kind === "learned" &&
    descriptor.outputFreq !== Math.ceil(descriptor.residual.inputFreq / descriptor.residual.strideFreq)
  ) {
    throw new Error(`${descriptor.label} learned residual shape mismatch`);
  }
}

function validateSlice(slice: CampPlusArenaSlice, needed: number, arenaBytes: number): void {
  if (slice.byteLength < needed || slice.byteOffset + slice.byteLength > arenaBytes) {
    throw new RangeError(`FCM arena slice ${slice.label} is too small`);
  }
}

function rangesOverlap(left: CampPlusArenaSlice, right: CampPlusArenaSlice): boolean {
  return left.byteOffset < right.byteOffset + right.byteLength && right.byteOffset < left.byteOffset + left.byteLength;
}

function readStorage(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } };
}

function writeStorage(binding: number): GPUBindGroupLayoutEntry {
  return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } };
}

function uniformEntry(binding: number): GPUBindGroupLayoutEntry {
  return {
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: {
      type: "uniform",
      minBindingSize: FCM_DISPATCH_GPU_BUFFER_BYTES,
    },
  };
}

function arenaEntry(binding: number, arena: CampPlusActivationArena): GPUBindGroupEntry {
  return { binding, resource: { buffer: arena.buffer, size: arena.byteLength } };
}

function sectionEntry(binding: number, buffer: GPUBuffer, section: CampPlusPackedSection): GPUBindGroupEntry {
  return { binding, resource: { buffer, offset: section.byteOffset, size: section.byteLength } };
}

function dummyEntry(binding: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding, resource: { buffer, offset: 0, size: 256 } };
}

function initializedUniform(device: GPUDevice, label: string, data: Uint32Array<ArrayBuffer>): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: FCM_DISPATCH_GPU_BUFFER_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

async function checkedPipeline(device: GPUDevice, label: string, code: string, layout: GPUBindGroupLayout): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label, code });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length > 0) throw new Error(`${label} WGSL failed: ${errors.map((item) => item.message).join("; ")}`);
  return device.createComputePipelineAsync({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: "main" },
  });
}

function ceilDiv(value: number, divisor: number): number {
  return Math.floor((value + divisor - 1) / divisor);
}

function product(values: readonly number[]): number {
  return values.reduce((result, value) => result * value, 1);
}

export const FCM_FIRST_WGSL = /* wgsl */ `
enable f16;

struct Parameters {
  output_offset: u32,
  batch_size: u32,
  features: u32,
  frames: u32,
  output_groups: u32,
  r0: u32, r1: u32, r2: u32, r3: u32, r4: u32, r5: u32,
  r6: u32, r7: u32, r8: u32, r9: u32, r10: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> arena: array<f16>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;
var<workgroup> weight_cache: array<vec4<f16>, 9>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let output_group = group.x;
  let batch = group.y / parameters.features;
  let output_freq = group.y % parameters.features;
  var index = local_id.x;
  while (index < 9u) {
    // Packed convolution weights are [kernel, output_group, input_group, input_lane].
    // The first convolution has one padded input group and only lane zero is live.
    weight_cache[index] =
      weights[(index * parameters.output_groups + output_group) * 4u];
    index += 128u;
  }
  workgroupBarrier();
  let output_time = group.z * 128u + local_id.x;
  if (output_time >= parameters.frames || batch >= parameters.batch_size) { return; }
  var accumulator = vec4<f32>(biases[output_group]);
  for (var kernel_freq = 0u; kernel_freq < 3u; kernel_freq += 1u) {
    let source_freq = i32(output_freq + kernel_freq) - 1;
    if (source_freq < 0 || source_freq >= i32(parameters.features)) { continue; }
    for (var kernel_time = 0u; kernel_time < 3u; kernel_time += 1u) {
      let source_time = i32(output_time + kernel_time) - 1;
      if (source_time < 0 || source_time >= i32(parameters.frames)) { continue; }
      let input_index =
        (batch * parameters.frames + u32(source_time)) * parameters.features + u32(source_freq);
      let value = f32(f16(input[input_index]));
      accumulator = fma(
        vec4<f32>(value),
        vec4<f32>(weight_cache[kernel_freq * 3u + kernel_time]),
        accumulator,
      );
    }
  }
  let rounded = max(vec4<f16>(accumulator), vec4<f16>(f16(0.0)));
  let output_channel = output_group * 4u;
  for (var lane = 0u; lane < 4u; lane += 1u) {
    let output_index =
      parameters.output_offset +
      (((batch * 32u + output_channel + lane) * parameters.features + output_freq) * parameters.frames + output_time);
    arena[output_index] = rounded[lane];
  }
}
`;

export const FCM_CONV_WGSL = /* wgsl */ `
enable f16;

struct Parameters {
  input_offset: u32,
  residual_offset: u32,
  output_offset: u32,
  batch_size: u32,
  input_freq: u32,
  output_freq: u32,
  frames: u32,
  stride_freq: u32,
  residual_input_freq: u32,
  residual_stride_freq: u32,
  residual_mode: u32,
  output_relu: u32,
  r0: u32, r1: u32, r2: u32, r3: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> shortcut_weights: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> shortcut_biases: array<vec4<f16>>;
@group(0) @binding(5) var<uniform> parameters: Parameters;
var<workgroup> weight_cache: array<vec4<f16>, 320>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let output_group = group.x;
  let batch = group.y / parameters.output_freq;
  let output_freq = group.y % parameters.output_freq;
  var cache_index = local_id.x;
  while (cache_index < 288u) {
    let kernel = cache_index / 32u;
    let input_channel = cache_index - kernel * 32u;
    let input_group = input_channel / 4u;
    let input_lane = input_channel & 3u;
    let packed_index = ((kernel * 8u + output_group) * 8u + input_group) * 4u + input_lane;
    weight_cache[cache_index] = weights[packed_index];
    cache_index += 128u;
  }
  if (parameters.residual_mode == 2u) {
    var shortcut_index = local_id.x;
    while (shortcut_index < 32u) {
      let input_group = shortcut_index / 4u;
      let input_lane = shortcut_index & 3u;
      let packed_index = (output_group * 8u + input_group) * 4u + input_lane;
      weight_cache[288u + shortcut_index] = shortcut_weights[packed_index];
      shortcut_index += 128u;
    }
  }
  workgroupBarrier();
  let output_time = group.z * 128u + local_id.x;
  if (output_time >= parameters.frames || batch >= parameters.batch_size) { return; }

  var main = vec4<f32>(biases[output_group]);
  for (var kernel_freq = 0u; kernel_freq < 3u; kernel_freq += 1u) {
    let source_freq = i32(output_freq * parameters.stride_freq + kernel_freq) - 1;
    if (source_freq < 0 || source_freq >= i32(parameters.input_freq)) { continue; }
    for (var kernel_time = 0u; kernel_time < 3u; kernel_time += 1u) {
      let source_time = i32(output_time + kernel_time) - 1;
      if (source_time < 0 || source_time >= i32(parameters.frames)) { continue; }
      let kernel = kernel_freq * 3u + kernel_time;
      let channel_stride = parameters.input_freq * parameters.frames;
      for (var input_group = 0u; input_group < 8u; input_group += 1u) {
        let input_channel = input_group * 4u;
        let input_index =
          parameters.input_offset +
          (((batch * 32u + input_channel) * parameters.input_freq + u32(source_freq)) * parameters.frames + u32(source_time));
        let weight_index = kernel * 32u + input_channel;
        main = fma(vec4<f32>(f32(arena[input_index])), vec4<f32>(weight_cache[weight_index]), main);
        main = fma(vec4<f32>(f32(arena[input_index + channel_stride])), vec4<f32>(weight_cache[weight_index + 1u]), main);
        main = fma(vec4<f32>(f32(arena[input_index + 2u * channel_stride])), vec4<f32>(weight_cache[weight_index + 2u]), main);
        main = fma(vec4<f32>(f32(arena[input_index + 3u * channel_stride])), vec4<f32>(weight_cache[weight_index + 3u]), main);
      }
    }
  }
  let main_rounded = vec4<f16>(main);
  var result = main_rounded;
  if (parameters.residual_mode == 1u) {
    let output_channel = output_group * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let residual_index =
        parameters.residual_offset +
        (((batch * 32u + output_channel + lane) * parameters.residual_input_freq + output_freq) * parameters.frames + output_time);
      result[lane] = f16(main_rounded[lane] + arena[residual_index]);
    }
  } else if (parameters.residual_mode == 2u) {
    var shortcut = vec4<f32>(shortcut_biases[output_group]);
    let source_freq = output_freq * parameters.residual_stride_freq;
    let residual_channel_stride = parameters.residual_input_freq * parameters.frames;
    for (var input_group = 0u; input_group < 8u; input_group += 1u) {
      let input_channel = input_group * 4u;
      let input_index =
        parameters.residual_offset +
        (((batch * 32u + input_channel) * parameters.residual_input_freq + source_freq) * parameters.frames + output_time);
      shortcut = fma(vec4<f32>(f32(arena[input_index])), vec4<f32>(weight_cache[288u + input_channel]), shortcut);
      shortcut = fma(vec4<f32>(f32(arena[input_index + residual_channel_stride])), vec4<f32>(weight_cache[289u + input_channel]), shortcut);
      shortcut = fma(vec4<f32>(f32(arena[input_index + 2u * residual_channel_stride])), vec4<f32>(weight_cache[290u + input_channel]), shortcut);
      shortcut = fma(vec4<f32>(f32(arena[input_index + 3u * residual_channel_stride])), vec4<f32>(weight_cache[291u + input_channel]), shortcut);
    }
    result = vec4<f16>(main_rounded + vec4<f16>(shortcut));
  }
  if (parameters.output_relu != 0u) {
    result = max(result, vec4<f16>(f16(0.0)));
  }
  let output_channel = output_group * 4u;
  for (var lane = 0u; lane < 4u; lane += 1u) {
    let output_index =
      parameters.output_offset +
      (((batch * 32u + output_channel + lane) * parameters.output_freq + output_freq) * parameters.frames + output_time);
    arena[output_index] = result[lane];
  }
}
`;

/** Materializes the selected FCM first-convolution kernel. */
export function fcmFirstWgsl(
  variant: FcmVariant,
  accumulation: FcmAccumulation = DEFAULT_FCM_ACCUMULATION,
): string {
  if (variant === LEGACY_FCM_VARIANT) {
    if (accumulation !== "float32") {
      throw new Error("The split-tail FCM diagnostic supports FP32 accumulation only");
    }
    return FCM_FIRST_WGSL;
  }
  const { outputTile, foldTimeTail } = fcmVariantConfiguration(variant);
  if (!foldTimeTail) {
    throw new Error(`FCM ${variant} must use the retained baseline shader`);
  }
  const accumulatorDeclarations = Array.from(
    { length: outputTile },
    (_, tile) => {
      if (accumulation === "float16") {
        return `    var accumulator_${tile} = biases[first_output_group + ${tile}u];`;
      }
      return `    var accumulator_${tile} = vec4<f32>(biases[first_output_group + ${tile}u]);`;
    },
  ).join("\n");
  const accumulationSteps = Array.from(
    { length: outputTile },
    (_, tile) => {
      if (accumulation === "float32") {
        return `        accumulator_${tile} = fma(
          vec4<f32>(value),
          vec4<f32>(weight_cache[${tile * 9}u + kernel]),
          accumulator_${tile},
        );`;
      }
      return `        accumulator_${tile} = fma(
          vec4<f16>(f16(value)),
          weight_cache[${tile * 9}u + kernel],
          accumulator_${tile},
        );`;
    },
  ).join("\n");
  const stores = Array.from(
    { length: outputTile },
    (_, tile) => {
      const value = accumulation === "float16"
        ? `accumulator_${tile}`
        : `vec4<f16>(accumulator_${tile})`;
      return `    store_output(
      first_output_group + ${tile}u,
      batch,
      output_freq,
      output_time,
      max(${value}, vec4<f16>(f16(0.0))),
    );`;
    },
  ).join("\n");
  return /* wgsl */ `
enable f16;

struct Parameters {
  output_offset: u32,
  batch_size: u32,
  features: u32,
  frames: u32,
  output_groups: u32,
  r0: u32, r1: u32, r2: u32, r3: u32, r4: u32, r5: u32,
  r6: u32, r7: u32, r8: u32, r9: u32, r10: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> arena: array<f16>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;
var<workgroup> weight_cache: array<vec4<f16>, ${outputTile * 9}>;

fn store_output(
  output_group: u32,
  batch: u32,
  output_freq: u32,
  output_time: u32,
  value: vec4<f16>,
) {
  let output_channel = output_group * 4u;
  let channel_stride = parameters.features * parameters.frames;
  let output_index =
    parameters.output_offset +
    (((batch * 32u + output_channel) * parameters.features + output_freq) * parameters.frames + output_time);
  arena[output_index] = value[0];
  arena[output_index + channel_stride] = value[1];
  arena[output_index + 2u * channel_stride] = value[2];
  arena[output_index + 3u * channel_stride] = value[3];
}

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let first_output_group = group.x * ${outputTile}u;
  let batch = group.y / parameters.features;
  let output_freq = group.y % parameters.features;
  var cache_index = local_id.x;
  while (cache_index < ${outputTile * 9}u) {
    let tile = cache_index / 9u;
    let kernel = cache_index - tile * 9u;
    let output_group = first_output_group + tile;
    // Packed convolution weights are [kernel, output_group, input_group, input_lane].
    // The first convolution has one padded input group and only lane zero is live.
    weight_cache[cache_index] =
      weights[(kernel * parameters.output_groups + output_group) * 4u];
    cache_index += 128u;
  }
  workgroupBarrier();
  if (batch >= parameters.batch_size) { return; }

  var output_time = local_id.x;
  while (output_time < parameters.frames) {
${accumulatorDeclarations}
    for (var kernel_freq = 0u; kernel_freq < 3u; kernel_freq += 1u) {
      let source_freq = i32(output_freq + kernel_freq) - 1;
      if (source_freq < 0 || source_freq >= i32(parameters.features)) { continue; }
      for (var kernel_time = 0u; kernel_time < 3u; kernel_time += 1u) {
        let source_time = i32(output_time + kernel_time) - 1;
        if (source_time < 0 || source_time >= i32(parameters.frames)) { continue; }
        let input_index =
          (batch * parameters.frames + u32(source_time)) * parameters.features + u32(source_freq);
        let value = f32(f16(input[input_index]));
        let kernel = kernel_freq * 3u + kernel_time;
${accumulationSteps}
      }
    }
${stores}
    output_time += 128u;
  }
}
`;
}

/** Materializes the selected FCM convolution kernel. */
export function fcmConvWgsl(
  variant: FcmVariant,
  accumulation: FcmAccumulation = DEFAULT_FCM_ACCUMULATION,
): string {
  if (variant === LEGACY_FCM_VARIANT) {
    if (accumulation !== "float32") {
      throw new Error("The split-tail FCM diagnostic supports FP32 accumulation only");
    }
    return FCM_CONV_WGSL;
  }
  const { outputTile, foldTimeTail } = fcmVariantConfiguration(variant);
  if (!foldTimeTail) {
    throw new Error(`FCM ${variant} must use the retained baseline shader`);
  }
  const mainDeclarations = Array.from(
    { length: outputTile },
    (_, tile) => {
      if (accumulation === "float16") {
        return `    var main_${tile} = biases[first_output_group + ${tile}u];`;
      }
      return `    var main_${tile} = vec4<f32>(biases[first_output_group + ${tile}u]);`;
    },
  ).join("\n");
  const mainAccumulations = Array.from({ length: outputTile }, (_, tile) => {
    const base = tile * 320;
    if (accumulation !== "float32") {
      return `        main_${tile} = fma(vec4<f16>(f16(input_0)), weight_cache[${base}u + weight_index], main_${tile});
        main_${tile} = fma(vec4<f16>(f16(input_1)), weight_cache[${base + 1}u + weight_index], main_${tile});
        main_${tile} = fma(vec4<f16>(f16(input_2)), weight_cache[${base + 2}u + weight_index], main_${tile});
        main_${tile} = fma(vec4<f16>(f16(input_3)), weight_cache[${base + 3}u + weight_index], main_${tile});`;
    }
    return `        main_${tile} = fma(vec4<f32>(input_0), vec4<f32>(weight_cache[${base}u + weight_index]), main_${tile});
        main_${tile} = fma(vec4<f32>(input_1), vec4<f32>(weight_cache[${base + 1}u + weight_index]), main_${tile});
        main_${tile} = fma(vec4<f32>(input_2), vec4<f32>(weight_cache[${base + 2}u + weight_index]), main_${tile});
        main_${tile} = fma(vec4<f32>(input_3), vec4<f32>(weight_cache[${base + 3}u + weight_index]), main_${tile});`;
  }).join("\n");
  const resultDeclarations = Array.from(
    { length: outputTile },
    (_, tile) => {
      const rounded = accumulation === "float16"
        ? `main_${tile}`
        : `vec4<f16>(main_${tile})`;
      return `    let main_rounded_${tile} = ${rounded};
    var result_${tile} = main_rounded_${tile};`;
    },
  ).join("\n");
  const identitySteps = Array.from(
    { length: outputTile },
    (_, tile) => `    {
      let output_channel = (first_output_group + ${tile}u) * 4u;
      for (var lane = 0u; lane < 4u; lane += 1u) {
        let residual_index =
          parameters.residual_offset +
          (((batch * 32u + output_channel + lane) * parameters.residual_input_freq + output_freq) * parameters.frames + output_time);
        result_${tile}[lane] = f16(main_rounded_${tile}[lane] + arena[residual_index]);
      }
    }`,
  ).join("\n");
  const shortcutDeclarations = Array.from(
    { length: outputTile },
    (_, tile) => {
      if (accumulation === "float16") {
        return `    var shortcut_${tile} = shortcut_biases[first_output_group + ${tile}u];`;
      }
      return `    var shortcut_${tile} = vec4<f32>(shortcut_biases[first_output_group + ${tile}u]);`;
    },
  ).join("\n");
  const shortcutAccumulations = Array.from({ length: outputTile }, (_, tile) => {
    const base = tile * 320 + 288;
    if (accumulation !== "float32") {
      return `      shortcut_${tile} = fma(vec4<f16>(f16(input_0)), weight_cache[${base}u + input_channel], shortcut_${tile});
      shortcut_${tile} = fma(vec4<f16>(f16(input_1)), weight_cache[${base + 1}u + input_channel], shortcut_${tile});
      shortcut_${tile} = fma(vec4<f16>(f16(input_2)), weight_cache[${base + 2}u + input_channel], shortcut_${tile});
      shortcut_${tile} = fma(vec4<f16>(f16(input_3)), weight_cache[${base + 3}u + input_channel], shortcut_${tile});`;
    }
    return `      shortcut_${tile} = fma(vec4<f32>(input_0), vec4<f32>(weight_cache[${base}u + input_channel]), shortcut_${tile});
      shortcut_${tile} = fma(vec4<f32>(input_1), vec4<f32>(weight_cache[${base + 1}u + input_channel]), shortcut_${tile});
      shortcut_${tile} = fma(vec4<f32>(input_2), vec4<f32>(weight_cache[${base + 2}u + input_channel]), shortcut_${tile});
      shortcut_${tile} = fma(vec4<f32>(input_3), vec4<f32>(weight_cache[${base + 3}u + input_channel]), shortcut_${tile});`;
  }).join("\n");
  const shortcutResults = Array.from(
    { length: outputTile },
    (_, tile) => {
      const shortcut = accumulation === "float16"
        ? `shortcut_${tile}`
        : `vec4<f16>(shortcut_${tile})`;
      return `    result_${tile} = vec4<f16>(main_rounded_${tile} + ${shortcut});`;
    },
  ).join("\n");
  const relus = Array.from(
    { length: outputTile },
    (_, tile) =>
      `      result_${tile} = max(result_${tile}, vec4<f16>(f16(0.0)));`,
  ).join("\n");
  const stores = Array.from(
    { length: outputTile },
    (_, tile) =>
      `    store_output(first_output_group + ${tile}u, batch, output_freq, output_time, result_${tile});`,
  ).join("\n");
  return /* wgsl */ `
enable f16;

struct Parameters {
  input_offset: u32,
  residual_offset: u32,
  output_offset: u32,
  batch_size: u32,
  input_freq: u32,
  output_freq: u32,
  frames: u32,
  stride_freq: u32,
  residual_input_freq: u32,
  residual_stride_freq: u32,
  residual_mode: u32,
  output_relu: u32,
  r0: u32, r1: u32, r2: u32, r3: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> shortcut_weights: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> shortcut_biases: array<vec4<f16>>;
@group(0) @binding(5) var<uniform> parameters: Parameters;
var<workgroup> weight_cache: array<vec4<f16>, ${outputTile * 320}>;

fn store_output(
  output_group: u32,
  batch: u32,
  output_freq: u32,
  output_time: u32,
  value: vec4<f16>,
) {
  let output_channel = output_group * 4u;
  let channel_stride = parameters.output_freq * parameters.frames;
  let output_index =
    parameters.output_offset +
    (((batch * 32u + output_channel) * parameters.output_freq + output_freq) * parameters.frames + output_time);
  arena[output_index] = value[0];
  arena[output_index + channel_stride] = value[1];
  arena[output_index + 2u * channel_stride] = value[2];
  arena[output_index + 3u * channel_stride] = value[3];
}

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let first_output_group = group.x * ${outputTile}u;
  let batch = group.y / parameters.output_freq;
  let output_freq = group.y % parameters.output_freq;
  var cache_index = local_id.x;
  while (cache_index < ${outputTile * 288}u) {
    let tile = cache_index / 288u;
    let tile_index = cache_index - tile * 288u;
    let kernel = tile_index / 32u;
    let input_channel = tile_index - kernel * 32u;
    let input_group = input_channel / 4u;
    let input_lane = input_channel & 3u;
    let output_group = first_output_group + tile;
    let packed_index =
      ((kernel * 8u + output_group) * 8u + input_group) * 4u + input_lane;
    weight_cache[tile * 320u + tile_index] = weights[packed_index];
    cache_index += 128u;
  }
  if (parameters.residual_mode == 2u) {
    var shortcut_index = local_id.x;
    while (shortcut_index < ${outputTile * 32}u) {
      let tile = shortcut_index / 32u;
      let input_channel = shortcut_index - tile * 32u;
      let input_group = input_channel / 4u;
      let input_lane = input_channel & 3u;
      let output_group = first_output_group + tile;
      let packed_index = (output_group * 8u + input_group) * 4u + input_lane;
      weight_cache[tile * 320u + 288u + input_channel] = shortcut_weights[packed_index];
      shortcut_index += 128u;
    }
  }
  workgroupBarrier();
  if (batch >= parameters.batch_size) { return; }

  var output_time = local_id.x;
  while (output_time < parameters.frames) {
${mainDeclarations}
    for (var kernel_freq = 0u; kernel_freq < 3u; kernel_freq += 1u) {
      let source_freq = i32(output_freq * parameters.stride_freq + kernel_freq) - 1;
      if (source_freq < 0 || source_freq >= i32(parameters.input_freq)) { continue; }
      for (var kernel_time = 0u; kernel_time < 3u; kernel_time += 1u) {
        let source_time = i32(output_time + kernel_time) - 1;
        if (source_time < 0 || source_time >= i32(parameters.frames)) { continue; }
        let kernel = kernel_freq * 3u + kernel_time;
        let channel_stride = parameters.input_freq * parameters.frames;
        for (var input_group = 0u; input_group < 8u; input_group += 1u) {
          let input_channel = input_group * 4u;
          let input_index =
            parameters.input_offset +
            (((batch * 32u + input_channel) * parameters.input_freq + u32(source_freq)) * parameters.frames + u32(source_time));
          let weight_index = kernel * 32u + input_channel;
          let input_0 = f32(arena[input_index]);
          let input_1 = f32(arena[input_index + channel_stride]);
          let input_2 = f32(arena[input_index + 2u * channel_stride]);
          let input_3 = f32(arena[input_index + 3u * channel_stride]);
${mainAccumulations}
        }
      }
    }
${resultDeclarations}
    if (parameters.residual_mode == 1u) {
${identitySteps}
    } else if (parameters.residual_mode == 2u) {
${shortcutDeclarations}
      let source_freq = output_freq * parameters.residual_stride_freq;
      let residual_channel_stride = parameters.residual_input_freq * parameters.frames;
      for (var input_group = 0u; input_group < 8u; input_group += 1u) {
        let input_channel = input_group * 4u;
        let input_index =
          parameters.residual_offset +
          (((batch * 32u + input_channel) * parameters.residual_input_freq + source_freq) * parameters.frames + output_time);
        let input_0 = f32(arena[input_index]);
        let input_1 = f32(arena[input_index + residual_channel_stride]);
        let input_2 = f32(arena[input_index + 2u * residual_channel_stride]);
        let input_3 = f32(arena[input_index + 3u * residual_channel_stride]);
${shortcutAccumulations}
      }
${shortcutResults}
    }
    if (parameters.output_relu != 0u) {
${relus}
    }
${stores}
    output_time += 128u;
  }
}
`;
}
