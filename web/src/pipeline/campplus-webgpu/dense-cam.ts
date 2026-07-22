/// <reference types="@webgpu/types" />

import type { CampPlusArenaSlice } from "./arena";
import { CampPlusActivationArena } from "./arena";
import type {
  CamDenseLayerMetadata,
  CampPlusPackedSection,
  PackedConvolutionRef,
} from "./metadata";
import { CampPlusGpuPackage } from "./package";

const WORKGROUP_SIZE = 128;
const BOTTLENECK_CHANNELS = 128;
const CAM_OUTPUT_CHANNELS = 32;
const MAX_DENSE_INPUT_CHANNELS = 992;
const BOTTLENECK_OUTPUT_TILE = 2;
export const DENSE_CAM_TILE2_WORKGROUP_STORAGE_BYTES =
  MAX_DENSE_INPUT_CHANNELS * BOTTLENECK_OUTPUT_TILE * 8 +
  WORKGROUP_SIZE * BOTTLENECK_OUTPUT_TILE * 16;
export const DENSE_CAM_REQUIRED_WORKGROUP_STORAGE_BYTES =
  MAX_DENSE_INPUT_CHANNELS * 8 + WORKGROUP_SIZE * 16;
const UNIFORM_BYTES = 64;

export type DenseBottleneckAccumulation =
  | "float32"
  | "float16-chunk32"
  | "float16";
export type DenseBottleneckOutputTile = 1 | 2 | 4;
export type DenseBottleneckWorkgroupSize = 96 | 128;
export type DenseBottleneckWeightSource = "workgroup-cache" | "direct";

export const DENSE_BOTTLENECK_VARIANTS = [
  "direct-tile1-wg128",
  "direct-tile2-wg128",
  "direct-tile4-wg128",
] as const;

export type DenseBottleneckVariant = (typeof DENSE_BOTTLENECK_VARIANTS)[number];

export const DEFAULT_DENSE_BOTTLENECK_VARIANT: DenseBottleneckVariant =
  "direct-tile4-wg128";

export interface DenseBottleneckVariantConfiguration {
  readonly accumulation: DenseBottleneckAccumulation;
  readonly outputTile: DenseBottleneckOutputTile;
  readonly workgroupSize: DenseBottleneckWorkgroupSize;
  readonly weightSource: DenseBottleneckWeightSource;
}

const DENSE_BOTTLENECK_VARIANT_CONFIGURATIONS: Readonly<
  Record<DenseBottleneckVariant, DenseBottleneckVariantConfiguration>
> = {
  "direct-tile1-wg128": {
    accumulation: "float32",
    outputTile: 1,
    workgroupSize: 128,
    weightSource: "direct",
  },
  "direct-tile2-wg128": {
    accumulation: "float32",
    outputTile: 2,
    workgroupSize: 128,
    weightSource: "direct",
  },
  "direct-tile4-wg128": {
    accumulation: "float32",
    outputTile: 4,
    workgroupSize: 128,
    weightSource: "direct",
  },
};

export function isDenseBottleneckVariant(value: string): value is DenseBottleneckVariant {
  return (DENSE_BOTTLENECK_VARIANTS as readonly string[]).includes(value);
}

export function denseBottleneckVariantConfiguration(
  variant: DenseBottleneckVariant,
): DenseBottleneckVariantConfiguration {
  return DENSE_BOTTLENECK_VARIANT_CONFIGURATIONS[variant];
}

export interface DenseBottleneckDescriptor {
  readonly label: string;
  readonly layer: CamDenseLayerMetadata;
  readonly slab: CampPlusArenaSlice;
  readonly slabChannels: number;
  readonly scratch: CampPlusArenaSlice;
  readonly doubledMean: CampPlusArenaSlice;
  readonly batchSize: number;
  readonly accumulation?: DenseBottleneckAccumulation;
  readonly outputTile?: DenseBottleneckOutputTile;
  readonly workgroupSize?: DenseBottleneckWorkgroupSize;
  readonly weightSource?: DenseBottleneckWeightSource;
}

export interface DenseLocalCamDescriptor {
  readonly label: string;
  readonly layer: CamDenseLayerMetadata;
  readonly slab: CampPlusArenaSlice;
  readonly slabChannels: number;
  readonly scratch: CampPlusArenaSlice;
  readonly doubledMean: CampPlusArenaSlice;
  readonly batchSize: number;
}

export class DenseCamDispatch {
  readonly gpuBufferBytes = UNIFORM_BYTES;
  private destroyed = false;

  constructor(
    readonly label: string,
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

/** Two-dispatch fused CAM dense layer: bottleneck+mean, then local/CAM+append. */
export class DenseCamKernels {
  private constructor(
    private readonly device: GPUDevice,
    private readonly gpuPackage: CampPlusGpuPackage,
    private readonly arena: CampPlusActivationArena,
    private readonly bottleneckPipelines: Map<string, GPUComputePipeline>,
    private readonly bottleneckLayout: GPUBindGroupLayout,
    private readonly localCamPipeline: GPUComputePipeline,
    private readonly localCamLayout: GPUBindGroupLayout,
  ) {}

  static async create(
    device: GPUDevice,
    gpuPackage: CampPlusGpuPackage,
    arena: CampPlusActivationArena,
  ): Promise<DenseCamKernels> {
    if (device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE) {
      throw new Error(`CAM++ dense kernels require ${WORKGROUP_SIZE} workgroup lanes`);
    }
    if (
      device.limits.maxComputeWorkgroupStorageSize <
      DENSE_CAM_REQUIRED_WORKGROUP_STORAGE_BYTES
    ) {
      throw new Error(
        `CAM++ dense bottleneck requires ${DENSE_CAM_REQUIRED_WORKGROUP_STORAGE_BYTES} workgroup bytes`,
      );
    }
    if (device.limits.maxStorageBuffersPerShaderStage < 7) {
      throw new Error("CAM++ local/CAM fusion requires seven storage buffers per shader stage");
    }

    const bottleneckLayout = device.createBindGroupLayout({
      label: "senko-campplus-dense-bottleneck-bindings",
      entries: storageEntries(4),
    });
    const localCamLayout = device.createBindGroupLayout({
      label: "senko-campplus-dense-local-cam-bindings",
      entries: storageEntries(7),
    });
    const defaultConfiguration = denseBottleneckVariantConfiguration(
      DEFAULT_DENSE_BOTTLENECK_VARIANT,
    );
    const [bottleneckPipeline, localCamPipeline] = await Promise.all([
      createCheckedPipeline(
        device,
        `senko-campplus-dense-bottleneck-${DEFAULT_DENSE_BOTTLENECK_VARIANT}`,
        denseBottleneckPipelineWgsl(defaultConfiguration),
        bottleneckLayout,
      ),
      createCheckedPipeline(
        device,
        "senko-campplus-dense-local-cam",
        DENSE_LOCAL_CAM_WGSL,
        localCamLayout,
      ),
    ]);
    return new DenseCamKernels(
      device,
      gpuPackage,
      arena,
      new Map([
        [
          bottleneckPipelineKey(
            defaultConfiguration.accumulation,
            defaultConfiguration.outputTile,
            defaultConfiguration.workgroupSize,
            defaultConfiguration.weightSource,
          ),
          bottleneckPipeline,
        ],
      ]),
      bottleneckLayout,
      localCamPipeline,
      localCamLayout,
    );
  }

  async prepareBottleneckVariant(
    accumulation: DenseBottleneckAccumulation,
    outputTile: DenseBottleneckOutputTile = 1,
    workgroupSize: DenseBottleneckWorkgroupSize = 128,
    weightSource: DenseBottleneckWeightSource = "direct",
  ): Promise<void> {
    if (outputTile === 1 && accumulation !== "float32") {
      throw new Error("The tile-1 CAM++ diagnostic currently supports FP32 accumulation only");
    }
    if (workgroupSize === 96 && (outputTile !== 1 || accumulation !== "float32")) {
      throw new Error("The 96-lane CAM++ diagnostic supports tile-1 FP32 only");
    }
    if (workgroupSize === 96 && weightSource === "direct") {
      throw new Error("The 96-lane CAM++ diagnostic currently requires cached weights");
    }
    if (weightSource === "direct" && accumulation !== "float32") {
      throw new Error("Direct CAM++ weights require FP32 accumulation");
    }
    if (outputTile === 4 && weightSource !== "direct") {
      throw new Error("The tile-4 CAM++ bottleneck requires direct weights");
    }
    if (
      outputTile === 2 &&
      weightSource === "workgroup-cache" &&
      this.device.limits.maxComputeWorkgroupStorageSize <
        DENSE_CAM_TILE2_WORKGROUP_STORAGE_BYTES
    ) {
      throw new Error(
        `The tile-2 CAM++ diagnostic requires ${DENSE_CAM_TILE2_WORKGROUP_STORAGE_BYTES} workgroup bytes`,
      );
    }
    const key = bottleneckPipelineKey(
      accumulation,
      outputTile,
      workgroupSize,
      weightSource,
    );
    if (this.bottleneckPipelines.has(key)) return;
    const pipeline = await createCheckedPipeline(
      this.device,
      `senko-campplus-dense-bottleneck-tile${outputTile}-wg${workgroupSize}-${weightSource}-${accumulation}`,
      denseBottleneckPipelineWgsl({
        accumulation,
        outputTile,
        workgroupSize,
        weightSource,
      }),
      this.bottleneckLayout,
    );
    this.bottleneckPipelines.set(key, pipeline);
  }

  createBottleneckDispatch(descriptor: DenseBottleneckDescriptor): DenseCamDispatch {
    validateDenseLayout(descriptor, this.arena.byteLength);
    const weight = this.convSection(
      descriptor.layer.bottleneck,
      BOTTLENECK_CHANNELS,
      descriptor.layer.inputChannels,
      1,
    );
    const bias = this.gpuPackage.section(descriptor.layer.bottleneck.bias);
    const affine = this.gpuPackage.section(descriptor.layer.preactivationAffine);
    const accumulation = descriptor.accumulation ?? "float32";
    const outputTile = descriptor.outputTile ?? 1;
    const workgroupSize = descriptor.workgroupSize ?? 128;
    const weightSource = descriptor.weightSource ?? "direct";
    const pipeline = this.bottleneckPipelines.get(
      bottleneckPipelineKey(accumulation, outputTile, workgroupSize, weightSource),
    );
    if (pipeline === undefined) {
      throw new Error(
        `CAM++ bottleneck tile${outputTile}/wg${workgroupSize}/${weightSource}/${accumulation} has not been prepared`,
      );
    }
    validateBias(bias, BOTTLENECK_CHANNELS);
    validateAffine(affine, descriptor.layer.inputChannels);
    const parameters = new Uint32Array([
      descriptor.slab.byteOffset / 2,
      descriptor.scratch.byteOffset / 2,
      descriptor.doubledMean.byteOffset / 2,
      descriptor.batchSize,
      descriptor.layer.inputChannels,
      descriptor.slabChannels,
      descriptor.layer.frames,
      ceilDiv(descriptor.layer.inputChannels, 4),
      BOTTLENECK_CHANNELS / 4,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
    ]);
    const uniform = createUniformBuffer(this.device, `${descriptor.label}-parameters`, parameters);
    try {
      const bindGroup = this.device.createBindGroup({
        label: `${descriptor.label}-bindings`,
        layout: this.bottleneckLayout,
        entries: [
          arenaEntry(this.arena),
          sectionEntry(1, this.gpuPackage.weightsBuffer, weight),
          sectionEntry(2, this.gpuPackage.weightsBuffer, bias),
          sectionEntry(3, this.gpuPackage.weightsBuffer, affine),
          { binding: 4, resource: { buffer: uniform, size: UNIFORM_BYTES } },
        ],
      });
      return new DenseCamDispatch(
        descriptor.label,
        pipeline,
        bindGroup,
        uniform,
        [BOTTLENECK_CHANNELS / 4 / outputTile, descriptor.batchSize, 1],
      );
    } catch (error) {
      uniform.destroy();
      throw error;
    }
  }

  createLocalCamDispatch(descriptor: DenseLocalCamDescriptor): DenseCamDispatch {
    validateDenseLayout(descriptor, this.arena.byteLength);
    if (descriptor.layer.appendChannel + CAM_OUTPUT_CHANNELS > descriptor.slabChannels) {
      throw new Error(`${descriptor.label} append exceeds its dense slab`);
    }
    const localWeight = this.convSection(
      descriptor.layer.local,
      CAM_OUTPUT_CHANNELS,
      BOTTLENECK_CHANNELS,
      3,
    );
    const localBias = this.gpuPackage.section(descriptor.layer.local.bias);
    const attention1Weight = this.convSection(
      descriptor.layer.attention1,
      64,
      BOTTLENECK_CHANNELS,
      1,
    );
    const attention1Bias = this.gpuPackage.section(descriptor.layer.attention1.bias);
    const attention2Weight = this.convSection(
      descriptor.layer.attention2,
      CAM_OUTPUT_CHANNELS,
      64,
      1,
    );
    const attention2Bias = this.gpuPackage.section(descriptor.layer.attention2.bias);
    validateBias(localBias, CAM_OUTPUT_CHANNELS);
    validateBias(attention1Bias, 64);
    validateBias(attention2Bias, CAM_OUTPUT_CHANNELS);
    const parameters = new Uint32Array([
      descriptor.scratch.byteOffset / 2,
      descriptor.doubledMean.byteOffset / 2,
      descriptor.slab.byteOffset / 2,
      descriptor.batchSize,
      descriptor.slabChannels,
      descriptor.layer.frames,
      descriptor.layer.appendChannel,
      descriptor.layer.localDilation,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
    ]);
    const uniform = createUniformBuffer(this.device, `${descriptor.label}-parameters`, parameters);
    try {
      const bindGroup = this.device.createBindGroup({
        label: `${descriptor.label}-bindings`,
        layout: this.localCamLayout,
        entries: [
          arenaEntry(this.arena),
          sectionEntry(1, this.gpuPackage.weightsBuffer, localWeight),
          sectionEntry(2, this.gpuPackage.weightsBuffer, localBias),
          sectionEntry(3, this.gpuPackage.weightsBuffer, attention1Weight),
          sectionEntry(4, this.gpuPackage.weightsBuffer, attention1Bias),
          sectionEntry(5, this.gpuPackage.weightsBuffer, attention2Weight),
          sectionEntry(6, this.gpuPackage.weightsBuffer, attention2Bias),
          { binding: 7, resource: { buffer: uniform, size: UNIFORM_BYTES } },
        ],
      });
      return new DenseCamDispatch(
        descriptor.label,
        this.localCamPipeline,
        bindGroup,
        uniform,
        [CAM_OUTPUT_CHANNELS / 4, descriptor.batchSize, 1],
      );
    } catch (error) {
      uniform.destroy();
      throw error;
    }
  }

  private convSection(
    convolution: PackedConvolutionRef,
    outputChannels: number,
    inputChannels: number,
    kernelElements: number,
  ): CampPlusPackedSection {
    const section = this.gpuPackage.section(convolution.weight);
    if (
      section.kind !== "conv_weight" ||
      section.logicalShape.length !== 3 ||
      section.logicalShape[0] !== outputChannels ||
      section.logicalShape[1] !== inputChannels ||
      section.logicalShape[2] !== kernelElements
    ) {
      throw new Error(`Unexpected CAM++ dense convolution section ${section.id}`);
    }
    return section;
  }
}

function validateDenseLayout(
  descriptor: DenseBottleneckDescriptor | DenseLocalCamDescriptor,
  arenaBytes: number,
): void {
  const { layer } = descriptor;
  if (
    !Number.isSafeInteger(descriptor.batchSize) ||
    descriptor.batchSize <= 0 ||
    !Number.isSafeInteger(descriptor.slabChannels) ||
    descriptor.slabChannels < layer.inputChannels ||
    layer.inputChannels > MAX_DENSE_INPUT_CHANNELS ||
    layer.bottleneckChannels !== BOTTLENECK_CHANNELS ||
    layer.outputChannels !== CAM_OUTPUT_CHANNELS ||
    layer.frames > WORKGROUP_SIZE
  ) {
    throw new Error(`${descriptor.label} has an unsupported dense-layer contract`);
  }
  validateSlice(
    descriptor.slab,
    descriptor.batchSize * descriptor.slabChannels * layer.frames * 2,
    arenaBytes,
  );
  validateSlice(
    descriptor.scratch,
    descriptor.batchSize * BOTTLENECK_CHANNELS * layer.frames * 2,
    arenaBytes,
  );
  validateSlice(
    descriptor.doubledMean,
    descriptor.batchSize * BOTTLENECK_CHANNELS * 2,
    arenaBytes,
  );
  if (
    rangesOverlap(descriptor.slab, descriptor.scratch) ||
    rangesOverlap(descriptor.slab, descriptor.doubledMean) ||
    rangesOverlap(descriptor.scratch, descriptor.doubledMean)
  ) {
    throw new Error(`${descriptor.label} dense arena ranges overlap`);
  }
}

function validateBias(section: CampPlusPackedSection, outputChannels: number): void {
  if (section.kind !== "conv_bias" || section.logicalShape[0] !== outputChannels) {
    throw new Error(`Unexpected CAM++ dense bias section ${section.id}`);
  }
}

function validateAffine(section: CampPlusPackedSection, inputChannels: number): void {
  if (section.kind !== "batch_norm_affine" || section.logicalShape[0] !== inputChannels) {
    throw new Error(`Unexpected CAM++ dense affine section ${section.id}`);
  }
}

function validateSlice(slice: CampPlusArenaSlice, requiredBytes: number, arenaBytes: number): void {
  if (
    slice.byteOffset % 256 !== 0 ||
    slice.byteLength < requiredBytes ||
    slice.byteOffset + slice.byteLength > arenaBytes
  ) {
    throw new RangeError(`CAM++ arena slice ${slice.label} does not fit its dense tensor`);
  }
}

function rangesOverlap(left: CampPlusArenaSlice, right: CampPlusArenaSlice): boolean {
  return (
    left.byteOffset < right.byteOffset + right.byteLength &&
    right.byteOffset < left.byteOffset + left.byteLength
  );
}

function storageEntries(storageCount: number): GPUBindGroupLayoutEntry[] {
  const entries: GPUBindGroupLayoutEntry[] = [];
  for (let binding = 0; binding < storageCount; binding += 1) {
    entries.push({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 ? "storage" : "read-only-storage" },
    });
  }
  entries.push({
    binding: storageCount,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: "uniform", minBindingSize: UNIFORM_BYTES },
  });
  return entries;
}

async function createCheckedPipeline(
  device: GPUDevice,
  label: string,
  code: string,
  bindGroupLayout: GPUBindGroupLayout,
): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label, code });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length > 0) {
    throw new Error(`${label} WGSL failed: ${errors.map((item) => item.message).join("; ")}`);
  }
  return device.createComputePipelineAsync({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module, entryPoint: "main" },
  });
}

function createUniformBuffer(
  device: GPUDevice,
  label: string,
  parameters: Uint32Array<ArrayBuffer>,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, parameters);
  return buffer;
}

function arenaEntry(arena: CampPlusActivationArena): GPUBindGroupEntry {
  return {
    binding: 0,
    resource: { buffer: arena.buffer, size: arena.byteLength },
  };
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

function ceilDiv(value: number, divisor: number): number {
  return Math.floor((value + divisor - 1) / divisor);
}

function bottleneckPipelineKey(
  accumulation: DenseBottleneckAccumulation,
  outputTile: DenseBottleneckOutputTile,
  workgroupSize: DenseBottleneckWorkgroupSize = 128,
  weightSource: DenseBottleneckWeightSource = "direct",
): string {
  return `${outputTile}:${workgroupSize}:${weightSource}:${accumulation}`;
}

function denseBottleneckPipelineWgsl(
  configuration: DenseBottleneckVariantConfiguration,
): string {
  const { accumulation, outputTile, workgroupSize, weightSource } = configuration;
  if (outputTile === 1) {
    if (weightSource === "direct") return DENSE_BOTTLENECK_TILE1_DIRECT_WGSL;
    return workgroupSize === 96
      ? DENSE_BOTTLENECK_TILE1_WG96_WGSL
      : DENSE_BOTTLENECK_TILE1_WGSL;
  }
  if (weightSource !== "direct") return denseBottleneckWgsl(accumulation);
  return outputTile === 2
    ? DENSE_BOTTLENECK_TILE2_DIRECT_WGSL
    : DENSE_BOTTLENECK_TILE4_DIRECT_WGSL;
}

export const DENSE_BOTTLENECK_TILE1_WGSL = /* wgsl */ `
enable f16;

struct Parameters {
  slab_offset: u32,
  scratch_offset: u32,
  doubled_mean_offset: u32,
  batch_size: u32,
  input_channels: u32,
  slab_channels: u32,
  frames: u32,
  input_groups: u32,
  output_groups: u32,
  reserved_0: u32,
  reserved_1: u32,
  reserved_2: u32,
  reserved_3: u32,
  reserved_4: u32,
  reserved_5: u32,
  reserved_6: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> affine: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

var<workgroup> weight_cache: array<vec4<f16>, 992>;
var<workgroup> mean_reduction: array<vec4<f32>, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let output_group = workgroup_id.x;
  let batch = workgroup_id.y;
  var cache_index = local_id.x;
  while (cache_index < parameters.input_channels) {
    let input_group = cache_index / 4u;
    let input_lane = cache_index & 3u;
    let packed_index =
      (output_group * parameters.input_groups + input_group) * 4u + input_lane;
    weight_cache[cache_index] = weights[packed_index];
    cache_index += 128u;
  }
  workgroupBarrier();

  let frame = local_id.x;
  var rounded = vec4<f16>(f16(0.0));
  if (frame < parameters.frames && batch < parameters.batch_size) {
    var accumulators = vec4<f32>(biases[output_group]);
    let batch_channel_base = batch * parameters.slab_channels;
    for (var input_group = 0u; input_group < parameters.input_groups; input_group += 1u) {
      let channel_base = input_group * 4u;
      let slab_index =
        parameters.slab_offset +
        ((batch_channel_base + channel_base) * parameters.frames + frame);
      let scale = affine[input_group * 2u];
      let shift = affine[input_group * 2u + 1u];
      let activated_0 = max(f16(f32(arena[slab_index]) * scale[0] + shift[0]), f16(0.0));
      let activated_1 = max(f16(f32(arena[slab_index + parameters.frames]) * scale[1] + shift[1]), f16(0.0));
      let activated_2 = max(f16(f32(arena[slab_index + 2u * parameters.frames]) * scale[2] + shift[2]), f16(0.0));
      let activated_3 = max(f16(f32(arena[slab_index + 3u * parameters.frames]) * scale[3] + shift[3]), f16(0.0));
      accumulators = fma(vec4<f32>(f32(activated_0)), vec4<f32>(weight_cache[channel_base]), accumulators);
      accumulators = fma(vec4<f32>(f32(activated_1)), vec4<f32>(weight_cache[channel_base + 1u]), accumulators);
      accumulators = fma(vec4<f32>(f32(activated_2)), vec4<f32>(weight_cache[channel_base + 2u]), accumulators);
      accumulators = fma(vec4<f32>(f32(activated_3)), vec4<f32>(weight_cache[channel_base + 3u]), accumulators);
    }
    rounded = max(vec4<f16>(accumulators), vec4<f16>(f16(0.0)));
    let output_channel_base = output_group * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let scratch_index =
        parameters.scratch_offset +
        ((batch * 128u + output_channel_base + lane) * parameters.frames + frame);
      arena[scratch_index] = rounded[lane];
    }
  }
  mean_reduction[local_id.x] = vec4<f32>(rounded);
  workgroupBarrier();

  var stride = 64u;
  loop {
    if (local_id.x < stride) {
      mean_reduction[local_id.x] += mean_reduction[local_id.x + stride];
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  if (local_id.x == 0u) {
    let mean = vec4<f16>(mean_reduction[0] / f32(parameters.frames));
    let doubled = vec4<f16>(mean * vec4<f16>(f16(2.0)));
    let output_channel_base = output_group * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let mean_index =
        parameters.doubled_mean_offset + batch * 128u + output_channel_base + lane;
      arena[mean_index] = doubled[lane];
    }
  }
}
`;

export const DENSE_BOTTLENECK_TILE1_WG96_WGSL =
  DENSE_BOTTLENECK_TILE1_WGSL.replace("array<vec4<f32>, 128>", "array<vec4<f32>, 96>")
    .replace("@workgroup_size(128)", "@workgroup_size(96)")
    .replace("cache_index += 128u;", "cache_index += 96u;")
    .replace(
      `  var stride = 64u;
  loop {
    if (local_id.x < stride) {
      mean_reduction[local_id.x] += mean_reduction[local_id.x + stride];
    }
    workgroupBarrier();`,
      `  if (local_id.x < 32u) {
    mean_reduction[local_id.x] += mean_reduction[local_id.x + 64u];
  }
  workgroupBarrier();
  var stride = 32u;
  loop {
    if (local_id.x < stride) {
      mean_reduction[local_id.x] += mean_reduction[local_id.x + stride];
    }
    workgroupBarrier();`,
    );

export const DENSE_BOTTLENECK_TILE1_DIRECT_WGSL =
  DENSE_BOTTLENECK_TILE1_WGSL.replace(
    "var<workgroup> weight_cache: array<vec4<f16>, 992>;\n",
    "",
  )
    .replace(
      `  var cache_index = local_id.x;
  while (cache_index < parameters.input_channels) {
    let input_group = cache_index / 4u;
    let input_lane = cache_index & 3u;
    let packed_index =
      (output_group * parameters.input_groups + input_group) * 4u + input_lane;
    weight_cache[cache_index] = weights[packed_index];
    cache_index += 128u;
  }
  workgroupBarrier();
`,
      "",
    )
    .replaceAll(
      "weight_cache[channel_base + 3u]",
      "weights[(output_group * parameters.input_groups + input_group) * 4u + 3u]",
    )
    .replaceAll(
      "weight_cache[channel_base + 2u]",
      "weights[(output_group * parameters.input_groups + input_group) * 4u + 2u]",
    )
    .replaceAll(
      "weight_cache[channel_base + 1u]",
      "weights[(output_group * parameters.input_groups + input_group) * 4u + 1u]",
    )
    .replaceAll(
      "weight_cache[channel_base]",
      "weights[(output_group * parameters.input_groups + input_group) * 4u]",
    );

/**
 * Direct-weight FP32 bottleneck that shares each BN/ReLU activation across two
 * adjacent output vec4s. Per-output FMA and reduction order matches tile 1.
 */
export const DENSE_BOTTLENECK_TILE2_DIRECT_WGSL = /* wgsl */ `
enable f16;

struct Parameters {
  slab_offset: u32,
  scratch_offset: u32,
  doubled_mean_offset: u32,
  batch_size: u32,
  input_channels: u32,
  slab_channels: u32,
  frames: u32,
  input_groups: u32,
  output_groups: u32,
  reserved_0: u32,
  reserved_1: u32,
  reserved_2: u32,
  reserved_3: u32,
  reserved_4: u32,
  reserved_5: u32,
  reserved_6: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> affine: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

struct OutputPair {
  first: vec4<f32>,
  second: vec4<f32>,
}

var<workgroup> mean_reduction: array<OutputPair, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let first_output_group = workgroup_id.x * 2u;
  let second_output_group = first_output_group + 1u;
  let batch = workgroup_id.y;
  let frame = local_id.x;
  var first_rounded = vec4<f16>(f16(0.0));
  var second_rounded = vec4<f16>(f16(0.0));
  if (frame < parameters.frames && batch < parameters.batch_size) {
    var first_accumulators = vec4<f32>(biases[first_output_group]);
    var second_accumulators = vec4<f32>(biases[second_output_group]);
    let batch_channel_base = batch * parameters.slab_channels;
    for (var input_group = 0u; input_group < parameters.input_groups; input_group += 1u) {
      let channel_base = input_group * 4u;
      let slab_index =
        parameters.slab_offset +
        ((batch_channel_base + channel_base) * parameters.frames + frame);
      let scale = affine[input_group * 2u];
      let shift = affine[input_group * 2u + 1u];
      let activated_0 = max(f16(f32(arena[slab_index]) * scale[0] + shift[0]), f16(0.0));
      let activated_1 = max(f16(f32(arena[slab_index + parameters.frames]) * scale[1] + shift[1]), f16(0.0));
      let activated_2 = max(f16(f32(arena[slab_index + 2u * parameters.frames]) * scale[2] + shift[2]), f16(0.0));
      let activated_3 = max(f16(f32(arena[slab_index + 3u * parameters.frames]) * scale[3] + shift[3]), f16(0.0));
      let first_weight_index =
        (first_output_group * parameters.input_groups + input_group) * 4u;
      let second_weight_index =
        (second_output_group * parameters.input_groups + input_group) * 4u;
      first_accumulators = fma(vec4<f32>(f32(activated_0)), vec4<f32>(weights[first_weight_index]), first_accumulators);
      first_accumulators = fma(vec4<f32>(f32(activated_1)), vec4<f32>(weights[first_weight_index + 1u]), first_accumulators);
      first_accumulators = fma(vec4<f32>(f32(activated_2)), vec4<f32>(weights[first_weight_index + 2u]), first_accumulators);
      first_accumulators = fma(vec4<f32>(f32(activated_3)), vec4<f32>(weights[first_weight_index + 3u]), first_accumulators);
      second_accumulators = fma(vec4<f32>(f32(activated_0)), vec4<f32>(weights[second_weight_index]), second_accumulators);
      second_accumulators = fma(vec4<f32>(f32(activated_1)), vec4<f32>(weights[second_weight_index + 1u]), second_accumulators);
      second_accumulators = fma(vec4<f32>(f32(activated_2)), vec4<f32>(weights[second_weight_index + 2u]), second_accumulators);
      second_accumulators = fma(vec4<f32>(f32(activated_3)), vec4<f32>(weights[second_weight_index + 3u]), second_accumulators);
    }
    first_rounded = max(vec4<f16>(first_accumulators), vec4<f16>(f16(0.0)));
    second_rounded = max(vec4<f16>(second_accumulators), vec4<f16>(f16(0.0)));
    let first_output_channel = first_output_group * 4u;
    let second_output_channel = second_output_group * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let first_scratch_index =
        parameters.scratch_offset +
        ((batch * 128u + first_output_channel + lane) * parameters.frames + frame);
      let second_scratch_index =
        parameters.scratch_offset +
        ((batch * 128u + second_output_channel + lane) * parameters.frames + frame);
      arena[first_scratch_index] = first_rounded[lane];
      arena[second_scratch_index] = second_rounded[lane];
    }
  }
  mean_reduction[local_id.x].first = vec4<f32>(first_rounded);
  mean_reduction[local_id.x].second = vec4<f32>(second_rounded);
  workgroupBarrier();

  var stride = 64u;
  loop {
    if (local_id.x < stride) {
      mean_reduction[local_id.x].first += mean_reduction[local_id.x + stride].first;
      mean_reduction[local_id.x].second += mean_reduction[local_id.x + stride].second;
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  if (local_id.x == 0u) {
    let first_mean = vec4<f16>(mean_reduction[0].first / f32(parameters.frames));
    let second_mean = vec4<f16>(mean_reduction[0].second / f32(parameters.frames));
    let first_doubled = vec4<f16>(first_mean * vec4<f16>(f16(2.0)));
    let second_doubled = vec4<f16>(second_mean * vec4<f16>(f16(2.0)));
    let first_output_channel = first_output_group * 4u;
    let second_output_channel = second_output_group * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let first_mean_index =
        parameters.doubled_mean_offset + batch * 128u + first_output_channel + lane;
      let second_mean_index =
        parameters.doubled_mean_offset + batch * 128u + second_output_channel + lane;
      arena[first_mean_index] = first_doubled[lane];
      arena[second_mean_index] = second_doubled[lane];
    }
  }
}
`;

/** Direct-weight FP32 bottleneck sharing each activation across four vec4 outputs. */
export const DENSE_BOTTLENECK_TILE4_DIRECT_WGSL = /* wgsl */ `
enable f16;

struct Parameters {
  slab_offset: u32,
  scratch_offset: u32,
  doubled_mean_offset: u32,
  batch_size: u32,
  input_channels: u32,
  slab_channels: u32,
  frames: u32,
  input_groups: u32,
  output_groups: u32,
  reserved_0: u32,
  reserved_1: u32,
  reserved_2: u32,
  reserved_3: u32,
  reserved_4: u32,
  reserved_5: u32,
  reserved_6: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> affine: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

struct OutputQuad {
  first: vec4<f32>,
  second: vec4<f32>,
  third: vec4<f32>,
  fourth: vec4<f32>,
}

var<workgroup> mean_reduction: array<OutputQuad, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let first_output_group = workgroup_id.x * 4u;
  let second_output_group = first_output_group + 1u;
  let third_output_group = first_output_group + 2u;
  let fourth_output_group = first_output_group + 3u;
  let batch = workgroup_id.y;
  let frame = local_id.x;
  var first_rounded = vec4<f16>(f16(0.0));
  var second_rounded = vec4<f16>(f16(0.0));
  var third_rounded = vec4<f16>(f16(0.0));
  var fourth_rounded = vec4<f16>(f16(0.0));
  if (frame < parameters.frames && batch < parameters.batch_size) {
    var first_accumulators = vec4<f32>(biases[first_output_group]);
    var second_accumulators = vec4<f32>(biases[second_output_group]);
    var third_accumulators = vec4<f32>(biases[third_output_group]);
    var fourth_accumulators = vec4<f32>(biases[fourth_output_group]);
    let batch_channel_base = batch * parameters.slab_channels;
    for (var input_group = 0u; input_group < parameters.input_groups; input_group += 1u) {
      let channel_base = input_group * 4u;
      let slab_index =
        parameters.slab_offset +
        ((batch_channel_base + channel_base) * parameters.frames + frame);
      let scale = affine[input_group * 2u];
      let shift = affine[input_group * 2u + 1u];
      let activated_0 = max(f16(f32(arena[slab_index]) * scale[0] + shift[0]), f16(0.0));
      let activated_1 = max(f16(f32(arena[slab_index + parameters.frames]) * scale[1] + shift[1]), f16(0.0));
      let activated_2 = max(f16(f32(arena[slab_index + 2u * parameters.frames]) * scale[2] + shift[2]), f16(0.0));
      let activated_3 = max(f16(f32(arena[slab_index + 3u * parameters.frames]) * scale[3] + shift[3]), f16(0.0));
      let first_weight_index =
        (first_output_group * parameters.input_groups + input_group) * 4u;
      let second_weight_index =
        (second_output_group * parameters.input_groups + input_group) * 4u;
      let third_weight_index =
        (third_output_group * parameters.input_groups + input_group) * 4u;
      let fourth_weight_index =
        (fourth_output_group * parameters.input_groups + input_group) * 4u;
      first_accumulators = fma(vec4<f32>(f32(activated_0)), vec4<f32>(weights[first_weight_index]), first_accumulators);
      first_accumulators = fma(vec4<f32>(f32(activated_1)), vec4<f32>(weights[first_weight_index + 1u]), first_accumulators);
      first_accumulators = fma(vec4<f32>(f32(activated_2)), vec4<f32>(weights[first_weight_index + 2u]), first_accumulators);
      first_accumulators = fma(vec4<f32>(f32(activated_3)), vec4<f32>(weights[first_weight_index + 3u]), first_accumulators);
      second_accumulators = fma(vec4<f32>(f32(activated_0)), vec4<f32>(weights[second_weight_index]), second_accumulators);
      second_accumulators = fma(vec4<f32>(f32(activated_1)), vec4<f32>(weights[second_weight_index + 1u]), second_accumulators);
      second_accumulators = fma(vec4<f32>(f32(activated_2)), vec4<f32>(weights[second_weight_index + 2u]), second_accumulators);
      second_accumulators = fma(vec4<f32>(f32(activated_3)), vec4<f32>(weights[second_weight_index + 3u]), second_accumulators);
      third_accumulators = fma(vec4<f32>(f32(activated_0)), vec4<f32>(weights[third_weight_index]), third_accumulators);
      third_accumulators = fma(vec4<f32>(f32(activated_1)), vec4<f32>(weights[third_weight_index + 1u]), third_accumulators);
      third_accumulators = fma(vec4<f32>(f32(activated_2)), vec4<f32>(weights[third_weight_index + 2u]), third_accumulators);
      third_accumulators = fma(vec4<f32>(f32(activated_3)), vec4<f32>(weights[third_weight_index + 3u]), third_accumulators);
      fourth_accumulators = fma(vec4<f32>(f32(activated_0)), vec4<f32>(weights[fourth_weight_index]), fourth_accumulators);
      fourth_accumulators = fma(vec4<f32>(f32(activated_1)), vec4<f32>(weights[fourth_weight_index + 1u]), fourth_accumulators);
      fourth_accumulators = fma(vec4<f32>(f32(activated_2)), vec4<f32>(weights[fourth_weight_index + 2u]), fourth_accumulators);
      fourth_accumulators = fma(vec4<f32>(f32(activated_3)), vec4<f32>(weights[fourth_weight_index + 3u]), fourth_accumulators);
    }
    first_rounded = max(vec4<f16>(first_accumulators), vec4<f16>(f16(0.0)));
    second_rounded = max(vec4<f16>(second_accumulators), vec4<f16>(f16(0.0)));
    third_rounded = max(vec4<f16>(third_accumulators), vec4<f16>(f16(0.0)));
    fourth_rounded = max(vec4<f16>(fourth_accumulators), vec4<f16>(f16(0.0)));
    let first_output_channel = first_output_group * 4u;
    let second_output_channel = second_output_group * 4u;
    let third_output_channel = third_output_group * 4u;
    let fourth_output_channel = fourth_output_group * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let first_scratch_index =
        parameters.scratch_offset +
        ((batch * 128u + first_output_channel + lane) * parameters.frames + frame);
      let second_scratch_index =
        parameters.scratch_offset +
        ((batch * 128u + second_output_channel + lane) * parameters.frames + frame);
      let third_scratch_index =
        parameters.scratch_offset +
        ((batch * 128u + third_output_channel + lane) * parameters.frames + frame);
      let fourth_scratch_index =
        parameters.scratch_offset +
        ((batch * 128u + fourth_output_channel + lane) * parameters.frames + frame);
      arena[first_scratch_index] = first_rounded[lane];
      arena[second_scratch_index] = second_rounded[lane];
      arena[third_scratch_index] = third_rounded[lane];
      arena[fourth_scratch_index] = fourth_rounded[lane];
    }
  }
  mean_reduction[local_id.x].first = vec4<f32>(first_rounded);
  mean_reduction[local_id.x].second = vec4<f32>(second_rounded);
  mean_reduction[local_id.x].third = vec4<f32>(third_rounded);
  mean_reduction[local_id.x].fourth = vec4<f32>(fourth_rounded);
  workgroupBarrier();

  var stride = 64u;
  loop {
    if (local_id.x < stride) {
      mean_reduction[local_id.x].first += mean_reduction[local_id.x + stride].first;
      mean_reduction[local_id.x].second += mean_reduction[local_id.x + stride].second;
      mean_reduction[local_id.x].third += mean_reduction[local_id.x + stride].third;
      mean_reduction[local_id.x].fourth += mean_reduction[local_id.x + stride].fourth;
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  if (local_id.x == 0u) {
    let first_mean = vec4<f16>(mean_reduction[0].first / f32(parameters.frames));
    let second_mean = vec4<f16>(mean_reduction[0].second / f32(parameters.frames));
    let third_mean = vec4<f16>(mean_reduction[0].third / f32(parameters.frames));
    let fourth_mean = vec4<f16>(mean_reduction[0].fourth / f32(parameters.frames));
    let first_doubled = vec4<f16>(first_mean * vec4<f16>(f16(2.0)));
    let second_doubled = vec4<f16>(second_mean * vec4<f16>(f16(2.0)));
    let third_doubled = vec4<f16>(third_mean * vec4<f16>(f16(2.0)));
    let fourth_doubled = vec4<f16>(fourth_mean * vec4<f16>(f16(2.0)));
    let first_output_channel = first_output_group * 4u;
    let second_output_channel = second_output_group * 4u;
    let third_output_channel = third_output_group * 4u;
    let fourth_output_channel = fourth_output_group * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let first_mean_index = parameters.doubled_mean_offset + batch * 128u + first_output_channel + lane;
      let second_mean_index = parameters.doubled_mean_offset + batch * 128u + second_output_channel + lane;
      let third_mean_index = parameters.doubled_mean_offset + batch * 128u + third_output_channel + lane;
      let fourth_mean_index = parameters.doubled_mean_offset + batch * 128u + fourth_output_channel + lane;
      arena[first_mean_index] = first_doubled[lane];
      arena[second_mean_index] = second_doubled[lane];
      arena[third_mean_index] = third_doubled[lane];
      arena[fourth_mean_index] = fourth_doubled[lane];
    }
  }
}
`;

const DENSE_BOTTLENECK_TEMPLATE = /* wgsl */ `
enable f16;

struct Parameters {
  slab_offset: u32,
  scratch_offset: u32,
  doubled_mean_offset: u32,
  batch_size: u32,
  input_channels: u32,
  slab_channels: u32,
  frames: u32,
  input_groups: u32,
  output_groups: u32,
  reserved_0: u32,
  reserved_1: u32,
  reserved_2: u32,
  reserved_3: u32,
  reserved_4: u32,
  reserved_5: u32,
  reserved_6: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> affine: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> parameters: Parameters;

struct OutputPair {
  first: vec4<f32>,
  second: vec4<f32>,
}

var<workgroup> weight_cache: array<vec4<f16>, 1984>;
var<workgroup> mean_reduction: array<OutputPair, 128>;

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let first_output_group = workgroup_id.x * 2u;
  let second_output_group = first_output_group + 1u;
  let batch = workgroup_id.y;
  var cache_index = local_id.x;
  let cached_vectors = parameters.input_channels * 2u;
  while (cache_index < cached_vectors) {
    let tile_index = cache_index / parameters.input_channels;
    let input_channel = cache_index - tile_index * parameters.input_channels;
    let input_group = input_channel / 4u;
    let input_lane = input_channel & 3u;
    let output_group = first_output_group + tile_index;
    let packed_index =
      (output_group * parameters.input_groups + input_group) * 4u + input_lane;
    weight_cache[cache_index] = weights[packed_index];
    cache_index += 128u;
  }
  workgroupBarrier();

  let frame = local_id.x;
  var first_rounded = vec4<f16>(f16(0.0));
  var second_rounded = vec4<f16>(f16(0.0));
  if (frame < parameters.frames && batch < parameters.batch_size) {
    __ACCUMULATOR_DECLARATION__
    for (var input_channel = 0u; input_channel < parameters.input_channels; input_channel += 1u) {
      let slab_index =
        parameters.slab_offset +
        ((batch * parameters.slab_channels + input_channel) * parameters.frames + frame);
      let input_group = input_channel / 4u;
      let input_lane = input_channel & 3u;
      let scale = affine[input_group * 2u][input_lane];
      let shift = affine[input_group * 2u + 1u][input_lane];
      let activated = max(f16(f32(arena[slab_index]) * scale + shift), f16(0.0));
      __ACCUMULATION_STEP__
    }
    __ACCUMULATOR_FINALIZE__
    let first_output_channel = first_output_group * 4u;
    let second_output_channel = second_output_group * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let first_scratch_index =
        parameters.scratch_offset +
        ((batch * 128u + first_output_channel + lane) * parameters.frames + frame);
      let second_scratch_index =
        parameters.scratch_offset +
        ((batch * 128u + second_output_channel + lane) * parameters.frames + frame);
      arena[first_scratch_index] = first_rounded[lane];
      arena[second_scratch_index] = second_rounded[lane];
    }
  }
  mean_reduction[local_id.x].first = vec4<f32>(first_rounded);
  mean_reduction[local_id.x].second = vec4<f32>(second_rounded);
  workgroupBarrier();

  var stride = 64u;
  loop {
    if (local_id.x < stride) {
      mean_reduction[local_id.x].first += mean_reduction[local_id.x + stride].first;
      mean_reduction[local_id.x].second += mean_reduction[local_id.x + stride].second;
    }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  if (local_id.x == 0u) {
    let first_mean = vec4<f16>(mean_reduction[0].first / f32(parameters.frames));
    let second_mean = vec4<f16>(mean_reduction[0].second / f32(parameters.frames));
    let first_doubled = vec4<f16>(first_mean * vec4<f16>(f16(2.0)));
    let second_doubled = vec4<f16>(second_mean * vec4<f16>(f16(2.0)));
    let first_output_channel = first_output_group * 4u;
    let second_output_channel = second_output_group * 4u;
    for (var lane = 0u; lane < 4u; lane += 1u) {
      let first_mean_index =
        parameters.doubled_mean_offset + batch * 128u + first_output_channel + lane;
      let second_mean_index =
        parameters.doubled_mean_offset + batch * 128u + second_output_channel + lane;
      arena[first_mean_index] = first_doubled[lane];
      arena[second_mean_index] = second_doubled[lane];
    }
  }
}
`;

export const DENSE_BOTTLENECK_WGSL = denseBottleneckWgsl("float32");

function denseBottleneckWgsl(accumulation: DenseBottleneckAccumulation): string {
  const snippets = bottleneckAccumulationSnippets(accumulation);
  return DENSE_BOTTLENECK_TEMPLATE.replace(
    "__ACCUMULATOR_DECLARATION__",
    snippets.declaration,
  )
    .replace("__ACCUMULATION_STEP__", snippets.step)
    .replace("__ACCUMULATOR_FINALIZE__", snippets.finalize);
}

function bottleneckAccumulationSnippets(accumulation: DenseBottleneckAccumulation): {
  readonly declaration: string;
  readonly step: string;
  readonly finalize: string;
} {
  if (accumulation === "float32") {
    return {
      declaration: `
    var first_accumulators = vec4<f32>(biases[first_output_group]);
    var second_accumulators = vec4<f32>(biases[second_output_group]);`,
      step: `
      first_accumulators = fma(
        vec4<f32>(f32(activated)),
        vec4<f32>(weight_cache[input_channel]),
        first_accumulators,
      );
      second_accumulators = fma(
        vec4<f32>(f32(activated)),
        vec4<f32>(weight_cache[parameters.input_channels + input_channel]),
        second_accumulators,
      );`,
      finalize: `
    first_rounded = max(vec4<f16>(first_accumulators), vec4<f16>(f16(0.0)));
    second_rounded = max(vec4<f16>(second_accumulators), vec4<f16>(f16(0.0)));`,
    };
  }
  if (accumulation === "float16-chunk32") {
    return {
      declaration: `
    var first_accumulators = vec4<f32>(biases[first_output_group]);
    var second_accumulators = vec4<f32>(biases[second_output_group]);
    var first_partial = vec4<f16>(f16(0.0));
    var second_partial = vec4<f16>(f16(0.0));`,
      step: `
      first_partial = fma(
        vec4<f16>(activated),
        weight_cache[input_channel],
        first_partial,
      );
      second_partial = fma(
        vec4<f16>(activated),
        weight_cache[parameters.input_channels + input_channel],
        second_partial,
      );
      if (((input_channel + 1u) % 32u) == 0u || input_channel + 1u == parameters.input_channels) {
        first_accumulators += vec4<f32>(first_partial);
        second_accumulators += vec4<f32>(second_partial);
        first_partial = vec4<f16>(f16(0.0));
        second_partial = vec4<f16>(f16(0.0));
      }`,
      finalize: `
    first_rounded = max(vec4<f16>(first_accumulators), vec4<f16>(f16(0.0)));
    second_rounded = max(vec4<f16>(second_accumulators), vec4<f16>(f16(0.0)));`,
    };
  }
  return {
    declaration: `
    var first_accumulators = biases[first_output_group];
    var second_accumulators = biases[second_output_group];`,
    step: `
      first_accumulators = fma(
        vec4<f16>(activated),
        weight_cache[input_channel],
        first_accumulators,
      );
      second_accumulators = fma(
        vec4<f16>(activated),
        weight_cache[parameters.input_channels + input_channel],
        second_accumulators,
      );`,
    finalize: `
    first_rounded = max(first_accumulators, vec4<f16>(f16(0.0)));
    second_rounded = max(second_accumulators, vec4<f16>(f16(0.0)));`,
  };
}

export const DENSE_LOCAL_CAM_WGSL = /* wgsl */ `
enable f16;

struct Parameters {
  scratch_offset: u32,
  doubled_mean_offset: u32,
  slab_offset: u32,
  batch_size: u32,
  slab_channels: u32,
  frames: u32,
  append_channel: u32,
  dilation: u32,
  reserved_0: u32,
  reserved_1: u32,
  reserved_2: u32,
  reserved_3: u32,
  reserved_4: u32,
  reserved_5: u32,
  reserved_6: u32,
  reserved_7: u32,
}

@group(0) @binding(0) var<storage, read_write> arena: array<f16>;
@group(0) @binding(1) var<storage, read> local_weights: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> local_biases: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> attention1_weights: array<vec4<f16>>;
@group(0) @binding(4) var<storage, read> attention1_biases: array<vec4<f16>>;
@group(0) @binding(5) var<storage, read> attention2_weights: array<vec4<f16>>;
@group(0) @binding(6) var<storage, read> attention2_biases: array<vec4<f16>>;
@group(0) @binding(7) var<uniform> parameters: Parameters;

var<workgroup> local_weight_cache: array<vec4<f16>, 384>;
var<workgroup> attention_hidden: array<f16, 64>;
var<workgroup> attention_gate: vec4<f16>;

@compute @workgroup_size(128)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let output_group = workgroup_id.x;
  let batch = workgroup_id.y;
  var cache_index = local_id.x;
  while (cache_index < 384u) {
    let kernel_index = cache_index / 128u;
    let input_channel = cache_index - kernel_index * 128u;
    let input_group = input_channel / 4u;
    let input_lane = input_channel & 3u;
    let packed_index =
      ((kernel_index * 8u + output_group) * 32u + input_group) * 4u + input_lane;
    local_weight_cache[cache_index] = local_weights[packed_index];
    cache_index += 128u;
  }

  if (local_id.x < 64u) {
    let hidden_channel = local_id.x;
    let hidden_group = hidden_channel / 4u;
    let hidden_lane = hidden_channel & 3u;
    var accumulator = f32(attention1_biases[hidden_group][hidden_lane]);
    for (var input_channel = 0u; input_channel < 128u; input_channel += 1u) {
      let input_group = input_channel / 4u;
      let input_lane = input_channel & 3u;
      let packed_index = (hidden_group * 32u + input_group) * 4u + input_lane;
      let weight = f32(attention1_weights[packed_index][hidden_lane]);
      let mean = f32(arena[parameters.doubled_mean_offset + batch * 128u + input_channel]);
      accumulator = fma(mean, weight, accumulator);
    }
    attention_hidden[hidden_channel] = max(f16(accumulator), f16(0.0));
  }
  workgroupBarrier();

  if (local_id.x == 0u) {
    var logits = vec4<f32>(attention2_biases[output_group]);
    for (var hidden_channel = 0u; hidden_channel < 64u; hidden_channel += 1u) {
      let hidden_group = hidden_channel / 4u;
      let hidden_lane = hidden_channel & 3u;
      let packed_index = (output_group * 16u + hidden_group) * 4u + hidden_lane;
      logits = fma(
        vec4<f32>(f32(attention_hidden[hidden_channel])),
        vec4<f32>(attention2_weights[packed_index]),
        logits,
      );
    }
    let rounded_logits = vec4<f16>(logits);
    attention_gate = vec4<f16>(
      1.0 / (1.0 + exp(-vec4<f32>(rounded_logits)))
    );
  }
  workgroupBarrier();

  let frame = local_id.x;
  if (frame >= parameters.frames || batch >= parameters.batch_size) { return; }
  var local = vec4<f32>(local_biases[output_group]);
  for (var kernel_index = 0u; kernel_index < 3u; kernel_index += 1u) {
    let source_frame = i32(frame) + (i32(kernel_index) - 1) * i32(parameters.dilation);
    if (source_frame < 0 || source_frame >= i32(parameters.frames)) { continue; }
    for (var input_channel = 0u; input_channel < 128u; input_channel += 1u) {
      let scratch_index =
        parameters.scratch_offset +
        ((batch * 128u + input_channel) * parameters.frames + u32(source_frame));
      local = fma(
        vec4<f32>(f32(arena[scratch_index])),
        vec4<f32>(local_weight_cache[kernel_index * 128u + input_channel]),
        local,
      );
    }
  }
  let gated = vec4<f16>(vec4<f16>(local) * attention_gate);
  let output_channel_base = parameters.append_channel + output_group * 4u;
  for (var lane = 0u; lane < 4u; lane += 1u) {
    let slab_index =
      parameters.slab_offset +
      ((batch * parameters.slab_channels + output_channel_base + lane) * parameters.frames + frame);
    arena[slab_index] = gated[lane];
  }
}
`;
