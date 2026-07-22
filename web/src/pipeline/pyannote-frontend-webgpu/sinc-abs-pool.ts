/// <reference types="@webgpu/types" />

import { PyannoteFrontendGpuPackage } from "./package";

const STATS_WORKGROUP_SIZE = 256;
const SINC_WORKGROUP_SIZE = 64;
const SAMPLES = 160_000;
const OUTPUT_CHANNELS = 80;
const KERNEL = 251;
const CONV_FRAMES = 15_975;
const POOL_FRAMES = 5_325;
const OUTPUT_GROUPS = OUTPUT_CHANNELS / 4;
const UNIFORM_BYTES = 64;

export interface PyannoteSincStageBuffers {
  readonly waveform: GPUBuffer;
  readonly pooled: GPUBuffer;
}

/**
 * The first direct-WebGPU frontend stage.
 *
 * Pass one reduces each waveform to its dynamic InstanceNorm scale/shift.
 * Pass two stages 2,161 normalized samples and 251 vec4 filters per workgroup,
 * then fuses Conv251/stride10, Abs, and MaxPool3/stride3 into one BCT write.
 */
export class PyannoteSincAbsPoolKernel {
  readonly statisticsBuffer: GPUBuffer;
  readonly uniformBuffer: GPUBuffer;
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly gpuPackage: PyannoteFrontendGpuPackage,
    private readonly statsPipeline: GPUComputePipeline,
    private readonly sincPipeline: GPUComputePipeline,
    statisticsBuffer: GPUBuffer,
    uniformBuffer: GPUBuffer,
  ) {
    this.statisticsBuffer = statisticsBuffer;
    this.uniformBuffer = uniformBuffer;
  }

  static async create(
    device: GPUDevice,
    gpuPackage: PyannoteFrontendGpuPackage,
  ): Promise<PyannoteSincAbsPoolKernel> {
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < STATS_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupStorageSize < 10_652
    ) {
      throw new Error("Raw pyannote Sinc stage exceeds this WebGPU device's compute limits");
    }
    validateSections(gpuPackage);
    const statsModule = device.createShaderModule({
      label: "senko-pyannote-waveform-instance-norm",
      code: WAVEFORM_STATS_WGSL,
    });
    const sincModule = device.createShaderModule({
      label: "senko-pyannote-sinc-abs-pool",
      code: SINC_ABS_POOL_WGSL,
    });
    const [statsInfo, sincInfo] = await Promise.all([
      statsModule.getCompilationInfo(),
      sincModule.getCompilationInfo(),
    ]);
    const errors = [...statsInfo.messages, ...sincInfo.messages].filter(
      (message) => message.type === "error",
    );
    if (errors.length > 0) {
      throw new Error(
        `Pyannote frontend WGSL failed: ${errors.map((item) => item.message).join("; ")}`,
      );
    }
    const [statsPipeline, sincPipeline] = await Promise.all([
      device.createComputePipelineAsync({
        label: "senko-pyannote-waveform-instance-norm",
        layout: "auto",
        compute: { module: statsModule, entryPoint: "main" },
      }),
      device.createComputePipelineAsync({
        label: "senko-pyannote-sinc-abs-pool",
        layout: "auto",
        compute: { module: sincModule, entryPoint: "main" },
      }),
    ]);
    const statisticsBuffer = device.createBuffer({
      label: "senko-pyannote-frontend-instance-norm-statistics",
      size: gpuPackage.metadata.memory.statisticsBytes,
      usage: GPUBufferUsage.STORAGE,
    });
    const uniformBuffer = device.createBuffer({
      label: "senko-pyannote-sinc-parameters",
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const parameters = new ArrayBuffer(UNIFORM_BYTES);
    const view = new DataView(parameters);
    const batch = gpuPackage.metadata.contract.inputShape[0];
    const integers = [
      batch,
      SAMPLES,
      OUTPUT_CHANNELS,
      CONV_FRAMES,
      POOL_FRAMES,
      OUTPUT_GROUPS,
      Math.ceil(POOL_FRAMES / SINC_WORKGROUP_SIZE),
      0,
    ];
    integers.forEach((value, index) => view.setUint32(index * 4, value, true));
    view.setFloat32(32, 1e-5, true);
    device.queue.writeBuffer(uniformBuffer, 0, parameters);
    return new PyannoteSincAbsPoolKernel(
      device,
      gpuPackage,
      statsPipeline,
      sincPipeline,
      statisticsBuffer,
      uniformBuffer,
    );
  }

  createDispatch(buffers: PyannoteSincStageBuffers): PyannoteSincAbsPoolDispatch {
    this.assertAlive();
    const inputBytes = this.gpuPackage.metadata.contract.inputShape[0] * SAMPLES * 4;
    const outputBytes =
      this.gpuPackage.metadata.contract.inputShape[0] * OUTPUT_CHANNELS * POOL_FRAMES * 2;
    if (buffers.waveform.size < inputBytes || buffers.pooled.size < outputBytes) {
      throw new Error("Pyannote Sinc stage buffer is smaller than its static B8 contract");
    }
    const affine = this.gpuPackage.section("instance_norm:0:affine");
    const weight = this.gpuPackage.section("conv:0:weight");
    const bias = this.gpuPackage.section("conv:0:bias");
    const statsBindings = this.device.createBindGroup({
      label: "senko-pyannote-waveform-instance-norm-bindings",
      layout: this.statsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.waveform, size: inputBytes } },
        {
          binding: 1,
          resource: {
            buffer: this.gpuPackage.weightsBuffer,
            offset: affine.byteOffset,
            size: affine.byteLength,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.statisticsBuffer,
            size: this.gpuPackage.metadata.memory.statisticsBytes,
          },
        },
        { binding: 3, resource: { buffer: this.uniformBuffer, size: UNIFORM_BYTES } },
      ],
    });
    const sincBindings = this.device.createBindGroup({
      label: "senko-pyannote-sinc-abs-pool-bindings",
      layout: this.sincPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: buffers.waveform, size: inputBytes } },
        {
          binding: 1,
          resource: {
            buffer: this.statisticsBuffer,
            size: this.gpuPackage.metadata.memory.statisticsBytes,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.gpuPackage.weightsBuffer,
            offset: weight.byteOffset,
            size: weight.byteLength,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.gpuPackage.weightsBuffer,
            offset: bias.byteOffset,
            size: bias.byteLength,
          },
        },
        { binding: 4, resource: { buffer: buffers.pooled, size: outputBytes } },
        { binding: 5, resource: { buffer: this.uniformBuffer, size: UNIFORM_BYTES } },
      ],
    });
    return new PyannoteSincAbsPoolDispatch(
      this.gpuPackage.metadata.contract.inputShape[0],
      this.statsPipeline,
      this.sincPipeline,
      statsBindings,
      sincBindings,
    );
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.statisticsBuffer.destroy();
    this.uniformBuffer.destroy();
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Pyannote Sinc stage has been destroyed");
  }
}

export class PyannoteSincAbsPoolDispatch {
  constructor(
    private readonly batch: number,
    private readonly statsPipeline: GPUComputePipeline,
    private readonly sincPipeline: GPUComputePipeline,
    private readonly statsBindings: GPUBindGroup,
    private readonly sincBindings: GPUBindGroup,
  ) {}

  encode(encoder: GPUCommandEncoder): void {
    const statsPass = encoder.beginComputePass({
      label: "senko-pyannote-waveform-instance-norm",
    });
    statsPass.setPipeline(this.statsPipeline);
    statsPass.setBindGroup(0, this.statsBindings);
    statsPass.dispatchWorkgroups(this.batch);
    statsPass.end();

    const sincPass = encoder.beginComputePass({ label: "senko-pyannote-sinc-abs-pool" });
    sincPass.setPipeline(this.sincPipeline);
    sincPass.setBindGroup(0, this.sincBindings);
    sincPass.dispatchWorkgroups(
      Math.ceil(POOL_FRAMES / SINC_WORKGROUP_SIZE),
      this.batch,
      OUTPUT_GROUPS,
    );
    sincPass.end();
  }
}

function validateSections(gpuPackage: PyannoteFrontendGpuPackage): void {
  const affine = gpuPackage.section("instance_norm:0:affine");
  const weight = gpuPackage.section("conv:0:weight");
  const bias = gpuPackage.section("conv:0:bias");
  if (
    affine.kind !== "instance_norm_affine" ||
    affine.layout !== "C4_GAMMA_BETA" ||
    affine.logicalShape[0] !== 1 ||
    weight.kind !== "conv_weight" ||
    weight.layout !== "K_I_O4_O" ||
    weight.dtype !== "float16" ||
    weight.logicalShape[0] !== OUTPUT_CHANNELS ||
    weight.logicalShape[1] !== 1 ||
    weight.logicalShape[2] !== KERNEL ||
    bias.kind !== "conv_bias" ||
    bias.layout !== "O4" ||
    bias.dtype !== "float16" ||
    bias.logicalShape[0] !== OUTPUT_CHANNELS
  ) {
    throw new Error("Packed pyannote Sinc sections do not match the kernel contract");
  }
}

const WAVEFORM_STATS_WGSL = /* wgsl */ `
struct FloatBuffer { values: array<f32> };

struct Parameters {
  batch: u32,
  samples: u32,
  output_channels: u32,
  conv_frames: u32,
  pool_frames: u32,
  output_groups: u32,
  time_tiles: u32,
  reserved: u32,
  epsilon: f32,
};

@group(0) @binding(0) var<storage, read> waveform: FloatBuffer;
@group(0) @binding(1) var<storage, read> affine: FloatBuffer;
@group(0) @binding(2) var<storage, read_write> statistics: FloatBuffer;
@group(0) @binding(3) var<uniform> parameters: Parameters;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch_index = workgroup_id.x;
  let lane = local_id.x;
  var sum = 0.0;
  var sample = lane;
  loop {
    if (sample >= parameters.samples) { break; }
    sum += waveform.values[batch_index * parameters.samples + sample];
    sample += 256u;
  }
  partial[lane] = sum;
  workgroupBarrier();
  var width = 128u;
  loop {
    if (lane < width) { partial[lane] += partial[lane + width]; }
    workgroupBarrier();
    if (width == 1u) { break; }
    width = width / 2u;
  }
  let mean = partial[0] / f32(parameters.samples);
  workgroupBarrier();

  var variance_sum = 0.0;
  sample = lane;
  loop {
    if (sample >= parameters.samples) { break; }
    let centered = waveform.values[batch_index * parameters.samples + sample] - mean;
    variance_sum += centered * centered;
    sample += 256u;
  }
  partial[lane] = variance_sum;
  workgroupBarrier();
  width = 128u;
  loop {
    if (lane < width) { partial[lane] += partial[lane + width]; }
    workgroupBarrier();
    if (width == 1u) { break; }
    width = width / 2u;
  }
  if (lane == 0u) {
    let variance = partial[0] / f32(parameters.samples);
    let scale = affine.values[0] * inverseSqrt(variance + parameters.epsilon);
    statistics.values[batch_index * 2u] = scale;
    statistics.values[batch_index * 2u + 1u] = affine.values[4] - mean * scale;
  }
}
`;

const SINC_ABS_POOL_WGSL = /* wgsl */ `
enable f16;

struct FloatBuffer { values: array<f32> };
struct HalfBuffer { values: array<f16> };
struct Half4Buffer { values: array<vec4<f16>> };

struct Parameters {
  batch: u32,
  samples: u32,
  output_channels: u32,
  conv_frames: u32,
  pool_frames: u32,
  output_groups: u32,
  time_tiles: u32,
  reserved: u32,
  epsilon: f32,
};

@group(0) @binding(0) var<storage, read> waveform: FloatBuffer;
@group(0) @binding(1) var<storage, read> statistics: FloatBuffer;
@group(0) @binding(2) var<storage, read> filters: Half4Buffer;
@group(0) @binding(3) var<storage, read> bias: Half4Buffer;
@group(0) @binding(4) var<storage, read_write> pooled: HalfBuffer;
@group(0) @binding(5) var<uniform> parameters: Parameters;

var<workgroup> signal_tile: array<f32, 2161>;
var<workgroup> filter_tile: array<vec4<f16>, 251>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let lane = local_id.x;
  let pooled_start = workgroup_id.x * 64u;
  let batch_index = workgroup_id.y;
  let output_group = workgroup_id.z;
  let waveform_start = pooled_start * 3u * 10u;
  let scale = statistics.values[batch_index * 2u];
  let shift = statistics.values[batch_index * 2u + 1u];

  var tile_index = lane;
  loop {
    if (tile_index >= 2161u) { break; }
    let waveform_index = waveform_start + tile_index;
    signal_tile[tile_index] = select(
      0.0,
      waveform.values[batch_index * parameters.samples + waveform_index] * scale + shift,
      waveform_index < parameters.samples,
    );
    tile_index += 64u;
  }
  var kernel_index = lane;
  loop {
    if (kernel_index >= 251u) { break; }
    filter_tile[kernel_index] =
      filters.values[kernel_index * parameters.output_groups + output_group];
    kernel_index += 64u;
  }
  workgroupBarrier();

  let pooled_frame = pooled_start + lane;
  if (pooled_frame >= parameters.pool_frames) { return; }
  var maximum = vec4<f32>(-1.0e30);
  for (var pool_lane = 0u; pool_lane < 3u; pool_lane += 1u) {
    var accumulated = vec4<f32>(bias.values[output_group]);
    let local_start = lane * 30u + pool_lane * 10u;
    for (var kernel = 0u; kernel < 251u; kernel += 1u) {
      accumulated = fma(
        vec4<f32>(signal_tile[local_start + kernel]),
        vec4<f32>(filter_tile[kernel]),
        accumulated,
      );
    }
    maximum = max(maximum, abs(accumulated));
  }
  for (var output_lane = 0u; output_lane < 4u; output_lane += 1u) {
    let channel = output_group * 4u + output_lane;
    let output_index =
      (batch_index * parameters.output_channels + channel) * parameters.pool_frames
      + pooled_frame;
    pooled.values[output_index] = f16(maximum[output_lane]);
  }
}
`;
