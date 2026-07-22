/// <reference types="@webgpu/types" />

import type { PyannoteFrontendPackedSection } from "./metadata";
import { PyannoteFrontendGpuPackage } from "./package";

const WORKGROUP_SIZE = 128;
const INPUT_BLOCK_CHANNELS = 16;
const KERNEL = 5;
const POOL = 3;
const INPUT_TILE_FRAMES = WORKGROUP_SIZE * POOL + KERNEL - 1;
const INPUT_TILE_ELEMENTS = INPUT_TILE_FRAMES * INPUT_BLOCK_CHANNELS;
const WORKGROUP_STORAGE_BYTES = INPUT_TILE_ELEMENTS * 4 + KERNEL * INPUT_BLOCK_CHANNELS * 8;
const UNIFORM_BYTES = 64;

export type ConvPoolOutputLayout = "f16-bct" | "f32-btf";

export interface PyannoteConvPoolDescriptor {
  readonly label: string;
  readonly input: GPUBuffer;
  readonly inputBytes: number;
  readonly output: GPUBuffer;
  readonly outputBytes: number;
  readonly statistics: GPUBuffer;
  readonly statisticsBytes: number;
  readonly weight: PyannoteFrontendPackedSection;
  readonly bias: PyannoteFrontendPackedSection;
  readonly batch: number;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly inputFrames: number;
  readonly outputFrames: number;
  readonly outputLayout: ConvPoolOutputLayout;
  readonly leakyAlpha: number;
}

export class PyannoteConvPoolDispatch {
  constructor(
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroup: GPUBindGroup,
    readonly uniformBuffer: GPUBuffer,
    private readonly workgroups: readonly [number, number, number],
    readonly label: string,
  ) {}

  encode(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: this.label });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(...this.workgroups);
    pass.end();
  }

  destroy(): void {
    this.uniformBuffer.destroy();
  }
}

export class PyannoteConvPoolKernel {
  private constructor(
    private readonly device: GPUDevice,
    private readonly gpuPackage: PyannoteFrontendGpuPackage,
    private readonly f16BctPipeline: GPUComputePipeline,
    private readonly f32BtfPipeline: GPUComputePipeline,
  ) {}

  static async create(
    device: GPUDevice,
    gpuPackage: PyannoteFrontendGpuPackage,
  ): Promise<PyannoteConvPoolKernel> {
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupStorageSize < WORKGROUP_STORAGE_BYTES
    ) {
      throw new Error(
        `Raw pyannote Conv5 needs ${WORKGROUP_STORAGE_BYTES} workgroup bytes and ${WORKGROUP_SIZE} lanes`,
      );
    }
    const f16Module = device.createShaderModule({
      label: "senko-pyannote-conv5-pool-f16-bct",
      code: convPoolWgsl("f16-bct"),
    });
    const f32Module = device.createShaderModule({
      label: "senko-pyannote-conv5-pool-f32-btf",
      code: convPoolWgsl("f32-btf"),
    });
    const [f16Info, f32Info] = await Promise.all([
      f16Module.getCompilationInfo(),
      f32Module.getCompilationInfo(),
    ]);
    const errors = [...f16Info.messages, ...f32Info.messages].filter(
      (message) => message.type === "error",
    );
    if (errors.length > 0) {
      throw new Error(
        `Pyannote Conv5/Pool WGSL failed: ${errors.map((item) => item.message).join("; ")}`,
      );
    }
    const [f16BctPipeline, f32BtfPipeline] = await Promise.all([
      device.createComputePipelineAsync({
        label: "senko-pyannote-conv5-pool-f16-bct",
        layout: "auto",
        compute: { module: f16Module, entryPoint: "main" },
      }),
      device.createComputePipelineAsync({
        label: "senko-pyannote-conv5-pool-f32-btf",
        layout: "auto",
        compute: { module: f32Module, entryPoint: "main" },
      }),
    ]);
    return new PyannoteConvPoolKernel(
      device,
      gpuPackage,
      f16BctPipeline,
      f32BtfPipeline,
    );
  }

  createDispatch(descriptor: PyannoteConvPoolDescriptor): PyannoteConvPoolDispatch {
    validateSections(descriptor);
    const pipeline =
      descriptor.outputLayout === "f16-bct"
        ? this.f16BctPipeline
        : this.f32BtfPipeline;
    const parameters = new ArrayBuffer(UNIFORM_BYTES);
    const view = new DataView(parameters);
    const integers = [
      descriptor.batch,
      descriptor.inputChannels,
      descriptor.outputChannels,
      descriptor.inputFrames,
      descriptor.outputFrames,
      descriptor.outputChannels / 4,
      Math.ceil(descriptor.inputChannels / INPUT_BLOCK_CHANNELS),
      0,
    ];
    integers.forEach((value, index) => view.setUint32(index * 4, value, true));
    view.setFloat32(32, descriptor.leakyAlpha, true);
    const uniformBuffer = this.device.createBuffer({
      label: `${descriptor.label}-parameters`,
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(uniformBuffer, 0, parameters);
    const bindGroup = this.device.createBindGroup({
      label: `${descriptor.label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: descriptor.input, size: descriptor.inputBytes } },
        {
          binding: 1,
          resource: {
            buffer: descriptor.statistics,
            size: descriptor.statisticsBytes,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.gpuPackage.weightsBuffer,
            offset: descriptor.weight.byteOffset,
            size: descriptor.weight.byteLength,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.gpuPackage.weightsBuffer,
            offset: descriptor.bias.byteOffset,
            size: descriptor.bias.byteLength,
          },
        },
        { binding: 4, resource: { buffer: descriptor.output, size: descriptor.outputBytes } },
        { binding: 5, resource: { buffer: uniformBuffer, size: UNIFORM_BYTES } },
      ],
    });
    return new PyannoteConvPoolDispatch(
      pipeline,
      bindGroup,
      uniformBuffer,
      [
        Math.ceil(descriptor.outputFrames / WORKGROUP_SIZE),
        descriptor.batch,
        descriptor.outputChannels / 4,
      ],
      descriptor.label,
    );
  }
}

function validateSections(descriptor: PyannoteConvPoolDescriptor): void {
  if (
    descriptor.inputChannels <= 0 ||
    descriptor.outputChannels <= 0 ||
    descriptor.outputChannels % 4 !== 0 ||
    descriptor.weight.kind !== "conv_weight" ||
    descriptor.weight.layout !== "K_I_O4_O" ||
    descriptor.weight.dtype !== "float16" ||
    descriptor.weight.logicalShape[0] !== descriptor.outputChannels ||
    descriptor.weight.logicalShape[1] !== descriptor.inputChannels ||
    descriptor.weight.logicalShape[2] !== KERNEL ||
    descriptor.bias.kind !== "conv_bias" ||
    descriptor.bias.layout !== "O4" ||
    descriptor.bias.dtype !== "float16" ||
    descriptor.bias.logicalShape[0] !== descriptor.outputChannels
  ) {
    throw new Error(`${descriptor.label} packed convolution sections are invalid`);
  }
  const expectedFrames = Math.floor((descriptor.inputFrames - KERNEL - (POOL - 1)) / POOL) + 1;
  if (descriptor.outputFrames !== expectedFrames) {
    throw new Error(`${descriptor.label} frame geometry is invalid`);
  }
}

function convPoolWgsl(outputLayout: ConvPoolOutputLayout): string {
  const outputDeclaration =
    outputLayout === "f16-bct"
      ? "@group(0) @binding(4) var<storage, read_write> output_values: HalfBuffer;"
      : "@group(0) @binding(4) var<storage, read_write> output_values: FloatBuffer;";
  const outputWrite =
    outputLayout === "f16-bct"
      ? `let output_index =
      (batch_index * parameters.output_channels + output_channel) * parameters.output_frames
      + output_frame;
    output_values.values[output_index] = f16(maximum[output_lane]);`
      : `let output_index =
      (batch_index * parameters.output_frames + output_frame) * parameters.output_channels
      + output_channel;
    output_values.values[output_index] = maximum[output_lane];`;
  return /* wgsl */ `
enable f16;
struct HalfBuffer { values: array<f16> };
struct FloatBuffer { values: array<f32> };
struct Half4Buffer { values: array<vec4<f16>> };
struct Parameters {
  batch: u32,
  input_channels: u32,
  output_channels: u32,
  input_frames: u32,
  output_frames: u32,
  output_groups: u32,
  input_blocks: u32,
  reserved: u32,
  leaky_alpha: f32,
};
@group(0) @binding(0) var<storage, read> input_values: HalfBuffer;
@group(0) @binding(1) var<storage, read> statistics: FloatBuffer;
@group(0) @binding(2) var<storage, read> filters: Half4Buffer;
@group(0) @binding(3) var<storage, read> bias: Half4Buffer;
${outputDeclaration}
@group(0) @binding(5) var<uniform> parameters: Parameters;

var<workgroup> input_tile: array<f32, ${INPUT_TILE_ELEMENTS}>;
var<workgroup> filter_tile: array<vec4<f16>, ${KERNEL * INPUT_BLOCK_CHANNELS}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let lane = local_id.x;
  let output_start = group_id.x * ${WORKGROUP_SIZE}u;
  let output_frame = output_start + lane;
  let batch_index = group_id.y;
  let output_group = group_id.z;
  var accumulated0 = vec4<f32>(bias.values[output_group]);
  var accumulated1 = accumulated0;
  var accumulated2 = accumulated0;

  for (var input_block = 0u; input_block < parameters.input_blocks; input_block += 1u) {
    var tile_index = lane;
    loop {
      if (tile_index >= ${INPUT_TILE_ELEMENTS}u) { break; }
      let block_channel = tile_index / ${INPUT_TILE_FRAMES}u;
      let tile_frame = tile_index % ${INPUT_TILE_FRAMES}u;
      let input_channel = input_block * ${INPUT_BLOCK_CHANNELS}u + block_channel;
      let input_frame = output_start * ${POOL}u + tile_frame;
      var activated = 0.0;
      if (input_channel < parameters.input_channels && input_frame < parameters.input_frames) {
        let input_index =
          (batch_index * parameters.input_channels + input_channel) * parameters.input_frames
          + input_frame;
        let statistics_index =
          (batch_index * parameters.input_channels + input_channel) * 2u;
        let normalized =
          f32(input_values.values[input_index]) * statistics.values[statistics_index]
          + statistics.values[statistics_index + 1u];
        activated = select(
          parameters.leaky_alpha * normalized,
          normalized,
          normalized >= 0.0,
        );
      }
      input_tile[tile_index] = activated;
      tile_index += ${WORKGROUP_SIZE}u;
    }
    var weight_index = lane;
    loop {
      if (weight_index >= ${KERNEL * INPUT_BLOCK_CHANNELS}u) { break; }
      let kernel = weight_index / ${INPUT_BLOCK_CHANNELS}u;
      let block_channel = weight_index % ${INPUT_BLOCK_CHANNELS}u;
      let input_channel = input_block * ${INPUT_BLOCK_CHANNELS}u + block_channel;
      filter_tile[weight_index] = vec4<f16>(0.0h);
      if (input_channel < parameters.input_channels) {
        let source =
          (kernel * parameters.input_channels + input_channel) * parameters.output_groups
          + output_group;
        filter_tile[weight_index] = filters.values[source];
      }
      weight_index += ${WORKGROUP_SIZE}u;
    }
    workgroupBarrier();

    if (output_frame < parameters.output_frames) {
      for (var block_channel = 0u; block_channel < ${INPUT_BLOCK_CHANNELS}u; block_channel += 1u) {
        for (var kernel = 0u; kernel < ${KERNEL}u; kernel += 1u) {
          let weight = vec4<f32>(
            filter_tile[kernel * ${INPUT_BLOCK_CHANNELS}u + block_channel]
          );
          let tile_base = block_channel * ${INPUT_TILE_FRAMES}u + lane * ${POOL}u + kernel;
          accumulated0 = fma(vec4<f32>(input_tile[tile_base]), weight, accumulated0);
          accumulated1 = fma(vec4<f32>(input_tile[tile_base + 1u]), weight, accumulated1);
          accumulated2 = fma(vec4<f32>(input_tile[tile_base + 2u]), weight, accumulated2);
        }
      }
    }
    workgroupBarrier();
  }

  if (output_frame >= parameters.output_frames) { return; }
  let maximum = max(accumulated0, max(accumulated1, accumulated2));
  for (var output_lane = 0u; output_lane < 4u; output_lane += 1u) {
    let output_channel = output_group * 4u + output_lane;
    ${outputWrite}
  }
}
`;
}
