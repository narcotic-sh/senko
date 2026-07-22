/// <reference types="@webgpu/types" />

import type { OrtModelAsset, OrtLoadProgress } from "./ort-backends";

const FRAMES = 589;
const HIDDEN = 128;
const OUTPUT_FEATURES = 256;
const LAYERS = 4;
const DIRECTIONS = 2;
const WORKGROUP_SIZE = 256;
const FLOAT_BYTES = 4;

export type PersistentLstmWeightPrecision = "float32" | "float16";

interface PackedTensor {
  readonly offset_bytes: number;
  readonly length_bytes: number;
  readonly shape: readonly number[];
  readonly packed_shape: readonly number[];
  readonly dtype: "float32-le" | "float16-le";
  readonly layout: "row-major" | "gate-column4-hidden-input4";
}

interface PackedDirection {
  readonly direction: "forward" | "reverse";
  readonly input_size: number;
  readonly hidden_size: 128;
  readonly gate_order: readonly ["input", "forget", "cell", "output"];
  readonly tensors: {
    readonly matrix: PackedTensor;
    readonly bias_ih: PackedTensor;
    readonly bias_hh: PackedTensor;
  };
}

interface PackedLayer {
  readonly layer: number;
  readonly input_size: number;
  readonly output_size: 256;
  readonly directions: readonly [PackedDirection, PackedDirection];
}

interface PackedLstmMetadata {
  readonly version: 2 | 3;
  readonly format:
    | "senko-persistent-lstm-f32-gc4h"
    | "senko-persistent-lstm-f16-gc4h";
  readonly byte_order: "little-endian";
  readonly alignment_bytes: 256;
  readonly boundary_layout: "batch,frame,feature";
  readonly frames: 589;
  readonly num_layers: 4;
  readonly bidirectional: true;
  readonly hidden_size: 128;
  readonly gate_order: readonly ["input", "forget", "cell", "output"];
  readonly weights: { readonly file: string; readonly bytes: number; readonly sha256: string };
  readonly layers: readonly PackedLayer[];
  readonly storage_dtype?: "float16";
  readonly accumulator_dtype?: "float32";
  readonly required_webgpu_features?: readonly ["shader-f16"];
  readonly weightPrecision: PersistentLstmWeightPrecision;
  readonly weightElementBytes: 2 | 4;
}

export interface PersistentLstmBufferBytes {
  readonly weights: number;
  readonly recurrentPingA: number;
  readonly recurrentPingB: number;
  readonly layerUniforms: number;
  readonly total: number;
}

export class PersistentWebGpuLstm {
  readonly outputBuffer: GPUBuffer;
  readonly bufferBytes: PersistentLstmBufferBytes;
  readonly weightPrecision: PersistentLstmWeightPrecision;

  private released = false;

  private constructor(
    private readonly batchSize: number,
    private readonly pipeline: GPUComputePipeline,
    private readonly bindGroups: readonly GPUBindGroup[],
    private readonly weightsBuffer: GPUBuffer,
    private readonly pingA: GPUBuffer,
    private readonly pingB: GPUBuffer,
    private readonly uniformBuffers: readonly GPUBuffer[],
    bufferBytes: PersistentLstmBufferBytes,
    weightPrecision: PersistentLstmWeightPrecision,
  ) {
    this.outputBuffer = pingB;
    this.bufferBytes = bufferBytes;
    this.weightPrecision = weightPrecision;
  }

  static async create(
    device: GPUDevice,
    batchSize: number,
    frontendOutput: GPUBuffer,
    weightsAsset: OrtModelAsset,
    metadataAsset: OrtModelAsset,
    onProgress?: OrtLoadProgress,
  ): Promise<PersistentWebGpuLstm> {
    if (device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE) {
      throw new Error(
        `Persistent LSTM needs ${WORKGROUP_SIZE} workgroup lanes; adapter supports ${device.limits.maxComputeInvocationsPerWorkgroup}`,
      );
    }

    onProgress?.("Fetching persistent LSTM package");
    const [weightsData, metadataData] = await Promise.all([
      fetchVerifiedAsset(weightsAsset),
      fetchVerifiedAsset(metadataAsset),
    ]);
    const metadata = parseMetadata(metadataData, weightsData.byteLength);
    if (
      metadata.weights.bytes !== weightsData.byteLength ||
      (weightsAsset.sha256 !== undefined &&
        metadata.weights.sha256.toLowerCase() !== weightsAsset.sha256.toLowerCase())
    ) {
      throw new Error("Persistent LSTM metadata does not match the selected weight asset");
    }
    const actualWeightsSha256 = bytesToHex(
      await crypto.subtle.digest("SHA-256", weightsData),
    );
    if (actualWeightsSha256 !== metadata.weights.sha256.toLowerCase()) {
      throw new Error("Persistent LSTM package SHA-256 does not match its metadata");
    }
    if (
      metadata.weightPrecision === "float16" &&
      !device.features.has("shader-f16")
    ) {
      throw new Error("FP16 persistent LSTM requires WebGPU shader-f16 support");
    }

    onProgress?.("Compiling persistent LSTM WebGPU kernel");
    const shader = device.createShaderModule({
      label: `senko-pyannote-persistent-lstm-${metadata.weightPrecision}`,
      code: persistentLstmWgsl(metadata.weightPrecision),
    });
    const compilation = await shader.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `Persistent LSTM WGSL failed: ${errors.map((message) => message.message).join("; ")}`,
      );
    }

    const bindGroupLayout = device.createBindGroupLayout({
      label: "senko-pyannote-persistent-lstm-bindings",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
      ],
    });
    const pipeline = await device.createComputePipelineAsync({
      label: "senko-pyannote-persistent-lstm",
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module: shader, entryPoint: "main" },
    });

    let weightsBuffer: GPUBuffer | undefined;
    let pingA: GPUBuffer | undefined;
    let pingB: GPUBuffer | undefined;
    const uniformBuffers: GPUBuffer[] = [];
    try {
      weightsBuffer = createInitializedBuffer(
        device,
        "senko-pyannote-lstm-weights",
        new Uint8Array(weightsData),
        GPUBufferUsage.STORAGE,
      );
      const recurrentBytes = batchSize * FRAMES * OUTPUT_FEATURES * FLOAT_BYTES;
      pingA = device.createBuffer({
        label: "senko-pyannote-lstm-ping-a",
        size: recurrentBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      pingB = device.createBuffer({
        label: "senko-pyannote-lstm-ping-b",
        size: recurrentBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      const bindGroups: GPUBindGroup[] = [];
      for (let layerIndex = 0; layerIndex < LAYERS; layerIndex += 1) {
        const layer = metadata.layers[layerIndex]!;
        const parameters = layerParameters(
          layer,
          layerIndex,
          metadata.weightElementBytes,
        );
        const uniform = createInitializedBuffer(
          device,
          `senko-pyannote-lstm-layer-${layerIndex}`,
          new Uint8Array(parameters.buffer),
          GPUBufferUsage.UNIFORM,
        );
        uniformBuffers.push(uniform);
        const input =
          layerIndex === 0 ? frontendOutput : layerIndex % 2 === 1 ? pingA : pingB;
        const output = layerIndex % 2 === 0 ? pingA : pingB;
        bindGroups.push(
          device.createBindGroup({
            label: `senko-pyannote-lstm-layer-${layerIndex}`,
            layout: bindGroupLayout,
            entries: [
              { binding: 0, resource: { buffer: weightsBuffer } },
              { binding: 1, resource: { buffer: input } },
              { binding: 2, resource: { buffer: output } },
              { binding: 3, resource: { buffer: uniform } },
            ],
          }),
        );
      }

      const layerUniforms = uniformBuffers.reduce((sum, buffer) => sum + buffer.size, 0);
      const bufferBytes = {
        weights: weightsBuffer.size,
        recurrentPingA: pingA.size,
        recurrentPingB: pingB.size,
        layerUniforms,
        total: weightsBuffer.size + pingA.size + pingB.size + layerUniforms,
      } satisfies PersistentLstmBufferBytes;
      onProgress?.("Persistent LSTM ready");
      return new PersistentWebGpuLstm(
        batchSize,
        pipeline,
        bindGroups,
        weightsBuffer,
        pingA,
        pingB,
        uniformBuffers,
        bufferBytes,
        metadata.weightPrecision,
      );
    } catch (error) {
      weightsBuffer?.destroy();
      pingA?.destroy();
      pingB?.destroy();
      for (const buffer of uniformBuffers) buffer.destroy();
      throw error;
    }
  }

  encode(encoder: GPUCommandEncoder): void {
    if (this.released) throw new Error("Persistent LSTM has been released");
    // A compute-pass boundary is the WebGPU synchronization point between
    // dependent storage-buffer dispatches. Four dispatches in one pass can
    // expose stale/partially-written ping data to later layers.
    for (let layerIndex = 0; layerIndex < LAYERS; layerIndex += 1) {
      this.encodeLayer(encoder, layerIndex);
    }
  }

  /** Diagnostic-only single-layer dispatch, preserving the normal ping-pong path. */
  encodeLayer(encoder: GPUCommandEncoder, layerIndex: number): void {
    if (this.released) throw new Error("Persistent LSTM has been released");
    const bindGroup = this.bindGroups[layerIndex];
    if (bindGroup === undefined) throw new RangeError(`Invalid LSTM layer ${layerIndex}`);
    const pass = encoder.beginComputePass({
      label: `senko-pyannote-persistent-lstm-layer-${layerIndex}`,
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.batchSize, DIRECTIONS, 1);
    pass.end();
  }

  layerOutputBuffer(layerIndex: number): GPUBuffer {
    if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= LAYERS) {
      throw new RangeError(`Invalid LSTM layer ${layerIndex}`);
    }
    return layerIndex % 2 === 0 ? this.pingA : this.pingB;
  }

  release(): void {
    if (this.released) return;
    this.released = true;
    this.weightsBuffer.destroy();
    this.pingA.destroy();
    this.pingB.destroy();
    for (const buffer of this.uniformBuffers) buffer.destroy();
  }
}

function createInitializedBuffer(
  device: GPUDevice,
  label: string,
  data: Uint8Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function layerParameters(
  layer: PackedLayer,
  expectedLayer: number,
  weightElementBytes: 2 | 4,
): Uint32Array {
  if (
    layer.layer !== expectedLayer ||
    layer.output_size !== OUTPUT_FEATURES ||
    layer.directions.length !== DIRECTIONS
  ) {
    throw new Error(`Malformed persistent LSTM layer ${expectedLayer}`);
  }
  const forward = layer.directions[0]!;
  const reverse = layer.directions[1]!;
  if (forward.direction !== "forward" || reverse.direction !== "reverse") {
    throw new Error(`Persistent LSTM layer ${expectedLayer} direction order is invalid`);
  }
  return new Uint32Array([
    FRAMES,
    layer.input_size,
    weightElementOffset(forward.tensors.matrix, weightElementBytes),
    weightElementOffset(forward.tensors.bias_ih, weightElementBytes),
    weightElementOffset(forward.tensors.bias_hh, weightElementBytes),
    weightElementOffset(reverse.tensors.matrix, weightElementBytes),
    weightElementOffset(reverse.tensors.bias_ih, weightElementBytes),
    weightElementOffset(reverse.tensors.bias_hh, weightElementBytes),
  ]);
}

function weightElementOffset(tensor: PackedTensor, elementBytes: 2 | 4): number {
  if (tensor.offset_bytes % 256 !== 0 || tensor.offset_bytes % elementBytes !== 0) {
    throw new Error(`Persistent LSTM tensor offset ${tensor.offset_bytes} is misaligned`);
  }
  return tensor.offset_bytes / elementBytes;
}

export function parsePersistentLstmMetadata(
  data: ArrayBuffer,
  weightsBytes: number,
): PackedLstmMetadata {
  const value: unknown = JSON.parse(new TextDecoder().decode(data));
  if (typeof value !== "object" || value === null) {
    throw new Error("Persistent LSTM metadata must be an object");
  }
  const metadata = value as Partial<PackedLstmMetadata>;
  let weightPrecision: PersistentLstmWeightPrecision;
  let weightElementBytes: 2 | 4;
  let tensorDtype: PackedTensor["dtype"];
  if (
    metadata.version === 2 &&
    metadata.format === "senko-persistent-lstm-f32-gc4h" &&
    metadata.storage_dtype === undefined &&
    metadata.accumulator_dtype === undefined &&
    metadata.required_webgpu_features === undefined
  ) {
    weightPrecision = "float32";
    weightElementBytes = 4;
    tensorDtype = "float32-le";
  } else if (
    metadata.version === 3 &&
    metadata.format === "senko-persistent-lstm-f16-gc4h" &&
    metadata.storage_dtype === "float16" &&
    metadata.accumulator_dtype === "float32" &&
    Array.isArray(metadata.required_webgpu_features) &&
    metadata.required_webgpu_features.length === 1 &&
    metadata.required_webgpu_features[0] === "shader-f16"
  ) {
    weightPrecision = "float16";
    weightElementBytes = 2;
    tensorDtype = "float16-le";
  } else {
    throw new Error("Persistent LSTM precision contract is invalid");
  }
  if (
    metadata.byte_order !== "little-endian" ||
    metadata.alignment_bytes !== 256 ||
    metadata.boundary_layout !== "batch,frame,feature" ||
    metadata.frames !== FRAMES ||
    metadata.num_layers !== LAYERS ||
    metadata.bidirectional !== true ||
    metadata.hidden_size !== HIDDEN ||
    !Array.isArray(metadata.gate_order) ||
    metadata.gate_order.join(",") !== "input,forget,cell,output" ||
    !Array.isArray(metadata.layers) ||
    metadata.layers.length !== LAYERS ||
    metadata.weights?.bytes !== weightsBytes ||
    typeof metadata.weights.file !== "string" ||
    metadata.weights.file.length === 0 ||
    !/^[0-9a-f]{64}$/i.test(metadata.weights.sha256 ?? "")
  ) {
    throw new Error("Persistent LSTM metadata contract is invalid");
  }
  let expectedOffset = 0;
  for (const [layerIndex, layer] of metadata.layers.entries()) {
    const expectedInput = layerIndex === 0 ? 60 : OUTPUT_FEATURES;
    if (
      typeof layer !== "object" ||
      layer === null ||
      layer.layer !== layerIndex ||
      layer.input_size !== expectedInput ||
      layer.output_size !== OUTPUT_FEATURES ||
      !Array.isArray(layer.directions) ||
      layer.directions.length !== DIRECTIONS
    ) {
      throw new Error(`Persistent LSTM layer ${layerIndex} shape is invalid`);
    }
    for (const [directionIndex, direction] of layer.directions.entries()) {
      const expectedDirection = directionIndex === 0 ? "forward" : "reverse";
      if (
        typeof direction !== "object" ||
        direction === null ||
        direction.direction !== expectedDirection ||
        direction.input_size !== expectedInput ||
        direction.hidden_size !== HIDDEN ||
        !Array.isArray(direction.gate_order) ||
        direction.gate_order.join(",") !== "input,forget,cell,output" ||
        typeof direction.tensors !== "object" ||
        direction.tensors === null
      ) {
        throw new Error(`Persistent LSTM layer ${layerIndex} direction is invalid`);
      }
      const columns = expectedInput + HIDDEN;
      const expectedMatrixBytes = 4 * HIDDEN * columns * weightElementBytes;
      const expectedBiasBytes = 4 * HIDDEN * weightElementBytes;
      const expectedTensorBytes = [expectedMatrixBytes, expectedBiasBytes, expectedBiasBytes];
      const tensors = [
        direction.tensors.matrix,
        direction.tensors.bias_ih,
        direction.tensors.bias_hh,
      ];
      for (const [tensorIndex, tensor] of tensors.entries()) {
        const expectedLayout =
          tensorIndex === 0 ? "gate-column4-hidden-input4" : "row-major";
        const expectedShape = tensorIndex === 0 ? [4 * HIDDEN, columns] : [4 * HIDDEN];
        const expectedPackedShape =
          tensorIndex === 0 ? [4, columns / 4, HIDDEN, 4] : [4 * HIDDEN];
        if (
          typeof tensor !== "object" ||
          tensor === null ||
          tensor.dtype !== tensorDtype ||
          tensor.layout !== expectedLayout ||
          !sameShape(tensor.shape, expectedShape) ||
          !sameShape(tensor.packed_shape, expectedPackedShape) ||
          !Number.isSafeInteger(tensor.offset_bytes) ||
          tensor.offset_bytes !== expectedOffset ||
          tensor.offset_bytes % 256 !== 0 ||
          !Number.isSafeInteger(tensor.length_bytes) ||
          tensor.length_bytes !== expectedTensorBytes[tensorIndex] ||
          tensor.offset_bytes + tensor.length_bytes > weightsBytes
        ) {
          throw new Error(`Persistent LSTM layer ${layerIndex} tensor is invalid`);
        }
        expectedOffset = alignTo(tensor.offset_bytes + tensor.length_bytes, 256);
      }
    }
  }
  if (expectedOffset !== weightsBytes) {
    throw new Error("Persistent LSTM package has trailing or missing weight bytes");
  }
  return Object.assign(metadata, { weightPrecision, weightElementBytes }) as PackedLstmMetadata;
}

function parseMetadata(data: ArrayBuffer, weightsBytes: number): PackedLstmMetadata {
  return parsePersistentLstmMetadata(data, weightsBytes);
}

function sameShape(actual: readonly number[] | undefined, expected: readonly number[]): boolean {
  return (
    Array.isArray(actual) &&
    actual.length === expected.length &&
    actual.every((value, index) => Number.isSafeInteger(value) && value === expected[index])
  );
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

async function fetchVerifiedAsset(asset: OrtModelAsset): Promise<ArrayBuffer> {
  const response = await fetch(asset.url);
  if (!response.ok) throw new Error(`Failed to load ${asset.url}: HTTP ${response.status}`);
  const data = await response.arrayBuffer();
  if (asset.byteLength !== undefined && data.byteLength !== asset.byteLength) {
    throw new Error(`${asset.url} has ${data.byteLength} bytes; expected ${asset.byteLength}`);
  }
  if (asset.sha256 !== undefined) {
    const digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", data))]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
    if (digest !== asset.sha256.toLowerCase()) {
      throw new Error(`SHA-256 mismatch for ${asset.url}`);
    }
  }
  return data;
}

function bytesToHex(buffer: ArrayBuffer): string {
  return [...new Uint8Array(buffer)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function persistentLstmWgsl(
  precision: PersistentLstmWeightPrecision,
): string {
  const halfPrecision = precision === "float16";
  return /* wgsl */ `
${halfPrecision ? "enable f16;" : ""}
struct FloatBuffer {
  values: array<f32>,
};

struct WeightBuffer {
  values: array<${halfPrecision ? "vec4<f16>" : "vec4<f32>"}>,
};

struct Parameters {
  frames: u32,
  input_size: u32,
  matrix_forward: u32,
  bias_ih_forward: u32,
  bias_hh_forward: u32,
  matrix_reverse: u32,
  bias_ih_reverse: u32,
  bias_hh_reverse: u32,
};

@group(0) @binding(0) var<storage, read> weights: WeightBuffer;
@group(0) @binding(1) var<storage, read> sequence_input: FloatBuffer;
@group(0) @binding(2) var<storage, read_write> sequence_output: FloatBuffer;
@group(0) @binding(3) var<uniform> params: Parameters;

var<workgroup> shared_input: array<f32, 256>;
var<workgroup> hidden_state: array<f32, 128>;
var<workgroup> gate_values: array<f32, 512>;

fn scalar_weight(index: u32) -> f32 {
  return f32(weights.values[index >> 2u][index & 3u]);
}

fn weight_vector(index: u32) -> vec4<f32> {
  return vec4<f32>(weights.values[index]);
}

fn logistic(value: f32) -> f32 {
  // The trained gates stay inside +/-19. Clamping outside that range is
  // mathematically saturated and avoids Metal fast-exp producing a non-finite
  // intermediate for a transient extreme value.
  return 1.0 / (1.0 + exp(-clamp(value, -30.0, 30.0)));
}

fn stable_tanh(value: f32) -> f32 {
  // tanh is already saturated to float32 precision at this magnitude. The
  // clamp avoids Metal fast-math's non-finite path for cell states in the
  // hundreds while preserving the model's float32 result.
  return tanh(clamp(value, -15.0, 15.0));
}

@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
  let worker = local_id.x;
  let lane = worker & 127u;
  let gate_pair = worker >> 7u;
  let batch = group_id.x;
  let reverse = group_id.y == 1u;
  let columns = params.input_size + 128u;
  let matrix_offset = select(params.matrix_forward, params.matrix_reverse, reverse);
  let bias_ih_offset = select(params.bias_ih_forward, params.bias_ih_reverse, reverse);
  let bias_hh_offset = select(params.bias_hh_forward, params.bias_hh_reverse, reverse);
  if (worker < 128u) {
    hidden_state[lane] = 0.0;
  }
  var cell = 0.0;

  for (var step = 0u; step < params.frames; step = step + 1u) {
    let frame = select(step, params.frames - 1u - step, reverse);
    let input_base = (batch * params.frames + frame) * params.input_size;
    if (worker < params.input_size) {
      shared_input[worker] = sequence_input.values[input_base + worker];
    }
    workgroupBarrier();

    let first_gate_index = gate_pair * 2u;
    let second_gate_index = first_gate_index + 1u;
    let first_row = first_gate_index * 128u + lane;
    let second_row = second_gate_index * 128u + lane;
    var first_gate = scalar_weight(bias_ih_offset + first_row) +
      scalar_weight(bias_hh_offset + first_row);
    var second_gate = scalar_weight(bias_ih_offset + second_row) +
      scalar_weight(bias_hh_offset + second_row);

    for (var column = 0u; column < params.input_size; column = column + 4u) {
      let values = vec4<f32>(
        shared_input[column],
        shared_input[column + 1u],
        shared_input[column + 2u],
        shared_input[column + 3u],
      );
      let column_group = column >> 2u;
      let groups = columns >> 2u;
      let matrix_vec4 = matrix_offset >> 2u;
      first_gate += dot(weight_vector(matrix_vec4 + ((first_gate_index * groups + column_group) * 128u) + lane), values);
      second_gate += dot(weight_vector(matrix_vec4 + ((second_gate_index * groups + column_group) * 128u) + lane), values);
    }

    for (var column = 0u; column < 128u; column = column + 4u) {
      let values = vec4<f32>(
        hidden_state[column],
        hidden_state[column + 1u],
        hidden_state[column + 2u],
        hidden_state[column + 3u],
      );
      let recurrent_column = params.input_size + column;
      let column_group = recurrent_column >> 2u;
      let groups = columns >> 2u;
      let matrix_vec4 = matrix_offset >> 2u;
      first_gate += dot(weight_vector(matrix_vec4 + ((first_gate_index * groups + column_group) * 128u) + lane), values);
      second_gate += dot(weight_vector(matrix_vec4 + ((second_gate_index * groups + column_group) * 128u) + lane), values);
    }

    gate_values[lane * 4u + first_gate_index] = first_gate;
    gate_values[lane * 4u + second_gate_index] = second_gate;
    // Both gate workers must finish reading the previous hidden state before
    // the first worker for each hidden unit updates it.
    workgroupBarrier();
    if (worker < 128u) {
      let input_gate = gate_values[lane * 4u];
      let forget_gate = gate_values[lane * 4u + 1u];
      let cell_gate = gate_values[lane * 4u + 2u];
      let output_gate = gate_values[lane * 4u + 3u];
      cell = logistic(forget_gate) * cell + logistic(input_gate) * stable_tanh(cell_gate);
      let hidden = logistic(output_gate) * stable_tanh(cell);
      hidden_state[lane] = hidden;
      let output_index = (batch * params.frames + frame) * 256u + group_id.y * 128u + lane;
      sequence_output.values[output_index] = hidden;
    }
    // The next frame's input-load barrier makes every hidden write visible
    // before either gate worker begins the following recurrent dot product.
  }
}
`;
}

/** Diagnostic FP32 baseline retained for direct A/B comparisons. */
export const PERSISTENT_LSTM_WGSL = persistentLstmWgsl("float32");

/** Production shader selected by the browser manifest's FP16 package. */
export const PERSISTENT_LSTM_F16_WGSL = persistentLstmWgsl("float16");
