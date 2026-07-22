/// <reference types="@webgpu/types" />

import type { CampPlusArenaSlice } from "./arena";
import {
  DEFAULT_DENSE_BOTTLENECK_VARIANT,
  denseBottleneckVariantConfiguration,
  type DenseBottleneckVariant,
  type DenseBottleneckVariantConfiguration,
  type DenseCamDispatch,
} from "./dense-cam";
import type { FcmDispatch, FcmVariant } from "./fcm";
import type { FinalStatsDenseDispatch } from "./final-stats-dense";
import type { PackedBctConvDispatch } from "./packed-bct-conv";
import type {
  PointwiseTransitDispatch,
  PointwiseTransitVariant,
} from "./pointwise-transit";
import {
  RawCampPlusFoundation,
  type RawCampPlusFoundationOptions,
} from "./runtime";

const FRAMES = 150;
const FEATURES = 80;
const TDNN_FRAMES = 75;
const DENSE_SLAB_CHANNELS = 1024;
const FINAL_CHANNELS = 512;
const BOTTLENECK_CHANNELS = 128;
const EMBEDDING_CHANNELS = 192;
export const CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS = 2;

export type CampPlusRawBatchSize = 4 | 8 | 16 | 32;

export interface CampPlusRawGraphOptions extends CampPlusPackageLoadOptionsOnly {
  readonly batchSize?: CampPlusRawBatchSize;
  /** Selects an FCM kernel variant; omission uses the measured production default. */
  readonly fcmVariant?: FcmVariant;
  /** Selects a dense bottleneck kernel; omission uses the production baseline. */
  readonly denseBottleneckVariant?: DenseBottleneckVariant;
  /** Selects a pointwise transit kernel; omission uses the production default. */
  readonly pointwiseTransitVariant?: PointwiseTransitVariant;
}

type CampPlusPackageLoadOptionsOnly = Pick<
  RawCampPlusFoundationOptions,
  "fetch" | "onProgress"
>;

export interface CampPlusRawGraphGpuBytes {
  readonly weights: number;
  readonly activationArena: number;
  readonly input: number;
  readonly output: number;
  readonly readback: number;
  readonly timestampBuffers: number;
  readonly dispatchUniforms: number;
  readonly total: number;
}

export interface CampPlusRawRunResult {
  readonly embeddings: Float32Array<ArrayBuffer>;
  readonly wallMs: number;
  readonly gpuMs?: number;
}

export interface CampPlusRawProfileGroup {
  readonly label: string;
  readonly gpuMs: number;
}

export interface CampPlusRawProfileResult {
  readonly wallMs: number;
  readonly groups: readonly CampPlusRawProfileGroup[];
  readonly transientGpuBufferBytes: number;
}

type GraphDispatch =
  | FcmDispatch
  | PackedBctConvDispatch
  | PointwiseTransitDispatch
  | DenseCamDispatch
  | FinalStatsDenseDispatch;

interface GraphSchedule {
  readonly first: FcmDispatch;
  readonly middle: readonly GraphDispatch[];
  readonly final: FinalStatsDenseDispatch;
  readonly all: readonly GraphDispatch[];
}

interface ProfileRange {
  readonly label: string;
  readonly firstDispatch: number;
  readonly lastDispatch: number;
}

interface CampPlusRawReadbackSlot {
  readonly embeddings: GPUBuffer;
  readonly querySet?: GPUQuerySet;
  readonly queryResolve?: GPUBuffer;
  readonly queryReadback?: GPUBuffer;
  inUse: boolean;
}

const PROFILE_RANGES = [
  { label: "fcm", firstDispatch: 0, lastDispatch: 9 },
  { label: "tdnn", firstDispatch: 10, lastDispatch: 10 },
  { label: "block1", firstDispatch: 11, lastDispatch: 34 },
  { label: "transit1", firstDispatch: 35, lastDispatch: 35 },
  { label: "block2", firstDispatch: 36, lastDispatch: 83 },
  { label: "transit2", firstDispatch: 84, lastDispatch: 84 },
  { label: "block3", firstDispatch: 85, lastDispatch: 116 },
  { label: "transit3", firstDispatch: 117, lastDispatch: 117 },
  { label: "final", firstDispatch: 118, lastDispatch: 118 },
] as const satisfies readonly ProfileRange[];

const ARENA_BYTES = {
  4: 7_372_800,
  8: 12_902_400,
  16: 25_190_400,
  32: 49_152_000,
} as const satisfies Record<CampPlusRawBatchSize, number>;

/** Static 119-dispatch CAM++ graph, encoded into one command buffer/submission. */
export class CampPlusRawGraph {
  readonly inputBuffer: GPUBuffer;
  readonly outputBuffer: GPUBuffer;
  readonly gpuBytes: CampPlusRawGraphGpuBytes;
  readonly dispatchCount: number;
  readonly fcmVariant: FcmVariant;
  readonly denseBottleneckVariant: DenseBottleneckVariant;
  readonly pointwiseTransitVariant: PointwiseTransitVariant;

  private readonly readbackSlots: readonly CampPlusRawReadbackSlot[];
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly foundation: RawCampPlusFoundation,
    readonly batchSize: CampPlusRawBatchSize,
    private readonly schedule: GraphSchedule,
    inputBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    readbackSlots: readonly CampPlusRawReadbackSlot[],
    denseBottleneckVariant: DenseBottleneckVariant,
  ) {
    this.inputBuffer = inputBuffer;
    this.outputBuffer = outputBuffer;
    if (readbackSlots.length !== CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS) {
      throw new Error(
        `CAM++ graph requires ${CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS} readback slots`,
      );
    }
    this.readbackSlots = readbackSlots;
    this.dispatchCount = schedule.all.length;
    this.fcmVariant = foundation.fcm.variant;
    this.denseBottleneckVariant = denseBottleneckVariant;
    this.pointwiseTransitVariant = foundation.pointwiseTransit.variant;
    const input = batchSize * FRAMES * FEATURES * 4;
    const output = batchSize * EMBEDDING_CHANNELS * 4;
    const readback = output * CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS;
    const timestampBuffers = readbackSlots.reduce(
      (sum, slot) =>
        sum +
        (slot.queryResolve?.size ?? 0) +
        (slot.queryReadback?.size ?? 0),
      0,
    );
    const dispatchUniforms = schedule.all.reduce(
      (sum, dispatch) => sum + dispatch.gpuBufferBytes,
      0,
    );
    this.gpuBytes = {
      weights: foundation.gpuBytes.weights,
      activationArena: foundation.gpuBytes.activationArena,
      input,
      output,
      readback,
      timestampBuffers,
      dispatchUniforms,
      total:
        foundation.gpuBytes.total +
        input +
        output +
        readback +
        timestampBuffers +
        dispatchUniforms,
    };
  }

  static async create(
    device: GPUDevice,
    metadataUrl: string,
    options: CampPlusRawGraphOptions = {},
  ): Promise<CampPlusRawGraph> {
    const batchSize = options.batchSize ?? 32;
    const denseBottleneckVariant =
      options.denseBottleneckVariant ?? DEFAULT_DENSE_BOTTLENECK_VARIANT;
    const denseBottleneckConfiguration =
      denseBottleneckVariantConfiguration(denseBottleneckVariant);
    const loadOptions: RawCampPlusFoundationOptions = {
      activationArenaBytes: ARENA_BYTES[batchSize],
      ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
      ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
      ...(options.fcmVariant === undefined
        ? {}
        : { fcmVariant: options.fcmVariant }),
      ...(options.pointwiseTransitVariant === undefined
        ? {}
        : { pointwiseTransitVariant: options.pointwiseTransitVariant }),
    };
    const foundation = await RawCampPlusFoundation.create(
      device,
      metadataUrl,
      loadOptions,
    );
    let inputBuffer: GPUBuffer | undefined;
    let outputBuffer: GPUBuffer | undefined;
    const readbackSlots: CampPlusRawReadbackSlot[] = [];
    let schedule: GraphSchedule | undefined;
    try {
      const declaredPlan = foundation.gpuPackage.metadata.memory.tradeoffs.find(
        (item) => item.frontendMicrobatch === batchSize,
      );
      if (declaredPlan?.activationArenaBytes !== ARENA_BYTES[batchSize]) {
        throw new Error(`Packed CAM++ metadata is missing the B${batchSize} arena plan`);
      }
      const inputBytes = batchSize * FRAMES * FEATURES * 4;
      const outputBytes = batchSize * EMBEDDING_CHANNELS * 4;
      inputBuffer = device.createBuffer({
        label: `senko-campplus-b${batchSize}-features`,
        size: inputBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      outputBuffer = device.createBuffer({
        label: `senko-campplus-b${batchSize}-embeddings`,
        size: outputBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      for (let slot = 0; slot < CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS; slot += 1) {
        readbackSlots.push(
          createReadbackSlot(
            device,
            batchSize,
            slot,
            outputBytes,
            device.features.has("timestamp-query"),
          ),
        );
      }
      await foundation.denseCam.prepareBottleneckVariant(
        denseBottleneckConfiguration.accumulation,
        denseBottleneckConfiguration.outputTile,
        denseBottleneckConfiguration.workgroupSize,
        denseBottleneckConfiguration.weightSource,
      );
      schedule = createSchedule(
        foundation,
        batchSize,
        inputBuffer,
        outputBuffer,
        denseBottleneckConfiguration,
      );
      if (schedule.all.length !== 119) {
        throw new Error(`CAM++ static schedule has ${schedule.all.length} dispatches, expected 119`);
      }
      return new CampPlusRawGraph(
        device,
        foundation,
        batchSize,
        schedule,
        inputBuffer,
        outputBuffer,
        readbackSlots,
        denseBottleneckVariant,
      );
    } catch (error) {
      schedule?.all.forEach((dispatch) => dispatch.destroy());
      readbackSlots.forEach(destroyReadbackSlot);
      outputBuffer?.destroy();
      inputBuffer?.destroy();
      foundation.destroy();
      throw error;
    }
  }

  upload(features: Float32Array<ArrayBuffer>): void {
    this.assertAlive();
    const expected = this.batchSize * FRAMES * FEATURES;
    if (features.length !== expected) {
      throw new RangeError(`CAM++ B${this.batchSize} expects ${expected} FP32 features`);
    }
    this.device.queue.writeBuffer(this.inputBuffer, 0, features);
  }

  /** Encode all dependent stages into one command buffer. Pass boundaries are GPU hazards. */
  encode(
    encoder: GPUCommandEncoder,
    timestamps = false,
    querySet?: GPUQuerySet,
  ): void {
    this.assertAlive();
    const timestampWrites = timestamps && querySet !== undefined;
    this.schedule.first.encode(
      encoder,
      timestampWrites
        ? { querySet, beginningOfPassWriteIndex: 0 }
        : undefined,
    );
    for (const dispatch of this.schedule.middle) dispatch.encode(encoder);
    this.schedule.final.encode(
      encoder,
      timestampWrites
        ? { querySet, endOfPassWriteIndex: 1 }
        : undefined,
    );
  }

  async run(
    features: Float32Array<ArrayBuffer>,
    options: { readonly timestamps?: boolean } = {},
  ): Promise<CampPlusRawRunResult> {
    this.assertAlive();
    const slot = this.readbackSlots.find((candidate) => !candidate.inUse);
    if (slot === undefined) {
      throw new Error(
        `CAM++ allows at most ${CAMPPLUS_RAW_MAX_IN_FLIGHT_RUNS} concurrent runs`,
      );
    }
    slot.inUse = true;
    try {
      const useTimestamps =
        options.timestamps === true && slot.querySet !== undefined;
      this.upload(features);
      const encoder = this.device.createCommandEncoder({
        label: `senko-campplus-b${this.batchSize}-full-graph`,
      });
      this.encode(encoder, useTimestamps, slot.querySet);
      const outputBytes = this.batchSize * EMBEDDING_CHANNELS * 4;
      encoder.copyBufferToBuffer(
        this.outputBuffer,
        0,
        slot.embeddings,
        0,
        outputBytes,
      );
      if (
        useTimestamps &&
        slot.querySet !== undefined &&
        slot.queryResolve !== undefined &&
        slot.queryReadback !== undefined
      ) {
        encoder.resolveQuerySet(slot.querySet, 0, 2, slot.queryResolve, 0);
        encoder.copyBufferToBuffer(
          slot.queryResolve,
          0,
          slot.queryReadback,
          0,
          16,
        );
      }
      const wallStart = performance.now();
      this.device.queue.submit([encoder.finish()]);
      await Promise.all([
        slot.embeddings.mapAsync(GPUMapMode.READ),
        useTimestamps ? slot.queryReadback!.mapAsync(GPUMapMode.READ) : undefined,
      ]);
      const wallMs = performance.now() - wallStart;
      const embeddings = new Float32Array(
        slot.embeddings.getMappedRange(),
      ).slice();
      let gpuMs: number | undefined;
      if (useTimestamps) {
        const values = new BigUint64Array(slot.queryReadback!.getMappedRange());
        gpuMs = Number(values[1]! - values[0]!) / 1_000_000;
      }
      return gpuMs === undefined ? { embeddings, wallMs } : { embeddings, wallMs, gpuMs };
    } finally {
      unmapIfMapped(slot.embeddings);
      if (slot.queryReadback !== undefined) unmapIfMapped(slot.queryReadback);
      slot.inUse = false;
    }
  }

  async profile(
    features: Float32Array<ArrayBuffer>,
  ): Promise<CampPlusRawProfileResult> {
    this.assertAlive();
    this.assertNoActiveRuns("profile");
    if (!this.device.features.has("timestamp-query")) {
      throw new Error("CAM++ group profiling requires timestamp-query");
    }
    const queryCount = PROFILE_RANGES.length * 2;
    const timestampBytes = queryCount * 8;
    const querySet = this.device.createQuerySet({ type: "timestamp", count: queryCount });
    const resolve = this.device.createBuffer({
      label: "senko-campplus-profile-resolve",
      size: timestampBytes,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readback = this.device.createBuffer({
      label: "senko-campplus-profile-readback",
      size: timestampBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      this.upload(features);
      const encoder = this.device.createCommandEncoder({
        label: `senko-campplus-b${this.batchSize}-profile`,
      });
      for (let index = 0; index < this.schedule.all.length; index += 1) {
        const rangeIndex = PROFILE_RANGES.findIndex(
          (range) => index >= range.firstDispatch && index <= range.lastDispatch,
        );
        if (rangeIndex < 0) throw new Error(`CAM++ dispatch ${index} has no profile group`);
        const range = PROFILE_RANGES[rangeIndex]!;
        const beginningOfPassWriteIndex =
          index === range.firstDispatch ? rangeIndex * 2 : undefined;
        const endOfPassWriteIndex =
          index === range.lastDispatch ? rangeIndex * 2 + 1 : undefined;
        const timestampWrites: GPUComputePassTimestampWrites | undefined =
          beginningOfPassWriteIndex === undefined && endOfPassWriteIndex === undefined
            ? undefined
            : {
                querySet,
                ...(beginningOfPassWriteIndex === undefined
                  ? {}
                  : { beginningOfPassWriteIndex }),
                ...(endOfPassWriteIndex === undefined ? {} : { endOfPassWriteIndex }),
              };
        this.schedule.all[index]!.encode(encoder, timestampWrites);
      }
      encoder.resolveQuerySet(querySet, 0, queryCount, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, readback, 0, timestampBytes);
      const wallStart = performance.now();
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const wallMs = performance.now() - wallStart;
      const timestamps = new BigUint64Array(readback.getMappedRange());
      const groups = PROFILE_RANGES.map((range, index) => ({
        label: range.label,
        gpuMs:
          Number(timestamps[index * 2 + 1]! - timestamps[index * 2]!) / 1_000_000,
      }));
      readback.unmap();
      return {
        wallMs,
        groups,
        transientGpuBufferBytes: timestampBytes * 2,
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
      resolve.destroy();
      querySet.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.assertNoActiveRuns("destroy");
    this.destroyed = true;
    this.schedule.all.forEach((dispatch) => dispatch.destroy());
    this.readbackSlots.forEach(destroyReadbackSlot);
    this.outputBuffer.destroy();
    this.inputBuffer.destroy();
    this.foundation.destroy();
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Raw CAM++ graph has been destroyed");
  }

  private assertNoActiveRuns(operation: string): void {
    if (this.readbackSlots.some((slot) => slot.inUse)) {
      throw new Error(`Cannot ${operation} CAM++ while inference is running`);
    }
  }
}

function createReadbackSlot(
  device: GPUDevice,
  batchSize: CampPlusRawBatchSize,
  slot: number,
  outputBytes: number,
  timestamps: boolean,
): CampPlusRawReadbackSlot {
  const embeddings = device.createBuffer({
    label: `senko-campplus-b${batchSize}-embedding-readback-${slot}`,
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let querySet: GPUQuerySet | undefined;
  let queryResolve: GPUBuffer | undefined;
  let queryReadback: GPUBuffer | undefined;
  try {
    if (timestamps) {
      querySet = device.createQuerySet({ type: "timestamp", count: 2 });
      queryResolve = device.createBuffer({
        label: `senko-campplus-timestamp-resolve-${slot}`,
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      queryReadback = device.createBuffer({
        label: `senko-campplus-timestamp-readback-${slot}`,
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
    return {
      embeddings,
      ...(querySet === undefined ? {} : { querySet }),
      ...(queryResolve === undefined ? {} : { queryResolve }),
      ...(queryReadback === undefined ? {} : { queryReadback }),
      inUse: false,
    };
  } catch (error) {
    queryReadback?.destroy();
    queryResolve?.destroy();
    querySet?.destroy();
    embeddings.destroy();
    throw error;
  }
}

function destroyReadbackSlot(slot: CampPlusRawReadbackSlot): void {
  unmapIfMapped(slot.embeddings);
  if (slot.queryReadback !== undefined) unmapIfMapped(slot.queryReadback);
  slot.queryReadback?.destroy();
  slot.queryResolve?.destroy();
  slot.querySet?.destroy();
  slot.embeddings.destroy();
}

function createSchedule(
  foundation: RawCampPlusFoundation,
  batchSize: CampPlusRawBatchSize,
  inputBuffer: GPUBuffer,
  outputBuffer: GPUBuffer,
  denseBottleneckConfiguration: DenseBottleneckVariantConfiguration,
): GraphSchedule {
  const arena = foundation.arena;
  const head = foundation.gpuPackage.metadata.fusedProgram.headConvolutions;
  if (head.length !== 12) throw new Error("CAM++ FCM metadata must contain 12 convolutions");

  const fcmA80 = tensorSlice(arena, "fcm-a-f80", 0, batchSize, 80);
  const fcmA40 = tensorSlice(arena, "fcm-a-f40", 0, batchSize, 40);
  const fcmA20 = tensorSlice(arena, "fcm-a-f20", 0, batchSize, 20);
  const fcmA10 = tensorSlice(arena, "fcm-a-f10", 0, batchSize, 10);
  const fcmABytes = fcmBytes(batchSize, 80);
  const fcmB40 = tensorSlice(arena, "fcm-b-f40", fcmABytes, batchSize, 40);
  const fcmB20 = tensorSlice(arena, "fcm-b-f20", fcmABytes, batchSize, 20);
  const fcmCOffset = fcmABytes + fcmBytes(batchSize, 40);
  const fcmC40 = tensorSlice(arena, "fcm-c-f40", fcmCOffset, batchSize, 40);
  const fcmC20 = tensorSlice(arena, "fcm-c-f20", fcmCOffset, batchSize, 20);

  const fcm: FcmDispatch[] = [];
  fcm.push(
    foundation.fcm.createFirstDispatch({
      label: "senko-campplus-fcm-conv1",
      convolution: head[0]!,
      input: inputBuffer,
      output: fcmA80,
      batchSize,
    }),
    foundation.fcm.createConvDispatch({
      label: "senko-campplus-fcm-layer1-0-conv1",
      convolution: head[1]!,
      input: fcmA80,
      inputFreq: 80,
      output: fcmB40,
      outputFreq: 40,
      strideFreq: 2,
      batchSize,
      residual: { kind: "none" },
      outputRelu: true,
    }),
    foundation.fcm.createConvDispatch({
      label: "senko-campplus-fcm-layer1-0-conv2-shortcut",
      convolution: head[2]!,
      input: fcmB40,
      inputFreq: 40,
      output: fcmC40,
      outputFreq: 40,
      strideFreq: 1,
      batchSize,
      residual: {
        kind: "learned",
        input: fcmA80,
        inputFreq: 80,
        strideFreq: 2,
        convolution: head[3]!,
      },
      outputRelu: true,
    }),
    foundation.fcm.createConvDispatch({
      label: "senko-campplus-fcm-layer1-1-conv1",
      convolution: head[4]!,
      input: fcmC40,
      inputFreq: 40,
      output: fcmA40,
      outputFreq: 40,
      strideFreq: 1,
      batchSize,
      residual: { kind: "none" },
      outputRelu: true,
    }),
    foundation.fcm.createConvDispatch({
      label: "senko-campplus-fcm-layer1-1-conv2-residual",
      convolution: head[5]!,
      input: fcmA40,
      inputFreq: 40,
      output: fcmB40,
      outputFreq: 40,
      strideFreq: 1,
      batchSize,
      residual: { kind: "identity", input: fcmC40 },
      outputRelu: true,
    }),
    foundation.fcm.createConvDispatch({
      label: "senko-campplus-fcm-layer2-0-conv1",
      convolution: head[6]!,
      input: fcmB40,
      inputFreq: 40,
      output: fcmA20,
      outputFreq: 20,
      strideFreq: 2,
      batchSize,
      residual: { kind: "none" },
      outputRelu: true,
    }),
    foundation.fcm.createConvDispatch({
      label: "senko-campplus-fcm-layer2-0-conv2-shortcut",
      convolution: head[7]!,
      input: fcmA20,
      inputFreq: 20,
      output: fcmC20,
      outputFreq: 20,
      strideFreq: 1,
      batchSize,
      residual: {
        kind: "learned",
        input: fcmB40,
        inputFreq: 40,
        strideFreq: 2,
        convolution: head[8]!,
      },
      outputRelu: true,
    }),
    foundation.fcm.createConvDispatch({
      label: "senko-campplus-fcm-layer2-1-conv1",
      convolution: head[9]!,
      input: fcmC20,
      inputFreq: 20,
      output: fcmA20,
      outputFreq: 20,
      strideFreq: 1,
      batchSize,
      residual: { kind: "none" },
      outputRelu: true,
    }),
    foundation.fcm.createConvDispatch({
      label: "senko-campplus-fcm-layer2-1-conv2-residual",
      convolution: head[10]!,
      input: fcmA20,
      inputFreq: 20,
      output: fcmB20,
      outputFreq: 20,
      strideFreq: 1,
      batchSize,
      residual: { kind: "identity", input: fcmC20 },
      outputRelu: true,
    }),
    foundation.fcm.createConvDispatch({
      label: "senko-campplus-fcm-conv2",
      convolution: head[11]!,
      input: fcmB20,
      inputFreq: 20,
      output: fcmA10,
      outputFreq: 10,
      strideFreq: 2,
      batchSize,
      residual: { kind: "none" },
      outputRelu: true,
    }),
  );

  const slabBytes = batchSize * DENSE_SLAB_CHANNELS * TDNN_FRAMES * 2;
  const slabA = arena.slice("dense-slab-a", 0, slabBytes);
  const slabB = arena.slice("dense-slab-b", slabBytes, slabBytes);
  const scratchOffset = slabBytes * 2;
  const scratchBytes = batchSize * BOTTLENECK_CHANNELS * TDNN_FRAMES * 2;
  const scratch = arena.slice("dense-bottleneck-scratch", scratchOffset, scratchBytes);
  const meanOffset = scratchOffset + scratchBytes;
  const doubledMean = arena.slice(
    "dense-doubled-mean",
    meanOffset,
    batchSize * BOTTLENECK_CHANNELS * 2,
  );

  const middle: GraphDispatch[] = [...fcm.slice(1)];
  const tdnn = foundation.createPackedConvolution({
    label: "senko-campplus-initial-tdnn",
    convolution: foundation.gpuPackage.metadata.fusedProgram.tdnn,
    input: fcmA10,
    output: slabB,
    batchSize,
    inputChannels: 320,
    inputFrames: FRAMES,
    outputFrames: TDNN_FRAMES,
    stride: 2,
    dilation: 1,
    padLeft: 2,
    padRight: 2,
    outputRelu: true,
    outputStorageChannels: DENSE_SLAB_CHANNELS,
  });
  middle.push(tdnn);

  let currentSlab = slabB;
  for (let blockIndex = 0; blockIndex < 3; blockIndex += 1) {
    const block = foundation.gpuPackage.metadata.fusedProgram.blocks[blockIndex];
    if (block === undefined) throw new Error(`CAM++ metadata is missing dense block ${blockIndex + 1}`);
    for (const layer of block.layers) {
      middle.push(
        foundation.denseCam.createBottleneckDispatch({
          label: `senko-campplus-${layer.id}-bottleneck`,
          layer,
          slab: currentSlab,
          slabChannels: DENSE_SLAB_CHANNELS,
          scratch,
          doubledMean,
          batchSize,
          ...denseBottleneckConfiguration,
        }),
        foundation.denseCam.createLocalCamDispatch({
          label: `senko-campplus-${layer.id}-local-cam`,
          layer,
          slab: currentSlab,
          slabChannels: DENSE_SLAB_CHANNELS,
          scratch,
          doubledMean,
          batchSize,
        }),
      );
    }
    const transit = foundation.gpuPackage.metadata.fusedProgram.transits[blockIndex];
    if (transit === undefined) throw new Error(`CAM++ metadata is missing transit ${blockIndex + 1}`);
    const weight = foundation.gpuPackage.section(transit.pointwise.weight);
    const outputChannels = weight.logicalShape[0]!;
    const inputChannels = weight.logicalShape[1]!;
    const expectedOutputChannels = blockIndex === 0 ? 256 : FINAL_CHANNELS;
    if (outputChannels !== expectedOutputChannels) {
      throw new Error(`${transit.id} has ${outputChannels} outputs, expected ${expectedOutputChannels}`);
    }
    const nextSlab = currentSlab === slabA ? slabB : slabA;
    middle.push(
      foundation.pointwiseTransit.createDispatch({
        label: `senko-campplus-${transit.id}`,
        convolution: transit.pointwise,
        preactivationAffine: transit.preactivationAffine,
        input: currentSlab,
        output: nextSlab,
        batchSize,
        inputChannels,
        outputChannels,
        frames: TDNN_FRAMES,
        outputRelu: transit.epilogue === "relu",
        inputStorageChannels: DENSE_SLAB_CHANNELS,
        outputStorageChannels:
          blockIndex === 2 ? FINAL_CHANNELS : DENSE_SLAB_CHANNELS,
      }),
    );
    currentSlab = nextSlab;
  }

  const final = foundation.finalStatsDense.createDispatch({
    label: "senko-campplus-final-stats-dense",
    input: currentSlab,
    inputStorageChannels: FINAL_CHANNELS,
    batchSize,
    dense: foundation.gpuPackage.metadata.fusedProgram.finalDense,
    outputAffine: foundation.gpuPackage.metadata.fusedProgram.finalOutputAffine,
    output: outputBuffer,
  });
  const all: GraphDispatch[] = [fcm[0]!, ...middle, final];
  return { first: fcm[0]!, middle, final, all };
}

function fcmBytes(batchSize: number, frequency: number): number {
  return batchSize * 32 * frequency * FRAMES * 2;
}

function unmapIfMapped(buffer: GPUBuffer): void {
  if (buffer.mapState === "mapped") buffer.unmap();
}

function tensorSlice(
  arena: RawCampPlusFoundation["arena"],
  label: string,
  byteOffset: number,
  batchSize: number,
  frequency: number,
): CampPlusArenaSlice {
  return arena.slice(label, byteOffset, fcmBytes(batchSize, frequency));
}
