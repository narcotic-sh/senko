/// <reference types="@webgpu/types" />

import {
  PyannoteSincAbsPoolKernel,
  type PyannoteSincAccumulationSchedule,
  type PyannoteSincAbsPoolDispatch,
} from "./sinc-abs-pool";
import {
  PyannoteF16BctNormKernel,
  PyannoteF32BtfNormKernel,
  type PyannoteF16BctNormDispatch,
  type PyannoteF32BtfNormDispatch,
} from "./instance-norm";
import {
  PyannoteConvPoolKernel,
  type PyannoteConvPoolActivationTilePrecision,
  type PyannoteConvPoolDispatch,
} from "./conv-pool";
import {
  PyannoteFrontendGpuPackage,
  type PyannoteFrontendPackageLoadOptions,
} from "./package";

export interface RawPyannoteFrontendGpuBytes {
  readonly weights: number;
  readonly activationSlots: number;
  readonly statistics: number;
  readonly uniforms: number;
  readonly total: number;
}

export interface RawPyannoteFrontendCreateOptions
  extends PyannoteFrontendPackageLoadOptions {
  readonly convActivationTilePrecision?: PyannoteConvPoolActivationTilePrecision;
  readonly sincAccumulationSchedule?: PyannoteSincAccumulationSchedule;
}

export const RAW_PYANNOTE_FRONTEND_PRODUCTION_KERNELS = {
  convActivationTilePrecision: "float16",
  sincAccumulationSchedule: "interleaved",
} as const satisfies Required<
  Pick<
    RawPyannoteFrontendCreateOptions,
    "convActivationTilePrecision" | "sincAccumulationSchedule"
  >
>;

/**
 * Complete direct-WebGPU frontend with a two-slot activation arena.
 *
 * Slot A holds the pooled Sinc activation and is later overwritten by the
 * final FP32 BTF features. Slot B first holds the FP32 waveform and is later
 * overwritten by the second FP16 pooled activation. No intermediate tensor
 * relies on JavaScript garbage collection for timely GPU reclamation.
 */
export class RawPyannoteFrontendFoundation {
  readonly waveformBuffer: GPUBuffer;
  readonly pooledSincBuffer: GPUBuffer;
  readonly frontendOutputBuffer: GPUBuffer;
  readonly sincDispatch: PyannoteSincAbsPoolDispatch;
  readonly gpuBytes: RawPyannoteFrontendGpuBytes;
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly gpuPackage: PyannoteFrontendGpuPackage,
    readonly sincKernel: PyannoteSincAbsPoolKernel,
    private readonly norm0Dispatch: PyannoteF16BctNormDispatch,
    private readonly conv1Dispatch: PyannoteConvPoolDispatch,
    private readonly norm1Dispatch: PyannoteF16BctNormDispatch,
    private readonly conv2Dispatch: PyannoteConvPoolDispatch,
    private readonly finalNormDispatch: PyannoteF32BtfNormDispatch,
    waveformBuffer: GPUBuffer,
    pooledSincBuffer: GPUBuffer,
  ) {
    this.waveformBuffer = waveformBuffer;
    this.pooledSincBuffer = pooledSincBuffer;
    this.frontendOutputBuffer = pooledSincBuffer;
    this.sincDispatch = sincKernel.createDispatch({
      waveform: waveformBuffer,
      pooled: pooledSincBuffer,
    });
    const metadata = gpuPackage.metadata;
    const activationSlots =
      metadata.memory.activationArenaBytes - metadata.memory.statisticsBytes;
    this.gpuBytes = {
      weights: metadata.binary.byteLength,
      activationSlots,
      statistics: metadata.memory.statisticsBytes,
      uniforms: 384,
      total:
        metadata.binary.byteLength +
        activationSlots +
        metadata.memory.statisticsBytes +
        384,
    };
  }

  static async create(
    device: GPUDevice,
    metadataUrl: string,
    options: RawPyannoteFrontendCreateOptions = {},
  ): Promise<RawPyannoteFrontendFoundation> {
    const gpuPackage = await PyannoteFrontendGpuPackage.load(
      device,
      metadataUrl,
      options,
    );
    let waveformBuffer: GPUBuffer | undefined;
    let pooledSincBuffer: GPUBuffer | undefined;
    let sincKernel: PyannoteSincAbsPoolKernel | undefined;
    let norm0Dispatch: PyannoteF16BctNormDispatch | undefined;
    let conv1Dispatch: PyannoteConvPoolDispatch | undefined;
    let norm1Dispatch: PyannoteF16BctNormDispatch | undefined;
    let conv2Dispatch: PyannoteConvPoolDispatch | undefined;
    let finalNormDispatch: PyannoteF32BtfNormDispatch | undefined;
    try {
      waveformBuffer = device.createBuffer({
        label: "senko-pyannote-frontend-slot-b-waveform-pool1",
        size: gpuPackage.metadata.memory.slotBBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      pooledSincBuffer = device.createBuffer({
        label: "senko-pyannote-frontend-slot-a-pool0-features",
        size: gpuPackage.metadata.memory.slotABytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const convActivationTilePrecision =
        options.convActivationTilePrecision ??
        RAW_PYANNOTE_FRONTEND_PRODUCTION_KERNELS.convActivationTilePrecision;
      const sincAccumulationSchedule =
        options.sincAccumulationSchedule ??
        RAW_PYANNOTE_FRONTEND_PRODUCTION_KERNELS.sincAccumulationSchedule;
      sincKernel = await PyannoteSincAbsPoolKernel.create(
        device,
        gpuPackage,
        sincAccumulationSchedule,
      );
      const [normKernel, convKernel, finalNormKernel] = await Promise.all([
        PyannoteF16BctNormKernel.create(device, gpuPackage),
        PyannoteConvPoolKernel.create(
          device,
          gpuPackage,
          convActivationTilePrecision,
        ),
        PyannoteF32BtfNormKernel.create(device, gpuPackage),
      ]);
      const batch = gpuPackage.metadata.contract.inputShape[0];
      const statistics = sincKernel.statisticsBuffer;
      const statisticsBytes = gpuPackage.metadata.memory.statisticsBytes;
      norm0Dispatch = normKernel.createDispatch({
        label: "senko-pyannote-pool0-instance-norm",
        input: pooledSincBuffer,
        inputBytes: batch * 80 * 5_325 * 2,
        statistics,
        statisticsBytes,
        affine: gpuPackage.section("instance_norm:1:affine"),
        batch,
        channels: 80,
        frames: 5_325,
        epsilon: 1e-5,
      });
      conv1Dispatch = convKernel.createDispatch({
        label: "senko-pyannote-conv1-pool",
        input: pooledSincBuffer,
        inputBytes: batch * 80 * 5_325 * 2,
        output: waveformBuffer,
        outputBytes: batch * 60 * 1_773 * 2,
        statistics,
        statisticsBytes,
        weight: gpuPackage.section("conv:1:weight"),
        bias: gpuPackage.section("conv:1:bias"),
        batch,
        inputChannels: 80,
        outputChannels: 60,
        inputFrames: 5_325,
        outputFrames: 1_773,
        outputLayout: "f16-bct",
        leakyAlpha: 0.01,
      });
      norm1Dispatch = normKernel.createDispatch({
        label: "senko-pyannote-pool1-instance-norm",
        input: waveformBuffer,
        inputBytes: batch * 60 * 1_773 * 2,
        statistics,
        statisticsBytes,
        affine: gpuPackage.section("instance_norm:2:affine"),
        batch,
        channels: 60,
        frames: 1_773,
        epsilon: 1e-5,
      });
      conv2Dispatch = convKernel.createDispatch({
        label: "senko-pyannote-conv2-pool",
        input: waveformBuffer,
        inputBytes: batch * 60 * 1_773 * 2,
        output: pooledSincBuffer,
        outputBytes: batch * 60 * 589 * 4,
        statistics,
        statisticsBytes,
        weight: gpuPackage.section("conv:2:weight"),
        bias: gpuPackage.section("conv:2:bias"),
        batch,
        inputChannels: 60,
        outputChannels: 60,
        inputFrames: 1_773,
        outputFrames: 589,
        outputLayout: "f32-btf",
        leakyAlpha: 0.01,
      });
      finalNormDispatch = finalNormKernel.createDispatch({
        label: "senko-pyannote-final-instance-norm-leaky",
        values: pooledSincBuffer,
        valueBytes: batch * 589 * 60 * 4,
        affine: gpuPackage.section("instance_norm:3:affine"),
        batch,
        channels: 60,
        frames: 589,
        epsilon: 1e-5,
        leakyAlpha: 0.01,
      });
      return new RawPyannoteFrontendFoundation(
        device,
        gpuPackage,
        sincKernel,
        norm0Dispatch,
        conv1Dispatch,
        norm1Dispatch,
        conv2Dispatch,
        finalNormDispatch,
        waveformBuffer,
        pooledSincBuffer,
      );
    } catch (error) {
      norm0Dispatch?.destroy();
      conv1Dispatch?.destroy();
      norm1Dispatch?.destroy();
      conv2Dispatch?.destroy();
      finalNormDispatch?.destroy();
      sincKernel?.destroy();
      waveformBuffer?.destroy();
      pooledSincBuffer?.destroy();
      gpuPackage.destroy();
      throw error;
    }
  }

  uploadWaveform(waveform: Float32Array): void {
    this.assertAlive();
    if (waveform.byteLength !== this.waveformBuffer.size) {
      throw new Error("Waveform does not match the static raw pyannote frontend batch");
    }
    const upload =
      waveform.buffer instanceof ArrayBuffer
        ? new Float32Array(waveform.buffer, waveform.byteOffset, waveform.length)
        : new Float32Array(waveform);
    this.device.queue.writeBuffer(this.waveformBuffer, 0, upload);
  }

  encodeSincStage(encoder: GPUCommandEncoder): void {
    this.assertAlive();
    this.sincDispatch.encode(encoder);
  }

  /** Diagnostic/profile boundary; requires a completed Sinc stage in slot A. */
  encodeConv1Stage(encoder: GPUCommandEncoder): void {
    this.assertAlive();
    this.norm0Dispatch.encode(encoder);
    this.conv1Dispatch.encode(encoder);
  }

  /** Diagnostic/profile boundary; requires a completed Conv1 stage in slot B. */
  encodeConv2AndFinalStage(encoder: GPUCommandEncoder): void {
    this.assertAlive();
    this.norm1Dispatch.encode(encoder);
    this.conv2Dispatch.encode(encoder);
    this.finalNormDispatch.encode(encoder);
  }

  encode(
    encoder: GPUCommandEncoder,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    this.assertAlive();
    this.sincDispatch.encode(encoder, timestampWrites);
    this.norm0Dispatch.encode(encoder);
    this.conv1Dispatch.encode(encoder);
    this.norm1Dispatch.encode(encoder);
    this.conv2Dispatch.encode(encoder);
    this.finalNormDispatch.encode(encoder);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.norm0Dispatch.destroy();
    this.conv1Dispatch.destroy();
    this.norm1Dispatch.destroy();
    this.conv2Dispatch.destroy();
    this.finalNormDispatch.destroy();
    this.sincKernel.destroy();
    this.waveformBuffer.destroy();
    this.pooledSincBuffer.destroy();
    this.gpuPackage.destroy();
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Raw pyannote frontend has been destroyed");
  }
}
