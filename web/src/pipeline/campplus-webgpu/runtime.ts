/// <reference types="@webgpu/types" />

import { CampPlusActivationArena, type CampPlusArenaSlice } from "./arena";
import {
  DENSE_CAM_REQUIRED_WORKGROUP_STORAGE_BYTES,
  DenseCamKernels,
} from "./dense-cam";
import {
  DEFAULT_FCM_VARIANT,
  FcmKernels,
  type FcmVariant,
} from "./fcm";
import { FinalStatsDenseKernel } from "./final-stats-dense";
import {
  PACKED_BCT_REQUIRED_WORKGROUP_STORAGE_BYTES,
  PackedBctConvKernel,
  type PackedBctConvDescriptor,
  type PackedBctConvDispatch,
} from "./packed-bct-conv";
import {
  POINTWISE_TRANSIT_REQUIRED_WORKGROUP_STORAGE_BYTES,
  POINTWISE_TRANSIT_TILE4_WORKGROUP_STORAGE_BYTES,
  PointwiseTransitKernels,
  type PointwiseTransitVariant,
  type PointwiseTransitDispatch,
} from "./pointwise-transit";
import {
  CampPlusGpuPackage,
  type CampPlusPackageLoadOptions,
} from "./package";

export interface CampPlusRawGpuBytes {
  readonly weights: number;
  readonly activationArena: number;
  readonly total: number;
}

export interface RawCampPlusFoundationOptions extends CampPlusPackageLoadOptions {
  /** Explicit arena plan for a supported front-end microbatch. */
  readonly activationArenaBytes?: number;
  /** Diagnostic FCM kernel selection; omission uses the measured production default. */
  readonly fcmVariant?: FcmVariant;
  /** Diagnostic transit kernel selection; omission uses the production default. */
  readonly pointwiseTransitVariant?: PointwiseTransitVariant;
}

export const RAW_CAMPPLUS_REQUIRED_LIMITS = {
  maxComputeWorkgroupStorageSize: Math.max(
    DENSE_CAM_REQUIRED_WORKGROUP_STORAGE_BYTES,
    PACKED_BCT_REQUIRED_WORKGROUP_STORAGE_BYTES,
    POINTWISE_TRANSIT_REQUIRED_WORKGROUP_STORAGE_BYTES,
  ),
} as const satisfies Record<string, number>;

export const RAW_CAMPPLUS_PREFERRED_LIMITS = {
  maxComputeWorkgroupStorageSize: POINTWISE_TRANSIT_TILE4_WORKGROUP_STORAGE_BYTES,
} as const satisfies Record<string, number>;

export function preferredRawCampPlusDeviceLimits(adapter: GPUAdapter): Record<string, number> {
  requireRawCampPlusAdapterLimits(adapter);
  return {
    maxComputeWorkgroupStorageSize: Math.min(
      adapter.limits.maxComputeWorkgroupStorageSize,
      RAW_CAMPPLUS_PREFERRED_LIMITS.maxComputeWorkgroupStorageSize,
    ),
  };
}

export function requireRawCampPlusAdapterLimits(adapter: GPUAdapter): void {
  if (
    adapter.limits.maxComputeWorkgroupStorageSize <
    RAW_CAMPPLUS_REQUIRED_LIMITS.maxComputeWorkgroupStorageSize
  ) {
    throw new Error(
      `Raw CAM++ needs ${RAW_CAMPPLUS_REQUIRED_LIMITS.maxComputeWorkgroupStorageSize} workgroup bytes; adapter exposes ${adapter.limits.maxComputeWorkgroupStorageSize}`,
    );
  }
}

/**
 * Direct-WebGPU CAM++ foundation. There is intentionally no ONNX execution
 * path here: unsupported kernels fail during construction instead of silently
 * allocating a second model runtime.
 */
export class RawCampPlusFoundation {
  readonly arena: CampPlusActivationArena;
  readonly packedConvolution: PackedBctConvKernel;
  readonly denseCam: DenseCamKernels;
  readonly fcm: FcmKernels;
  readonly finalStatsDense: FinalStatsDenseKernel;
  readonly pointwiseTransit: PointwiseTransitKernels;
  readonly gpuBytes: CampPlusRawGpuBytes;
  private destroyed = false;

  private constructor(
    readonly gpuPackage: CampPlusGpuPackage,
    arena: CampPlusActivationArena,
    packedConvolution: PackedBctConvKernel,
    denseCam: DenseCamKernels,
    fcm: FcmKernels,
    finalStatsDense: FinalStatsDenseKernel,
    pointwiseTransit: PointwiseTransitKernels,
  ) {
    this.arena = arena;
    this.packedConvolution = packedConvolution;
    this.denseCam = denseCam;
    this.fcm = fcm;
    this.finalStatsDense = finalStatsDense;
    this.pointwiseTransit = pointwiseTransit;
    this.gpuBytes = {
      weights: gpuPackage.metadata.binary.byteLength,
      activationArena: arena.byteLength,
      total: gpuPackage.metadata.binary.byteLength + arena.byteLength,
    };
  }

  static async create(
    device: GPUDevice,
    metadataUrl: string,
    options: RawCampPlusFoundationOptions = {},
  ): Promise<RawCampPlusFoundation> {
    const gpuPackage = await CampPlusGpuPackage.load(device, metadataUrl, options);
    let arena: CampPlusActivationArena | undefined;
    try {
      arena = new CampPlusActivationArena(
        device,
        options.activationArenaBytes ?? gpuPackage.metadata.memory.activationArenaBytes,
      );
      const [
        packedConvolution,
        denseCam,
        fcm,
        finalStatsDense,
        pointwiseTransit,
      ] = await Promise.all([
        PackedBctConvKernel.create(device, gpuPackage, arena),
        DenseCamKernels.create(device, gpuPackage, arena),
        FcmKernels.create(
          device,
          gpuPackage,
          arena,
          options.fcmVariant ?? DEFAULT_FCM_VARIANT,
        ),
        FinalStatsDenseKernel.create(device, gpuPackage, arena),
        PointwiseTransitKernels.create(
          device,
          gpuPackage,
          arena,
          options.pointwiseTransitVariant,
        ),
      ]);
      return new RawCampPlusFoundation(
        gpuPackage,
        arena,
        packedConvolution,
        denseCam,
        fcm,
        finalStatsDense,
        pointwiseTransit,
      );
    } catch (error) {
      arena?.destroy();
      gpuPackage.destroy();
      throw error;
    }
  }

  createTdnnDispatch(
    input: CampPlusArenaSlice,
    output: CampPlusArenaSlice,
    batchSize: number,
  ): PackedBctConvDispatch {
    this.assertAlive();
    return this.packedConvolution.createDispatch({
      label: "senko-campplus-initial-tdnn",
      convolution: this.gpuPackage.metadata.fusedProgram.tdnn,
      input,
      output,
      batchSize,
      inputChannels: 320,
      inputFrames: 150,
      outputFrames: 75,
      stride: 2,
      dilation: 1,
      padLeft: 2,
      padRight: 2,
      outputRelu: true,
    });
  }

  createTransitDispatch(
    transitIndex: number,
    input: CampPlusArenaSlice,
    output: CampPlusArenaSlice,
    batchSize: number,
    frames = 75,
  ): PointwiseTransitDispatch {
    this.assertAlive();
    const transit = this.gpuPackage.metadata.fusedProgram.transits[transitIndex];
    if (transit === undefined) throw new RangeError(`Invalid CAM++ transit ${transitIndex}`);
    const weight = this.gpuPackage.section(transit.pointwise.weight);
    const outputChannels = weight.logicalShape[0]!;
    return this.pointwiseTransit.createDispatch({
      label: `senko-campplus-${transit.id}`,
      convolution: transit.pointwise,
      preactivationAffine: transit.preactivationAffine,
      input,
      output,
      batchSize,
      inputChannels: weight.logicalShape[1]!,
      outputChannels,
      frames,
      inputStorageChannels: weight.logicalShape[1]!,
      outputStorageChannels: outputChannels,
      outputRelu: transit.epilogue === "relu",
    });
  }

  createPackedConvolution(descriptor: PackedBctConvDescriptor): PackedBctConvDispatch {
    this.assertAlive();
    return this.packedConvolution.createDispatch(descriptor);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.arena.destroy();
    this.gpuPackage.destroy();
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("Raw CAM++ runtime has been destroyed");
  }
}
