/// <reference types="@webgpu/types" />

const ARENA_ALIGNMENT = 256;

export interface CampPlusArenaSlice {
  readonly label: string;
  readonly byteOffset: number;
  readonly byteLength: number;
}

/** One fixed-size GPU allocation whose offsets are reused according to tensor lifetimes. */
export class CampPlusActivationArena {
  readonly buffer: GPUBuffer;
  private destroyed = false;

  constructor(
    device: GPUDevice,
    readonly byteLength: number,
  ) {
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || byteLength % ARENA_ALIGNMENT !== 0) {
      throw new RangeError("CAM++ activation arena size must be a positive 256-byte multiple");
    }
    if (
      byteLength > device.limits.maxBufferSize ||
      byteLength > device.limits.maxStorageBufferBindingSize
    ) {
      throw new Error("CAM++ activation arena exceeds this WebGPU device's buffer limits");
    }
    this.buffer = device.createBuffer({
      label: "senko-campplus-activation-arena",
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  slice(label: string, byteOffset: number, byteLength: number): CampPlusArenaSlice {
    if (this.destroyed) throw new Error("CAM++ activation arena has been destroyed");
    if (
      !Number.isSafeInteger(byteOffset) ||
      !Number.isSafeInteger(byteLength) ||
      byteOffset < 0 ||
      byteLength <= 0 ||
      byteOffset % ARENA_ALIGNMENT !== 0 ||
      byteLength % 4 !== 0 ||
      byteOffset + byteLength > this.byteLength
    ) {
      throw new RangeError(`Invalid CAM++ activation slice ${label}`);
    }
    return { label, byteOffset, byteLength };
  }

  upload(
    device: GPUDevice,
    destination: CampPlusArenaSlice,
    data: ArrayBuffer | ArrayBufferView<ArrayBuffer>,
  ): void {
    if (this.destroyed) throw new Error("CAM++ activation arena has been destroyed");
    if (data.byteLength > destination.byteLength || data.byteLength % 4 !== 0) {
      throw new RangeError(`Upload does not fit CAM++ activation slice ${destination.label}`);
    }
    device.queue.writeBuffer(this.buffer, destination.byteOffset, data);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.buffer.destroy();
  }
}
