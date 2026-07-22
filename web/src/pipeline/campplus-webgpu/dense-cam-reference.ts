import { float16BitsToFloat32, float32ToFloat16Bits, roundToFloat16 } from "./reference";

export interface DenseBottleneckReferenceParameters {
  readonly batchSize: number;
  readonly inputChannels: number;
  readonly slabChannels: number;
  readonly frames: number;
}

export interface DenseBottleneckReferenceResult {
  readonly scratch: Uint16Array<ArrayBuffer>;
  readonly doubledMean: Uint16Array<ArrayBuffer>;
}

export interface DenseLocalCamReferenceParameters {
  readonly batchSize: number;
  readonly slabChannels: number;
  readonly frames: number;
  readonly appendChannel: number;
  readonly dilation: number;
}

/** Independent CPU oracle for the fused dense bottleneck and doubled time mean. */
export function evaluateDenseBottleneckReference(
  slab: Uint16Array,
  packedWeight: Uint16Array,
  packedBias: Uint16Array,
  packedAffine: Float32Array,
  parameters: DenseBottleneckReferenceParameters,
): DenseBottleneckReferenceResult {
  const { batchSize, inputChannels, slabChannels, frames } = parameters;
  if (slab.length !== batchSize * slabChannels * frames) {
    throw new RangeError("Dense reference slab length disagrees with its physical shape");
  }
  const inputGroups = ceilDiv(inputChannels, 4);
  if (packedWeight.length !== 32 * inputGroups * 16) {
    throw new RangeError("Dense bottleneck packed weight length disagrees with input channels");
  }
  if (packedBias.length !== 128 || packedAffine.length !== inputGroups * 8) {
    throw new RangeError("Dense bottleneck bias/affine shape mismatch");
  }
  const scratch = new Uint16Array(batchSize * 128 * frames);
  const doubledMean = new Uint16Array(batchSize * 128);
  for (let batch = 0; batch < batchSize; batch += 1) {
    for (let outputGroup = 0; outputGroup < 32; outputGroup += 1) {
      for (let frame = 0; frame < frames; frame += 1) {
        const accumulators = [
          float16BitsToFloat32(packedBias[outputGroup * 4]!),
          float16BitsToFloat32(packedBias[outputGroup * 4 + 1]!),
          float16BitsToFloat32(packedBias[outputGroup * 4 + 2]!),
          float16BitsToFloat32(packedBias[outputGroup * 4 + 3]!),
        ];
        for (let inputChannel = 0; inputChannel < inputChannels; inputChannel += 1) {
          const inputGroup = Math.floor(inputChannel / 4);
          const inputLane = inputChannel % 4;
          const inputIndex =
            (batch * slabChannels + inputChannel) * frames + frame;
          const activated = Math.max(
            roundToFloat16(
              float16BitsToFloat32(slab[inputIndex]!) *
                packedAffine[inputGroup * 8 + inputLane]! +
                packedAffine[inputGroup * 8 + 4 + inputLane]!,
            ),
            0,
          );
          const weightBase = ((outputGroup * inputGroups + inputGroup) * 4 + inputLane) * 4;
          for (let lane = 0; lane < 4; lane += 1) {
            accumulators[lane] =
              accumulators[lane]! +
              activated * float16BitsToFloat32(packedWeight[weightBase + lane]!);
          }
        }
        for (let lane = 0; lane < 4; lane += 1) {
          const outputChannel = outputGroup * 4 + lane;
          scratch[(batch * 128 + outputChannel) * frames + frame] =
            float32ToFloat16Bits(Math.max(roundToFloat16(accumulators[lane]!), 0));
        }
      }
      for (let lane = 0; lane < 4; lane += 1) {
        const outputChannel = outputGroup * 4 + lane;
        const values = new Float32Array(128);
        for (let frame = 0; frame < frames; frame += 1) {
          values[frame] = float16BitsToFloat32(
            scratch[(batch * 128 + outputChannel) * frames + frame]!,
          );
        }
        for (let stride = 64; stride >= 1; stride /= 2) {
          for (let index = 0; index < stride; index += 1) {
            values[index] = Math.fround(values[index]! + values[index + stride]!);
          }
        }
        const mean = roundToFloat16(values[0]! / frames);
        doubledMean[batch * 128 + outputChannel] = float32ToFloat16Bits(
          roundToFloat16(mean * 2),
        );
      }
    }
  }
  return { scratch, doubledMean };
}

/** CPU oracle for local K3, attention MLP/sigmoid, gate, and dense append. */
export function evaluateDenseLocalCamReference(
  scratch: Uint16Array,
  doubledMean: Uint16Array,
  localWeight: Uint16Array,
  localBias: Uint16Array,
  attention1Weight: Uint16Array,
  attention1Bias: Uint16Array,
  attention2Weight: Uint16Array,
  attention2Bias: Uint16Array,
  parameters: DenseLocalCamReferenceParameters,
): Uint16Array<ArrayBuffer> {
  const { batchSize, frames, dilation } = parameters;
  if (
    scratch.length !== batchSize * 128 * frames ||
    doubledMean.length !== batchSize * 128 ||
    localWeight.length !== 3 * 8 * 32 * 16 ||
    localBias.length !== 32 ||
    attention1Weight.length !== 16 * 32 * 16 ||
    attention1Bias.length !== 64 ||
    attention2Weight.length !== 8 * 16 * 16 ||
    attention2Bias.length !== 32
  ) {
    throw new RangeError("Dense local/CAM reference tensor shape mismatch");
  }
  const output = new Uint16Array(batchSize * 32 * frames);
  const hidden = new Uint16Array(64);
  const gates = new Uint16Array(32);
  for (let batch = 0; batch < batchSize; batch += 1) {
    for (let hiddenChannel = 0; hiddenChannel < 64; hiddenChannel += 1) {
      let accumulator = float16BitsToFloat32(attention1Bias[hiddenChannel]!);
      for (let inputChannel = 0; inputChannel < 128; inputChannel += 1) {
        accumulator +=
          float16BitsToFloat32(doubledMean[batch * 128 + inputChannel]!) *
          packedWeightScalar(
            attention1Weight,
            64,
            128,
            hiddenChannel,
            inputChannel,
            0,
          );
      }
      hidden[hiddenChannel] = float32ToFloat16Bits(
        Math.max(roundToFloat16(accumulator), 0),
      );
    }
    for (let outputChannel = 0; outputChannel < 32; outputChannel += 1) {
      let accumulator = float16BitsToFloat32(attention2Bias[outputChannel]!);
      for (let hiddenChannel = 0; hiddenChannel < 64; hiddenChannel += 1) {
        accumulator +=
          float16BitsToFloat32(hidden[hiddenChannel]!) *
          packedWeightScalar(
            attention2Weight,
            32,
            64,
            outputChannel,
            hiddenChannel,
            0,
          );
      }
      const roundedLogit = roundToFloat16(accumulator);
      gates[outputChannel] = float32ToFloat16Bits(
        roundToFloat16(1 / (1 + Math.exp(-roundedLogit))),
      );
    }

    for (let outputGroup = 0; outputGroup < 8; outputGroup += 1) {
      for (let frame = 0; frame < frames; frame += 1) {
        const accumulators = [
          float16BitsToFloat32(localBias[outputGroup * 4]!),
          float16BitsToFloat32(localBias[outputGroup * 4 + 1]!),
          float16BitsToFloat32(localBias[outputGroup * 4 + 2]!),
          float16BitsToFloat32(localBias[outputGroup * 4 + 3]!),
        ];
        for (let kernel = 0; kernel < 3; kernel += 1) {
          const sourceFrame = frame + (kernel - 1) * dilation;
          if (sourceFrame < 0 || sourceFrame >= frames) continue;
          for (let inputChannel = 0; inputChannel < 128; inputChannel += 1) {
            const input = float16BitsToFloat32(
              scratch[(batch * 128 + inputChannel) * frames + sourceFrame]!,
            );
            const inputGroup = Math.floor(inputChannel / 4);
            const inputLane = inputChannel % 4;
            const weightBase =
              (((kernel * 8 + outputGroup) * 32 + inputGroup) * 4 + inputLane) * 4;
            for (let lane = 0; lane < 4; lane += 1) {
              accumulators[lane] =
                accumulators[lane]! +
                input * float16BitsToFloat32(localWeight[weightBase + lane]!);
            }
          }
        }
        for (let lane = 0; lane < 4; lane += 1) {
          const outputChannel = outputGroup * 4 + lane;
          const local = roundToFloat16(accumulators[lane]!);
          const gate = float16BitsToFloat32(gates[outputChannel]!);
          output[(batch * 32 + outputChannel) * frames + frame] =
            float32ToFloat16Bits(roundToFloat16(local * gate));
        }
      }
    }
  }
  return output;
}

function packedWeightScalar(
  packed: Uint16Array,
  outputChannels: number,
  inputChannels: number,
  outputChannel: number,
  inputChannel: number,
  kernel: number,
): number {
  const outputGroups = ceilDiv(outputChannels, 4);
  const inputGroups = ceilDiv(inputChannels, 4);
  const outputGroup = Math.floor(outputChannel / 4);
  const outputLane = outputChannel % 4;
  const inputGroup = Math.floor(inputChannel / 4);
  const inputLane = inputChannel % 4;
  const index =
    ((((kernel * outputGroups + outputGroup) * inputGroups + inputGroup) * 4 + inputLane) *
      4 +
      outputLane);
  return float16BitsToFloat32(packed[index]!);
}

function ceilDiv(value: number, divisor: number): number {
  return Math.floor((value + divisor - 1) / divisor);
}
