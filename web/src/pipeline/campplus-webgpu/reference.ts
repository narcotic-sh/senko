export interface PackedBctReferenceParameters {
  readonly batchSize: number;
  readonly inputChannels: number;
  readonly outputChannels: number;
  readonly inputFrames: number;
  readonly outputFrames: number;
  readonly kernelElements: number;
  readonly stride: number;
  readonly dilation: number;
  readonly padLeft: number;
  readonly outputRelu: boolean;
}

/** Test/oracle packer for logical [output,input,kernel] FP32 weights. */
export function packOikWeightReference(
  logical: Float32Array,
  outputChannels: number,
  inputChannels: number,
  kernelElements: number,
): Uint16Array {
  if (logical.length !== outputChannels * inputChannels * kernelElements) {
    throw new RangeError("Logical convolution weight length disagrees with its shape");
  }
  const outputGroups = ceilDiv(outputChannels, 4);
  const inputGroups = ceilDiv(inputChannels, 4);
  const packed = new Uint16Array(kernelElements * outputGroups * inputGroups * 16);
  for (let kernel = 0; kernel < kernelElements; kernel += 1) {
    for (let output = 0; output < outputChannels; output += 1) {
      for (let input = 0; input < inputChannels; input += 1) {
        const outputGroup = Math.floor(output / 4);
        const outputLane = output % 4;
        const inputGroup = Math.floor(input / 4);
        const inputLane = input % 4;
        const packedIndex =
          ((((kernel * outputGroups + outputGroup) * inputGroups + inputGroup) * 4 +
            inputLane) *
            4 +
            outputLane);
        const logicalIndex = (output * inputChannels + input) * kernelElements + kernel;
        packed[packedIndex] = float32ToFloat16Bits(logical[logicalIndex]!);
      }
    }
  }
  return packed;
}

export function packBiasReference(logical: Float32Array): Uint16Array {
  const packed = new Uint16Array(ceilDiv(logical.length, 4) * 4);
  for (let index = 0; index < logical.length; index += 1) {
    packed[index] = float32ToFloat16Bits(logical[index]!);
  }
  return packed;
}

export function packAffineReference(
  scale: Float32Array,
  shift: Float32Array,
): Float32Array {
  if (scale.length !== shift.length) throw new RangeError("Affine scale and shift lengths differ");
  const groups = ceilDiv(scale.length, 4);
  const packed = new Float32Array(groups * 8);
  for (let channel = 0; channel < scale.length; channel += 1) {
    const group = Math.floor(channel / 4);
    const lane = channel % 4;
    packed[group * 8 + lane] = scale[channel]!;
    packed[group * 8 + 4 + lane] = shift[channel]!;
  }
  return packed;
}

/** CPU evaluator with the same K_O4_I4_I_O indexing and FP16 boundaries as WGSL. */
export function evaluatePackedBctConvReference(
  input: Uint16Array,
  packedWeight: Uint16Array,
  packedBias: Uint16Array,
  parameters: PackedBctReferenceParameters,
  packedAffine?: Float32Array,
): Uint16Array {
  const {
    batchSize,
    inputChannels,
    outputChannels,
    inputFrames,
    outputFrames,
    kernelElements,
  } = parameters;
  if (input.length !== batchSize * inputChannels * inputFrames) {
    throw new RangeError("Packed reference input length disagrees with BCT dimensions");
  }
  const inputGroups = ceilDiv(inputChannels, 4);
  const outputGroups = ceilDiv(outputChannels, 4);
  if (packedWeight.length !== kernelElements * outputGroups * inputGroups * 16) {
    throw new RangeError("Packed reference weight length disagrees with dimensions");
  }
  if (packedBias.length !== outputGroups * 4) {
    throw new RangeError("Packed reference bias length disagrees with output channels");
  }
  if (packedAffine !== undefined && packedAffine.length !== inputGroups * 8) {
    throw new RangeError("Packed reference affine length disagrees with input channels");
  }

  const output = new Uint16Array(batchSize * outputChannels * outputFrames);
  for (let batch = 0; batch < batchSize; batch += 1) {
    for (let outputGroup = 0; outputGroup < outputGroups; outputGroup += 1) {
      for (let frame = 0; frame < outputFrames; frame += 1) {
        const accumulators = [0, 0, 0, 0];
        for (let lane = 0; lane < 4; lane += 1) {
          accumulators[lane] = float16BitsToFloat32(packedBias[outputGroup * 4 + lane]!);
        }
        for (let kernel = 0; kernel < kernelElements; kernel += 1) {
          const sourceFrame =
            frame * parameters.stride + kernel * parameters.dilation - parameters.padLeft;
          if (sourceFrame < 0 || sourceFrame >= inputFrames) continue;
          for (let inputChannel = 0; inputChannel < inputChannels; inputChannel += 1) {
            const inputIndex =
              (batch * inputChannels + inputChannel) * inputFrames + sourceFrame;
            let inputValue = float16BitsToFloat32(input[inputIndex]!);
            if (packedAffine !== undefined) {
              const inputGroup = Math.floor(inputChannel / 4);
              const inputLane = inputChannel % 4;
              const scale = packedAffine[inputGroup * 8 + inputLane]!;
              const shift = packedAffine[inputGroup * 8 + 4 + inputLane]!;
              inputValue = Math.max(roundToFloat16(inputValue * scale + shift), 0);
            }
            const inputGroup = Math.floor(inputChannel / 4);
            const inputLane = inputChannel % 4;
            const packedBase =
              ((((kernel * outputGroups + outputGroup) * inputGroups + inputGroup) * 4 +
                inputLane) *
                4);
            for (let lane = 0; lane < 4; lane += 1) {
              const weight = float16BitsToFloat32(packedWeight[packedBase + lane]!);
              accumulators[lane] = accumulators[lane]! + inputValue * weight;
            }
          }
        }
        for (let lane = 0; lane < 4; lane += 1) {
          const outputChannel = outputGroup * 4 + lane;
          if (outputChannel >= outputChannels) continue;
          let value = roundToFloat16(accumulators[lane]!);
          if (parameters.outputRelu) value = Math.max(value, 0);
          const outputIndex =
            (batch * outputChannels + outputChannel) * outputFrames + frame;
          output[outputIndex] = float32ToFloat16Bits(value);
        }
      }
    }
  }
  return output;
}

export function roundToFloat16(value: number): number {
  return float16BitsToFloat32(float32ToFloat16Bits(value));
}

const FLOAT_VIEW = new Float32Array(1);
const UINT_VIEW = new Uint32Array(FLOAT_VIEW.buffer);

export function float32ToFloat16Bits(value: number): number {
  FLOAT_VIEW[0] = value;
  const bits = UINT_VIEW[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (exponent === 0xff) {
    return sign | (mantissa === 0 ? 0x7c00 : 0x7e00 | (mantissa >>> 13));
  }

  let halfExponent = exponent - 127 + 15;
  if (halfExponent >= 31) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significand = mantissa | 0x800000;
    const shift = 14 - halfExponent;
    let halfMantissa = significand >>> shift;
    const remainder = significand & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (remainder > halfway || (remainder === halfway && (halfMantissa & 1) !== 0)) {
      halfMantissa += 1;
    }
    return sign | halfMantissa;
  }

  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) !== 0)) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      halfMantissa = 0;
      halfExponent += 1;
      if (halfExponent >= 31) return sign | 0x7c00;
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

export function float16BitsToFloat32(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function ceilDiv(value: number, divisor: number): number {
  return Math.floor((value + divisor - 1) / divisor);
}
