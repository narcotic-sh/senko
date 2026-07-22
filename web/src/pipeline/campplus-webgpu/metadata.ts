export type CampPlusSectionKind =
  | "conv_weight"
  | "conv_bias"
  | "batch_norm_affine";
export type CampPlusSectionDtype = "float16" | "float32";
export type CampPlusSectionLayout = "K_O4_I4_I_O" | "O4" | "C4_SCALE_SHIFT";

export interface CampPlusPackedSection {
  readonly id: string;
  readonly kind: CampPlusSectionKind;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly elementCount: number;
  readonly dtype: CampPlusSectionDtype;
  readonly logicalShape: readonly number[];
  readonly packedShape: readonly number[];
  readonly layout: CampPlusSectionLayout;
}

export interface PackedConvolutionRef {
  readonly weight: string;
  readonly bias: string;
}

export interface CamDenseLayerMetadata {
  readonly id: string;
  readonly layer: number;
  readonly inputChannels: number;
  readonly appendChannel: number;
  readonly bottleneckChannels: number;
  readonly outputChannels: number;
  readonly frames: number;
  readonly preactivationAffine: string;
  readonly bottleneck: PackedConvolutionRef;
  readonly local: PackedConvolutionRef;
  readonly localDilation: number;
  readonly attention1: PackedConvolutionRef;
  readonly attention2: PackedConvolutionRef;
}

export interface CamDenseBlockMetadata {
  readonly id: string;
  readonly layers: readonly CamDenseLayerMetadata[];
}

export interface CamTransitMetadata {
  readonly id: string;
  readonly preactivationAffine: string;
  readonly pointwise: PackedConvolutionRef;
  readonly epilogue: "identity" | "relu";
}

export interface CampPlusFusedProgram {
  readonly headConvolutions: readonly PackedConvolutionRef[];
  readonly tdnn: PackedConvolutionRef;
  readonly blocks: readonly CamDenseBlockMetadata[];
  readonly transits: readonly CamTransitMetadata[];
  readonly finalDense: PackedConvolutionRef;
  readonly finalOutputAffine: string;
}

export interface CampPlusMemoryPlan {
  readonly frontendMicrobatch: number;
  readonly activationArenaBytes: number;
  readonly weightBufferBytes: number;
  readonly minimumResidentGpuBytes: number;
  readonly tradeoffs: readonly {
    readonly frontendMicrobatch: number;
    readonly activationArenaBytes: number;
    readonly minimumResidentGpuBytes: number;
    readonly frontendTdnnDispatches: number;
  }[];
}

export interface CampPlusPackageMetadata {
  readonly source: {
    readonly file: string;
    readonly byteLength: number;
    readonly sha256: string;
  };
  readonly binary: {
    readonly file: string;
    readonly byteLength: number;
    readonly sha256: string;
    readonly payloadSha256: string;
    readonly headerBytes: 256;
    readonly sectionAlignment: 256;
    readonly sectionCount: number;
  };
  readonly contract: {
    readonly inputShape: readonly [number, 150, 80];
    readonly outputShape: readonly [number, 192];
    readonly requiredWebGpuFeatures: readonly ["shader-f16"];
  };
  readonly memory: CampPlusMemoryPlan;
  readonly sections: readonly CampPlusPackedSection[];
  readonly fusedProgram: CampPlusFusedProgram;
}

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const HEADER_BYTES = 256;
const SECTION_ALIGNMENT = 256;

export function parseCampPlusMetadata(value: unknown): CampPlusPackageMetadata {
  const root = expectObject(value, "metadata");
  if (expectString(root.schema, "metadata.schema") !== "senko.campplus.webgpu-pack") {
    throw new Error("Unsupported CAM++ package schema");
  }
  if (expectInteger(root.format_version, "metadata.format_version") !== 1) {
    throw new Error("Unsupported CAM++ package format version");
  }

  const sourceObject = expectObject(root.source, "metadata.source");
  const source = {
    file: expectString(sourceObject.file, "metadata.source.file"),
    byteLength: expectPositiveInteger(sourceObject.byte_length, "metadata.source.byte_length"),
    sha256: expectSha256(sourceObject.sha256, "metadata.source.sha256"),
  };

  const binaryObject = expectObject(root.binary, "metadata.binary");
  const binary = {
    file: expectString(binaryObject.file, "metadata.binary.file"),
    byteLength: expectPositiveInteger(binaryObject.byte_length, "metadata.binary.byte_length"),
    sha256: expectSha256(binaryObject.sha256, "metadata.binary.sha256"),
    payloadSha256: expectSha256(
      binaryObject.payload_sha256,
      "metadata.binary.payload_sha256",
    ),
    headerBytes: expectInteger(binaryObject.header_bytes, "metadata.binary.header_bytes"),
    sectionAlignment: expectInteger(
      binaryObject.section_alignment,
      "metadata.binary.section_alignment",
    ),
    sectionCount: expectPositiveInteger(
      binaryObject.section_count,
      "metadata.binary.section_count",
    ),
  };
  if (binary.headerBytes !== HEADER_BYTES || binary.sectionAlignment !== SECTION_ALIGNMENT) {
    throw new Error("CAM++ package v1 requires a 256-byte header and section alignment");
  }
  if (binary.byteLength % SECTION_ALIGNMENT !== 0) {
    throw new Error("CAM++ binary length must be section aligned");
  }

  const contractObject = expectObject(root.contract, "metadata.contract");
  if (contractObject.internal_dtype !== "float16" || contractObject.channel_tile !== 4) {
    throw new Error("CAM++ package requires FP16 internals and four-channel packing");
  }
  if (contractObject.weights_are_batch_independent !== true) {
    throw new Error("CAM++ packed weights must be batch independent");
  }
  const inputObject = expectObject(contractObject.input, "metadata.contract.input");
  const outputObject = expectObject(contractObject.output, "metadata.contract.output");
  if (inputObject.dtype !== "float32" || outputObject.dtype !== "float32") {
    throw new Error("CAM++ package API boundaries must be FP32");
  }
  const inputShape = expectIntegerArray(inputObject.shape, "metadata.contract.input.shape");
  const outputShape = expectIntegerArray(outputObject.shape, "metadata.contract.output.shape");
  if (
    inputShape.length !== 3 ||
    inputShape[0]! <= 0 ||
    inputShape[1] !== 150 ||
    inputShape[2] !== 80
  ) {
    throw new Error("CAM++ input shape must be [batch,150,80]");
  }
  if (
    outputShape.length !== 2 ||
    outputShape[0] !== inputShape[0] ||
    outputShape[1] !== 192
  ) {
    throw new Error("CAM++ output shape must be [batch,192]");
  }
  const requiredFeatures = expectArray(
    contractObject.required_webgpu_features,
    "metadata.contract.required_webgpu_features",
  );
  if (requiredFeatures.length !== 1 || requiredFeatures[0] !== "shader-f16") {
    throw new Error("CAM++ package v1 requires exactly the shader-f16 WebGPU feature");
  }

  const sections = parseSections(root.sections, binary);
  const sectionMap = new Map(sections.map((section) => [section.id, section]));
  const fusedProgram = parseFusedProgram(root.fused_program, sectionMap);
  const memory = parseMemoryPlan(root.memory, binary.byteLength, inputShape[0]!);

  return {
    source,
    binary: {
      ...binary,
      headerBytes: HEADER_BYTES,
      sectionAlignment: SECTION_ALIGNMENT,
    },
    contract: {
      inputShape: [inputShape[0]!, 150, 80],
      outputShape: [outputShape[0]!, 192],
      requiredWebGpuFeatures: ["shader-f16"],
    },
    memory,
    sections,
    fusedProgram,
  };
}

function parseSections(
  value: unknown,
  binary: {
    readonly byteLength: number;
    readonly sectionCount: number;
    readonly headerBytes: number;
    readonly sectionAlignment: number;
  },
): readonly CampPlusPackedSection[] {
  const values = expectArray(value, "metadata.sections");
  if (values.length !== binary.sectionCount) {
    throw new Error("CAM++ metadata section count does not match its binary declaration");
  }
  const ids = new Set<string>();
  const sections = values.map((item, index): CampPlusPackedSection => {
    const path = `metadata.sections[${index}]`;
    const object = expectObject(item, path);
    const id = expectString(object.id, `${path}.id`);
    if (ids.has(id)) throw new Error(`Duplicate CAM++ section id: ${id}`);
    ids.add(id);
    const kind = expectEnum(
      object.kind,
      ["conv_weight", "conv_bias", "batch_norm_affine"] as const,
      `${path}.kind`,
    );
    const dtype = expectEnum(
      object.dtype,
      ["float16", "float32"] as const,
      `${path}.dtype`,
    );
    const layout = expectEnum(
      object.layout,
      ["K_O4_I4_I_O", "O4", "C4_SCALE_SHIFT"] as const,
      `${path}.layout`,
    );
    const byteOffset = expectPositiveInteger(object.byte_offset, `${path}.byte_offset`);
    const byteLength = expectPositiveInteger(object.byte_length, `${path}.byte_length`);
    const elementCount = expectPositiveInteger(object.element_count, `${path}.element_count`);
    const logicalShape = expectPositiveIntegerArray(object.logical_shape, `${path}.logical_shape`);
    const packedShape = expectPositiveIntegerArray(object.packed_shape, `${path}.packed_shape`);
    if (byteOffset < binary.headerBytes || byteOffset % binary.sectionAlignment !== 0) {
      throw new Error(`${path} is not aligned after the package header`);
    }
    if (byteLength % 4 !== 0 || byteOffset + byteLength > binary.byteLength) {
      throw new Error(`${path} has an invalid byte range`);
    }
    const width = dtype === "float16" ? 2 : 4;
    if (byteLength !== elementCount * width || product(packedShape) !== elementCount) {
      throw new Error(`${path} byte length and packed shape disagree`);
    }
    validateSectionLayout(path, kind, dtype, layout, logicalShape, packedShape);
    return {
      id,
      kind,
      byteOffset,
      byteLength,
      elementCount,
      dtype,
      logicalShape,
      packedShape,
      layout,
    };
  });
  const sorted = [...sections].sort((left, right) => left.byteOffset - right.byteOffset);
  for (let index = 1; index < sorted.length; index += 1) {
    const previous = sorted[index - 1]!;
    const current = sorted[index]!;
    if (current.byteOffset < previous.byteOffset + previous.byteLength) {
      throw new Error(`CAM++ sections overlap: ${previous.id} and ${current.id}`);
    }
  }
  return sections;
}

function validateSectionLayout(
  path: string,
  kind: CampPlusSectionKind,
  dtype: CampPlusSectionDtype,
  layout: CampPlusSectionLayout,
  logical: readonly number[],
  packed: readonly number[],
): void {
  if (kind === "conv_weight") {
    if (dtype !== "float16" || layout !== "K_O4_I4_I_O" || logical.length < 3) {
      throw new Error(`${path} is not a packed FP16 convolution weight`);
    }
    const kernelElements = product(logical.slice(2));
    const expected = [kernelElements, ceilDiv(logical[0]!, 4), ceilDiv(logical[1]!, 4), 4, 4];
    if (!sameShape(packed, expected)) throw new Error(`${path} has an invalid weight tile shape`);
    return;
  }
  if (kind === "conv_bias") {
    if (
      dtype !== "float16" ||
      layout !== "O4" ||
      logical.length !== 1 ||
      !sameShape(packed, [ceilDiv(logical[0]!, 4), 4])
    ) {
      throw new Error(`${path} is not a packed FP16 convolution bias`);
    }
    return;
  }
  if (
    dtype !== "float32" ||
    layout !== "C4_SCALE_SHIFT" ||
    logical.length !== 2 ||
    logical[1] !== 2 ||
    !sameShape(packed, [ceilDiv(logical[0]!, 4), 2, 4])
  ) {
    throw new Error(`${path} is not a packed FP32 BatchNorm affine`);
  }
}

function parseFusedProgram(
  value: unknown,
  sections: ReadonlyMap<string, CampPlusPackedSection>,
): CampPlusFusedProgram {
  const object = expectObject(value, "metadata.fused_program");
  const head = expectObject(object.head, "metadata.fused_program.head");
  const headConvolutions = expectArray(
    head.convolutions,
    "metadata.fused_program.head.convolutions",
  ).map((item, index) =>
    parseConvolutionRef(
      item,
      `metadata.fused_program.head.convolutions[${index}]`,
      sections,
    ),
  );
  const tdnnObject = expectObject(object.tdnn, "metadata.fused_program.tdnn");
  const tdnn = parseConvolutionRef(
    tdnnObject.convolution,
    "metadata.fused_program.tdnn.convolution",
    sections,
  );

  const blocks = expectArray(object.blocks, "metadata.fused_program.blocks").map(
    (blockValue, blockIndex): CamDenseBlockMetadata => {
      const path = `metadata.fused_program.blocks[${blockIndex}]`;
      const block = expectObject(blockValue, path);
      const layers = expectArray(block.layers, `${path}.layers`).map(
        (layerValue, layerIndex): CamDenseLayerMetadata => {
          const layerPath = `${path}.layers[${layerIndex}]`;
          const layer = expectObject(layerValue, layerPath);
          const affine = expectString(
            layer.preactivation_affine,
            `${layerPath}.preactivation_affine`,
          );
          expectSection(sections, affine, "batch_norm_affine", layerPath);
          return {
            id: expectString(layer.id, `${layerPath}.id`),
            layer: expectPositiveInteger(layer.layer, `${layerPath}.layer`),
            inputChannels: expectPositiveInteger(
              layer.input_channels,
              `${layerPath}.input_channels`,
            ),
            appendChannel: expectPositiveInteger(
              layer.append_channel,
              `${layerPath}.append_channel`,
            ),
            bottleneckChannels: expectPositiveInteger(
              layer.bottleneck_channels,
              `${layerPath}.bottleneck_channels`,
            ),
            outputChannels: expectPositiveInteger(
              layer.output_channels,
              `${layerPath}.output_channels`,
            ),
            frames: expectPositiveInteger(layer.frames, `${layerPath}.frames`),
            preactivationAffine: affine,
            bottleneck: parseConvolutionRef(
              layer.bottleneck,
              `${layerPath}.bottleneck`,
              sections,
            ),
            local: parseConvolutionRef(layer.local, `${layerPath}.local`, sections),
            localDilation: expectPositiveInteger(
              layer.local_dilation,
              `${layerPath}.local_dilation`,
            ),
            attention1: parseConvolutionRef(
              layer.attention1,
              `${layerPath}.attention1`,
              sections,
            ),
            attention2: parseConvolutionRef(
              layer.attention2,
              `${layerPath}.attention2`,
              sections,
            ),
          };
        },
      );
      return { id: expectString(block.id, `${path}.id`), layers };
    },
  );

  const transits = expectArray(object.transits, "metadata.fused_program.transits").map(
    (transitValue, index): CamTransitMetadata => {
      const path = `metadata.fused_program.transits[${index}]`;
      const transit = expectObject(transitValue, path);
      const affine = expectString(transit.preactivation_affine, `${path}.preactivation_affine`);
      expectSection(sections, affine, "batch_norm_affine", path);
      return {
        id: expectString(transit.id, `${path}.id`),
        preactivationAffine: affine,
        pointwise: parseConvolutionRef(transit.pointwise, `${path}.pointwise`, sections),
        epilogue: expectEnum(transit.epilogue, ["identity", "relu"] as const, `${path}.epilogue`),
      };
    },
  );

  const finalObject = expectObject(object.final, "metadata.fused_program.final");
  const finalOutputAffine = expectString(
    finalObject.output_affine,
    "metadata.fused_program.final.output_affine",
  );
  expectSection(sections, finalOutputAffine, "batch_norm_affine", "metadata.fused_program.final");
  return {
    headConvolutions,
    tdnn,
    blocks,
    transits,
    finalDense: parseConvolutionRef(
      finalObject.dense,
      "metadata.fused_program.final.dense",
      sections,
    ),
    finalOutputAffine,
  };
}

function parseConvolutionRef(
  value: unknown,
  path: string,
  sections: ReadonlyMap<string, CampPlusPackedSection>,
): PackedConvolutionRef {
  const object = expectObject(value, path);
  const weight = expectString(object.weight, `${path}.weight`);
  const bias = expectString(object.bias, `${path}.bias`);
  const weightSection = expectSection(sections, weight, "conv_weight", path);
  const biasSection = expectSection(sections, bias, "conv_bias", path);
  if (weightSection.logicalShape[0] !== biasSection.logicalShape[0]) {
    throw new Error(`${path} weight and bias output channels disagree`);
  }
  return { weight, bias };
}

function parseMemoryPlan(
  value: unknown,
  binaryBytes: number,
  sourceBatch: number,
): CampPlusMemoryPlan {
  const memory = expectObject(value, "metadata.memory");
  const planned = expectObject(memory.planned_webgpu, "metadata.memory.planned_webgpu");
  const recommended = expectObject(
    planned.recommended,
    "metadata.memory.planned_webgpu.recommended",
  );
  const plan = {
    frontendMicrobatch: expectPositiveInteger(
      recommended.frontend_microbatch,
      "metadata.memory.planned_webgpu.recommended.frontend_microbatch",
    ),
    activationArenaBytes: expectPositiveInteger(
      recommended.activation_arena_bytes,
      "metadata.memory.planned_webgpu.recommended.activation_arena_bytes",
    ),
    weightBufferBytes: expectPositiveInteger(
      recommended.weight_buffer_bytes,
      "metadata.memory.planned_webgpu.recommended.weight_buffer_bytes",
    ),
    minimumResidentGpuBytes: expectPositiveInteger(
      recommended.minimum_resident_gpu_bytes,
      "metadata.memory.planned_webgpu.recommended.minimum_resident_gpu_bytes",
    ),
  };
  if (
    plan.frontendMicrobatch !== sourceBatch ||
    plan.weightBufferBytes !== binaryBytes ||
    plan.minimumResidentGpuBytes !== plan.activationArenaBytes + binaryBytes ||
    plan.activationArenaBytes % SECTION_ALIGNMENT !== 0
  ) {
    throw new Error("CAM++ recommended memory plan is internally inconsistent");
  }
  const tradeoffs = expectArray(
    planned.frontend_microbatch_tradeoffs,
    "metadata.memory.planned_webgpu.frontend_microbatch_tradeoffs",
  ).map((item, index) => {
    const path = `metadata.memory.planned_webgpu.frontend_microbatch_tradeoffs[${index}]`;
    const object = expectObject(item, path);
    const activationArenaBytes = expectPositiveInteger(
      object.activation_arena_bytes,
      `${path}.activation_arena_bytes`,
    );
    const minimumResidentGpuBytes = expectPositiveInteger(
      object.minimum_resident_gpu_bytes,
      `${path}.minimum_resident_gpu_bytes`,
    );
    if (minimumResidentGpuBytes !== activationArenaBytes + binaryBytes) {
      throw new Error(`${path} has inconsistent resident memory accounting`);
    }
    return {
      frontendMicrobatch: expectPositiveInteger(
        object.frontend_microbatch,
        `${path}.frontend_microbatch`,
      ),
      activationArenaBytes,
      minimumResidentGpuBytes,
      frontendTdnnDispatches: expectPositiveInteger(
        object.frontend_tdnn_dispatches,
        `${path}.frontend_tdnn_dispatches`,
      ),
    };
  });
  return { ...plan, tradeoffs };
}

function expectSection(
  sections: ReadonlyMap<string, CampPlusPackedSection>,
  id: string,
  kind: CampPlusSectionKind,
  path: string,
): CampPlusPackedSection {
  const section = sections.get(id);
  if (section === undefined || section.kind !== kind) {
    throw new Error(`${path} references missing or incompatible section ${id}`);
  }
  return section;
}

function expectObject(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function expectArray(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value)) throw new TypeError(`${path} must be an array`);
  return value;
}

function expectString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new TypeError(`${path} must be a non-empty string`);
  }
  return value;
}

function expectInteger(value: unknown, path: string): number {
  if (!Number.isSafeInteger(value)) throw new TypeError(`${path} must be a safe integer`);
  return value as number;
}

function expectPositiveInteger(value: unknown, path: string): number {
  const result = expectInteger(value, path);
  if (result <= 0) throw new RangeError(`${path} must be positive`);
  return result;
}

function expectIntegerArray(value: unknown, path: string): readonly number[] {
  return expectArray(value, path).map((item, index) => expectInteger(item, `${path}[${index}]`));
}

function expectPositiveIntegerArray(value: unknown, path: string): readonly number[] {
  return expectArray(value, path).map((item, index) =>
    expectPositiveInteger(item, `${path}[${index}]`),
  );
}

function expectSha256(value: unknown, path: string): string {
  const result = expectString(value, path).toLowerCase();
  if (!SHA256_PATTERN.test(result)) throw new TypeError(`${path} must be a SHA-256 digest`);
  return result;
}

function expectEnum<const T extends readonly string[]>(
  value: unknown,
  choices: T,
  path: string,
): T[number] {
  if (typeof value !== "string" || !choices.includes(value)) {
    throw new TypeError(`${path} must be one of ${choices.join(", ")}`);
  }
  return value as T[number];
}

function product(values: readonly number[]): number {
  return values.reduce((result, value) => result * value, 1);
}

function ceilDiv(value: number, divisor: number): number {
  return Math.floor((value + divisor - 1) / divisor);
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
