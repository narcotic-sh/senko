export type PyannoteFrontendSectionKind =
  | "conv_weight"
  | "conv_bias"
  | "instance_norm_affine";

export type PyannoteFrontendSectionLayout =
  | "K_I_O4_O"
  | "O4"
  | "C4_GAMMA_BETA";

export interface PyannoteFrontendPackedSection {
  readonly id: string;
  readonly kind: PyannoteFrontendSectionKind;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly elementCount: number;
  readonly dtype: "float16" | "float32";
  readonly logicalShape: readonly number[];
  readonly packedShape: readonly number[];
  readonly layout: PyannoteFrontendSectionLayout;
}

export interface PyannoteFrontendPackageMetadata {
  readonly source: {
    readonly sha256: string;
  };
  readonly binary: {
    readonly file: string;
    readonly byteLength: number;
    readonly sha256: string;
    readonly payloadSha256: string;
    readonly headerBytes: number;
    readonly sectionAlignment: number;
    readonly sectionCount: number;
  };
  readonly contract: {
    readonly inputShape: readonly [number, 1, 160000];
    readonly outputShape: readonly [number, 589, 60];
    readonly intermediateDtype: "float16" | "float32";
    readonly weightDtype: "float16" | "float32";
  };
  readonly memory: {
    readonly slotABytes: number;
    readonly slotBBytes: number;
    readonly activationArenaBytes: number;
    readonly statisticsBytes: number;
    readonly minimumResidentGpuBytes: number;
  };
  readonly sections: readonly PyannoteFrontendPackedSection[];
}

export function parsePyannoteFrontendMetadata(
  value: unknown,
): PyannoteFrontendPackageMetadata {
  const root = object(value, "metadata");
  if (
    root.schema !== "senko.pyannote-frontend.webgpu-pack" ||
    root.format_version !== 1
  ) {
    throw new Error("Unsupported pyannote frontend WebGPU package");
  }
  const source = object(root.source, "metadata.source");
  const binary = object(root.binary, "metadata.binary");
  const contract = object(root.contract, "metadata.contract");
  const input = object(contract.input, "metadata.contract.input");
  const output = object(contract.output, "metadata.contract.output");
  const inputShape = tuple(input.shape, [0, 1, 160_000], "metadata.contract.input.shape");
  const outputShape = tuple(output.shape, [inputShape[0]!, 589, 60], "metadata.contract.output.shape");
  if (
    inputShape[0] !== 1 &&
    inputShape[0] !== 8 &&
    inputShape[0] !== 16 &&
    inputShape[0] !== 32
  ) {
    throw new Error("Unsupported pyannote frontend batch size");
  }
  if (
    input.dtype !== "float32" ||
    input.layout !== "BCT" ||
    output.dtype !== "float32" ||
    output.layout !== "BTF" ||
    contract.boundary_dtype !== "float32" ||
    (contract.intermediate_dtype !== "float16" &&
      contract.intermediate_dtype !== "float32") ||
    contract.reduction_dtype !== "float32" ||
    (contract.weight_dtype !== "float16" && contract.weight_dtype !== "float32") ||
    contract.channel_tile !== 4
  ) {
    throw new Error("Unsupported pyannote frontend tensor contract");
  }

  const memory = object(root.memory, "metadata.memory");
  const planned = object(memory.planned_webgpu, "metadata.memory.planned_webgpu");
  const arena = object(planned.aliased_arena, "metadata.memory.planned_webgpu.aliased_arena");
  const sectionsRaw = array(root.sections, "metadata.sections");
  const sections = sectionsRaw.map((item, index) => parseSection(item, index));
  const sectionIds = new Set(sections.map((section) => section.id));
  if (sectionIds.size !== sections.length) {
    throw new Error("Duplicate pyannote frontend section id");
  }
  const sectionAlignment = positiveInteger(
    binary.section_alignment,
    "metadata.binary.section_alignment",
  );
  if (
    sectionAlignment !== 256 ||
    sections.some((section) => section.byteOffset % sectionAlignment !== 0)
  ) {
    throw new Error("Invalid pyannote frontend section alignment");
  }
  const binaryBytes = positiveInteger(binary.byte_length, "metadata.binary.byte_length");
  if (
    sections.some(
      (section) => section.byteOffset + section.byteLength > binaryBytes,
    ) ||
    positiveInteger(binary.section_count, "metadata.binary.section_count") !== sections.length
  ) {
    throw new Error("Pyannote frontend sections exceed the packed binary");
  }

  return {
    source: { sha256: hexDigest(source.sha256, "metadata.source.sha256") },
    binary: {
      file: string(binary.file, "metadata.binary.file"),
      byteLength: binaryBytes,
      sha256: hexDigest(binary.sha256, "metadata.binary.sha256"),
      payloadSha256: hexDigest(
        binary.payload_sha256,
        "metadata.binary.payload_sha256",
      ),
      headerBytes: positiveInteger(binary.header_bytes, "metadata.binary.header_bytes"),
      sectionAlignment,
      sectionCount: sections.length,
    },
    contract: {
      inputShape: inputShape as [number, 1, 160000],
      outputShape: outputShape as [number, 589, 60],
      intermediateDtype: contract.intermediate_dtype,
      weightDtype: contract.weight_dtype,
    },
    memory: {
      slotABytes: positiveInteger(
        arena.slot_a_bytes,
        "metadata.memory.planned_webgpu.aliased_arena.slot_a_bytes",
      ),
      slotBBytes: positiveInteger(
        arena.slot_b_bytes,
        "metadata.memory.planned_webgpu.aliased_arena.slot_b_bytes",
      ),
      activationArenaBytes: positiveInteger(
        arena.activation_arena_bytes,
        "metadata.memory.planned_webgpu.aliased_arena.activation_arena_bytes",
      ),
      statisticsBytes: positiveInteger(
        arena.statistics_bytes,
        "metadata.memory.planned_webgpu.aliased_arena.statistics_bytes",
      ),
      minimumResidentGpuBytes: positiveInteger(
        arena.minimum_resident_gpu_bytes,
        "metadata.memory.planned_webgpu.aliased_arena.minimum_resident_gpu_bytes",
      ),
    },
    sections,
  };
}

function parseSection(value: unknown, index: number): PyannoteFrontendPackedSection {
  const path = `metadata.sections[${index}]`;
  const section = object(value, path);
  const kind = section.kind;
  if (
    kind !== "conv_weight" &&
    kind !== "conv_bias" &&
    kind !== "instance_norm_affine"
  ) {
    throw new Error(`${path}.kind is unsupported`);
  }
  const layout = section.layout;
  if (
    layout !== "K_I_O4_O" &&
    layout !== "O4" &&
    layout !== "C4_GAMMA_BETA"
  ) {
    throw new Error(`${path}.layout is unsupported`);
  }
  if (section.dtype !== "float16" && section.dtype !== "float32") {
    throw new Error(`${path}.dtype must be float16/float32`);
  }
  return {
    id: string(section.id, `${path}.id`),
    kind,
    byteOffset: positiveInteger(section.byte_offset, `${path}.byte_offset`),
    byteLength: positiveInteger(section.byte_length, `${path}.byte_length`),
    elementCount: positiveInteger(section.element_count, `${path}.element_count`),
    dtype: section.dtype,
    logicalShape: dimensions(section.logical_shape, `${path}.logical_shape`),
    packedShape: dimensions(section.packed_shape, `${path}.packed_shape`),
    layout,
  };
}

function object(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value)) throw new Error(`${path} must be an array`);
  return value;
}

function string(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${path} must be a non-empty string`);
  }
  return value;
}

function positiveInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${path} must be a positive integer`);
  }
  return value;
}

function dimensions(value: unknown, path: string): readonly number[] {
  const result = array(value, path).map((item, index) =>
    positiveInteger(item, `${path}[${index}]`),
  );
  if (result.length === 0) throw new Error(`${path} must not be empty`);
  return result;
}

function tuple(
  value: unknown,
  expected: readonly number[],
  path: string,
): readonly number[] {
  const result = dimensions(value, path);
  if (
    result.length !== expected.length ||
    result.some((item, index) => expected[index] !== 0 && item !== expected[index])
  ) {
    throw new Error(`${path} does not match the static frontend contract`);
  }
  return result;
}

function hexDigest(value: unknown, path: string): string {
  const result = string(value, path);
  if (!/^[0-9a-f]{64}$/.test(result)) throw new Error(`${path} must be SHA-256 hex`);
  return result;
}
