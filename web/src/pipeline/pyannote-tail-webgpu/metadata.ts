export interface PyannoteTailSection {
  readonly id: string;
  readonly kind: "matrix" | "bias";
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly logicalShape: readonly number[];
  readonly layout: "I_O4_O" | "O4";
}

export interface PyannoteTailMetadata {
  readonly sourceSha256: string;
  readonly binary: {
    readonly file: string;
    readonly byteLength: number;
    readonly sha256: string;
    readonly payloadSha256: string;
    readonly sectionCount: number;
  };
  readonly batch: number;
  readonly weightBytes: number;
  readonly outputBytes: number;
  readonly readbackBytes: number;
  readonly uniformBytes: number;
  readonly explicitGpuBytes: number;
  readonly sections: ReadonlyMap<string, PyannoteTailSection>;
}

export function parsePyannoteTailMetadata(value: unknown): PyannoteTailMetadata {
  const root = object(value, "metadata");
  if (root.schema !== "senko.pyannote-tail.webgpu-pack" || root.format_version !== 1) {
    throw new Error("Unsupported raw pyannote tail package");
  }
  const source = object(root.source, "metadata.source");
  const binary = object(root.binary, "metadata.binary");
  const contract = object(root.contract, "metadata.contract");
  const inputShape = dimensions(contract.input_shape, "metadata.contract.input_shape");
  const outputShape = dimensions(contract.output_shape, "metadata.contract.output_shape");
  if (
    inputShape.length !== 3 ||
    outputShape.length !== 3 ||
    inputShape[0] !== outputShape[0] ||
    inputShape[1] !== 589 ||
    inputShape[2] !== 256 ||
    outputShape[1] !== 589 ||
    outputShape[2] !== 7 ||
    contract.boundary_dtype !== "float32" ||
    contract.weight_dtype !== "float16" ||
    contract.accumulator_dtype !== "float32"
  ) {
    throw new Error("Unsupported raw pyannote tail tensor contract");
  }
  const sectionList = array(root.sections, "metadata.sections").map((item, index) =>
    parseSection(item, index),
  );
  const sections = new Map(sectionList.map((section) => [section.id, section]));
  if (sections.size !== 6 || sections.size !== sectionList.length) {
    throw new Error("Raw pyannote tail requires six unique sections");
  }
  const binaryBytes = positive(binary.byte_length, "metadata.binary.byte_length");
  if (
    binary.header_bytes !== 256 ||
    binary.section_alignment !== 256 ||
    binary.section_count !== sectionList.length ||
    sectionList.some(
      (section) =>
        section.byteOffset % 256 !== 0 ||
        section.byteOffset + section.byteLength > binaryBytes,
    )
  ) {
    throw new Error("Raw pyannote tail binary layout is invalid");
  }
  const memory = object(root.memory, "metadata.memory");
  return {
    sourceSha256: digest(source.sha256, "metadata.source.sha256"),
    binary: {
      file: text(binary.file, "metadata.binary.file"),
      byteLength: binaryBytes,
      sha256: digest(binary.sha256, "metadata.binary.sha256"),
      payloadSha256: digest(binary.payload_sha256, "metadata.binary.payload_sha256"),
      sectionCount: sectionList.length,
    },
    batch: inputShape[0]!,
    weightBytes: positive(memory.weight_buffer_bytes, "metadata.memory.weight_buffer_bytes"),
    outputBytes: positive(memory.output_buffer_bytes, "metadata.memory.output_buffer_bytes"),
    readbackBytes: positive(memory.readback_buffer_bytes, "metadata.memory.readback_buffer_bytes"),
    uniformBytes: positive(memory.uniform_bytes, "metadata.memory.uniform_bytes"),
    explicitGpuBytes: positive(memory.explicit_gpu_bytes, "metadata.memory.explicit_gpu_bytes"),
    sections,
  };
}

function parseSection(value: unknown, index: number): PyannoteTailSection {
  const path = `metadata.sections[${index}]`;
  const section = object(value, path);
  if (
    (section.kind !== "matrix" && section.kind !== "bias") ||
    (section.layout !== "I_O4_O" && section.layout !== "O4") ||
    section.dtype !== "float16"
  ) {
    throw new Error(`${path} has an unsupported storage contract`);
  }
  return {
    id: text(section.id, `${path}.id`),
    kind: section.kind,
    byteOffset: positive(section.byte_offset, `${path}.byte_offset`),
    byteLength: positive(section.byte_length, `${path}.byte_length`),
    logicalShape: dimensions(section.logical_shape, `${path}.logical_shape`),
    layout: section.layout,
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

function dimensions(value: unknown, path: string): readonly number[] {
  return array(value, path).map((item, index) => positive(item, `${path}[${index}]`));
}

function positive(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${path} must be a positive integer`);
  }
  return value;
}

function text(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${path} must be a non-empty string`);
  }
  return value;
}

function digest(value: unknown, path: string): string {
  const result = text(value, path);
  if (!/^[0-9a-f]{64}$/.test(result)) throw new Error(`${path} must be SHA-256 hex`);
  return result;
}
