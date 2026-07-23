import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type FixtureDtype =
  | "float32-le"
  | "float64-le"
  | "int32-le"
  | "int64-le";

export interface FixtureArtifact {
  readonly file: string;
  readonly dtype: FixtureDtype;
  readonly shape: readonly number[];
  readonly byteLength: number;
  readonly sha256: string;
  readonly columns?: readonly string[];
}

export interface ClusteringParityManifest {
  readonly schemaVersion: 1;
  readonly fixtureKind: string;
  readonly mode: "seeded-diagnostic" | "unseeded-production-sample";
  readonly source: unknown;
  readonly offlineSources: unknown;
  readonly configuration: unknown;
  readonly environment: unknown;
  readonly timingsMs: unknown;
  readonly labelSummaries: unknown;
  readonly validations: unknown;
  readonly artifacts: Readonly<Record<string, FixtureArtifact>>;
}

export type LoadedFixtureArray =
  | Float32Array
  | Float64Array
  | Int32Array
  | BigInt64Array;

export interface HdbscanParityFixture {
  readonly directory: string;
  readonly manifest: ClusteringParityManifest;
  /** Exact dtype returned by native UMAP: Float32 in umap-learn 0.5.12. */
  readonly projection: Float32Array;
  /** Raw HDBSCAN labels, before Senko CommonClustering post-processing. */
  readonly rawLabels: Int32Array;
}

/**
 * Load only the inputs and oracle labels needed for an HDBSCAN crossover test.
 * Larger optional diagnostics remain lazy via loadFixtureArtifact.
 */
export async function loadHdbscanParityFixture(
  directory: string | URL,
): Promise<HdbscanParityFixture> {
  const fixtureDirectory = normalizeDirectory(directory);
  const manifest = await loadParityManifest(fixtureDirectory);
  const projection = await loadFixtureArtifact(
    fixtureDirectory,
    manifest,
    "umapProjection",
  );
  const rawLabels = await loadFixtureArtifact(
    fixtureDirectory,
    manifest,
    "hdbscanRawLabels",
  );
  if (!(projection instanceof Float32Array)) {
    throw new TypeError(
      `umapProjection must be float32-le, got ${
        manifest.artifacts.umapProjection?.dtype ?? "missing"
      }`,
    );
  }
  if (!(rawLabels instanceof Int32Array)) {
    throw new TypeError(
      `hdbscanRawLabels must be int32-le, got ${
        manifest.artifacts.hdbscanRawLabels?.dtype ?? "missing"
      }`,
    );
  }
  const projectionArtifact = manifest.artifacts.umapProjection!;
  if (
    projectionArtifact.shape.length !== 2 ||
    projectionArtifact.shape[0] !== rawLabels.length
  ) {
    throw new RangeError(
      "umapProjection must have shape [raw-label count, dimension]",
    );
  }

  return {
    directory: fixtureDirectory,
    manifest,
    projection,
    rawLabels,
  };
}

export async function loadParityManifest(
  directory: string | URL,
): Promise<ClusteringParityManifest> {
  const fixtureDirectory = normalizeDirectory(directory);
  const manifestPath = resolve(fixtureDirectory, "manifest.json");
  const parsed: unknown = JSON.parse(await readFile(manifestPath, "utf8"));
  if (!isRecord(parsed)) {
    throw new TypeError("parity fixture manifest must be an object");
  }
  if (parsed.schemaVersion !== 1) {
    throw new RangeError(
      `unsupported parity fixture schema ${String(parsed.schemaVersion)}`,
    );
  }
  if (
    parsed.mode !== "seeded-diagnostic" &&
    parsed.mode !== "unseeded-production-sample"
  ) {
    throw new TypeError(`invalid parity fixture mode ${String(parsed.mode)}`);
  }
  if (!isRecord(parsed.artifacts)) {
    throw new TypeError("parity fixture manifest must contain artifacts");
  }
  for (const [key, value] of Object.entries(parsed.artifacts)) {
    validateArtifact(key, value);
  }
  return parsed as unknown as ClusteringParityManifest;
}

/**
 * Load and SHA-256 validate one typed artifact without retaining unrelated
 * fixture arrays. This keeps long-recording diagnostics memory-bounded.
 */
export async function loadFixtureArtifact(
  directory: string | URL,
  manifest: ClusteringParityManifest,
  key: string,
): Promise<LoadedFixtureArray> {
  const fixtureDirectory = normalizeDirectory(directory);
  const artifact = manifest.artifacts[key];
  if (artifact === undefined) {
    throw new RangeError(`parity fixture is missing artifact ${key}`);
  }
  const artifactPath = resolveInside(fixtureDirectory, artifact.file);
  const bytes = await readFile(artifactPath);
  if (bytes.byteLength !== artifact.byteLength) {
    throw new RangeError(
      `${key} byte length ${bytes.byteLength} does not match manifest ${artifact.byteLength}`,
    );
  }
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (sha256 !== artifact.sha256.toLowerCase()) {
    throw new Error(
      `${key} SHA-256 ${sha256} does not match manifest ${artifact.sha256}`,
    );
  }
  const elementBytes = bytesPerElement(artifact.dtype);
  const expectedElements = checkedShapeProduct(artifact.shape);
  if (expectedElements * elementBytes !== bytes.byteLength) {
    throw new RangeError(
      `${key} shape [${artifact.shape.join(",")}] and dtype ${artifact.dtype} ` +
        `do not match byte length ${bytes.byteLength}`,
    );
  }

  // The native fixture format is explicitly little-endian. WebAssembly and all
  // supported browser hosts are little-endian, but fail loudly in an unusual
  // test environment rather than silently byte-swapping the oracle.
  requireLittleEndianHost();
  const copy = Uint8Array.from(bytes);
  switch (artifact.dtype) {
    case "float32-le":
      return new Float32Array(copy.buffer);
    case "float64-le":
      return new Float64Array(copy.buffer);
    case "int32-le":
      return new Int32Array(copy.buffer);
    case "int64-le":
      return new BigInt64Array(copy.buffer);
  }
}

function normalizeDirectory(directory: string | URL): string {
  return resolve(
    directory instanceof URL ? fileURLToPath(directory) : directory,
  );
}

function resolveInside(directory: string, file: string): string {
  if (isAbsolute(file)) {
    throw new RangeError("fixture artifact paths must be relative");
  }
  const path = resolve(directory, file);
  const relativePath = relative(directory, path);
  if (
    relativePath === ".." ||
    relativePath.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`)
  ) {
    throw new RangeError(`fixture artifact escapes its directory: ${file}`);
  }
  return path;
}

function validateArtifact(key: string, value: unknown): void {
  if (!isRecord(value)) {
    throw new TypeError(`artifact ${key} must be an object`);
  }
  if (typeof value.file !== "string" || value.file.length === 0) {
    throw new TypeError(`artifact ${key} must have a file`);
  }
  if (!isFixtureDtype(value.dtype)) {
    throw new TypeError(`artifact ${key} has invalid dtype ${String(value.dtype)}`);
  }
  if (
    !Array.isArray(value.shape) ||
    value.shape.some(
      (dimension) => !Number.isSafeInteger(dimension) || dimension < 0,
    )
  ) {
    throw new TypeError(`artifact ${key} has an invalid shape`);
  }
  if (!Number.isSafeInteger(value.byteLength) || value.byteLength < 0) {
    throw new TypeError(`artifact ${key} has an invalid byteLength`);
  }
  if (
    typeof value.sha256 !== "string" ||
    !/^[\da-f]{64}$/iu.test(value.sha256)
  ) {
    throw new TypeError(`artifact ${key} has an invalid SHA-256`);
  }
  if (
    value.columns !== undefined &&
    (!Array.isArray(value.columns) ||
      value.columns.some((column) => typeof column !== "string"))
  ) {
    throw new TypeError(`artifact ${key} has invalid columns`);
  }
}

function checkedShapeProduct(shape: readonly number[]): number {
  let product = 1;
  for (const dimension of shape) {
    product *= dimension;
    if (!Number.isSafeInteger(product)) {
      throw new RangeError("fixture artifact shape is too large");
    }
  }
  return product;
}

function bytesPerElement(dtype: FixtureDtype): number {
  switch (dtype) {
    case "float32-le":
    case "int32-le":
      return 4;
    case "float64-le":
    case "int64-le":
      return 8;
  }
}

function isFixtureDtype(value: unknown): value is FixtureDtype {
  return (
    value === "float32-le" ||
    value === "float64-le" ||
    value === "int32-le" ||
    value === "int64-le"
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0001]);
  if (new Uint8Array(words.buffer)[0] !== 1) {
    throw new Error("little-endian parity fixtures require a little-endian host");
  }
}
