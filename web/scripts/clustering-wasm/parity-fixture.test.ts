import { createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  loadFixtureArtifact,
  loadHdbscanParityFixture,
  type ClusteringParityManifest,
  type FixtureArtifact,
} from "./parity-fixture";
import { compareLabelPartitions } from "./parity-diagnostics";
import { WasmClusteringKernels } from "../../src/clustering/wasm-kernels";

const realFixtureEnabled =
  process.env.SENKO_RUN_CLUSTERING_PARITY_FIXTURE === "1";

describe("native clustering parity fixture", () => {
  let directory: string;

  beforeEach(async () => {
    const temporaryRoot = join(process.cwd(), ".tmp");
    await mkdir(temporaryRoot, { recursive: true });
    directory = await mkdtemp(join(temporaryRoot, "clustering-parity-"));
  });

  afterEach(async () => {
    await rm(directory, { recursive: true, force: true });
  });

  it("loads and validates the native projection and raw labels lazily", async () => {
    const projection = new Float32Array([1, 2, 3, 4, 5, 6]);
    const labels = new Int32Array([9, -1, 4]);
    const coreDistances = new Float64Array([0.2, 0.4, 0.6]);
    const manifest = makeManifest({
      umapProjection: await writeArtifact(
        directory,
        "umap-projection.f32",
        projection,
        "float32-le",
        [3, 2],
      ),
      hdbscanRawLabels: await writeArtifact(
        directory,
        "hdbscan-raw-labels.i32",
        labels,
        "int32-le",
        [3],
      ),
      hdbscanCoreDistances: await writeArtifact(
        directory,
        "hdbscan-core-distances.f64",
        coreDistances,
        "float64-le",
        [3],
      ),
    });
    await writeFile(
      join(directory, "manifest.json"),
      JSON.stringify(manifest),
    );

    const fixture = await loadHdbscanParityFixture(directory);

    expect(fixture.projection).toEqual(projection);
    expect(fixture.rawLabels).toEqual(labels);
    expect(
      await loadFixtureArtifact(
        fixture.directory,
        fixture.manifest,
        "hdbscanCoreDistances",
      ),
    ).toEqual(coreDistances);
  });

  it("rejects a binary whose SHA-256 no longer matches its manifest", async () => {
    const projection = new Float32Array([1, 2]);
    const labels = new Int32Array([0]);
    const manifest = makeManifest({
      umapProjection: await writeArtifact(
        directory,
        "umap-projection.f32",
        projection,
        "float32-le",
        [1, 2],
      ),
      hdbscanRawLabels: await writeArtifact(
        directory,
        "hdbscan-raw-labels.i32",
        labels,
        "int32-le",
        [1],
      ),
    });
    await writeFile(
      join(directory, "manifest.json"),
      JSON.stringify(manifest),
    );
    await writeFile(
      join(directory, "hdbscan-raw-labels.i32"),
      new Int32Array([1]),
    );

    await expect(loadHdbscanParityFixture(directory)).rejects.toThrow(
      /SHA-256/,
    );
  });

  it("rejects artifacts that escape the fixture directory", async () => {
    const labels = new Int32Array([0]);
    const artifact = await writeArtifact(
      directory,
      "labels.i32",
      labels,
      "int32-le",
      [1],
    );
    const manifest = makeManifest({
      umapProjection: {
        ...artifact,
        file: "../outside.f32",
        dtype: "float32-le",
        shape: [1, 1],
      },
      hdbscanRawLabels: artifact,
    });
    await writeFile(
      join(directory, "manifest.json"),
      JSON.stringify(manifest),
    );

    await expect(loadHdbscanParityFixture(directory)).rejects.toThrow(
      /escapes its directory/,
    );
  });
});

describe.skipIf(!realFixtureEnabled)("real native clustering parity fixture", () => {
  it("loads the seeded one-hour HDBSCAN oracle with validated hashes", async () => {
    const fixture = await loadHdbscanParityFixture(
      new URL(
        "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
        import.meta.url,
      ),
    );

    expect(fixture.projection.length).toBe(5_713 * 60);
    expect(fixture.rawLabels.length).toBe(5_713);
    expect(fixture.manifest.mode).toBe("seeded-diagnostic");
    expect(new Set(fixture.rawLabels)).toEqual(
      new Set([0, 1, 2, 3, 4, 5, 6]),
    );
    expect(
      await loadFixtureArtifact(
        fixture.directory,
        fixture.manifest,
        "hdbscanMst",
      ),
    ).toHaveLength(5_712 * 3);
  });

  it("matches the seeded one-hour native HDBSCAN partition and noise mask", async () => {
    const fixture = await loadHdbscanParityFixture(
      new URL(
        "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
        import.meta.url,
      ),
    );
    const wasmBytes = await readFile(
      new URL("../../src/clustering/wasm/senko-clustering.wasm", import.meta.url),
    );
    const kernels = await WasmClusteringKernels.fromBytes(wasmBytes);
    try {
      const started = performance.now();
      const candidate = kernels.clusterHdbscanF64Semantics(
        fixture.projection,
        fixture.rawLabels.length,
        60,
        20,
        10,
      );
      const diagnostics = compareLabelPartitions(
        fixture.rawLabels,
        candidate,
      );
      console.info(
        JSON.stringify({
          backend: "wasm-exact-hdbscan",
          elapsedMs: performance.now() - started,
          diagnostics,
          memory: kernels.memoryStats,
        }),
      );
      expect(diagnostics.adjustedRandIndex).toBe(1);
      expect(diagnostics.exactPartition).toBe(true);
      expect(diagnostics.exactNoiseMask).toBe(true);
    } finally {
      kernels.dispose();
    }
  }, 60_000);
});

function makeManifest(
  artifacts: Record<string, FixtureArtifact>,
): ClusteringParityManifest {
  return {
    schemaVersion: 1,
    fixtureKind: "senko-native-clustering-parity",
    mode: "seeded-diagnostic",
    source: {},
    offlineSources: {},
    configuration: {},
    environment: {},
    timingsMs: {},
    labelSummaries: {},
    validations: {},
    artifacts,
  };
}

async function writeArtifact(
  directory: string,
  file: string,
  values: ArrayBufferView,
  dtype: FixtureArtifact["dtype"],
  shape: readonly number[],
): Promise<FixtureArtifact> {
  const bytes = new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  );
  await writeFile(join(directory, file), bytes);
  return {
    file,
    dtype,
    shape,
    byteLength: bytes.byteLength,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  };
}
