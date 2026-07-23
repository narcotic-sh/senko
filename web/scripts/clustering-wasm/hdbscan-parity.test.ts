import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import { WasmClusteringKernels } from "../../src/clustering/wasm-kernels";
import {
  compareLabelPartitions,
  compareMstEdges,
  compareNumericArrays,
  type UndirectedWeightedEdge,
} from "./parity-diagnostics";
import {
  loadFixtureArtifact,
  loadHdbscanParityFixture,
} from "./parity-fixture";

const parityIt =
  process.env.SENKO_RUN_HDBSCAN_PARITY === "1" ? it : it.skip;
const diagnosticIt =
  process.env.SENKO_RUN_HDBSCAN_DIAGNOSTICS === "1" ? it : it.skip;
const longParityIt =
  process.env.SENKO_RUN_HDBSCAN_LONG_PARITY === "1" ? it : it.skip;

describe("HDBSCAN native parity", () => {
  parityIt(
    "matches the seeded short-recording native projection exactly",
    async () => {
      await expectNativeParity(
        new URL(
          "../../../.research/native-reference/clustering-parity/test-audio-short/seed-42/",
          import.meta.url,
        ),
        [806, 60],
      );
    },
    120_000,
  );

  parityIt(
    "matches the seeded one-hour native projection exactly",
    async () => {
      await expectNativeParity(
        new URL(
          "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
          import.meta.url,
        ),
        [5_713, 60],
      );
    },
    120_000,
  );

  longParityIt(
    "matches the seeded 43,804-point long-recording partition exactly",
    async () => {
      await expectNativeParity(
        new URL(
          "../../../.research/native-reference/clustering-parity/test-audio-long/seed-42/",
          import.meta.url,
        ),
        [43_804, 60],
        70 * 1024 * 1024,
      );
    },
    120_000,
  );

  diagnosticIt(
    "matches one-hour native core distances and approximate MST exactly",
    async () => {
      const fixture = await loadHdbscanParityFixture(
        new URL(
          "../../../.research/native-reference/clustering-parity/test-audio/seed-42/",
          import.meta.url,
        ),
      );
      const nativeCore = await loadFixtureArtifact(
        fixture.directory,
        fixture.manifest,
        "hdbscanCoreDistances",
      );
      const nativeMst = await loadFixtureArtifact(
        fixture.directory,
        fixture.manifest,
        "hdbscanMst",
      );
      expect(nativeCore).toBeInstanceOf(Float64Array);
      expect(nativeMst).toBeInstanceOf(Float64Array);

      const wasmBytes = await readFile(
        new URL(
          "../../src/clustering/wasm/senko-clustering.wasm",
          import.meta.url,
        ),
      );
      const kernels = await WasmClusteringKernels.fromBytes(wasmBytes);
      try {
        const diagnostic = kernels.diagnoseHdbscanF64Semantics(
          fixture.projection,
          fixture.rawLabels.length,
          60,
          20,
          10,
        );
        const coreParity = compareNumericArrays(
          nativeCore as Float64Array,
          diagnostic.coreDistances,
          { absolute: 1e-12, relative: 1e-12 },
        );
        const mstParity = compareMstEdges(
          unpackMst(nativeMst as Float64Array),
          unpackMst(diagnostic.mstRows),
          { absolute: 1e-12, relative: 1e-12 },
        );
        const mstWeightMultisetParity = compareNumericArrays(
          sortedMstWeights(nativeMst as Float64Array),
          sortedMstWeights(diagnostic.mstRows),
          { absolute: 1e-12, relative: 1e-12 },
        );
        const labels = compareLabelPartitions(
          fixture.rawLabels,
          diagnostic.labels,
        );
        console.info(
          JSON.stringify({
            coreParity,
            mstParity: {
              exactEndpoints: mstParity.exactEndpoints,
              missingEndpointCount: mstParity.missingEndpointPairs.length,
              unexpectedEndpointCount:
                mstParity.unexpectedEndpointPairs.length,
              sharedEndpointWeightParity: mstParity.weights,
            },
            mstWeightMultisetParity,
            labels,
          }),
        );
        expect(coreParity.mismatchCount).toBe(0);
        expect(mstParity.exactEndpoints).toBe(true);
        expect(mstParity.weights.mismatchCount).toBe(0);
        expect(mstWeightMultisetParity.mismatchCount).toBe(0);
        expect(labels.exactPartition).toBe(true);
        expect(labels.exactNoiseMask).toBe(true);
      } finally {
        kernels.dispose();
      }
    },
    120_000,
  );
});

async function expectNativeParity(
  directory: URL,
  expectedShape: readonly [number, number],
  maximumScratchBytes = 10 * 1024 * 1024,
): Promise<void> {
  const fixture = await loadHdbscanParityFixture(directory);
  const wasmBytes = await readFile(
    new URL(
      "../../src/clustering/wasm/senko-clustering.wasm",
      import.meta.url,
    ),
  );
  const kernels = await WasmClusteringKernels.fromBytes(wasmBytes);
  const shape =
    fixture.manifest.artifacts.umapProjection?.shape ?? [];
  expect(shape).toEqual(expectedShape);

  try {
    const startedAt = performance.now();
    const labels = kernels.clusterHdbscanF64Semantics(
      fixture.projection,
      shape[0]!,
      shape[1]!,
      20,
      10,
    );
    const elapsedMs = performance.now() - startedAt;
    const diagnostics = compareLabelPartitions(
      fixture.rawLabels,
      labels,
    );

    expect(diagnostics.exactPartition).toBe(true);
    expect(diagnostics.exactNoiseMask).toBe(true);
    expect(diagnostics.adjustedRandIndex).toBe(1);
    expect(kernels.memoryStats.peakArenaUsedBytes).toBeLessThan(
      maximumScratchBytes,
    );
    console.info(
      `HDBSCAN ${shape[0]}-row native parity: ${elapsedMs.toFixed(1)} ms, ` +
        `${kernels.memoryStats.peakArenaUsedBytes.toLocaleString()} scratch bytes`,
    );
  } finally {
    kernels.dispose();
  }
}

function unpackMst(values: Float64Array): UndirectedWeightedEdge[] {
  if (values.length % 3 !== 0) {
    throw new RangeError("MST diagnostic must contain three values per edge");
  }
  const result: UndirectedWeightedEdge[] = [];
  for (let offset = 0; offset < values.length; offset += 3) {
    result.push({
      from: values[offset]!,
      to: values[offset + 1]!,
      weight: values[offset + 2]!,
    });
  }
  return result;
}

function sortedMstWeights(values: Float64Array): Float64Array {
  if (values.length % 3 !== 0) {
    throw new RangeError("MST diagnostic must contain three values per edge");
  }
  const result = new Float64Array(values.length / 3);
  for (let offset = 2; offset < values.length; offset += 3) {
    result[(offset - 2) / 3] = values[offset]!;
  }
  result.sort();
  return result;
}
