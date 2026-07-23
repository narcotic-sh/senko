import {
  clusterEmbeddingsNativeThreaded,
  type NativeUmapThreadedResult,
} from "./native-umap";
import {
  ThreadedUmapLayoutPool,
  type ThreadedUmapLayoutPoolOptions,
} from "./threaded-umap-layout";
import { WasmClusteringKernels } from "./wasm-kernels";

export interface BrowserClusteringMemoryStats {
  readonly heapBytes: number;
  readonly ordinaryWasmHeapBytes: number;
  readonly activeSharedLayoutBytes: number;
  readonly peakSharedLayoutBytes: number;
}

export interface BrowserClusteringResourceOptions {
  readonly layout?: ThreadedUmapLayoutPoolOptions;
}

/** Owns every reusable CPU clustering resource in the pipeline worker. */
export class BrowserClusteringResources {
  readonly kernels: WasmClusteringKernels;
  readonly layout: ThreadedUmapLayoutPool;

  private disposed = false;

  private constructor(
    kernels: WasmClusteringKernels,
    layout: ThreadedUmapLayoutPool,
  ) {
    this.kernels = kernels;
    this.layout = layout;
  }

  static async create(
    options: BrowserClusteringResourceOptions = {},
  ): Promise<BrowserClusteringResources> {
    const [kernelsResult, layoutResult] = await Promise.allSettled([
      WasmClusteringKernels.create(),
      ThreadedUmapLayoutPool.create(options.layout),
    ] as const);
    if (
      kernelsResult.status === "rejected" ||
      layoutResult.status === "rejected"
    ) {
      if (kernelsResult.status === "fulfilled") {
        kernelsResult.value.dispose();
      }
      if (layoutResult.status === "fulfilled") {
        layoutResult.value.dispose();
      }
      throw kernelsResult.status === "rejected"
        ? kernelsResult.reason
        : layoutResult.status === "rejected"
          ? layoutResult.reason
          : new Error("Failed to create browser clustering resources");
    }
    return new BrowserClusteringResources(
      kernelsResult.value,
      layoutResult.value,
    );
  }

  get memoryStats(): BrowserClusteringMemoryStats {
    const ordinary = this.kernels.memoryStats.heapBytes;
    const shared = this.layout.memoryStats;
    return {
      // The ordinary arena remains resident while layout runs, so summing its
      // current heap with layout's run high-water is a true concurrent peak.
      heapBytes: ordinary + shared.peakSharedBytes,
      ordinaryWasmHeapBytes: ordinary,
      activeSharedLayoutBytes: shared.activeSharedBytes,
      peakSharedLayoutBytes: shared.peakSharedBytes,
    };
  }

  async warmup(): Promise<void> {
    this.assertUsable();
    this.kernels.warmup();
    await this.layout.warmup();
  }

  resetTransientMemoryStats(): void {
    this.assertUsable();
    this.layout.resetMemoryStats();
  }

  async clusterNativeUmap(
    embeddings: Float32Array,
    count: number,
    dimension: number,
    signal?: AbortSignal,
  ): Promise<NativeUmapThreadedResult> {
    this.assertUsable();
    const randomSeed = randomUint32();
    return clusterEmbeddingsNativeThreaded(
      embeddings,
      count,
      dimension,
      randomSeed,
      this.kernels,
      this.layout,
      signal,
    );
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.layout.dispose();
    this.kernels.dispose();
  }

  private assertUsable(): void {
    if (this.disposed) {
      throw new Error("Browser clustering resources are disposed");
    }
  }
}

function randomUint32(): number {
  if (globalThis.crypto === undefined) {
    throw new Error("Native UMAP clustering requires a secure random source");
  }
  return globalThis.crypto.getRandomValues(new Uint32Array(1))[0]!;
}
