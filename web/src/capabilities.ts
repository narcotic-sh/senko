export interface WebGpuRuntimeCapabilities {
  readonly available: boolean;
  readonly adapterInfo?: {
    readonly architecture: string;
    readonly description: string;
    readonly device: string;
    readonly vendor: string;
  };
  readonly features: readonly string[];
  readonly maxBufferSize?: number;
  readonly maxStorageBufferBindingSize?: number;
}

export interface RuntimeCapabilities {
  readonly secureContext: boolean;
  readonly crossOriginIsolated: boolean;
  readonly dedicatedWorker: boolean;
  readonly sharedArrayBuffer: boolean;
  readonly wasm: boolean;
  readonly wasmSimd: boolean;
  readonly wasmThreads: boolean;
  readonly webgpu: WebGpuRuntimeCapabilities;
}

export interface CapabilityAssessment {
  readonly canRun: boolean;
  readonly errors: readonly string[];
  readonly warnings: readonly string[];
}

const WASM_SIMD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x04, 0x01, 0x60,
  0x00, 0x00, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0x41,
  0x00, 0xfd, 0x0f, 0x1a, 0x0b,
]);

function detectWasmThreads(): boolean {
  if (
    !globalThis.crossOriginIsolated ||
    typeof globalThis.SharedArrayBuffer === "undefined" ||
    typeof globalThis.WebAssembly === "undefined"
  ) {
    return false;
  }

  try {
    const memory = new WebAssembly.Memory({
      initial: 1,
      maximum: 1,
      shared: true,
    });
    return memory.buffer instanceof SharedArrayBuffer;
  } catch {
    return false;
  }
}

function adapterInfo(
  adapter: GPUAdapter,
): NonNullable<WebGpuRuntimeCapabilities["adapterInfo"]> {
  return {
    architecture: adapter.info.architecture,
    description: adapter.info.description,
    device: adapter.info.device,
    vendor: adapter.info.vendor,
  };
}

async function detectWebGpu(): Promise<WebGpuRuntimeCapabilities> {
  if (navigator.gpu === undefined) {
    return { available: false, features: [] };
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) {
    return { available: false, features: [] };
  }

  return {
    available: true,
    adapterInfo: adapterInfo(adapter),
    features: [...adapter.features].sort(),
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
}

export async function detectRuntimeCapabilities(): Promise<RuntimeCapabilities> {
  const wasm = typeof globalThis.WebAssembly !== "undefined";
  return {
    secureContext: globalThis.isSecureContext,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    dedicatedWorker: typeof globalThis.Worker !== "undefined",
    sharedArrayBuffer: typeof globalThis.SharedArrayBuffer !== "undefined",
    wasm,
    wasmSimd: wasm && WebAssembly.validate(WASM_SIMD_PROBE),
    wasmThreads: detectWasmThreads(),
    webgpu: await detectWebGpu(),
  };
}

export function assessRuntimeCapabilities(
  capabilities: RuntimeCapabilities,
): CapabilityAssessment {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!capabilities.secureContext) {
    errors.push("WebGPU requires a secure context (HTTPS or localhost).");
  }
  if (!capabilities.dedicatedWorker) {
    errors.push("Dedicated module workers are unavailable.");
  }
  if (!capabilities.wasm) {
    errors.push("WebAssembly is unavailable.");
  }
  if (!capabilities.webgpu.available) {
    errors.push("No WebGPU adapter is available.");
  }
  if (!capabilities.crossOriginIsolated || !capabilities.wasmThreads) {
    errors.push(
      "Native clustering requires cross-origin isolation and shared WASM memory.",
    );
  }
  if (!capabilities.wasmSimd) {
    errors.push("Native clustering requires WASM SIMD.");
  }
  if (!capabilities.webgpu.features.includes("shader-f16")) {
    warnings.push("shader-f16 is unavailable; neural stages will use FP32 kernels.");
  }

  return { canRun: errors.length === 0, errors, warnings };
}
