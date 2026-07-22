interface ChromiumPerformanceMemory {
  readonly usedJSHeapSize: number;
  readonly totalJSHeapSize: number;
}

interface ByobReadObservation {
  readonly index: number;
  readonly requestedCapacity: number;
  readonly returnedLength: number;
  readonly returnedCapacity: number;
  readonly passedViewLengthAfterRead: number;
  readonly passedBufferLengthAfterRead: number;
  readonly sameArrayBufferObject: boolean;
  readonly returnedByteOffset: number;
}

const inputElement = document.querySelector<HTMLInputElement>("#audio");
const outputElement = document.querySelector<HTMLPreElement>("#output");
if (inputElement === null || outputElement === null) throw new Error("Missing probe UI");
const input = inputElement;
const output = outputElement;

function heap(): ChromiumPerformanceMemory | undefined {
  return (
    performance as Performance & { readonly memory?: ChromiumPerformanceMemory }
  ).memory;
}

input.addEventListener("change", () => {
  const file = input.files?.[0];
  if (file === undefined) return;
  input.disabled = true;
  void probe(file).catch((error: unknown) => {
    output.textContent = `ERROR ${error instanceof Error ? error.stack : String(error)}`;
    throw error;
  });
});

async function probe(file: File): Promise<void> {
  const started = performance.now();
  const initialHeap = heap();
  const reader = file.stream().getReader({ mode: "byob" });
  let reusable = new Uint8Array(320_000);
  let bytesRead = 0;
  let reads = 0;
  let peakUsedJsHeap = initialHeap?.usedJSHeapSize ?? 0;
  let allPassedViewsDetached = true;
  let allReturnedBuffersReusable = true;
  let minimumReturnedCapacity = Number.POSITIVE_INFINITY;
  let maximumReturnedCapacity = 0;
  const observations: ByobReadObservation[] = [];

  while (bytesRead < file.size) {
    const passedView = reusable;
    const passedBuffer = passedView.buffer;
    const requestedCapacity = passedView.byteLength;
    const result = await reader.read(passedView);
    const value = result.value;
    if (result.done || value === undefined || value.byteLength === 0) {
      throw new Error(`BYOB ended after ${bytesRead}/${file.size} bytes`);
    }
    const observation = {
      index: reads,
      requestedCapacity,
      returnedLength: value.byteLength,
      returnedCapacity: value.buffer.byteLength,
      passedViewLengthAfterRead: passedView.byteLength,
      passedBufferLengthAfterRead: passedBuffer.byteLength,
      sameArrayBufferObject: value.buffer === passedBuffer,
      returnedByteOffset: value.byteOffset,
    } satisfies ByobReadObservation;
    if (reads < 8 || bytesRead + value.byteLength >= file.size) {
      observations.push(observation);
    }
    allPassedViewsDetached &&=
      observation.passedViewLengthAfterRead === 0 &&
      observation.passedBufferLengthAfterRead === 0;
    allReturnedBuffersReusable &&=
      observation.returnedCapacity > 0 && observation.returnedByteOffset === 0;
    minimumReturnedCapacity = Math.min(
      minimumReturnedCapacity,
      observation.returnedCapacity,
    );
    maximumReturnedCapacity = Math.max(
      maximumReturnedCapacity,
      observation.returnedCapacity,
    );
    bytesRead += value.byteLength;
    reads += 1;
    reusable = new Uint8Array(value.buffer);
    peakUsedJsHeap = Math.max(peakUsedJsHeap, heap()?.usedJSHeapSize ?? 0);
  }
  await reader.cancel();
  reader.releaseLock();
  const finalHeap = heap();
  const result = {
    fileBytes: file.size,
    elapsedMs: performance.now() - started,
    reads,
    bytesRead,
    initialScratchBytes: 320_000,
    finalScratchBytes: reusable.byteLength,
    allPassedViewsDetached,
    allReturnedBuffersReusable,
    minimumReturnedCapacity,
    maximumReturnedCapacity,
    observations,
    heap: {
      initialUsedJsHeap: initialHeap?.usedJSHeapSize,
      peakUsedJsHeap,
      finalUsedJsHeap: finalHeap?.usedJSHeapSize,
      initialTotalJsHeap: initialHeap?.totalJSHeapSize,
      finalTotalJsHeap: finalHeap?.totalJSHeapSize,
    },
  };
  Object.assign(globalThis, { __senkoByobProbe: result });
  output.textContent = JSON.stringify(result, null, 2);
}
