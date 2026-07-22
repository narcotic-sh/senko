export type CampPlusStorageDtype = "float16" | "float32";

export function campPlusStorageBytes(dtype: CampPlusStorageDtype): 2 | 4 {
  return dtype === "float16" ? 2 : 4;
}

/**
 * The FP32 fallback intentionally reuses the exact production shader
 * schedules. Widening scalar storage and explicit rounding conversions is a
 * mechanical WGSL transform, which keeps the FP16 source byte-for-byte
 * unchanged while guaranteeing that the fallback contains no optional f16
 * language feature.
 */
export function campPlusStorageWgsl(
  source: string,
  dtype: CampPlusStorageDtype,
): string {
  if (dtype === "float16") return source;
  const widened = source
    .replace(/^\s*enable\s+f16;\s*/m, "")
    .replace(/\bf16\b/g, "f32");
  if (/\bf16\b|enable\s+f16\s*;/.test(widened)) {
    throw new Error("CAM++ FP32 WGSL still contains shader-f16 syntax");
  }
  return widened;
}
