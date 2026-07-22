import { IncrementalSha256 } from "./sha256";

const SOURCE_SHA256 = "33".repeat(32);

export interface SyntheticCampPlusFixture {
  readonly binary: Uint8Array;
  readonly metadata: Record<string, unknown>;
}

export function makeSyntheticCampPlusFixture(): SyntheticCampPlusFixture {
  const binary = new Uint8Array(1024);
  for (let index = 256; index < binary.length; index += 1) {
    binary[index] = index * 29 + 7;
  }
  const payloadSha256 = IncrementalSha256.hex(binary.subarray(256));
  const header = new DataView(binary.buffer);
  binary.set(new TextEncoder().encode("SNKCAMW1"), 0);
  header.setUint32(8, 1, true);
  header.setUint32(12, 256, true);
  header.setUint32(16, 256, true);
  header.setUint32(20, 3, true);
  header.setBigUint64(24, 1024n, true);
  binary.set(hexToBytes(SOURCE_SHA256), 32);
  binary.set(hexToBytes(payloadSha256), 64);
  header.setUint32(96, 32, true);
  header.setUint32(100, 150, true);
  header.setUint32(104, 80, true);
  header.setUint32(108, 192, true);
  const sha256 = IncrementalSha256.hex(binary);

  const convolution = { weight: "test-weight", bias: "test-bias" };
  const metadata: Record<string, unknown> = {
    schema: "senko.campplus.webgpu-pack",
    format_version: 1,
    source: {
      file: "source.onnx",
      byte_length: 2048,
      sha256: SOURCE_SHA256,
      opset: [{ domain: "", version: 17 }],
    },
    binary: {
      file: "test.bin",
      byte_length: 1024,
      sha256,
      payload_sha256: payloadSha256,
      header_bytes: 256,
      section_alignment: 256,
      section_count: 3,
      endianness: "little",
    },
    contract: {
      input: { name: "features", shape: [32, 150, 80], dtype: "float32" },
      output: { name: "embeddings", shape: [32, 192], dtype: "float32" },
      internal_dtype: "float16",
      required_webgpu_features: ["shader-f16"],
      channel_tile: 4,
      weights_are_batch_independent: true,
    },
    inventory: {},
    compute: {},
    memory: {
      onnx_reference: {},
      planned_webgpu: {
        recommended: {
          frontend_microbatch: 32,
          activation_arena_bytes: 256,
          weight_buffer_bytes: 1024,
          minimum_resident_gpu_bytes: 1280,
        },
        frontend_microbatch_tradeoffs: [
          {
            frontend_microbatch: 32,
            activation_arena_bytes: 256,
            minimum_resident_gpu_bytes: 1280,
            frontend_tdnn_dispatches: 11,
          },
        ],
      },
    },
    sections: [
      {
        id: "test-weight",
        kind: "conv_weight",
        byte_offset: 256,
        byte_length: 32,
        element_count: 16,
        dtype: "float16",
        logical_shape: [4, 4, 1],
        packed_shape: [1, 1, 1, 4, 4],
        layout: "K_O4_I4_I_O",
      },
      {
        id: "test-bias",
        kind: "conv_bias",
        byte_offset: 512,
        byte_length: 8,
        element_count: 4,
        dtype: "float16",
        logical_shape: [4],
        packed_shape: [1, 4],
        layout: "O4",
      },
      {
        id: "test-affine",
        kind: "batch_norm_affine",
        byte_offset: 768,
        byte_length: 32,
        element_count: 8,
        dtype: "float32",
        logical_shape: [4, 2],
        packed_shape: [1, 2, 4],
        layout: "C4_SCALE_SHIFT",
      },
    ],
    fused_program: {
      head: { convolutions: [convolution] },
      tdnn: { convolution },
      blocks: [],
      transits: [],
      final: { dense: convolution, output_affine: "test-affine" },
    },
  };
  return { binary, metadata };
}

function hexToBytes(hex: string): Uint8Array {
  const bytes = new Uint8Array(hex.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}
