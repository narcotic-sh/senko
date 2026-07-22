import { describe, expect, it } from "vitest";

import { IncrementalSha256 } from "./sha256";

const encoder = new TextEncoder();

describe("IncrementalSha256", () => {
  it("matches standard SHA-256 vectors", () => {
    expect(IncrementalSha256.hex(new Uint8Array())).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    expect(IncrementalSha256.hex(encoder.encode("abc"))).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    expect(
      IncrementalSha256.hex(
        encoder.encode("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
      ),
    ).toBe("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  });

  it("is independent of stream chunk boundaries", () => {
    const bytes = new Uint8Array(100_003);
    for (let index = 0; index < bytes.length; index += 1) bytes[index] = index * 37 + 11;
    const expected = IncrementalSha256.hex(bytes);
    const digest = new IncrementalSha256();
    let offset = 0;
    const chunks = [1, 2, 63, 4, 129, 7, 4097];
    let chunkIndex = 0;
    while (offset < bytes.length) {
      const length = Math.min(chunks[chunkIndex % chunks.length]!, bytes.length - offset);
      digest.update(bytes.subarray(offset, offset + length));
      offset += length;
      chunkIndex += 1;
    }
    expect(digest.digestHex()).toBe(expected);
  });
});
