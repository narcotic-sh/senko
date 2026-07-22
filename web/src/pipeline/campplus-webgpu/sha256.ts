const INITIAL_STATE = new Uint32Array([
  0x6a09e667,
  0xbb67ae85,
  0x3c6ef372,
  0xa54ff53a,
  0x510e527f,
  0x9b05688c,
  0x1f83d9ab,
  0x5be0cd19,
]);

const ROUND_CONSTANTS = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const BLOCK_BYTES = 64;

/** Incremental SHA-256 with a fixed 320-byte working set. */
export class IncrementalSha256 {
  private readonly state = INITIAL_STATE.slice();
  private readonly pending = new Uint8Array(BLOCK_BYTES);
  private readonly schedule = new Uint32Array(64);
  private pendingBytes = 0;
  private totalBytes = 0;
  private finished = false;

  update(data: Uint8Array): this {
    if (this.finished) throw new Error("SHA-256 digest has already been finalized");
    if (data.byteLength === 0) return this;
    this.totalBytes += data.byteLength;
    if (!Number.isSafeInteger(this.totalBytes)) {
      throw new RangeError("SHA-256 input is too large for exact JavaScript length accounting");
    }

    let offset = 0;
    if (this.pendingBytes > 0) {
      const copied = Math.min(BLOCK_BYTES - this.pendingBytes, data.byteLength);
      this.pending.set(data.subarray(0, copied), this.pendingBytes);
      this.pendingBytes += copied;
      offset += copied;
      if (this.pendingBytes === BLOCK_BYTES) {
        this.compress(this.pending, 0);
        this.pendingBytes = 0;
      }
    }

    while (offset + BLOCK_BYTES <= data.byteLength) {
      this.compress(data, offset);
      offset += BLOCK_BYTES;
    }
    if (offset < data.byteLength) {
      const trailing = data.subarray(offset);
      this.pending.set(trailing, 0);
      this.pendingBytes = trailing.byteLength;
    }
    return this;
  }

  digest(): Uint8Array {
    if (this.finished) throw new Error("SHA-256 digest has already been finalized");
    this.finished = true;

    this.pending[this.pendingBytes] = 0x80;
    this.pending.fill(0, this.pendingBytes + 1);
    if (this.pendingBytes >= 56) {
      this.compress(this.pending, 0);
      this.pending.fill(0);
    }

    const bitLength = this.totalBytes * 8;
    const high = Math.floor(bitLength / 0x1_0000_0000);
    const low = bitLength >>> 0;
    const view = new DataView(this.pending.buffer);
    view.setUint32(56, high, false);
    view.setUint32(60, low, false);
    this.compress(this.pending, 0);

    const result = new Uint8Array(32);
    const resultView = new DataView(result.buffer);
    for (let index = 0; index < this.state.length; index += 1) {
      resultView.setUint32(index * 4, this.state[index]!, false);
    }
    return result;
  }

  digestHex(): string {
    return bytesToHex(this.digest());
  }

  static hex(data: Uint8Array): string {
    return new IncrementalSha256().update(data).digestHex();
  }

  private compress(bytes: Uint8Array, offset: number): void {
    const schedule = this.schedule;
    for (let index = 0; index < 16; index += 1) {
      const start = offset + index * 4;
      schedule[index] =
        ((bytes[start]! << 24) |
          (bytes[start + 1]! << 16) |
          (bytes[start + 2]! << 8) |
          bytes[start + 3]!) >>>
        0;
    }
    for (let index = 16; index < 64; index += 1) {
      const previous15 = schedule[index - 15]!;
      const previous2 = schedule[index - 2]!;
      const sigma0 =
        rotateRight(previous15, 7) ^ rotateRight(previous15, 18) ^ (previous15 >>> 3);
      const sigma1 =
        rotateRight(previous2, 17) ^ rotateRight(previous2, 19) ^ (previous2 >>> 10);
      schedule[index] =
        (schedule[index - 16]! + sigma0 + schedule[index - 7]! + sigma1) >>> 0;
    }

    let a = this.state[0]!;
    let b = this.state[1]!;
    let c = this.state[2]!;
    let d = this.state[3]!;
    let e = this.state[4]!;
    let f = this.state[5]!;
    let g = this.state[6]!;
    let h = this.state[7]!;
    for (let index = 0; index < 64; index += 1) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choice = (e & f) ^ (~e & g);
      const temporary1 =
        (h + sum1 + choice + ROUND_CONSTANTS[index]! + schedule[index]!) >>> 0;
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const temporary2 = (sum0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temporary1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temporary1 + temporary2) >>> 0;
    }

    this.state[0] = (this.state[0]! + a) >>> 0;
    this.state[1] = (this.state[1]! + b) >>> 0;
    this.state[2] = (this.state[2]! + c) >>> 0;
    this.state[3] = (this.state[3]! + d) >>> 0;
    this.state[4] = (this.state[4]! + e) >>> 0;
    this.state[5] = (this.state[5]! + f) >>> 0;
    this.state[6] = (this.state[6]! + g) >>> 0;
    this.state[7] = (this.state[7]! + h) >>> 0;
  }
}

export function bytesToHex(bytes: Uint8Array): string {
  let result = "";
  for (const value of bytes) result += value.toString(16).padStart(2, "0");
  return result;
}

function rotateRight(value: number, bits: number): number {
  return ((value >>> bits) | (value << (32 - bits))) >>> 0;
}
