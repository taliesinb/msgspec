// A tight MessagePack reader/writer whose byte output is identical to
// `msgspec.msgpack`. It is deliberately small and targeted: the generated
// codecs from `msgspec.javascript.codec` drive it directly, so there is no
// general-purpose schema machinery here - just the primitives msgspec emits.
//
// Integers are encoded in the smallest form (matching msgspec). 64-bit integer
// types round-trip as native `bigint` (exact, no safe-integer narrowing);
// ordinary `int`/`float` stay as JS `number`.
//
// This module is the source of truth for the runtime that
// `msgspec.javascript.codec` inlines into each generated (dependency-free)
// codec module.

const _utf8enc = new TextEncoder();
const _utf8dec = new TextDecoder();

export class Writer {
  constructor() {
    this.b = new Uint8Array(64);
    this.dv = new DataView(this.b.buffer);
    this.n = 0;
  }

  _ensure(k) {
    if (this.n + k <= this.b.length) return;
    let s = this.b.length * 2;
    while (s < this.n + k) s *= 2;
    const nb = new Uint8Array(s);
    nb.set(this.b.subarray(0, this.n));
    this.b = nb;
    this.dv = new DataView(nb.buffer);
  }

  bytes() {
    return this.b.subarray(0, this.n);
  }

  u8(x) {
    this._ensure(1);
    this.b[this.n++] = x;
  }

  raw(u) {
    this._ensure(u.length);
    this.b.set(u, this.n);
    this.n += u.length;
  }

  nil() {
    this.u8(0xc0);
  }

  bool(x) {
    this.u8(x ? 0xc3 : 0xc2);
  }

  // A JS number, assumed to be an integer. Picks the smallest msgpack int form.
  int(x) {
    if (x >= 0) {
      if (x < 0x80) {
        this.u8(x);
      } else if (x < 0x100) {
        this._ensure(2);
        this.b[this.n++] = 0xcc;
        this.b[this.n++] = x;
      } else if (x < 0x10000) {
        this._ensure(3);
        this.b[this.n++] = 0xcd;
        this.dv.setUint16(this.n, x);
        this.n += 2;
      } else if (x < 0x100000000) {
        this._ensure(5);
        this.b[this.n++] = 0xce;
        this.dv.setUint32(this.n, x);
        this.n += 4;
      } else {
        this.int64(BigInt(x));
      }
    } else {
      if (x >= -0x20) {
        this._ensure(1);
        this.b[this.n++] = x & 0xff;
      } else if (x >= -0x80) {
        this._ensure(2);
        this.b[this.n++] = 0xd0;
        this.b[this.n++] = x & 0xff;
      } else if (x >= -0x8000) {
        this._ensure(3);
        this.b[this.n++] = 0xd1;
        this.dv.setInt16(this.n, x);
        this.n += 2;
      } else if (x >= -0x80000000) {
        this._ensure(5);
        this.b[this.n++] = 0xd2;
        this.dv.setInt32(this.n, x);
        this.n += 4;
      } else {
        this.int64(BigInt(x));
      }
    }
  }

  // A bigint. Picks the smallest msgpack int form by value (matching msgspec,
  // which is value-directed, not type-directed).
  int64(x) {
    if (x >= 0n) {
      if (x < 0x80n) {
        this.u8(Number(x));
      } else if (x < 0x100n) {
        this._ensure(2);
        this.b[this.n++] = 0xcc;
        this.b[this.n++] = Number(x);
      } else if (x < 0x10000n) {
        this._ensure(3);
        this.b[this.n++] = 0xcd;
        this.dv.setUint16(this.n, Number(x));
        this.n += 2;
      } else if (x < 0x100000000n) {
        this._ensure(5);
        this.b[this.n++] = 0xce;
        this.dv.setUint32(this.n, Number(x));
        this.n += 4;
      } else {
        this._ensure(9);
        this.b[this.n++] = 0xcf;
        this.dv.setBigUint64(this.n, x);
        this.n += 8;
      }
    } else {
      if (x >= -0x20n) {
        this._ensure(1);
        this.b[this.n++] = Number(x) & 0xff;
      } else if (x >= -0x80n) {
        this._ensure(2);
        this.b[this.n++] = 0xd0;
        this.b[this.n++] = Number(x) & 0xff;
      } else if (x >= -0x8000n) {
        this._ensure(3);
        this.b[this.n++] = 0xd1;
        this.dv.setInt16(this.n, Number(x));
        this.n += 2;
      } else if (x >= -0x80000000n) {
        this._ensure(5);
        this.b[this.n++] = 0xd2;
        this.dv.setInt32(this.n, Number(x));
        this.n += 4;
      } else {
        this._ensure(9);
        this.b[this.n++] = 0xd3;
        this.dv.setBigInt64(this.n, x);
        this.n += 8;
      }
    }
  }

  float(x) {
    this._ensure(9);
    this.b[this.n++] = 0xcb;
    this.dv.setFloat64(this.n, x);
    this.n += 8;
  }

  // A 5-byte float32 (for `Float32`-typed values).
  float32(x) {
    this._ensure(5);
    this.b[this.n++] = 0xca;
    this.dv.setFloat32(this.n, x);
    this.n += 4;
  }

  str(s) {
    const u = _utf8enc.encode(s);
    const len = u.length;
    if (len < 0x20) {
      this.u8(0xa0 | len);
    } else if (len < 0x100) {
      this._ensure(2);
      this.b[this.n++] = 0xd9;
      this.b[this.n++] = len;
    } else if (len < 0x10000) {
      this._ensure(3);
      this.b[this.n++] = 0xda;
      this.dv.setUint16(this.n, len);
      this.n += 2;
    } else {
      this._ensure(5);
      this.b[this.n++] = 0xdb;
      this.dv.setUint32(this.n, len);
      this.n += 4;
    }
    this.raw(u);
  }

  bin(u) {
    const len = u.length;
    if (len < 0x100) {
      this._ensure(2);
      this.b[this.n++] = 0xc4;
      this.b[this.n++] = len;
    } else if (len < 0x10000) {
      this._ensure(3);
      this.b[this.n++] = 0xc5;
      this.dv.setUint16(this.n, len);
      this.n += 2;
    } else {
      this._ensure(5);
      this.b[this.n++] = 0xc6;
      this.dv.setUint32(this.n, len);
      this.n += 4;
    }
    this.raw(u);
  }

  arrayHeader(n) {
    if (n < 0x10) {
      this.u8(0x90 | n);
    } else if (n < 0x10000) {
      this._ensure(3);
      this.b[this.n++] = 0xdc;
      this.dv.setUint16(this.n, n);
      this.n += 2;
    } else {
      this._ensure(5);
      this.b[this.n++] = 0xdd;
      this.dv.setUint32(this.n, n);
      this.n += 4;
    }
  }

  mapHeader(n) {
    if (n < 0x10) {
      this.u8(0x80 | n);
    } else if (n < 0x10000) {
      this._ensure(3);
      this.b[this.n++] = 0xde;
      this.dv.setUint16(this.n, n);
      this.n += 2;
    } else {
      this._ensure(5);
      this.b[this.n++] = 0xdf;
      this.dv.setUint32(this.n, n);
      this.n += 4;
    }
  }

  // A MessagePack extension: `code` is the (signed) ext type, `u` the payload.
  ext(code, u) {
    const len = u.length;
    if (len === 1) {
      this._ensure(2);
      this.b[this.n++] = 0xd4;
    } else if (len === 2) {
      this._ensure(2);
      this.b[this.n++] = 0xd5;
    } else if (len === 4) {
      this._ensure(2);
      this.b[this.n++] = 0xd6;
    } else if (len === 8) {
      this._ensure(2);
      this.b[this.n++] = 0xd7;
    } else if (len === 16) {
      this._ensure(2);
      this.b[this.n++] = 0xd8;
    } else if (len < 0x100) {
      this._ensure(3);
      this.b[this.n++] = 0xc7;
      this.b[this.n++] = len;
    } else if (len < 0x10000) {
      this._ensure(4);
      this.b[this.n++] = 0xc8;
      this.dv.setUint16(this.n, len);
      this.n += 2;
    } else {
      this._ensure(6);
      this.b[this.n++] = 0xc9;
      this.dv.setUint32(this.n, len);
      this.n += 4;
    }
    this.b[this.n++] = code & 0xff;
    this.raw(u);
  }

  // A msgspec.data tensor (a `TensorHandle`) as ext type 84 ('T'), body:
  // [version=1][dtype:u8][ndims:u8][shape...:i64-LE][packed little-endian data].
  tensor(h) {
    const spec = _TDTYPE[h.dtype];
    if (spec === undefined) throw new Error("msgpack: unknown tensor dtype " + h.dtype);
    const shape = h.shape;
    const nd = shape.length;
    const a = h.array;
    const raw = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    const body = new Uint8Array(3 + nd * 8 + raw.length);
    const dv = new DataView(body.buffer);
    body[0] = 1;
    body[1] = spec[0];
    body[2] = nd;
    let off = 3;
    // Shape dims are big-endian int64 (msgpack byte order); the packed data
    // that follows is native little-endian.
    for (let i = 0; i < nd; i++) { dv.setBigInt64(off, BigInt(shape[i]), false); off += 8; }
    body.set(raw, off);
    this.ext(84, body);
  }

  // Encode an arbitrary JS value (for `Any`-typed positions). Integers use the
  // smallest int form, bigints go through the 64-bit path, plain objects become
  // string-keyed maps.
  value(x) {
    if (x === null || x === undefined) this.nil();
    else if (typeof x === "boolean") this.bool(x);
    else if (typeof x === "bigint") this.int64(x);
    else if (typeof x === "number") {
      if (Number.isInteger(x)) this.int(x);
      else this.float(x);
    } else if (typeof x === "string") this.str(x);
    else if (x instanceof Uint8Array) this.bin(x);
    else if (Array.isArray(x)) {
      this.arrayHeader(x.length);
      for (const e of x) this.value(e);
    } else {
      const ks = Object.keys(x);
      this.mapHeader(ks.length);
      for (const k of ks) {
        this.str(k);
        this.value(x[k]);
      }
    }
  }
}

export class Reader {
  constructor(bytes) {
    this.b = bytes;
    this.dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    this.p = 0;
  }

  // Consume a nil if the next value is one; returns whether it did.
  tryNil() {
    if (this.b[this.p] === 0xc0) {
      this.p++;
      return true;
    }
    return false;
  }

  bool() {
    return this.b[this.p++] === 0xc3;
  }

  int() {
    const t = this.b[this.p++];
    if (t < 0x80) return t;
    if (t >= 0xe0) return t - 0x100;
    let v;
    switch (t) {
      case 0xcc: return this.b[this.p++];
      case 0xcd: v = this.dv.getUint16(this.p); this.p += 2; return v;
      case 0xce: v = this.dv.getUint32(this.p); this.p += 4; return v;
      case 0xcf: v = this.dv.getBigUint64(this.p); this.p += 8; return Number(v);
      case 0xd0: v = this.dv.getInt8(this.p); this.p += 1; return v;
      case 0xd1: v = this.dv.getInt16(this.p); this.p += 2; return v;
      case 0xd2: v = this.dv.getInt32(this.p); this.p += 4; return v;
      case 0xd3: v = this.dv.getBigInt64(this.p); this.p += 8; return Number(v);
    }
    throw new Error("msgpack: expected int, got tag 0x" + t.toString(16));
  }

  int64() {
    const t = this.b[this.p++];
    if (t < 0x80) return BigInt(t);
    if (t >= 0xe0) return BigInt(t - 0x100);
    let v;
    switch (t) {
      case 0xcc: return BigInt(this.b[this.p++]);
      case 0xcd: v = this.dv.getUint16(this.p); this.p += 2; return BigInt(v);
      case 0xce: v = this.dv.getUint32(this.p); this.p += 4; return BigInt(v);
      case 0xcf: v = this.dv.getBigUint64(this.p); this.p += 8; return v;
      case 0xd0: v = this.dv.getInt8(this.p); this.p += 1; return BigInt(v);
      case 0xd1: v = this.dv.getInt16(this.p); this.p += 2; return BigInt(v);
      case 0xd2: v = this.dv.getInt32(this.p); this.p += 4; return BigInt(v);
      case 0xd3: v = this.dv.getBigInt64(this.p); this.p += 8; return v;
    }
    throw new Error("msgpack: expected int, got tag 0x" + t.toString(16));
  }

  float() {
    const t = this.b[this.p];
    let v;
    if (t === 0xcb) {
      this.p++;
      v = this.dv.getFloat64(this.p);
      this.p += 8;
      return v;
    }
    if (t === 0xca) {
      this.p++;
      v = this.dv.getFloat32(this.p);
      this.p += 4;
      return v;
    }
    // msgspec accepts an integer on the wire where a float is expected.
    return this.int();
  }

  str() {
    const t = this.b[this.p++];
    let len;
    if (t >= 0xa0 && t <= 0xbf) len = t & 0x1f;
    else if (t === 0xd9) len = this.b[this.p++];
    else if (t === 0xda) { len = this.dv.getUint16(this.p); this.p += 2; }
    else if (t === 0xdb) { len = this.dv.getUint32(this.p); this.p += 4; }
    else throw new Error("msgpack: expected str, got tag 0x" + t.toString(16));
    const s = _utf8dec.decode(this.b.subarray(this.p, this.p + len));
    this.p += len;
    return s;
  }

  bin() {
    const t = this.b[this.p++];
    let len;
    if (t === 0xc4) len = this.b[this.p++];
    else if (t === 0xc5) { len = this.dv.getUint16(this.p); this.p += 2; }
    else if (t === 0xc6) { len = this.dv.getUint32(this.p); this.p += 4; }
    else throw new Error("msgpack: expected bin, got tag 0x" + t.toString(16));
    const v = this.b.slice(this.p, this.p + len);
    this.p += len;
    return v;
  }

  arrayHeader() {
    const t = this.b[this.p++];
    if (t >= 0x90 && t <= 0x9f) return t & 0x0f;
    if (t === 0xdc) { const v = this.dv.getUint16(this.p); this.p += 2; return v; }
    if (t === 0xdd) { const v = this.dv.getUint32(this.p); this.p += 4; return v; }
    throw new Error("msgpack: expected array, got tag 0x" + t.toString(16));
  }

  mapHeader() {
    const t = this.b[this.p++];
    if (t >= 0x80 && t <= 0x8f) return t & 0x0f;
    if (t === 0xde) { const v = this.dv.getUint16(this.p); this.p += 2; return v; }
    if (t === 0xdf) { const v = this.dv.getUint32(this.p); this.p += 4; return v; }
    throw new Error("msgpack: expected map, got tag 0x" + t.toString(16));
  }

  // Returns [code, Uint8Array].
  ext() {
    const t = this.b[this.p++];
    let len;
    if (t === 0xd4) len = 1;
    else if (t === 0xd5) len = 2;
    else if (t === 0xd6) len = 4;
    else if (t === 0xd7) len = 8;
    else if (t === 0xd8) len = 16;
    else if (t === 0xc7) len = this.b[this.p++];
    else if (t === 0xc8) { len = this.dv.getUint16(this.p); this.p += 2; }
    else if (t === 0xc9) { len = this.dv.getUint32(this.p); this.p += 4; }
    else throw new Error("msgpack: expected ext, got tag 0x" + t.toString(16));
    const code = this.dv.getInt8(this.p);
    this.p += 1;
    const data = this.b.slice(this.p, this.p + len);
    this.p += len;
    return [code, data];
  }

  // A msgspec.data tensor (ext type 84) -> a `TensorHandle`. The packed data is
  // copied into a fresh, element-aligned buffer so the typed-array view is valid
  // regardless of the tensor's offset within the message.
  tensor() {
    const data = this.ext()[1];
    const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const spec = _TCODE[data[1]];
    if (spec === undefined) throw new Error("msgpack: unknown tensor dtype code " + data[1]);
    const nd = data[2];
    const shape = new Array(nd);
    let off = 3;
    for (let i = 0; i < nd; i++) { shape[i] = Number(dv.getBigInt64(off, false)); off += 8; }
    const sub = data.slice(off);
    const Ctor = spec[1];
    const array = new Ctor(sub.buffer, sub.byteOffset, sub.byteLength / Ctor.BYTES_PER_ELEMENT);
    return new TensorHandle(array, spec[0], shape);
  }

  // Skip exactly one value (used for unknown struct fields).
  skip() {
    const t = this.b[this.p++];
    if (t < 0x80 || t >= 0xe0) return;
    if (t <= 0x8f) { let n = t & 0x0f; while (n-- > 0) { this.skip(); this.skip(); } return; }
    if (t <= 0x9f) { let n = t & 0x0f; while (n-- > 0) this.skip(); return; }
    if (t <= 0xbf) { this.p += t & 0x1f; return; }
    let n;
    switch (t) {
      case 0xc0: case 0xc2: case 0xc3: return;
      case 0xcc: case 0xd0: this.p += 1; return;
      case 0xcd: case 0xd1: this.p += 2; return;
      case 0xca: case 0xce: case 0xd2: this.p += 4; return;
      case 0xcb: case 0xcf: case 0xd3: this.p += 8; return;
      case 0xd9: n = this.b[this.p++]; this.p += n; return;
      case 0xda: n = this.dv.getUint16(this.p); this.p += 2 + n; return;
      case 0xdb: n = this.dv.getUint32(this.p); this.p += 4 + n; return;
      case 0xc4: n = this.b[this.p++]; this.p += n; return;
      case 0xc5: n = this.dv.getUint16(this.p); this.p += 2 + n; return;
      case 0xc6: n = this.dv.getUint32(this.p); this.p += 4 + n; return;
      case 0xdc: n = this.dv.getUint16(this.p); this.p += 2; while (n-- > 0) this.skip(); return;
      case 0xdd: n = this.dv.getUint32(this.p); this.p += 4; while (n-- > 0) this.skip(); return;
      case 0xde: n = this.dv.getUint16(this.p); this.p += 2; while (n-- > 0) { this.skip(); this.skip(); } return;
      case 0xdf: n = this.dv.getUint32(this.p); this.p += 4; while (n-- > 0) { this.skip(); this.skip(); } return;
      case 0xd4: this.p += 2; return;
      case 0xd5: this.p += 3; return;
      case 0xd6: this.p += 5; return;
      case 0xd7: this.p += 9; return;
      case 0xd8: this.p += 17; return;
      case 0xc7: n = this.b[this.p++]; this.p += 1 + n; return;
      case 0xc8: n = this.dv.getUint16(this.p); this.p += 2; this.p += 1 + n; return;
      case 0xc9: n = this.dv.getUint32(this.p); this.p += 4; this.p += 1 + n; return;
    }
    throw new Error("msgpack: bad tag 0x" + t.toString(16));
  }

  // Decode an arbitrary value (for `Any`-typed positions). 64-bit ints become
  // bigint; everything else maps to the natural JS type.
  value() {
    const t = this.b[this.p];
    if (t < 0x80 || t >= 0xe0) return this.int();
    if (t <= 0x8f || t === 0xde || t === 0xdf) {
      const n = this.mapHeader();
      const o = {};
      for (let i = 0; i < n; i++) {
        const k = this.value();
        o[k] = this.value();
      }
      return o;
    }
    if (t <= 0x9f || t === 0xdc || t === 0xdd) {
      const n = this.arrayHeader();
      const a = new Array(n);
      for (let i = 0; i < n; i++) a[i] = this.value();
      return a;
    }
    if (t <= 0xbf || t === 0xd9 || t === 0xda || t === 0xdb) return this.str();
    switch (t) {
      case 0xc0: this.p++; return null;
      case 0xc2: this.p++; return false;
      case 0xc3: this.p++; return true;
      case 0xca: case 0xcb: return this.float();
      case 0xcc: case 0xcd: case 0xce:
      case 0xd0: case 0xd1: case 0xd2: return this.int();
      case 0xcf: case 0xd3: return this.int64();
      case 0xc4: case 0xc5: case 0xc6: return this.bin();
      case 0xd4: case 0xd5: case 0xd6: case 0xd7: case 0xd8:
      case 0xc7: case 0xc8: case 0xc9: return this.ext();
    }
    throw new Error("msgpack: bad tag 0x" + t.toString(16));
  }
}

// --- Collection decode helpers (used by generated codecs) ------------------

export function decArray(r, fn) {
  const n = r.arrayHeader();
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = fn(r);
  return a;
}

export function decSet(r, fn) {
  const n = r.arrayHeader();
  const s = new Set();
  for (let i = 0; i < n; i++) s.add(fn(r));
  return s;
}

export function decMap(r, kfn, vfn) {
  const n = r.mapHeader();
  const o = {};
  for (let i = 0; i < n; i++) {
    const k = kfn(r);
    o[k] = vfn(r);
  }
  return o;
}

// --- JSON codec helpers ----------------------------------------------------
//
// The generated JSON codec is a structural transform over `JSON.parse` /
// `JSON.stringify`. These helpers handle the values JSON can't carry natively:
// `bytes` (base64), and 64-bit/generic integers (a hex string when they'd
// exceed the JS safe-integer range), byte-compatible with `msgspec.json`.

// --- Tensors (msgspec.data.Tensor) -----------------------------------------
// A packed n-dimensional array. dtype name -> [wire code, TypedArray ctor].
// `int64`/`uint64` use BigInt typed arrays; `bool` packs into a Uint8Array.
const _TDTYPE = {
  uint8: [0, Uint8Array],
  uint16: [1, Uint16Array],
  uint32: [2, Uint32Array],
  uint64: [3, BigUint64Array],
  int8: [4, Int8Array],
  int16: [5, Int16Array],
  int32: [6, Int32Array],
  int64: [7, BigInt64Array],
  float32: [8, Float32Array],
  float64: [9, Float64Array],
  bool: [10, Uint8Array],
};
const _TCODE = [];
for (const _k in _TDTYPE) _TCODE[_TDTYPE[_k][0]] = [_k, _TDTYPE[_k][1]];

// A decoded tensor: a TypedArray plus its dtype string and shape, mirroring
// `msgspec.data`'s TensorHandle. Construct one to hand a tensor to `encode`.
export class TensorHandle {
  constructor(array, dtype, shape) {
    this.type = "TensorHandle";
    this.array = array;
    this.dtype = dtype;
    this.shape = shape;
  }
}

// The raw little-endian bytes backing a TensorHandle's TypedArray.
function _tbytes(h) {
  const a = h.array;
  return new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
}

// A TensorHandle -> its JSON object form (data base64-encoded), byte-compatible
// with `msgspec.json`'s tensor encoding.
export function encTensorJSON(h) {
  return { type: "tensor", shape: h.shape, dtype: h.dtype, data: b64encode(_tbytes(h)) };
}

// The inverse: a parsed JSON tensor object -> TensorHandle.
export function decTensorJSON(o) {
  const spec = _TDTYPE[o.dtype];
  if (spec === undefined) throw new Error("msgpack: unknown tensor dtype " + o.dtype);
  const raw = b64decode(o.data);
  const Ctor = spec[1];
  const array = new Ctor(raw.buffer, raw.byteOffset, raw.byteLength / Ctor.BYTES_PER_ELEMENT);
  return new TensorHandle(array, o.dtype, o.shape);
}

const _B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
let _B64R = null;

// A Uint8Array -> a standard (padded) base64 string.
export function b64encode(u8) {
  let s = "";
  const n = u8.length;
  for (let i = 0; i < n; i += 3) {
    const a = u8[i];
    const b = i + 1 < n ? u8[i + 1] : 0;
    const c = i + 2 < n ? u8[i + 2] : 0;
    s += _B64[a >> 2];
    s += _B64[((a & 3) << 4) | (b >> 4)];
    s += i + 1 < n ? _B64[((b & 15) << 2) | (c >> 6)] : "=";
    s += i + 2 < n ? _B64[c & 63] : "=";
  }
  return s;
}

// A base64 string -> Uint8Array.
export function b64decode(str) {
  if (_B64R === null) {
    _B64R = {};
    for (let i = 0; i < _B64.length; i++) _B64R[_B64[i]] = i;
  }
  let len = str.length;
  while (len > 0 && str[len - 1] === "=") len--;
  const out = new Uint8Array((len * 3) >> 2);
  let bits = 0, nbits = 0, oi = 0;
  for (let i = 0; i < len; i++) {
    bits = (bits << 6) | _B64R[str[i]];
    nbits += 6;
    if (nbits >= 8) {
      nbits -= 8;
      out[oi++] = (bits >> nbits) & 0xff;
    }
  }
  return out;
}

const _SAFE = 9007199254740991n; // 2**53 - 1

// A bigint -> a JSON value: a number when small enough, else a hex string.
// `signed` adds an explicit +/- sign; `force` always emits the hex string
// (for the ambiguous `Int | Float` Scalar union).
export function encHexInt(v, signed, force) {
  const neg = v < 0n;
  const mag = neg ? -v : v;
  if (!force && mag <= _SAFE) return Number(v);
  return (signed ? (neg ? "-" : "+") : "") + "0x" + mag.toString(16);
}

// The inverse: a JSON value (number or hex string) -> bigint.
export function decHexInt(x) {
  if (typeof x === "string") {
    const neg = x[0] === "-";
    const body = x[0] === "+" || x[0] === "-" ? x.slice(1) : x;
    const mag = BigInt(body);
    return neg ? -mag : mag;
  }
  return BigInt(x);
}
