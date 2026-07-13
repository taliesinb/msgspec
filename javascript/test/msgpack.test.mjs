import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Writer, Reader } from "../src/msgpack.mjs";

const fixtures = JSON.parse(
  readFileSync(new URL("./fixtures.json", import.meta.url)),
);

const hex = (u) => Buffer.from(u).toString("hex");

// Each fixture is [name, expected-hex, value, kind]. `kind` picks the Writer
// method so we exercise int vs int64 vs float explicitly.
const ENC = [
  ["int_0", 0, "int"],
  ["int_127", 127, "int"],
  ["int_128", 128, "int"],
  ["int_255", 255, "int"],
  ["int_256", 256, "int"],
  ["int_65535", 65535, "int"],
  ["int_65536", 65536, "int"],
  ["int_2p32m1", 2 ** 32 - 1, "int"],
  ["int_2p32", 2 ** 32, "int"],
  ["int_2p63m1", 2n ** 63n - 1n, "int64"],
  ["int_2p64m1", 2n ** 64n - 1n, "int64"],
  ["int_neg1", -1, "int"],
  ["int_neg32", -32, "int"],
  ["int_neg33", -33, "int"],
  ["int_neg128", -128, "int"],
  ["int_neg129", -129, "int"],
  ["int_neg32768", -32768, "int"],
  ["int_neg32769", -32769, "int"],
  ["int_neg2p31", -(2 ** 31), "int"],
  ["int_neg2p31m1", -(2 ** 31) - 1, "int"],
  ["int_neg2p63", -(2n ** 63n), "int64"],
  ["float_1p5", 1.5, "float"],
  ["float_0", 0.0, "float"],
  ["true", true, "bool"],
  ["false", false, "bool"],
  ["none", null, "nil"],
  ["str_empty", "", "str"],
  ["str_hi", "hi", "str"],
  ["str_31", "a".repeat(31), "str"],
  ["str_32", "a".repeat(32), "str"],
  ["str_256", "a".repeat(256), "str"],
  ["str_unicode", "héllo→", "str"],
];

test("writer bytes match msgspec fixtures", () => {
  for (const [name, value, kind] of ENC) {
    const w = new Writer();
    switch (kind) {
      case "int": w.int(value); break;
      case "int64": w.int64(value); break;
      case "float": w.float(value); break;
      case "bool": w.bool(value); break;
      case "nil": w.nil(); break;
      case "str": w.str(value); break;
    }
    assert.equal(hex(w.bytes()), fixtures[name], `encode ${name}`);
  }
});

test("bytes / list / dict encode match msgspec", () => {
  let w = new Writer();
  w.bin(new Uint8Array([0, 1, 2]));
  assert.equal(hex(w.bytes()), fixtures.bytes_3, "bin");

  w = new Writer();
  w.arrayHeader(3);
  w.int(1); w.int(2); w.int(3);
  assert.equal(hex(w.bytes()), fixtures.list_123, "list_123");

  w = new Writer();
  w.arrayHeader(0);
  assert.equal(hex(w.bytes()), fixtures.list_empty, "list_empty");

  w = new Writer();
  w.arrayHeader(16);
  for (let i = 0; i < 16; i++) w.int(i);
  assert.equal(hex(w.bytes()), fixtures.list_16, "list_16 (array16 header)");

  w = new Writer();
  w.mapHeader(2);
  w.str("a"); w.int(1); w.str("b"); w.int(2);
  assert.equal(hex(w.bytes()), fixtures.dict_ab, "dict_ab");
});

test("reader round-trips its own output", () => {
  const vals = [0, 127, 128, 65536, -1, -33, -70000, 1.5, true, false, null, "hi", "a".repeat(300)];
  for (const v of vals) {
    const w = new Writer();
    if (v === null) w.nil();
    else if (typeof v === "boolean") w.bool(v);
    else if (typeof v === "string") w.str(v);
    else if (Number.isInteger(v)) w.int(v);
    else w.float(v);
    const r = new Reader(w.bytes());
    assert.deepEqual(r.value(), v, `roundtrip ${v}`);
  }
});

test("reader decodes 64-bit ints to exact bigint", () => {
  for (const big of [2n ** 63n - 1n, 2n ** 64n - 1n, -(2n ** 63n)]) {
    const w = new Writer();
    w.int64(big);
    const r = new Reader(w.bytes());
    assert.equal(r.int64(), big, `bigint ${big}`);
  }
});

test("reader decodes msgspec-encoded nested structure", () => {
  // fixtures.nested = encode({"xs":[1,2,{"y":true}],"s":"hi"})
  const bytes = Uint8Array.from(Buffer.from(fixtures.nested, "hex"));
  const r = new Reader(bytes);
  assert.deepEqual(r.value(), { xs: [1, 2, { y: true }], s: "hi" });
});
