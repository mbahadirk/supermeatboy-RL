// Execute the native adapter with fake memory: observed enemies, laser phases, shots and frame messages.
// No game process or real memory is accessed. This does not validate executable offsets against a live game.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const heap = new Map();
class Mem {
  constructor(address) { this.address = Number(address); }
  add(n) { return new Mem(this.address + Number(n)); }
  sub(n) { return new Mem(this.address - Number(n)); }
  isNull() { return this.address === 0; }
  equals(other) { return this.address === other.address; }
  toString() { return '0x' + this.address.toString(16); }
  toInt32() { return this.address | 0; }
  readFloat() { return heap.get(this.address) || 0; }
  readDouble() { return this.readFloat(); }
  readU32() { return this.readFloat(); }
  readU16() { return this.readFloat(); }
  readU8() { return this.readFloat(); }
  readS32() { return this.readFloat(); }
  readPointer() { return new Mem(this.readFloat()); }
  readByteArray(n) { return new ArrayBuffer(n); }
  writeFloat(v) { heap.set(this.address, v); return this; }
  writeDouble(v) { return this.writeFloat(v); }
  writeU32(v) { return this.writeFloat(v); }
  writeU16(v) { return this.writeFloat(v); }
  writeU8(v) { return this.writeFloat(v); }
  writePointer(v) { return this.writeFloat(v.address); }
  writeByteArray() { return this; }
}
let allocated = 0x1000000;
const alloc = n => { const p = new Mem(allocated); allocated += Number(n) || 256; return p; };
const sent = [];
const context = vm.createContext({
  Mem, assert, Date, Math, ArrayBuffer, Uint8Array, Uint32Array, Float32Array,
  Process: { id: 1, getCurrentThreadId: () => 1,
    getModuleByName: () => ({base: new Mem(0x400000), findExportByName: () => new Mem(0x1000)}) },
  Memory: {alloc, allocUtf16String: () => alloc(256), copy() {}},
  Interceptor: {attach() { return {detach() {}}; }, replace() {}},
  NativeFunction: function () { return () => new Mem(0); },
  NativeCallback: function (fn) { return fn; },
  ptr: n => new Mem(n), rpc: {}, setInterval() {},
  send: m => sent.push(JSON.parse(JSON.stringify(m))),
  recv: (type, fn) => ({wait() { if (type === 'act') fn({payload: {a: [], leader: 0}}); }}),
});
const source = fs.readFileSync(path.join(__dirname, '..', 'realgame.js'), 'utf8');
vm.runInContext(source.replace('__CONFIG__', JSON.stringify({agents: 2, pool: 40, wave_frames: 1200,
                                                          finish_frames: 60, finish_quota: 1})), context);
vm.runInContext(`
  const ea = new Mem(0x200000);
  ea.add(0x13c).writeU32(1); ea.add(0x140).writeFloat(8);
  ea.add(0x144).writeFloat(20); ea.add(0x148).writeFloat(30);
  enemies = [{a: ea, kind: 0}];
  assert.equal(enemyRows()[0][7][0][0], 1);
  assert.equal(enemyRows()[0][7][0][3], 8);
  ea.add(0x1d0).writeU32(5);
  assert.equal(enemyRows()[0][7].length, 0);
  const la = new Mem(0x210000);
  la.add(0x14).writeFloat(-40); la.add(0x18).writeFloat(10);
  la.add(0x1c).writeFloat(80); la.add(0x20).writeFloat(60);
  la.add(8).writeU32(1);
  lasers = [{a: la, piece: new Mem(0)}];
  assert.equal(laserRows()[0][7][0][0], 3);
  assert.equal(laserRows()[0][7][0][3], 80);
  la.add(8).writeU32(2);
  assert.equal(laserRows()[0][7].length, 0);
  assert.equal(laserState()[0][0], 2);
  const w = new Mem(0x220000), entity = new Mem(0x230000);
  w.add(0x2c30).writeU16(1); w.add(0x2c40).writePointer(entity); w.add(0x2f40).writeFloat(5);
  entity.add(0x14).writeU32(1);
  for (let j = 0; j < 10; j++) entity.add(j * 0xf8 + 0xfc).writeU32(1);
  entity.add(0xfc).writeU32(0);
  entity.add(0xa8).writeFloat(25); entity.add(0xac).writeFloat(35);
  entity.add(0xc0).writeFloat(300); entity.add(0xc4).writeFloat(-60);
  assert.equal(shots(w)[0].length, 1);
  assert.equal(shots(w)[0][0][3], 5);
  assert.equal(shots(w)[1][0][1], 300);
  entity.add(0x14).writeU32(5);
  assert.equal(shots(w)[0][0][3], 1); // kind 5's native default radius
  assert.equal(stepMs(), null);       // timer reads exist; no time-step patch is installed
  kb = [-100, 100, -100, 100];
  assert.equal(outOfLevel(101, 0), true);
  assert.equal(outOfLevel(100, 0), false);
  kbPending = kb; pieceSig = sig(pieces(w));
  exchange(w);
`, context);
const msg = sent.findLast(m => m.ev === 'frame');
assert.ok(msg);
assert.deepEqual(msg.sh, [[0, 25, 35, 1]]);
assert.deepEqual(msg.shx, [[5, 300, -60]]);
assert.deepEqual(msg.kb, [-100, 100, -100, 100]);
assert.equal(msg.hz.length, 2);  // enemy and laser rows, both currently harmless
assert.ok(msg.hz.every(h => h[7].length === 0));
assert.ok(!source.includes('Memory.patchCode'));  // preserve ONLYPPO's existing timing behaviour
console.log('Native perception producers and frame wire format passed.');
