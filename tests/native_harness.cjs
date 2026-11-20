// Execute the native adapter with fake memory: observed enemies, laser phases, shots and frame messages.
// No game process or real memory is accessed. This does not validate executable offsets against a live game.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
module.exports = function loadAdapter(config = {}) {
const heap = new Map();
const hooks = new Map(), replacements = new Map();
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
  Interceptor: {attach(p,h) { hooks.set(p.address,h); return {detach() {}}; }, replace(p,f) { replacements.set(p.address,f); }},
  NativeFunction: function () { return () => new Mem(0); },
  NativeCallback: function (fn) { return fn; },
  ptr: n => new Mem(n), rpc: {}, setInterval() {},
  send: m => sent.push(JSON.parse(JSON.stringify(m))),
  recv: (type, fn) => ({wait() { if (type === 'act') fn({payload: {a: [], leader: 0}}); if (type === 'go') fn({payload: context.goPayload || {}}); }}),
});
const source = fs.readFileSync(path.join(__dirname, '..', 'realgame.js'), 'utf8');
vm.runInContext(source.replace('__CONFIG__', JSON.stringify({agents: 2, pool: 40, wave_frames: 1200,
                                                          finish_frames: 60, finish_quota: 1, ...config})), context);

return {context, heap, Mem, sent, source, hooks, replacements};
};
