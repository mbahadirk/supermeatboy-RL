const vm = require('node:vm');
const assert = require('node:assert/strict');
const {context, sent, source} = require('./native_harness.cjs')();
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
