// Frida: per-level capture for Super Meat Boy.
// On every level load it re-arms, and on the first simulated frame it writes
// level_<seq>_<name>.json with tilemap, set pieces, type table and player spawn.
// Every lethal contact reported by FUN_004cd350 is appended to deaths.jsonl.
//
// Addresses are image VAs (base 0x400000); ASLR is resolved at runtime.
//   FUN_004e9020  level load ("Loading SMB Level: %s"), arg0 -> {char *name, ...}
//   FUN_004657e0  world->grid, ecx = tilemap
//   FUN_00465b00  GetTile(gx, gy, layer), ecx = tilemap
//   FUN_004cd350  player-vs-world contact test, ecx = world, arg0 = player, returns 1 on lethal contact
const DIR = 'C:/Users/pahstoner/PycharmProjects/REA-MeatBoy/smbrl/imported/capture/';  // gitignored; personal-use captures
const base = Process.getModuleByName('SuperMeatBoy.exe').base;
const img = a => base.add(a - 0x400000);
function rd(fn) { try { return fn(); } catch (e) { return null; } }
function hexAt(p, n) { try { return Array.from(new Uint8Array(p.readByteArray(n))).map(b => b.toString(16).padStart(2, '0')).join(''); } catch (e) { return null; } }
function writeFile(name, text, mode) { try { const f = new File(DIR + name, mode || 'w'); f.write(text); f.close(); return true; } catch (e) { send({ err: 'write ' + name + ': ' + e.message }); return false; } }

let levelName = null, armed = true, tilemap = null, seq = 0, lastKey = null, frame = 0, lastDeathFrame = -1000;
const RUN = Date.now().toString(36);  // distinguishes captures across script reloads (seq restarts at 1)
// per-frame trajectory, buffered and appended to traj.jsonl
let traj = [];
function flushTraj() { if (traj.length) { writeFile('traj.jsonl', traj.join('\n') + '\n', 'a'); traj = []; } }
// RTTI class name from an object's vtable: vtable[-1] -> CompleteObjectLocator, +0xc -> TypeDescriptor, +0x8 -> ".?AV...@@"
const clsCache = {};
function className(obj) {
  const vt = rd(() => obj.readPointer()); if (!vt) return null;
  const k = vt.toString(); if (k in clsCache) return clsCache[k];
  const n = rd(() => vt.sub(4).readPointer().add(12).readPointer().add(8).readCString());
  clsCache[k] = n; return n;
}
let pframe = 0;
const entSeen = {};

function tilemapKey(G) {
  return [rd(() => G.add(0x20).readU16()), rd(() => G.add(0x22).readU16()),
          rd(() => G.add(0x18).readFloat()), rd(() => G.add(0x1c).readFloat())].join(',');
}

function dump(W, P) {
  const G = tilemap;
  const Wd = rd(() => G.add(0x20).readU16()), Hd = rd(() => G.add(0x22).readU16());
  const out = {
    name: levelName, seq: seq, run: RUN,
    spawn: [rd(() => P.add(0x84).readFloat()), rd(() => P.add(0x88).readFloat())],
    tiles: { W: Wd, H: Hd, originX: rd(() => G.add(0x18).readFloat()), originY: rd(() => G.add(0x1c).readFloat()),
             tileSize: rd(() => img(0x6f6ab4).readFloat()), layers: [] },
    pieces: [], typeTable: [],
  };
  if (Wd && Hd && Wd * Hd < 200000)
    for (let l = 0; l < 3; l++) { const lp = rd(() => G.add(l * 4).readPointer());
      out.tiles.layers.push(lp && !lp.isNull() ? hexAt(lp, Wd * Hd * 4) : null); }
  for (let i = 0; i < 450; i++) {
    const o = rd(() => W.add(0x124 + i * 8).readPointer()); if (!o || o.isNull()) break;
    out.pieces.push({ type: rd(() => o.add(0x10).readU16()), x: rd(() => o.add(0x14).readFloat()), y: rd(() => o.add(0x18).readFloat()),
                      angle: rd(() => W.add(0x124 + i * 8 + 4).readFloat()), raw: hexAt(o, 0x80) });
  }
  const nT = rd(() => W.add(0x1d84).readS32());
  if (nT > 0 && nT < 400) for (let i = 0; i < nT; i++) out.typeTable.push(hexAt(W.add(0xde4 + i * 0x28), 0x28));
  out.special = { t2fe4: rd(() => W.add(0x2fe4).readU32()), t2fe8: rd(() => W.add(0x2fe8).readU32()), t2fec: rd(() => W.add(0x2fec).readU32()),
                  t2ffc: rd(() => W.add(0x2ffc).readU32()), t2eec: rd(() => W.add(0x2eec).readU32()), flagsDB4: rd(() => W.add(0xdb4).readU32()) };
  const nC = rd(() => W.add(0x2c30).readU16()), aC = rd(() => W.add(0x2c40).readPointer());
  out.circles = { count: nC, elems: [] };
  if (aC && nC) for (let i = 0; i < Math.min(nC, 16); i++) out.circles.elems.push(hexAt(aC.add(i * 0x9d4), 0x9d4));
  // A: count @+0x2d00, arr @+0x2d10, stride 0x24 (contact returns lethal); B: count @+0x1db8, arr @+0x1dc8, stride 0x1c,
  // elem+0 -> object whose +0x10 type and +0x14..+0x20 box are copied into the overlap tester (non-lethal trigger zones?)
  const nA = rd(() => W.add(0x2d00).readU16()), aA = rd(() => W.add(0x2d10).readPointer());
  out.A = { count: nA, elems: [] };
  if (aA && nA) for (let i = 0; i < Math.min(nA, 500); i++) out.A.elems.push(hexAt(aA.add(i * 0x24), 0x24));
  const nB = rd(() => W.add(0x1db8).readU16()), aB = rd(() => W.add(0x1dc8).readPointer());
  out.B = { count: nB, elems: [] };
  if (aB && nB) for (let i = 0; i < Math.min(nB, 500); i++) {
    const e = aB.add(i * 0x1c), o = rd(() => e.readPointer());
    out.B.elems.push({ raw: hexAt(e, 0x1c), obj: o && !o.isNull() ? hexAt(o, 0x40) : null, cls: o && !o.isNull() ? className(o) : null });
  }
  const safe = (levelName || 'unknown').replace(/[^A-Za-z0-9_.-]/g, '_');
  const file = 'level_' + RUN + '_' + String(seq).padStart(3, '0') + '_' + safe + '.json';
  writeFile(file, JSON.stringify(out));
  send({ evt: 'level_dumped', file: file, name: levelName, W: Wd, H: Hd, pieces: out.pieces.length, typeRows: out.typeTable.length,
         circles: nC, A: nA, B: nB, spawn: out.spawn });
}

Interceptor.attach(img(0x4e9020), { onEnter(args) {
  levelName = rd(() => args[0].readPointer().readCString());
  armed = true; send({ evt: 'level_load', name: levelName });
} });
Interceptor.attach(img(0x4657e0), { onEnter() { tilemap = this.context.ecx; } });
Interceptor.attach(img(0x465b00), { onEnter() { tilemap = this.context.ecx; } });
Interceptor.attach(img(0x4cd350), {
  onEnter(args) {
    frame++;
    this.W = this.context.ecx; this.P = args[0];
    if (!tilemap) return;
    const key = tilemapKey(tilemap);
    if (armed || key !== lastKey) { flushTraj(); armed = false; lastKey = key; seq++; dump(this.W, this.P); }
    // other entities passing through the contact test: log class + position, sampled per entity
    const P = this.P, pk = P.toString();
    const n = (entSeen[pk] = (entSeen[pk] || 0) + 1);
    if (n === 1 || n % 30 === 0)
      traj.push(JSON.stringify({ k: 'ent', r: RUN, s: seq, f: frame, p: pk, c: className(P),
        x: rd(() => P.add(0x84).readFloat()), y: rd(() => P.add(0x88).readFloat()) }));
    if (traj.length >= 200) flushTraj();
  },
  onLeave(ret) {
    if (ret.toInt32() !== 1 || frame - lastDeathFrame < 30) return;
    lastDeathFrame = frame;
    const rec = { name: levelName, run: RUN, seq: seq, frame: frame, reason: rd(() => this.W.add(0x50).readU32()),
                  pos: [rd(() => this.P.add(0x84).readFloat()), rd(() => this.P.add(0x88).readFloat())] };
    writeFile('deaths.jsonl', JSON.stringify(rec) + '\n', 'a');
    send({ evt: 'lethal_contact', rec: rec });
  },
});
// special-object handlers called from FUN_004cd350 by type-table flag bit; log each hit with the live player position
let lastPlayer = null, lastSpecialFlag = null;
function logEvent(kind, extra) {
  const rec = Object.assign({ ev: kind, name: levelName, run: RUN, seq: seq, t: pframe, pos: lastPlayer }, extra || {});
  writeFile('events.jsonl', JSON.stringify(rec) + '\n', 'a');
  send({ evt: 'special_event', rec: rec });
}
const HANDLERS = { 0x4c1910: 'bit02_levelend', 0x4a5c50: 'bit02_levelend_alt', 0x4a5b90: 'bit10_swap', 0x4c29b0: 'bit40_handler' };
for (const [a, kind] of Object.entries(HANDLERS)) {
  try { Interceptor.attach(img(Number(a)), { onEnter() { logEvent(kind); } }); }
  catch (e) { send({ err: 'hook ' + kind + ': ' + e.message }); }
}
// bit 0x08 sets bit0 of a global (image 0x6f0758); report changes, polled from the contact test
Interceptor.attach(img(0x4cd350), { onLeave() {
  const v = rd(() => img(0x6f0758).readU32());
  if (lastSpecialFlag !== null && v !== lastSpecialFlag) logEvent('bit08_global', { from: lastSpecialFlag, to: v });
  lastSpecialFlag = v;
} });

// player trajectory from the character update (FUN_00520d30, ecx = MeatBoyCharactor; field offsets validated live)
Interceptor.attach(img(0x520d30), { onEnter() {
  const C = this.context.ecx; if (C.isNull()) return;
  pframe++;
  lastPlayer = [rd(() => C.add(0x84).readFloat()), rd(() => C.add(0x88).readFloat())];
  traj.push(JSON.stringify({ k: 'pl', r: RUN, s: seq, t: pframe, p: C.toString(), c: className(C),
    x: rd(() => C.add(0x84).readFloat()), y: rd(() => C.add(0x88).readFloat()),
    vx: rd(() => C.add(0x9c).readFloat()), vy: rd(() => C.add(0xa0).readFloat()),
    st: rd(() => C.add(0x7b0).readU32()), fl: rd(() => C.add(0x7b4).readU32()) }));
  if (traj.length >= 200) flushTraj();
} });
send({ info: 'level_dumper_installed' });
