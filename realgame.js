// Frida: agents that are REAL Meat Boy characters inside Super Meat Boy (personal use).
//
// The game keeps a pool of 40 extra DefaultMeatBoy characters per level (built by the main character,
// vtable+0x2c -> FUN_0051eb60, stored at main+0x7c0; normally used for the level-complete replay). Each
// agent is one of them, switched to mode 2 (+0x9e0: 0 player, 1 replay, 2 = ours: full physics and
// collision, no replay recording, no global death/level-reset code). Python (realgame.py) gets every
// agent's state each frame and answers with its inputs; the game's own input handlers, update, collision
// and hazards do the rest, so agents hit exactly what Meat Boy hits.
//
//   waves: all agents spawn together after the level is reset through the main character's respawn
//          (vtable+0x30 -> FUN_00520150 -> FUN_004d2060 / FUN_00514740 / ...: saws, platforms, crumble
//          blocks back to their start). One life each: a dead agent stays where it died until the next
//          wave, which starts when every agent is dead or finished, or after CFG.wave_frames.
//   death: the update asks vtable+0x70 (FUN_0051e7d0, "return 1") whether a lethal contact kills; for an
//          agent we record the death and answer 0, so the game's own death code never runs for it.
//   goal:  FUN_004c1910 / FUN_004a5c50 (level end) are swallowed while an agent updates and count as that
//          agent's finish. Bandages, keys and warps are off for agents (world+0xdb4 bits 0x8/0x10; the
//          game's key pickup is gated by 0x10): keys are per agent instead (below).
//   keys:  each agent picks up a key by touching it (within KEY_R of the key piece); the lock tiles that
//          key opens (groups from Python) are opened only while that agent updates (tile words swapped
//          before its update and put back after every agent), so one agent's key opens nothing for others.
//   the real character: hidden and frozen, used as the camera (parked on the leading agent) and for the
//          level reset; S (game focused) finishes the level the game's way, recording the time Python sends
//          (agent.py: the planned agents' best; 999.99 without one).
//   paths: the reward paths from Python are drawn over the picture through Direct3D 9 (see below).
//   map:   every frame the collision tile layer is compared with the last frame and changed cells are sent
//          (crumble blocks breaking, locks opening, the level reset putting them back), so Python's map of
//          the level (agents' view, monitor) stays live. Pieces (TileLevelSetPiece): type +0x10, x +0x14,
//          y +0x18, scale x/y +0x24/+0x28, angle +0x2c (radians).
//   time:  the game's time step includes any wait on Python, so long work (paths, PPO) only happens at
//          the level start and between waves, and agents are held still for the frame after a long wait.
//          With lock_step (default) the game always takes its fixed step instead (see "game time step").
//
// Addresses are image VAs (base 0x400000), resolved against the ASLR base at runtime.
const CFG = __CONFIG__;   // {agents, pool, wave_frames, finish_frames, finish_quota, probe_frames, lock_step}
const base = Process.getModuleByName('SuperMeatBoy.exe').base;
const img = a => base.add(a - 0x400000);
const rd = fn => { try { return fn(); } catch (e) { return null; } };

const UPD = img(0x519d40), DRAW = img(0x51fae0), CANDIE = img(0x51e7d0);
const updateChar = new NativeFunction(UPD, 'void', ['pointer'], 'thiscall');
const drawChar = new NativeFunction(DRAW, 'void', ['pointer', 'pointer', 'pointer', 'int'], 'thiscall');
const setState = new NativeFunction(img(0x520900), 'void', ['pointer', 'int'], 'thiscall');
const GAME = img(0x7035f4);                       // [GAME] = game object, [game+0x24] = level (spawn +0x1d2c)
const WORLD = img(0x7035c4);                      // [WORLD] = collision world (ecx of FUN_004cd350)
const vfn = {};
function vcall(obj, off, ...args) {               // thiscall virtual with int-sized args
  const f = obj.readPointer().add(off).readPointer();
  const k = f.toString() + ':' + args.length;
  if (!vfn[k]) vfn[k] = new NativeFunction(f, 'void', ['pointer'].concat(args.map(() => 'pointer')), 'thiscall');
  return vfn[k](obj, ...args);
}
const V_RESPAWN = 0x30, V_RUN = 0x38, V_JUMP = 0x3c, V_LEFT = 0x40, V_RIGHT = 0x44;

// ---- game time step ---------------------------------------------------------------------------------
// FUN_00450e70 runs once per frame on the game timer (ecx): it adds the real elapsed microseconds to an
// accumulator and, when a step is due (timer+0x88 = 1), sets the step length timer+0x68 (float s; +0x70 double):
// the fixed step timer+0x18 if the frame came in time, else the whole real elapsed time (0x450ef0: compare,
// then `jl 0x450f4b` = take the fixed step). So a slow frame (waiting on Python) moved everything by the wait.
// lock_step turns that jl into jmp: always the fixed step. Physics then no longer depends on Python's speed
// (the game runs in slow motion while Python is slow). The two bytes are checked first and put back on stop.
const LOCK_AT = img(0x450ef6), LOCK_ORIG = [0x7c, 0x53], LOCK_ON = [0xeb, 0x53];
let timer = null, lockOn = false;
const timerHook = Interceptor.attach(img(0x450e70), {
  onEnter() {
    if (timer) return;
    timer = this.context.ecx;
    send({ ev: 'debug', msg: `game timer found: fixed step ${(1000 * timer.add(0x18).readFloat()).toFixed(2)} ms` +
                             (lockOn ? ' (locked: every step is this long)' : '') });
  }
});
function setLock(on) {
  const cur = new Uint8Array(LOCK_AT.readByteArray(2));
  if (cur[1] !== 0x53 || (cur[0] !== LOCK_ORIG[0] && cur[0] !== LOCK_ON[0])) {
    send({ ev: 'error', msg: `time step lock: unexpected bytes ${cur[0].toString(16)} ${cur[1].toString(16)} at 0x450ef6, left alone` });
    return false;
  }
  Memory.patchCode(LOCK_AT, 2, code => code.writeByteArray(on ? LOCK_ON : LOCK_ORIG));
  lockOn = on;
  return true;
}
if (CFG.lock_step && setLock(true)) send({ ev: 'debug', msg: "time step locked to the game's fixed step" });
function stepMs() { return timer ? rd(() => 1000 * timer.add(0x68).readFloat()) : null; }   // this update's step

let active = true;
rpc.exports = { shutdown() { if (lockOn) setLock(false); if (active) { active = false; releaseMain(); } return true; } };   // only when no frame waits

// ---- level / main character -------------------------------------------------------------------
let curName = null, main = null, settle = 0, agents = [], agentSet = new Set(), ready = false;
let crowd = [], wantAgents = CFG.agents;           // all usable pool characters; the first wantAgents are agents
let wave = 0, frame = 0, status = [], held = [], prevJump = [], finishFrame = [];
let curAgent = -1, leader = -1, best = null, finishing = 0, finishRequested = false, hud = null;
let goalPieces = [];

function world() { return rd(() => WORLD.readPointer()); }
function levelObj() { return rd(() => GAME.readPointer().add(0x24).readPointer()); }

Interceptor.attach(img(0x4e9020), { onEnter(a) {
  const n = rd(() => a[0].readPointer().readCString());
  if (!n) return;
  curName = n; main = null; settle = 0; ready = false; agents = []; crowd = []; agentSet = new Set(); pendingWave = false; holdFrames = 0;
  keyPos = []; keyLocks = []; lockWord = new Map(); lockOpen = new Map(); keysOf = [];
  finishing = 0; finishRequested = false; best = null; leader = -1; wave = 0; frame = 0;
  tilemap = null; levelOY = null; levelBox = null; paths = []; prevTiles = null; kb = null; kbShown = false; kbPending = null;   // the new level has its own tile map, pool and paths
  send({ ev: 'level', name: n });
} });


// More than the game's 40: right after the game builds its pool (FUN_0051eb60: pool[i] =
// FUN_00527190(main, i), thiscall on the factory [0x704678], i < 40) more characters are built the same
// way (indices 40...), during the level load like the game's own. The clone keeps the index only for the
// replay (+0xa44). They are kept here (the game's pool array has 40 slots) and never freed (a little memory
// per level load).
// The clone constructor (FUN_0051e950) logs "Cloning Charactor" through OutputDebugStringA, which works by
// raising a debug-print exception (0x40010006) that Windows handles inside that call. With Frida's default
// ('steal') the exception is taken as a crash of the call ("system error") and nothing is built, so
// exceptions here go to the game's own handlers ('propagate').
const CLONE = new NativeFunction(img(0x527190), 'pointer', ['pointer', 'pointer', 'int'],
                                 { abi: 'thiscall', exceptions: 'propagate' });
let extra = [], extraFor = null;
function buildExtra(m, when) {
  const want = Math.min(160, (CFG.pool || 40) - 40);
  extra = []; extraFor = m;
  if (!active || want <= 0) return;
  try {
    for (let i = 0; i < want; i++) {
      const c = CLONE(img(0x704678).readPointer(), m, 40 + i);
      if (c.isNull()) break;
      c.add(0x9e0).writeU32(1);                      // like the pool: not the player
      extra.push(c);
    }
  } catch (e) { send({ ev: 'error', msg: 'extra characters: ' + e.message }); }
  send({ ev: 'debug', msg: `built ${extra.length} extra characters (pool ${40 + extra.length}, ${when})` });
}
Interceptor.attach(img(0x51eb60), {
  onEnter() { this.m = this.context.ecx; },
  onLeave() { buildExtra(this.m, 'level load'); }
});

function setupAgents() {
  const pool = rd(() => main.add(0x7c0).readPointer());
  if (!pool || pool.isNull()) return false;
  const read = () => {
    const out = [];
    for (let i = 0; i < 40; i++) {
      const c = rd(() => pool.add(4 * i).readPointer());
      if (c && !c.isNull()) out.push(c);
    }
    return out;
  };
  crowd = read();
  // attached in the middle of a level: its pool was built before this script ran. Clone the rest now, from the
  // game thread (this runs inside the player's update), the same way the pool builder does.
  if (crowd.length && (CFG.pool || 40) > 40 && !(extraFor && extraFor.equals(main))) buildExtra(main, 'attached mid-level');
  if (extraFor && main && extraFor.equals(main)) crowd = crowd.concat(extra);   // built with this level's pool
  agents = crowd.slice(0, Math.max(1, Math.min(wantAgents, crowd.length)));
  // (no pool yet: wait for the game to build it; building it ourselves calls the character factory outside
  // the level load, which is not safe)
  if (!agents.length) return false;
  // every pool character is ours (never updated or drawn by the game), so the count can change between waves
  agentSet = new Set(crowd.map(c => c.toString()));
  crowd.forEach(c => c.add(0x9e0).writeU32(2));
  resetArrays();
  return sendLevelInfo();
}

// static level facts from the game's memory: tile layer 0, origin, size; goal pieces (type flag 0x02)
let tilemap = null, levelOY = null, levelBox = null;   // levelBox: [left, right, top] edges of the map
// out of the level: the player (mode 0 only) dies when its position (+0x84, +0x88) leaves the camera's view
// enlarged by 15% (FUN_00520d30, 0x520fd3..0x521092: the view box is copied from camera+0x144, camera =
// [game+0x20]; half extents at box+0x8c / +0x90, times 1.15 (0x3f933333); centre box+0x50 / +0x60). The camera
// (FUN_004a30c0) is clamped so its view stays inside the level's camera bounds ([game+0x24] = the level
// object: min x +0x1d38, min y +0x1d3c, max x +0x1d40, max y +0x1d44; about +-1e8 when the level has none),
// centred on an axis where the bounds are smaller than the view. Our agents are mode 2 (no such check), so for
// a player the camera follows the kill lines per axis are: bounds wider than the view: [min - 0.15 h,
// max + 0.15 h]; else [c - 1.15 h, c + 1.15 h] around the bounds' centre c (h: view half extent). That is
// killBox(); an axis without bounds keeps the older rule: OUT_SIDE past the map's left / right / top edge,
// OUT_BOTTOM below it.
const OUT_SIDE = 10, OUT_BOTTOM = 60, VIEW_KILL = 1.15, NO_BOUNDS = 1e7;
let kb = null, kbShown = false, kbPending = null;   // kb: [left, right, bottom, top] now (null: older rule)
function camera() { return rd(() => GAME.readPointer().add(0x20).readPointer()); }
function killBox() {                                 // {kb, cam_bounds, view_half, centre} or null
  const lv = levelObj(), cam = camera();
  if (!lv || lv.isNull() || !cam || cam.isNull() || !levelBox) return null;
  const b = rd(() => [0x1d38, 0x1d3c, 0x1d40, 0x1d44].map(o => lv.add(o).readFloat()));
  const view = cam.add(0x144);
  const h = rd(() => [view.add(0x8c).readFloat(), view.add(0x90).readFloat()]);
  const centre = rd(() => [view.add(0x50).readFloat(), view.add(0x60).readFloat()]);
  if (!b || !h || !b.every(Number.isFinite) || !h.every(v => Number.isFinite(v) && v > 0 && v < 1e5)) return null;
  const axis = (lo, hi, hh) => {
    if (!(Math.abs(lo) < NO_BOUNDS && Math.abs(hi) < NO_BOUNDS && hi > lo)) return null;   // no bounds here
    if (hi - lo >= 2 * hh) return [lo - (VIEW_KILL - 1) * hh, hi + (VIEW_KILL - 1) * hh];
    const c = (lo + hi) / 2;
    return [c - VIEW_KILL * hh, c + VIEW_KILL * hh];
  };
  const x = axis(b[0], b[2], h[0]), y = axis(b[1], b[3], h[1]);
  if (!x && !y) return null;                         // no camera bounds at all: the older rule
  const r = v => +v.toFixed(2);
  const box = [...(x || [levelBox[0] - OUT_SIDE, levelBox[1] + OUT_SIDE]),
               ...(y || [levelOY - OUT_BOTTOM, levelBox[2] + OUT_SIDE])].map(r);
  return { kb: box, cam_bounds: b.map(r), view_half: h.map(r), centre: centre ? centre.map(r) : null };
}
function updateKillBox(where) {                     // at the level info and every wave start (camera zoom)
  const k = killBox();
  if (!k) return null;
  const moved = !kb || k.kb.some((v, j) => Math.abs(v - kb[j]) > 1);
  if (moved) {
    if (where !== 'info') kbPending = k.kb;          // changed after the level info: goes with the next frame
    kb = k.kb;
  }
  if (!kbShown || (moved && where !== 'info')) {
    kbShown = true;
    send({ ev: 'debug', msg: `level kill box: x ${kb[0].toFixed(1)}..${kb[1].toFixed(1)}, y ${kb[2].toFixed(1)}..${kb[3].toFixed(1)} ` +
                             `(camera bounds ${k.cam_bounds.map(v => v.toFixed(1)).join(', ')}, view half ${k.view_half[0].toFixed(1)} x ${k.view_half[1].toFixed(1)}; ` +
                             `camera centre ${k.centre ? k.centre.map(v => v.toFixed(1)).join(', ') : '?'}; map x ${levelBox[0].toFixed(0)}..${levelBox[1].toFixed(0)}, ` +
                             `y ${levelOY.toFixed(0)}..${levelBox[2].toFixed(0)}; ${where})` });
  }
  return k;
}
Interceptor.attach(img(0x465b00), { onEnter() { if (!tilemap) tilemap = this.context.ecx; } });
function validTilemap(G) {                           // levels are centred: origin = -(size * 10)
  const W = rd(() => G.add(0x20).readU16()), H = rd(() => G.add(0x22).readU16());
  const ox = rd(() => G.add(0x18).readFloat()), oy = rd(() => G.add(0x1c).readFloat());
  return W >= 4 && H >= 4 && W <= 2000 && H <= 2000 && Math.abs(ox + W * 10) < 1 && Math.abs(oy + H * 10) < 1;
}
function typeFlags(w, t) {                         // world type table: rows {w0 id, ..., w9 flags} at +0xde4
  const n = w.add(0x1d84).readS32();
  for (let i = 0; i < n; i++) {
    const row = w.add(0xde4 + i * 0x28);
    if (row.readU32() === t) return row.add(0x24).readU32();
  }
  return null;
}
function pieces(w) {                               // null-terminated {piece*, angle} list at world+0x124
  const out = [];
  for (let p = w.add(0x124); ; p = p.add(8)) {
    const pc = p.readPointer();
    if (pc.isNull() || out.length > 4000) break;
    out.push(pc);
  }
  return out;
}
function sendLevelInfo() {
  const w = world();
  const G = tilemap;
  if (!w || !G) return false;
  if (!validTilemap(G)) { tilemap = null; return false; }   // not this level's map: capture it again
  const W = G.add(0x20).readU16(), H = G.add(0x22).readU16();
  const ox = G.add(0x18).readFloat(), oy = G.add(0x1c).readFloat();
  levelOY = oy;
  levelBox = [ox, ox + W * 20, oy + H * 20];
  const tiles = G.readPointer().readByteArray(W * H * 4);
  tileCount = W * H; prevTiles = new Uint32Array(tiles.slice(0));
  goalPieces = [];
  const objs = [];                                   // [type, x, y, type-table flags or -1] (keys 0x40, bandages 0x08)
  for (const pc of pieces(w)) {
    const t = pc.add(0x10).readU16() & 0xff, fl = typeFlags(w, t);
    const x = pc.add(0x14).readFloat(), y = pc.add(0x18).readFloat();
    objs.push([t, x, y, fl === null ? -1 : fl]);
    if (fl !== null && (fl & 2)) goalPieces.push([x, y]);
  }
  const sp = levelObj().add(0x1d2c);
  spawnX = sp.readFloat(); spawnY = sp.add(4).readFloat();
  buildLethal(w);
  enemyProbe = 3;                                    // log the enemies' fields on the first 3 frames
  kb = null; kbPending = null;
  const k = rd(() => updateKillBox('info'));
  if (!k && !kbShown) { kbShown = true; send({ ev: 'debug', msg: 'level kill box: none (no camera bounds or unreadable): map edges +10 / -60 rule' }); }
  send({ ev: 'info', name: curName, W, H, ox, oy, goals: goalPieces, spawn: [sp.readFloat(), sp.add(4).readFloat()],
         agents: agents.length, lethal: lethal.length, hz: hazards(), objs,
         lethalTypes: lethalInfo().types, lethalIds: lethalInfo().ids, enemies: enemies.length,
         fx: rd(() => forceFields(w)), portalTypes: rd(() => portalTypes(w)) || [], pf: rd(() => platforms(w)) || [],
         kb: kb, cam_bounds: k ? k.cam_bounds : null, view_half: k ? k.view_half : null }, tiles);
  waitGo();                                          // Python builds the reward paths now
  return true;
}
// lethal pieces, decided like FUN_004cd350: some types never collide (world +0x2ffc/+0x2eec/+0x2fe4/
// +0x2fe8/+0x2fec, global table +0x34..+0x44); a type with a type-table row is lethal when its flags have
// bit 0; a type without a row is lethal. Built once per level; positions are re-read every frame (movers).
let lethal = [], lethalSize = [];
const sizeBuf = Memory.alloc(16), flagBuf = Memory.alloc(16), offBuf = Memory.alloc(16);
function pieceSize(w, pc) {   // the game's own size query (world+8 vtable +0x1c = FUN_004ca860): [w, h] at scale 1
  try {
    const obj = w.add(8), fn = obj.readPointer().add(0x1c).readPointer();
    const k = 'size:' + fn;
    if (!vfn[k]) vfn[k] = new NativeFunction(fn, 'void', ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'thiscall');
    sizeBuf.writeFloat(0); sizeBuf.add(4).writeFloat(0);
    vfn[k](obj, pc, sizeBuf, flagBuf, offBuf);
    return [sizeBuf.readFloat(), sizeBuf.add(4).readFloat()];
  } catch (e) { return [0, 0]; }
}
function buildLethal(w) {
  const g = img(0x703794).readPointer();
  const skip = new Set([0x2ffc, 0x2eec, 0x2fe4, 0x2fe8, 0x2fec].map(o => w.add(o).readU32())
    .concat([0x34, 0x38, 0x3c, 0x40, 0x44].map(o => g.add(o).readU32())));
  const old = new Map(lethal.map((pc, k) => [pc.toString(), lethalSize[k]]));
  const all = pieces(w);
  pieceSig = sig(all);
  lethal = [];
  for (const pc of all) {
    const t = pc.add(0x10).readU16() & 0xff;
    if (skip.has(t)) continue;
    const fl = typeFlags(w, t);
    if (fl === null || (fl & 1)) lethal.push(pc);
  }
  lethalSize = lethal.map(pc => old.get(pc.toString()) || pieceSize(w, pc));
  rd(() => Memory.copy(KILL_SHAPE, w.add(0x70).readPointer(), 0xa0));   // this level's animation library (+0x20)
  buildEnemies(w);
}

// ---- kill regions: what the game's narrow phase really tests ----------------------------------------
// FUN_004cd350 (the pieces check) copies each lethal piece into the world's scratch shape [world+0x70]
// (0x4cd745..0x4cd7cb: +0x14..+0x20 -> +0x24..+0x30; scale +0x24/+0x28 times the flips of +0x10 bits 10-11
// (FUN_004976c0: 1 -> x, 2 -> y, 3 -> both) -> +0x34; angle +0x2c -> +0x3c and +0x40; type -> +0x10; animation
// time +0x38 + +0x30 -> +0x4c; +0x34 -> +0x60) and calls FUN_0045a0e0(shape, character box). That builds the
// piece's matrix (FUN_00494b20: scale + rotation, translation = position) and asks the type's animation
// (library [shape+0x20]+0x30, vtable 0x6b7474, +0xc = FUN_0045ad90 -> FUN_0045a990) at that time: a Flash-like
// timeline (+0x10 loop flag, +0x18 length s, +0x24 layers {keyframes: length, start, tween flag, 0x98-byte
// children}) whose active children are composed (FUN_0045a1b0: position, scale, rotation, pivot, tweened to the
// next keyframe) and recursed into. A child with collision kind +0x84 = 1 / 2 ends in FUN_0045b160 (sprite
// entries, vtable 0x6b745c; the other entry class, 0x6b7448, never collides), which builds
//   kind 1: a circle at the matrix's translation, radius |max(w, h) x max(scale x, scale y)| (entry +0xc/+0x10,
//           scale from FUN_004952f0) -> {flags 0x11, r +4, centre +8/+0xc}, tested at 0x45b225;
//   kind 2: a box of half size |w x scale x x 0.5|, |h x scale y x 0.5|, turned with the matrix
//           (FUN_00475e30: axis 0 +0x44/+0x54, centre +0x50/+0x60, half sizes +0x84/+0x88), tested at 0x45b39a,
// each through FUN_00476720 against the character's box (an OBB of half size 6). The shapes move inside the
// animation while the piece stays put: on 1-14 the "movin" saws keep +0x14/+0x18 and their circle (r 20.35 at
// scale 0.6) travels 93.6 units over the 2.667 s loop; the "particle" pieces only follow the saws and test no
// shape at all. The wave reset (FUN_004d2060) sets every piece's +0x30 to 0, so the same wave frame has the
// same shapes in every wave.
// So the shapes are read from the game itself: a private copy of the scratch shape (game memory is never
// written), filled as above, goes through FUN_0045a0e0 against a copy of a real character box moved far away
// (no test can succeed, so every active shape is tested), and the two test sites record what they are given.
// killShapes must run inside a replaced function (here: the update, Interceptor.replace): called from an
// Interceptor.attach callback, Frida does not run the test sites' listeners and every piece reads as empty.
const NARROW = new NativeFunction(img(0x45a0e0), 'int', ['pointer', 'pointer'], 'thiscall');
const KILL_SHAPE = Memory.alloc(0x100), FAR_BOX = Memory.alloc(0x100);
const KILL_MAX = 32;                               // shapes per piece (more are dropped)
let killGot = null, killTid = -1, killReady = false, killErr = 0;
const r2 = v => Math.round(v * 100) / 100;
function killErrMsg(e) { if (killErr++ < 3) send({ ev: 'debug', msg: 'kill shapes: ' + e.message }); }
Interceptor.attach(img(0x45b225), function () {      // circle: [esp] = {flags, r, centre x, centre y}
  if (killGot === null || this.threadId !== killTid || killGot.length >= KILL_MAX) return;
  try {
    const s = this.context.esp.readPointer(), r = r2(s.add(4).readFloat());
    killGot.push([1, r2(s.add(8).readFloat()), r2(s.add(12).readFloat()), r, r, 0]);
  } catch (e) { killErrMsg(e); }
});
Interceptor.attach(img(0x45b39a), function () {      // box: eax = the OBB
  if (killGot === null || this.threadId !== killTid || killGot.length >= KILL_MAX) return;
  try {
    const s = this.context.eax;
    killGot.push([2, r2(s.add(0x50).readFloat()), r2(s.add(0x60).readFloat()), r2(s.add(0x84).readFloat()),
                  r2(s.add(0x88).readFloat()),
                  Math.round(Math.atan2(s.add(0x54).readFloat(), s.add(0x44).readFloat()) * 1000) / 1000]);
  } catch (e) { killErrMsg(e); }
});
let boxHook = Interceptor.attach(img(0x4cd350), { onEnter(a) {   // a real character box (args[1], else char+0x44)
  if (killReady) return;
  try {
    const bx = a[1].isNull() ? a[0].add(0x44).readPointer() : a[1];
    if ((bx.readU32() & 0xf) !== 0) return;          // an OBB (Bounds type 0), as the characters' boxes are
    Memory.copy(FAR_BOX, bx, 0xa0);
    for (const o of [0x10, 0x20, 0x50, 0x60]) FAR_BOX.add(o).writeFloat(1e7);   // matrix and centre copies
    killReady = true;
  } catch (e) { killErrMsg(e); }
} });
function killShapes(pc) {   // the piece's shapes now: [[1, x, y, r, r, 0] | [2, x, y, half w, half h, angle]], null unknown
  if (!killReady) return null;
  if (boxHook) { boxHook.detach(); boxHook = null; }
  try {
    const S = KILL_SHAPE, u = (pc.add(0x10).readU16() >> 10) & 3, a = pc.add(0x2c).readFloat();
    for (const o of [0, 4, 8, 12]) S.add(0x24 + o).writeU32(pc.add(0x14 + o).readU32());
    S.add(0x34).writeFloat(pc.add(0x24).readFloat() * (u === 1 || u === 3 ? -1 : 1));
    S.add(0x38).writeFloat(pc.add(0x28).readFloat() * (u >= 2 ? -1 : 1));
    S.add(0x3c).writeFloat(a); S.add(0x40).writeFloat(a);
    S.add(0x10).writeU32(pc.add(0x10).readU16() & 0xff);
    S.add(0x4c).writeFloat(pc.add(0x38).readFloat() + pc.add(0x30).readFloat());
    S.add(0x60).writeU32(pc.add(0x34).readU32());
    killGot = []; killTid = Process.getCurrentThreadId();
    NARROW(S, FAR_BOX);
    return killGot;
  } catch (e) { killErrMsg(e); return null; } finally { killGot = null; }
}
// pieces can be added or removed while the level runs (shots): the list is checked every frame
let pieceSig = '';
function sig(all) { return all.length + ':' + (all.length ? all[0] + ',' + all[all.length - 1] : ''); }
function lethalInfo() {   // types and ids (addresses) of the lethal pieces, in the order hazards() lists them;
                          // then the enemies (ENEMY_TYPE + their kind, id "enemy<address>")
  return { types: lethal.map(pc => pc.add(0x10).readU16() & 0xff).concat(enemies.map(e => ENEMY_TYPE + e.kind)),
           ids: lethal.map(pc => pc.toString()).concat(enemies.map(e => 'enemy' + e.a)) };
}
// per lethal piece, now: [x, y, size w, size h, scale x, scale y, angle, kill shapes] (the piece itself, size
// from FUN_004ca860 at scale 1: the world model's inputs; then what kills: killShapes, [] = nothing now); then one
// row per enemy (enemyRows)
function hazards() {
  return lethal.map((pc, k) => [pc.add(0x14).readFloat(), pc.add(0x18).readFloat(), lethalSize[k][0],
                                lethalSize[k][1], pc.add(0x24).readFloat(), pc.add(0x28).readFloat(),
                                pc.add(0x2c).readFloat(), killShapes(pc)]).concat(enemyRows());
}

// ---- enemies: the crawlers and fliers (Hospital and later) are not pieces --------------------------------
// They are SMBAnimal objects (SMBAnimal.cpp; vtable 0x682488, ctor FUN_005129e0, 0x1e8 bytes) of the animal
// manager [[0x7035c4] + 0x2e48] (FUN_00513430 adds one: u16 count +0x18, u16 capacity +0x1a, pointer array +0x28).
// Kind +0x1cc indexes the name table at 0x6f1a00: 0-3 groundcritter, 4-7 aircritter (background, never lethal),
// 8-11 groundenemy, 12-15 airenemy, 16 bosscritter, 17 charger, 18 boo, 19 hellbossboy. The character update
// (FUN_00520d30 at 0x5218ef, right after the pieces check FUN_004cd350) calls FUN_00513200(char, char box +0x71c):
// every animal of kind 8-15 or 17-19 whose state +0x1d0 is not 5 is tested with the bounds-vs-bounds test
// FUN_00476720 against its bounds at +0x13c: type 1 = a circle {flags (& 0xf = 1), radius +0x140, centre
// +0x144 / +0x148} (set to radius 10 in the ctor). A touch kills (the same death path as a lethal piece, so agents
// die through CANDIE as for pieces). Read straight from the game: nothing is written.
const ENEMY_TYPE = 1000;                             // lethal type ids of enemies: ENEMY_TYPE + kind (not a piece type)
const ENEMY_KINDS = new Set([8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]);
let enemies = [];                                    // [{a: address, kind}] of this level, in manager order
function buildEnemies(w) {
  enemies = [];
  const mgr = rd(() => w.add(0x2e48).readPointer());
  if (!mgr || mgr.isNull()) return;
  const n = rd(() => mgr.add(0x18).readU16()) || 0, arr = rd(() => mgr.add(0x28).readPointer());
  if (!arr || arr.isNull() || n > 4096) return;
  for (let i = 0; i < n; i++) {
    const a = rd(() => arr.add(4 * i).readPointer());
    if (!a || a.isNull()) continue;
    const kind = rd(() => a.add(0x1cc).readS32());
    if (kind !== null && ENEMY_KINDS.has(kind)) enemies.push({ a, kind });
  }
}
// per enemy, now: [x, y, 2r, 2r, 1, 1, 0, kill shapes]: the circle's centre and diameter as the piece box (the
// world model's inputs), and [[1, x, y, r, r, 0]] while it can kill (state not 5; bounds a circle), else []
function enemyRows() {
  return enemies.map(e => {
    const a = e.a;
    const fl = rd(() => a.add(0x13c).readU32()), r = rd(() => a.add(0x140).readFloat());
    const x = rd(() => a.add(0x144).readFloat()), y = rd(() => a.add(0x148).readFloat());
    const st = rd(() => a.add(0x1d0).readS32());
    if (x === null || y === null || r === null) return [0, 0, 0, 0, 1, 1, 0, null];
    const on = st !== 5 && fl !== null && (fl & 0xf) === 1;
    return [r2(x), r2(y), r2(2 * r), r2(2 * r), 1, 1, 0, on ? [[1, r2(x), r2(y), r2(r), r2(r), 0]] : []];
  });
}
// the first frames of a level: everything read about each enemy (to check the layout above against the game)
let enemyProbe = 0;
function probeEnemies() {
  if (enemyProbe <= 0 || !enemies.length) return;
  enemyProbe--;
  send({ ev: 'debug', msg: 'enemies: ' + JSON.stringify(enemies.slice(0, 12).map(e => {
    const a = e.a, f = o => rd(() => r2(a.add(o).readFloat()));
    return { kind: e.kind, state: rd(() => a.add(0x1d0).readS32()), bounds: rd(() => a.add(0x13c).readU32()),
             r: f(0x140), centre: [f(0x144), f(0x148)], pos: [f(0xd8), f(0xdc)], spawn: [f(0xe8), f(0xec)] };
  })) });
}
// Shots (thrown saws, missiles): not pieces. Each shooter / launcher is an entity in the world's array
// (u16 count +0x2c30, pointer +0x2c40, 0x9d4 bytes each; +0 its piece, +0x14 kind) with 10 shot slots of
// 0xf8 bytes: position +0xa8/+0xac, state +0xfc (0 = flying, the only state that kills). Radius per kind
// (world +0x2f20 kinds 0/3, +0x2f40 kind 1, +0x2f60 kind 2, +0x2fb8 kind 4) x the piece's scale
// (FUN_004cd350 tests exactly these). [[id, x, y, radius], ...] of the flying ones.
const SHOT_R = { 0: 0x2f20, 3: 0x2f20, 1: 0x2f40, 2: 0x2f60, 4: 0x2fb8 };
function shots(w) {
  const out = [];
  const n = w.add(0x2c30).readU16();
  if (!n || n > 64) return out;
  const arr = w.add(0x2c40).readPointer();
  if (arr.isNull()) return out;
  for (let i = 0; i < n; i++) {
    const e = arr.add(i * 0x9d4), kind = e.add(0x14).readS32();
    const ro = SHOT_R[kind];
    if (ro === undefined) continue;
    const pc = e.readPointer();
    const r = w.add(ro).readFloat() * (pc.isNull() ? 1 : Math.abs(pc.add(0x24).readFloat() || 1));
    for (let j = 0; j < 10; j++) {
      const o = e.add(j * 0xf8);
      if (o.add(0xfc).readU32() !== 0) continue;
      out.push([i * 10 + j, o.add(0xa8).readFloat(), o.add(0xac).readFloat(), r]);
    }
  }
  return out;
}
// Fans and orbs (FUN_004c40e0, run in each character's update):
//   fans: array at world+0x2cb4 (u16 count +0x2ccc, data pointer +0x2cdc, 0x24 bytes each): +0 the fan
//         piece, +4 its x axis, +0x14 (half width, 0) turned by its angle, +0x1c its up axis. Wind acts in
//         front of the fan (up axis) for piece+0x40 units, strength piece+0x3c (x5 in the near half).
//   orbs: world+0x2e14 (u16 count +0x2e2c, data pointer +0x2e3c, 4-byte piece pointers): a round field,
//         inner radius s32 piece+0x44 (strength +0x3c), outer radius s32 piece+0x48 (strength +0x40).
function forceFields(w) {
  const fans = [], orbs = [];
  const nf = w.add(0x2ccc).readU16();
  if (nf && nf < 64) {
    const a = w.add(0x2cdc).readPointer();
    for (let i = 0; i < nf; i++) {
      const e = a.add(i * 0x24), pc = e.readPointer();
      if (pc.isNull()) continue;
      const hx = e.add(0x14).readFloat(), hy = e.add(0x18).readFloat();
      fans.push([pc.add(0x14).readFloat(), pc.add(0x18).readFloat(), e.add(0x1c).readFloat(), e.add(0x20).readFloat(),
                 Math.hypot(hx, hy), pc.add(0x40).readFloat(), pc.add(0x3c).readFloat()]);
    }
  }
  const no = w.add(0x2e2c).readU16();
  if (no && no < 64) {
    const a = w.add(0x2e3c).readPointer();
    for (let i = 0; i < no; i++) {
      const pc = a.add(4 * i).readPointer();
      if (pc.isNull()) continue;
      orbs.push([pc.add(0x14).readFloat(), pc.add(0x18).readFloat(), pc.add(0x44).readS32(), pc.add(0x48).readS32(),
                 pc.add(0x3c).readFloat(), pc.add(0x40).readFloat()]);
    }
  }
  return { fans, orbs };
}
// Moving platforms: the level build (FUN_004c8c10) puts every piece of the world's platform type (+0x2ffc) in
// a ResizeableArray (count u16 world+0x2dc4, data +0x2dd4, 0x14 bytes: +0 piece, +4..+0xc three tile shapes,
// +0x10 = 1 when cos(angle) is ~0, i.e. upright). Collision (around 0x4cde59) tests the three shapes at the
// piece's position and 20 units either side along x (or along y when upright): a 60 x 20 solid bar. The piece
// moves, so its position is re-read every frame. [x, y, upright 0/1]
function platforms(w) {
  const n = w.add(0x2dc4).readU16();
  const out = [];
  if (!n || n > 256) return out;
  const a = w.add(0x2dd4).readPointer();
  for (let i = 0; i < n; i++) {
    const e = a.add(i * 0x14), pc = e.readPointer();
    if (pc.isNull()) continue;
    out.push([+pc.add(0x14).readFloat().toFixed(1), +pc.add(0x18).readFloat().toFixed(1), e.add(0x10).readU32() ? 1 : 0]);
  }
  return out;
}
// world type ids of portal pieces (portalidle/enter/exit, paired in piece order by the level build FUN_004c8c10)
function portalTypes(w) { return [0x2fe4, 0x2fe8, 0x2fec].map(o => w.add(o).readS32()).filter(t => t >= 0); }

// collision tiles that changed since the last frame: [[cell index, new word], ...]
let prevTiles = null, tileCount = 0;
function tileChanges() {
  if (!prevTiles || !tilemap) return null;
  const a = rd(() => tilemap.readPointer());
  if (!a || a.isNull()) return null;
  const now = new Uint32Array(a.readByteArray(tileCount * 4));
  let out = null;
  for (let i = 0; i < tileCount; i++) {
    if (now[i] !== prevTiles[i]) {
      if (!out) out = [];
      if (out.length < 2000) out.push([i, now[i]]);
    }
  }
  prevTiles = now;
  return out;
}

// ---- inputs through the game's own handlers: thiscall(char, {state 1 press / 2 held / 3 release, ms}) -------
const inBuf = Memory.alloc(8);
function press(c, off, state, ms) { inBuf.writeU32(state); inBuf.add(4).writeU32(ms); vcall(c, off, inBuf); }
const WALL_REPRESS = 6;
function applyInput(i, a) {                        // a = [left, right, jump, run]
  const c = agents[i], h = held[i];
  [[0, V_LEFT], [1, V_RIGHT], [3, V_RUN]].forEach(([k, off]) => {
    if (a[k]) { h[k]++; press(c, off, h[k] === 1 ? 1 : 2, Math.round(h[k] * 1000 / 60)); }
    else if (h[k]) { h[k] = 0; press(c, off, 3, 0); }
  });
  // a wall jump needs a new press: jump kept held while in the air against a wall (flags 0x200 wall, 0x2
  // airborne) for WALL_REPRESS frames or more is let go for this frame and pressed again on the next, so
  // "jump" on a wall always jumps off it (realgame.py mirrors this for the agents' "jump held" input)
  const fl = c.add(0x7b4).readU32();
  if (a[2] && h[2] >= WALL_REPRESS && (fl & 0x200) && (fl & 2)) { h[2] = 0; press(c, V_JUMP, 3, 0); }
  else if (a[2]) { h[2]++; press(c, V_JUMP, h[2] === 1 ? 1 : 2, Math.round(h[2] * 1000 / 60)); }
  else if (h[2]) { h[2] = 0; press(c, V_JUMP, 3, 0); }
}

// the hidden real character sits at the camera position (often an agent on the goal): put it back on the
// spawn before the game moves it again, or the game would count its own finish there
function releaseMain() {
  if (!main) return;
  try { vcall(main, V_RESPAWN); } catch (e) {}
  const sp = levelObj().add(0x1d2c);
  for (const o of [0x84, 0x81c, 0x824]) { main.add(o).writeFloat(sp.readFloat()); main.add(o + 4).writeFloat(sp.add(4).readFloat()); }
}

function resetArrays() {
  status = agents.map(() => 1); held = agents.map(() => [0, 0, 0, 0]); prevJump = agents.map(() => false);
  finishFrame = agents.map(() => -1);
}
function startWave() {
  const want = Math.max(1, Math.min(wantAgents, crowd.length));
  if (want !== agents.length) { agents = crowd.slice(0, want); resetArrays(); leader = 0; }   // monitor slider
  wave++; frame = 0;
  rd(() => updateKillBox('wave ' + wave));         // the view size can change (camera zoom): read it live
  vcall(main, V_RESPAWN);                          // resets the level's moving parts
  agents.forEach((c, i) => {
    const fl = c.add(0x7b4).readU32();
    // with flag 0x10000000 the respawn (FUN_00520150) only moves the character; without it, it also clears
    // the contact list (+0xe0, 40 x 0x28) and the rest of the per-life state. Kept, a contact with the goal
    // from the last wave made agents that finished there finish again on their first frame.
    c.add(0x7b4).writeU32(fl & ~0x10000000);
    vcall(c, V_RESPAWN);
    // respawn moves the character (+0x84) but not where it is drawn: the render position (+0x824, copied
    // from +0x84 at the end of each update) only follows in its next update, so without this the frame drawn
    // right after the reset (and frozen during a PPO update) shows each agent where it died
    c.add(0x824).writeFloat(c.add(0x84).readFloat()); c.add(0x828).writeFloat(c.add(0x88).readFloat());
    c.add(0x9e0).writeU32(2);
    c.add(0x7b4).writeU32(0x50081 | (fl & 0x10000000));
    setState(c, 0);
    status[i] = 0; held[i] = [0, 0, 0, 0]; finishFrame[i] = -1; keysOf[i] = 0;
  });
  send({ ev: 'wave', wave });
}

function agentState(c, i) {
  return [c.add(0x84).readFloat(), c.add(0x88).readFloat(), c.add(0x9c).readFloat(), c.add(0xa0).readFloat(),
          c.add(0x7b4).readU32(), c.add(0x7b0).readU32(), status[i], keysOf[i] || 0];
}

// ---- per-agent keys ---------------------------------------------------------------------------------------
const KEY_R = 22;                                   // touching a key: centre within this of the key piece
let keyPos = [], keyLocks = [], lockWord = new Map(), keysOf = [], lockOpen = new Map();
function onKeys(p) {                                // from Python with the level info: keys and their lock cells
  keyPos = (p && p.keys) || []; keyLocks = (p && p.locks) || [];
  lockWord = new Map(); lockOpen = new Map();
  const a = rd(() => tilemap.readPointer());
  if (a && !a.isNull()) keyLocks.forEach(g => g.forEach(ci => { if (!lockWord.has(ci)) { lockWord.set(ci, a.add(4 * ci).readU32()); lockOpen.set(ci, false); } }));
}
function setLocks(a, mask) {                        // lock tiles as agent `mask` (bit k = has key k) sees them
  for (let k = 0; k < keyLocks.length; k++) {
    const open = ((mask >> k) & 1) === 1;
    for (const ci of keyLocks[k]) {
      if (lockOpen.get(ci) === open) continue;
      a.add(4 * ci).writeU32(open ? 0x100 : lockWord.get(ci));
      lockOpen.set(ci, open);
    }
  }
}
function pickKeys(c, i) {
  if (!keyPos.length) return;
  const x = c.add(0x84).readFloat(), y = c.add(0x88).readFloat();
  for (let k = 0; k < keyPos.length; k++)
    if (Math.hypot(x - keyPos[k][0], y - keyPos[k][1]) < KEY_R) keysOf[i] = (keysOf[i] || 0) | (1 << k);
}

// Long waits on Python. The game measures the time between its frames, so the frame after a long wait
// moves everything by speed x the whole wait: agents jumped hundreds of units (onto the goal) on the first
// frame after a PPO update. Long work (paths, PPO) therefore happens only at the level start and between
// waves, and after any long wait one frame passes with the agents held still before they move again.
let holdFrames = 0, pendingWave = false;
function afterWait(t0) { if (!lockOn && Date.now() - t0 > 50) holdFrames = Math.max(holdFrames, 1); }
function waitGo() {                                // Python's answer may carry a new agent count (next wave)
  const t0 = Date.now();
  recv('go', m => {
    if (m.payload && m.payload.agents) wantAgents = m.payload.agents;
    if (m.payload && m.payload.keys) onKeys(m.payload);
  }).wait();
  afterWait(t0);
}
let actions = null;
function onAct(m) { actions = m.payload; }
// world-model probe (--probe-char N): the first agent's whole character object (CHAR_BYTES, the default
// Meat Boy's allocation size in the factory FUN_00527190) goes with the next N frames, to find the hidden
// timers (jump release, wall-jump lockout) in it
const CHAR_BYTES = 0xa4c;
let probeLeft = CFG.probe_frames || 0;
function exchange(w, gap) {                         // lockstep: send states, wait for Python's inputs
  actions = null;
  const t0 = Date.now();
  const changed = sig(pieces(w)) !== pieceSig;
  if (changed) buildLethal(w);                       // pieces came or went: the hazard list changes
  const msg = { ev: 'frame', wave, frame, s: agents.map(agentState), hz: hazards(), sh: rd(() => shots(w)) || [] };
  const st = stepMs();                               // the game's time step for this update (from its timer)
  msg.dt = st !== null ? +st.toFixed(3) : gap;
  if (gap !== undefined) msg.gap = gap;              // real ms since the last step
  if ((frame & 7) === 0) msg.fx = rd(() => forceFields(w));   // fans / orbs (they rarely move)
  const pf = rd(() => platforms(w));
  if (pf && pf.length) msg.pf = pf;                  // moving platforms: every frame
  if (changed) msg.lt = lethalInfo();
  probeEnemies();
  if (kbPending) { msg.kb = kbPending; kbPending = null; }   // the kill box moved (rare)
  const tc = tileChanges();
  if (tc) msg.tc = tc;
  if (probeLeft > 0 && agents.length && status[0] === 0) {
    probeLeft--;
    send(msg, agents[0].readByteArray(CHAR_BYTES));
  } else send(msg);
  recv('act', onAct).wait();
  afterWait(t0);
  return actions;
}

// A frame that came much later than the last one (the game hitched, or ran slowly in the background) moves
// everything by speed x the whole gap: agents are held still on such a frame instead.
let lastStep = 0, longFrames = 0;
function stepAgents() {
  const w = world();
  if (!w) return;
  const now = Date.now(), gap = now - lastStep;
  lastStep = now;
  if (holdFrames > 0) { holdFrames--; return; }
  if (!lockOn && gap > 80 && gap < 60000 && frame > 0) {   // (locked: a slow frame is still one fixed step)
    if ((longFrames++ % 50) === 0) send({ ev: 'debug', msg: `long frame (${gap} ms): agents held still (${longFrames} so far)` });
    return;
  }
  if (pendingWave || (frame === 0 && wave === 0)) { pendingWave = false; startWave(); }
  const act = exchange(w, gap);
  if (!act) return;
  leader = act.leader;
  if (act.stop) {                                   // hand the characters back to the game
    active = false;
    if (lockOn) setLock(false);
    crowd.forEach(c => c.add(0x9e0).writeU32(1));
    releaseMain();
    send({ ev: 'stopped' });
    return;
  }
  if (act.reset) { endWave('reset'); return; }
  if (act.end) { endWave(String(act.end)); return; }   // Python ends the wave (agent.py: a demo wave is over)
  const mask = w.add(0xdb4), m0 = mask.readU32();
  mask.writeU32(m0 & ~0x18);                        // agents: no bandage / key pickup (row flags 0x08 / 0x40 are
                                                    // gated by mask bit 0x10, FUN_004cd350) and no warps (0x08)
  const ta = keyLocks.length ? rd(() => tilemap.readPointer()) : null;
  try {
    for (let i = 0; i < agents.length; i++) {
      if (status[i] !== 0) continue;
      applyInput(i, act.a[i]);
      vcall(agents[i], 0x58);                       // run/walk acceleration (the update does this only for the player)
      if (ta) setLocks(ta, keysOf[i] || 0);
      curAgent = i;
      const x0 = agents[i].add(0x84).readFloat(), y0 = agents[i].add(0x88).readFloat();
      updateChar(agents[i]);
      curAgent = -1;
      pickKeys(agents[i], i);
      const x1 = agents[i].add(0x84).readFloat(), y1 = agents[i].add(0x88).readFloat();
      // the game moved it back to the spawn: it died some way our death hook does not see (counted as a death)
      if (status[i] === 0 && frame > 2 && Math.hypot(x1 - x0, y1 - y0) > 90 && Math.hypot(x1 - spawnX, y1 - spawnY) < 40) {
        status[i] = 1;
        if ((respawnDbg++ % 200) === 0) send({ ev: 'debug', msg: `agent ${i}: the game put it back on the spawn (a death the hook missed; ${respawnDbg} so far)` });
      }
      if (Math.hypot(x1 - x0, y1 - y0) > 150 && (jumpDbg++ < 40))   // (falling tops out near 50 per frame)
        send({ ev: 'debug', msg: `jump: agent ${i} wave ${wave} frame ${frame}: (${x0.toFixed(0)}, ${y0.toFixed(0)}) -> (${x1.toFixed(0)}, ${y1.toFixed(0)}) status ${status[i]}, frame gap ${gap} ms` });
      if (status[i] === 0 && outOfLevel(x1, y1)) status[i] = 1;
    }
  } finally {
    curAgent = -1; mask.writeU32(m0);
    if (ta) {                                       // locks closed again for drawing and the next frame (also
      for (const [ci, wd] of lockWord) {            // if the game itself started opening one)
        if (ta.add(4 * ci).readU32() !== wd) ta.add(4 * ci).writeU32(wd);
        lockOpen.set(ci, false);
      }
    }
  }
  frame++;
  // the wave ends when everyone is dead or finished, after wave_frames, or once finish_quota agents finished
  const quota = CFG.finish_quota > 0 ? Math.min(CFG.finish_quota, agents.length) : Infinity;
  if (status.filter(s => s === 2).length >= quota) endWave('quota');
  else if (status.every(s => s !== 0) || frame >= CFG.wave_frames) endWave(frame >= CFG.wave_frames ? 'time' : 'done');
}
function outOfLevel(x, y) {
  if (kb) return x < kb[0] || x > kb[1] || y < kb[2] || y > kb[3];
  if (levelOY !== null && y < levelOY - OUT_BOTTOM) return true;
  return levelBox !== null && (x < levelBox[0] - OUT_SIDE || x > levelBox[1] + OUT_SIDE || y > levelBox[2] + OUT_SIDE);
}
function endWave(why) {
  // s / hz / sh: the agents' final states and the hazards and shots now (as in a frame message), so a death
  // on the wave's last update (no frame message shows it) has its exact position. Read failures never stop
  // the wave from ending.
  send({ ev: 'wave_end', wave, frame, why, status: status.slice(), finish: finishFrame.slice(),
         s: rd(() => agents.map(agentState)) || [], hz: rd(hazards) || [], sh: rd(() => shots(world())) || [] });
  waitGo();                                          // Python learns (PPO) now
  frame = 0;
  pendingWave = true;                                // the next wave starts on a following frame
}

// agents: lethal contact -> dead (the game's death code is skipped)
Interceptor.replace(CANDIE, new NativeCallback(function (self) {
  if (curAgent >= 0 && self.equals(agents[curAgent])) { status[curAgent] = 1; return 0; }
  return 1;
}, 'int', ['pointer'], 'thiscall'));

// level end: an agent touching the goal finishes (for itself only)
const LE1 = img(0x4c1910), LE2 = img(0x4a5c50);
let badFinish = 0, jumpDbg = 0, respawnDbg = 0, spawnX = 0, spawnY = 0;
const levelEnd1 = new NativeFunction(LE1, 'void', ['pointer', 'int'], 'thiscall');
const levelEnd2 = new NativeFunction(LE2, 'void', ['pointer', 'int'], 'thiscall');
function goalHook(orig) {
  return new NativeCallback(function (self, arg) {
    if (curAgent >= 0) {
      // a finish counts only next to a goal piece (a stale contact elsewhere must not set a record)
      const c = agents[curAgent], x = c.add(0x84).readFloat(), y = c.add(0x88).readFloat();
      const near = goalPieces.some(([gx, gy]) => Math.hypot(gx - x, gy - y) < 60);
      if (status[curAgent] === 0 && near) { status[curAgent] = 2; finishFrame[curAgent] = frame; }
      else if (!near && (badFinish++ % 50) === 0) send({ ev: 'debug', msg: `ignored a level end for agent ${curAgent} at (${x.toFixed(0)}, ${y.toFixed(0)}), frame ${frame}: no goal near` });
      return;
    }
    orig(self, arg);
    if (finishRequested) send({ ev: 'level_end', level: curName });
  }, 'void', ['pointer', 'int'], 'thiscall');
}
Interceptor.replace(LE1, goalHook(levelEnd1));
Interceptor.replace(LE2, goalHook(levelEnd2));

// ---- main character: hidden + frozen camera; agents step inside its update ------------------------
let camX = 0, camY = 0;
Interceptor.replace(UPD, new NativeCallback(function (self) {
  if (!active) { updateChar(self); return; }
  const key = self.toString();
  if (agentSet.has(key)) return;                    // agents move only through stepAgents (original calls)
  if (main === null && rd(() => self.add(0x9e0).readU32()) === 0) {
    main = self;
    if (!curName) { curName = 'unknown'; send({ ev: 'level', name: curName }); }   // attached mid-level
  }
  if (!main || !self.equals(main)) { updateChar(self); return; }
  const st = rd(() => self.add(0x7b0).readU32());
  if (finishing > 0) {                              // S: walk into the goal through the game's own update
    finishing--;
    // hud+0x384: the level timer (float seconds; the HUD adds each step's dt, 0x4a28ae). The level end copies it
    // and FUN_004ba960 stores it as the level's record only when lower than the stored one. finishTime: the
    // agent's best from Python, or 999.99 (slower than any real run: it never replaces a real record)
    if (hud) hud.add(0x384).writeFloat(finishTime);
    if (goalPieces.length) { self.add(0x84).writeFloat(goalPieces[0][0]); self.add(0x88).writeFloat(goalPieces[0][1]); }
    updateChar(self);
    return;
  }
  if (st === 16 || settle < 30) { if (st !== 16) settle++; updateChar(self); return; }
  if (!ready) {
    try { ready = setupAgents(); } catch (e) { send({ ev: 'error', msg: 'setup: ' + e.message }); ready = false; }
    if (!ready) { updateChar(self); return; }
    return;                                         // the level info wait: the first wave starts on a later frame
  }
  stepAgents();
  const L = agents[leader] || agents[0];
  if (L) { camX = L.add(0x84).readFloat(); camY = L.add(0x88).readFloat(); }
  for (const o of [0x84, 0x81c, 0x824]) { self.add(o).writeFloat(camX); self.add(o + 4).writeFloat(camY); }
}, 'void', ['pointer'], 'thiscall'));

// draw: no main character; every agent with its own animation (leader and finished solid, others faded)
let drawingAgents = false;
Interceptor.replace(DRAW, new NativeCallback(function (drawer, ch, sprite, flag) {
  if (!active || drawingAgents || finishing > 0 || !main || !ch.equals(main)) {
    if (agentSet.has(ch.toString()) && !drawingAgents) return;
    drawChar(drawer, ch, sprite, flag);
    return;
  }
  if (!ready) { drawChar(drawer, ch, sprite, flag); return; }
  if (!sprite.isNull()) return;
  drawingAgents = true;
  try {
    agents.forEach((c, i) => {
      c.add(0xa40).writeU8(i === leader || status[i] === 2 ? 0 : 1);   // leader / finished solid, rest faded
      drawChar(drawer, c, ptr(0), 1);
    });
  } catch (e) { send({ ev: 'error', msg: 'draw: ' + e.message }); }
  finally { drawingAgents = false; }   // must always reset, or no agent is drawn again
}, 'void', ['pointer', 'pointer', 'pointer', 'int'], 'thiscall'));

// ---- reward paths (from Python), drawn into the game as thick lines through Direct3D 9 ---------------------
// The device is the first field of the graphics object [0x709068] (FUN_00627850 creates it). The game draws
// with vertex shaders whose constants c0..c3 hold world -> clip. FUN_004ac4c0 renders the scene layer by
// layer (into separate render targets, composited later); right after each layer's geometry is drawn
// (0x4ac6a9) c0..c3 hold the plain camera matrix, which is kept. At Present the paths are drawn onto the
// back buffer: each world point through that matrix to the screen, then pre-transformed coloured triangles
// (no shaders, no texture), with every device state used saved before and restored after.
let paths = [];                                      // [{pts: [[x, y], ...], col: 0xAARRGGBB, w: px, dash: bool}]
function onPaths(m) { paths = m.payload || []; recv('path', onPaths); }
recv('path', onPaths);
const GFX = img(0x709068);
let dev = null, D = null;
function d3d() {
  const g = rd(() => GFX.readPointer());
  const d = g && !g.isNull() ? rd(() => g.readPointer()) : null;
  if (!d || d.isNull()) return null;
  if (!dev || !dev.equals(d)) {
    dev = d;
    const vt = d.readPointer();
    const V = (i, args) => new NativeFunction(vt.add(4 * i).readPointer(), 'int', ['pointer'].concat(args), 'stdcall');
    D = { getVP: V(48, ['pointer']), setRS: V(57, ['uint', 'uint']), getRS: V(58, ['uint', 'pointer']),
          getTex: V(64, ['uint', 'pointer']), setTex: V(65, ['uint', 'pointer']),
          getTSS: V(66, ['uint', 'uint', 'pointer']), setTSS: V(67, ['uint', 'uint', 'uint']),
          dpup: V(83, ['uint', 'uint', 'pointer', 'uint']),
          setDecl: V(87, ['pointer']), getDecl: V(88, ['pointer']), setFVF: V(89, ['uint']), getFVF: V(90, ['pointer']),
          setVS: V(92, ['pointer']), getVS: V(93, ['pointer']), getC: V(95, ['uint', 'pointer', 'uint']),
          setSS: V(100, ['uint', 'pointer', 'uint', 'uint']), getSS: V(101, ['uint', 'pointer', 'pointer', 'pointer']),
          setPS: V(107, ['pointer']), getPS: V(108, ['pointer']), present: vt.add(4 * 17).readPointer(),
          getBB: V(18, ['uint', 'uint', 'uint', 'pointer']), setRT: V(37, ['uint', 'pointer']), getRT: V(38, ['uint', 'pointer']),
          begin: V(41, []), end: V(42, []), setVP: V(47, ['pointer']) };
  }
  return D;
}
function release(p) { if (p && !p.isNull()) new NativeFunction(p.readPointer().add(8).readPointer(), 'uint', ['pointer'], 'stdcall')(p); }
// render states: z, z write, alpha test, src/dest blend, cull, alpha blend, fog, stencil, lighting,
// colour write (RGB only: the game keeps something in alpha), blend op, scissor, sRGB write, separate alpha
const RS = [[7, 0], [14, 0], [15, 0], [19, 5], [20, 6], [22, 1], [27, 1], [28, 0], [52, 0], [137, 0], [168, 7],
            [171, 1], [174, 0], [194, 0], [206, 0]];
const TSS = [[0, 1, 2], [0, 2, 0], [0, 4, 2], [0, 5, 0], [1, 1, 1], [1, 4, 1]];   // stage 0: diffuse only; stage 1 off
const MAXV = 12000, vbuf = Memory.alloc(MAXV * 20), tmp = Memory.alloc(64), mat = Memory.alloc(64), vpb = Memory.alloc(24);
const save = { rs: Memory.alloc(4 * RS.length), tss: Memory.alloc(4 * TSS.length), p: Memory.alloc(32) };
let nv = 0;
function quad(ax, ay, bx, by, hw, col) {             // a segment as two triangles, hw pixels each side
  if (nv + 6 > MAXV) return;
  let dx = bx - ax, dy = by - ay;
  const L = Math.hypot(dx, dy) || 1;
  const nx = -dy / L * hw, ny = dx / L * hw;
  const v = [[ax + nx, ay + ny], [bx + nx, by + ny], [bx - nx, by - ny], [ax + nx, ay + ny], [bx - nx, by - ny], [ax - nx, ay - ny]];
  for (const [x, y] of v) {
    const o = vbuf.add(nv * 20);
    o.writeFloat(x); o.add(4).writeFloat(y); o.add(8).writeFloat(0); o.add(12).writeFloat(1); o.add(16).writeU32(col);
    nv++;
  }
}
function dot(x, y, r, col) { quad(x - r, y, x + r, y, r, col); }
let camM = null, camFrame = false, pathErr = 0;
const vpSave = Memory.alloc(24), rtp = Memory.alloc(8);
function drawPaths(d, M) {
  d.getVP(dev, vpb);
  const VX = vpb.readU32(), VY = vpb.add(4).readU32(), VW = vpb.add(8).readU32(), VH = vpb.add(12).readU32();
  const scr = ([x, y]) => {
    const w = M[12] * x + M[13] * y + M[15];
    return [VX + ((M[0] * x + M[1] * y + M[3]) / w + 1) * 0.5 * VW, VY + (1 - (M[4] * x + M[5] * y + M[7]) / w) * 0.5 * VH];
  };
  nv = 0;
  for (const p of paths) {                          // dark outline first, then every colour on top
    const s = p.pts.map(scr);
    for (let k = 1; k < s.length; k++) {
      if (p.dash && !(k & 1)) continue;             // dashed: every other segment
      quad(s[k - 1][0], s[k - 1][1], s[k][0], s[k][1], (p.w || 3) + 2, 0x90000000);
    }
  }
  for (const p of paths) {
    const s = p.pts.map(scr), hw = p.w || 3;
    for (let k = 1; k < s.length; k++) {
      if (p.dash && !(k & 1)) continue;
      quad(s[k - 1][0], s[k - 1][1], s[k][0], s[k][1], hw, p.col);
    }
    if (!p.dash || s.length === 1) for (const q of s) dot(q[0], q[1], hw + 1, p.col);
  }
  if (!nv) return;
  // save what is changed
  const P = save.p;   // 0 vs, 4 ps, 8 decl, 12 fvf, 16 tex0, 20 stream vb, 24 stream offset, 28 stream stride
  d.getVS(dev, P); d.getPS(dev, P.add(4)); d.getDecl(dev, P.add(8)); d.getFVF(dev, P.add(12)); d.getTex(dev, 0, P.add(16));
  d.getSS(dev, 0, P.add(20), P.add(24), P.add(28));
  RS.forEach(([k], i) => d.getRS(dev, k, save.rs.add(4 * i)));
  TSS.forEach(([st, k], i) => d.getTSS(dev, st, k, save.tss.add(4 * i)));
  try {
    d.setVS(dev, ptr(0)); d.setPS(dev, ptr(0)); d.setTex(dev, 0, ptr(0)); d.setFVF(dev, 0x44);   // XYZRHW | DIFFUSE
    RS.forEach(([k, v]) => d.setRS(dev, k, v));
    TSS.forEach(([st, k, v]) => d.setTSS(dev, st, k, v));
    d.dpup(dev, 4, nv / 3, vbuf, 20);                // triangle list
  } finally {
    RS.forEach(([k], i) => d.setRS(dev, k, save.rs.add(4 * i).readU32()));
    TSS.forEach(([st, k], i) => d.setTSS(dev, st, k, save.tss.add(4 * i).readU32()));
    const vs = P.readPointer(), ps = P.add(4).readPointer(), decl = P.add(8).readPointer(), tex = P.add(16).readPointer(), vb = P.add(20).readPointer();
    d.setVS(dev, vs); d.setPS(dev, ps);
    if (!decl.isNull()) d.setDecl(dev, decl); else d.setFVF(dev, P.add(12).readU32());
    d.setTex(dev, 0, tex);
    d.setSS(dev, 0, vb, P.add(24).readU32(), P.add(28).readU32());   // DrawPrimitiveUP clears stream 0
    [vs, ps, decl, tex, vb].forEach(release);         // the Get* calls added references
  }
}
Interceptor.attach(img(0x4ac6a9), function () {    // a scene layer was just drawn: c0..c3 = camera
  if (camFrame || !ready || !paths.length) return;
  const d = d3d();
  if (!d) return;
  d.getC(dev, 0, mat, 4);
  const M = Array.from({ length: 16 }, (_, i) => mat.add(4 * i).readFloat());
  if (Math.abs(M[12]) < 1e-3 && Math.abs(M[13]) < 1e-3 && Math.abs(M[14] - 1) < 0.01 && M[15] > 10) { camM = M; camFrame = true; }
});
{ const d0 = d3d(); if (d0) Interceptor.attach(d0.present, { onEnter() {
  const fresh = camFrame;
  camFrame = false;
  if (!active || !ready || !paths.length || !camM || !fresh) return;
  const d = d3d();
  if (!d) return;
  let old = null, bb = null, begun = false;
  try {
    d.getVP(dev, vpSave);
    if (d.getRT(dev, 0, rtp) === 0) old = rtp.readPointer();
    if (d.getBB(dev, 0, 0, 0, rtp) !== 0) return;
    bb = rtp.readPointer();
    d.setRT(dev, 0, bb);                             // also sets the viewport to the whole back buffer
    begun = d.begin(dev) === 0;
    drawPaths(d, camM);
  } catch (e) { if ((pathErr++ % 600) === 0) send({ ev: 'error', msg: 'paths: ' + e.message }); }
  finally {
    if (begun) d.end(dev);
    if (old) d.setRT(dev, 0, old);
    d.setVP(dev, vpSave);
    release(old); release(bb);
  }
} }); }

// ---- HUD: best time on the timer sign; continue past the level-complete screen after S -------------
const TIMER_TEXT = img(0x702d0c);
const fmtBest = Memory.allocUtf16String('BEST %.02f'), fmtNone = Memory.allocUtf16String('BEST --.--');
function onBest(m) { best = m.payload; recv('best', onBest); }
recv('best', onBest);
Interceptor.attach(img(0x45c220), { onEnter(a) {
  if (!ready || !a[0].equals(TIMER_TEXT.readPointer())) return;
  if (best === null) { a[1] = fmtNone; return; }
  a[1] = fmtBest;
  this.context.esp.add(12).writeDouble(best);
} });
let completeFrames = 0, pressA = 0;
Interceptor.attach(img(0x4a1810), { onEnter() {
  hud = this.context.ecx;
  const st = rd(() => hud.add(0x38c).readS32());
  if (st === 13 && finishRequested) {
    if (++completeFrames === 60) { pressA = 8; send({ ev: 'continue', level: curName }); }
  } else completeFrames = 0;
} });
const xinput = ['XINPUT9_1_0.dll', 'xinput1_3.dll', 'XInput1_4.dll'].map(m => rd(() => Process.getModuleByName(m))).find(m => m);
let packet = 1;
if (xinput) Interceptor.attach(xinput.findExportByName('XInputGetState'), {
  onEnter(a) { this.idx = a[0].toInt32(); this.out = a[1]; },
  onLeave(r) {
    if (this.idx !== 0 || pressA <= 0) return;
    pressA--;
    const o = this.out;
    o.writeU32(packet++); o.add(4).writeU16(0x1000);
    for (let q = 6; q < 16; q++) o.add(q).writeU8(0);
    r.replace(ptr(0));
  },
});
let finishTime = 999.99;                            // S: the level time this finish records (payload.time)
function onFinish(m) {
  if (ready && main) {
    const t = m && m.payload ? m.payload.time : null;
    finishTime = typeof t === 'number' && t > 0 && t < 999.99 ? t : 999.99;
    finishRequested = true; finishing = CFG.finish_frames;
  }
  recv('finish', onFinish);
}
recv('finish', onFinish);

const user32 = Process.getModuleByName('user32.dll');
const GetAsyncKeyState = new NativeFunction(user32.findExportByName('GetAsyncKeyState'), 'int16', ['int']);
const GetForegroundWindow = new NativeFunction(user32.findExportByName('GetForegroundWindow'), 'pointer', []);
const GetWindowThreadProcessId = new NativeFunction(user32.findExportByName('GetWindowThreadProcessId'), 'uint32', ['pointer', 'pointer']);
const pidBuf = Memory.alloc(4);
function gameFocused() {
  const h = GetForegroundWindow();
  if (h.isNull()) return false;
  GetWindowThreadProcessId(h, pidBuf);
  return pidBuf.readU32() === Process.id;
}
let sDown = false;
setInterval(() => {
  const down = (GetAsyncKeyState(0x53) & 0x8000) !== 0;
  if (down && !sDown && gameFocused() && ready && !finishRequested) send({ ev: 'save', level: curName });
  sDown = down;
}, 40);
send({ ev: 'ready' });
