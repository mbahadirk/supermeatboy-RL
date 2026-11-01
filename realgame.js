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
//          agent's finish. Keys, bandages and warp portals are off for agents (world+0xdb4 bits 0x8/0x10)
//          until per-agent keys are done.
//   the real character: hidden and frozen, used as the camera (parked on the leading agent) and for the
//          level reset; S (game focused) saves the best agent and finishes the level the game's way.
//   paths: the reward paths from Python are drawn over the picture through Direct3D 9 (see below).
//   time:  the game's time step includes any wait on Python, so long work (paths, PPO) only happens at
//          the level start and between waves, and agents are held still for the frame after a long wait.
//
// Addresses are image VAs (base 0x400000), resolved against the ASLR base at runtime.
const CFG = __CONFIG__;   // {agents, wave_frames, finish_frames}
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

let active = true;
rpc.exports = { shutdown() { if (active) { active = false; releaseMain(); } return true; } };   // only when no frame waits

// ---- level / main character -------------------------------------------------------------------
let curName = null, main = null, settle = 0, agents = [], agentSet = new Set(), ready = false;
let wave = 0, frame = 0, status = [], held = [], prevJump = [], finishFrame = [];
let curAgent = -1, leader = -1, best = null, finishing = 0, finishRequested = false, hud = null;
let goalPieces = [];

function world() { return rd(() => WORLD.readPointer()); }
function levelObj() { return rd(() => GAME.readPointer().add(0x24).readPointer()); }

Interceptor.attach(img(0x4e9020), { onEnter(a) {
  const n = rd(() => a[0].readPointer().readCString());
  if (!n) return;
  curName = n; main = null; settle = 0; ready = false; agents = []; agentSet = new Set(); pendingWave = false; holdFrames = 0;
  finishing = 0; finishRequested = false; best = null; leader = -1; wave = 0; frame = 0;
  tilemap = null; levelOY = null; paths = [];   // the new level has its own tile map, pool and paths
  send({ ev: 'level', name: n });
} });


function setupAgents() {
  const pool = rd(() => main.add(0x7c0).readPointer());
  if (!pool || pool.isNull()) return false;
  const read = () => {
    const out = [];
    for (let i = 0; i < 40 && out.length < CFG.agents; i++) {
      const c = rd(() => pool.add(4 * i).readPointer());
      if (c && !c.isNull()) out.push(c);
    }
    return out;
  };
  agents = read();
  // (no pool yet: wait for the game to build it; building it ourselves calls the character factory outside
  // the level load, which is not safe)
  if (!agents.length) return false;
  agentSet = new Set(agents.map(c => c.toString()));
  agents.forEach(c => c.add(0x9e0).writeU32(2));
  status = agents.map(() => 1); held = agents.map(() => [0, 0, 0, 0]); prevJump = agents.map(() => false);
  finishFrame = agents.map(() => -1);
  return sendLevelInfo();
}

// static level facts from the game's memory: tile layer 0, origin, size; goal pieces (type flag 0x02)
let tilemap = null, levelOY = null;
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
  const tiles = G.readPointer().readByteArray(W * H * 4);
  goalPieces = [];
  for (const pc of pieces(w)) {
    const t = pc.add(0x10).readU16() & 0xff, fl = typeFlags(w, t);
    if (fl !== null && (fl & 2)) goalPieces.push([pc.add(0x14).readFloat(), pc.add(0x18).readFloat()]);
  }
  const sp = levelObj().add(0x1d2c);
  buildLethal(w);
  send({ ev: 'info', name: curName, W, H, ox, oy, goals: goalPieces, spawn: [sp.readFloat(), sp.add(4).readFloat()],
         agents: agents.length, lethal: lethal.length, hz: hazards() }, tiles);
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
  lethal = [];
  for (const pc of pieces(w)) {
    const t = pc.add(0x10).readU16() & 0xff;
    if (skip.has(t)) continue;
    const fl = typeFlags(w, t);
    if (fl === null || (fl & 1)) lethal.push(pc);
  }
  lethalSize = lethal.map(pc => pieceSize(w, pc));
}
function hazards() {                               // [x, y, size w, size h, scale] per lethal piece, now
  return lethal.map((pc, k) => [pc.add(0x14).readFloat(), pc.add(0x18).readFloat(), lethalSize[k][0],
                                lethalSize[k][1], pc.add(0x2c).readFloat()]);
}

// ---- inputs through the game's own handlers: thiscall(char, {state 1 press / 2 held / 3 release, ms}) -------
const inBuf = Memory.alloc(8);
function press(c, off, state, ms) { inBuf.writeU32(state); inBuf.add(4).writeU32(ms); vcall(c, off, inBuf); }
function applyInput(i, a) {                        // a = [left, right, jump, run]
  const c = agents[i], h = held[i];
  [[0, V_LEFT], [1, V_RIGHT], [3, V_RUN]].forEach(([k, off]) => {
    if (a[k]) { h[k]++; press(c, off, h[k] === 1 ? 1 : 2, Math.round(h[k] * 1000 / 60)); }
    else if (h[k]) { h[k] = 0; press(c, off, 3, 0); }
  });
  if (a[2]) { h[2]++; press(c, V_JUMP, h[2] === 1 ? 1 : 2, Math.round(h[2] * 1000 / 60)); }
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

function startWave() {
  wave++; frame = 0;
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
    status[i] = 0; held[i] = [0, 0, 0, 0]; finishFrame[i] = -1;
  });
  send({ ev: 'wave', wave });
}

function agentState(c, i) {
  return [c.add(0x84).readFloat(), c.add(0x88).readFloat(), c.add(0x9c).readFloat(), c.add(0xa0).readFloat(),
          c.add(0x7b4).readU32(), c.add(0x7b0).readU32(), status[i]];
}

// Long waits on Python. The game measures the time between its frames, so the frame after a long wait
// moves everything by speed x the whole wait: agents jumped hundreds of units (onto the goal) on the first
// frame after a PPO update. Long work (paths, PPO) therefore happens only at the level start and between
// waves, and after any long wait one frame passes with the agents held still before they move again.
let holdFrames = 0, pendingWave = false;
function afterWait(t0) { if (Date.now() - t0 > 50) holdFrames = Math.max(holdFrames, 1); }
function waitGo() { const t0 = Date.now(); recv('go', () => {}).wait(); afterWait(t0); }
let actions = null;
function onAct(m) { actions = m.payload; }
function exchange(w) {                              // lockstep: send states, wait for Python's inputs
  actions = null;
  const t0 = Date.now();
  send({ ev: 'frame', wave, frame, s: agents.map(agentState), hz: hazards() });
  recv('act', onAct).wait();
  afterWait(t0);
  return actions;
}

function stepAgents() {
  const w = world();
  if (!w) return;
  if (holdFrames > 0) { holdFrames--; return; }
  if (pendingWave || (frame === 0 && wave === 0)) { pendingWave = false; startWave(); }
  const act = exchange(w);
  if (!act) return;
  leader = act.leader;
  if (act.stop) {                                   // hand the characters back to the game
    active = false;
    agents.forEach(c => c.add(0x9e0).writeU32(1));
    releaseMain();
    send({ ev: 'stopped' });
    return;
  }
  if (act.reset) { endWave('reset'); return; }
  const mask = w.add(0xdb4), m0 = mask.readU32();
  mask.writeU32(m0 & ~0x18);                        // no keys / bandages / warps for agents yet
  try {
    for (let i = 0; i < agents.length; i++) {
      if (status[i] !== 0) continue;
      applyInput(i, act.a[i]);
      vcall(agents[i], 0x58);                       // run/walk acceleration (the update does this only for the player)
      curAgent = i;
      const x0 = agents[i].add(0x84).readFloat(), y0 = agents[i].add(0x88).readFloat();
      updateChar(agents[i]);
      curAgent = -1;
      if (frame < 30) {
        const x1 = agents[i].add(0x84).readFloat(), y1 = agents[i].add(0x88).readFloat();
        if (Math.hypot(x1 - x0, y1 - y0) > 150 && (jumpDbg++ < 40))   // (falling tops out near 50 per frame)
          send({ ev: 'debug', msg: `jump: agent ${i} wave ${wave} frame ${frame}: (${x0.toFixed(0)}, ${y0.toFixed(0)}) -> (${x1.toFixed(0)}, ${y1.toFixed(0)}) status ${status[i]}` });
      }
      if (status[i] === 0 && agents[i].add(0x88).readFloat() < levelOY - 60) status[i] = 1;
    }
  } finally { curAgent = -1; mask.writeU32(m0); }
  frame++;
  if (status.every(s => s !== 0) || frame >= CFG.wave_frames) endWave(frame >= CFG.wave_frames ? 'time' : 'done');
}
function endWave(why) {
  send({ ev: 'wave_end', wave, frame, why, status: status.slice(), finish: finishFrame.slice() });
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
let badFinish = 0, jumpDbg = 0;
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
    if (hud) hud.add(0x384).writeFloat(999.99);     // keep this forced finish out of the player's records
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
let paths = [];                                      // [{pts: [[x, y], ...], col: 0xAARRGGBB, w: px}]
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
    for (let k = 1; k < s.length; k++) quad(s[k - 1][0], s[k - 1][1], s[k][0], s[k][1], (p.w || 3) + 2, 0x90000000);
  }
  for (const p of paths) {
    const s = p.pts.map(scr), hw = p.w || 3;
    for (let k = 1; k < s.length; k++) quad(s[k - 1][0], s[k - 1][1], s[k][0], s[k][1], hw, p.col);
    for (const q of s) dot(q[0], q[1], hw + 1, p.col);
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
function onFinish() { if (ready && main) { finishRequested = true; finishing = CFG.finish_frames; } recv('finish', onFinish); }
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
