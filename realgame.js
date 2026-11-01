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
  curName = n; main = null; settle = 0; ready = false; agents = []; agentSet = new Set();
  finishing = 0; finishRequested = false; best = null; leader = -1; wave = 0; frame = 0;
  tilemap = null; levelOY = null;   // the new level has its own tile map and pool
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
         agents: agents.length, lethal: lethal.length }, tiles);
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
    vcall(c, V_RESPAWN);
    // respawn moves the character (+0x84) but not where it is drawn: the render position (+0x824, copied
    // from +0x84 at the end of each update) only follows in its next update, so without this the frame drawn
    // right after the reset (and frozen during a PPO update) shows each agent where it died
    c.add(0x824).writeFloat(c.add(0x84).readFloat()); c.add(0x828).writeFloat(c.add(0x88).readFloat());
    c.add(0x9e0).writeU32(2);
    const fl = c.add(0x7b4).readU32();
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

let actions = null;
function onAct(m) { actions = m.payload; }
function exchange(w) {                              // lockstep: send states, wait for Python's inputs
  actions = null;
  send({ ev: 'frame', wave, frame, s: agents.map(agentState), hz: hazards() });
  recv('act', onAct).wait();
  return actions;
}

function stepAgents() {
  const w = world();
  if (!w) return;
  if (pathAll.length && !(markerOwner && markerOwner.equals(main))) ensureMarkers();
  if (frame === 0 && wave === 0) startWave();
  const act = exchange(w);
  if (!act) return;
  leader = act.leader;
  if (act.stop) {                                   // hand the characters back to the game
    active = false;
    agents.forEach(c => c.add(0x9e0).writeU32(1));
    markers.forEach(c => c.add(0x9e0).writeU32(1));
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
      updateChar(agents[i]);
      curAgent = -1;
      if (status[i] === 0 && agents[i].add(0x88).readFloat() < levelOY - 60) status[i] = 1;
    }
  } finally { curAgent = -1; mask.writeU32(m0); }
  frame++;
  if (status.every(s => s !== 0) || frame >= CFG.wave_frames) endWave(frame >= CFG.wave_frames ? 'time' : 'done');
}
function endWave(why) {
  send({ ev: 'wave_end', wave, frame, why, status: status.slice(), finish: finishFrame.slice() });
  frame = 0;
  startWave();
}

// agents: lethal contact -> dead (the game's death code is skipped)
Interceptor.replace(CANDIE, new NativeCallback(function (self) {
  if (curAgent >= 0 && self.equals(agents[curAgent])) { status[curAgent] = 1; return 0; }
  return 1;
}, 'int', ['pointer'], 'thiscall'));

// level end: an agent touching the goal finishes (for itself only)
const LE1 = img(0x4c1910), LE2 = img(0x4a5c50);
const levelEnd1 = new NativeFunction(LE1, 'void', ['pointer', 'int'], 'thiscall');
const levelEnd2 = new NativeFunction(LE2, 'void', ['pointer', 'int'], 'thiscall');
function goalHook(orig) {
  return new NativeCallback(function (self, arg) {
    if (curAgent >= 0) {
      if (status[curAgent] === 0) { status[curAgent] = 2; finishFrame[curAgent] = frame; }
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
    drawPath(drawer);
  } catch (e) { send({ ev: 'error', msg: 'draw: ' + e.message }); }
  finally { drawingAgents = false; }   // must always reset, or no agent is drawn again
}, 'void', ['pointer', 'pointer', 'pointer', 'int'], 'thiscall'));

// reward path (from Python): small faded Meat Boys along it. The markers are the replay-pool characters the
// agents do not use (40 - agents), placed evenly along the path and never updated. (Building more characters
// with the game's factory, FUN_00527190, fails outside the level load and left the game unstable: not used.)
const PATH_SCALE = CFG.path_scale || 0.3, PATH_FADED = CFG.path_faded === false ? 0 : 1;
let pathAll = [], pathPts = [], markers = [], markerOwner = null, pathDbg = 0;
function onPath(m) { pathAll = m.payload || []; markerOwner = null; recv('path', onPath); }
recv('path', onPath);
function ensureMarkers() {                           // called from the update hook
  if (!main || (markerOwner && markerOwner.equals(main))) return;
  markerOwner = main;
  markers = [];
  const pool = rd(() => main.add(0x7c0).readPointer());
  for (let i = 0; pool && i < 40; i++) {
    const c = rd(() => pool.add(4 * i).readPointer());
    if (c && !c.isNull() && !agentSet.has(c.toString())) { setState(c, 0); markers.push(c); }
  }
  const n = pathAll.length, k = Math.min(markers.length, n);
  pathPts = k > 1 ? Array.from({ length: k }, (_, j) => pathAll[Math.round(j * (n - 1) / (k - 1))]) : pathAll.slice(0, k);
  send({ ev: 'debug', msg: `path markers: ${k} (spare pool characters)` });
}
function drawPath(drawer) {
  if (CFG.debug && (pathDbg++ % 300) === 0) send({ ev: 'debug', msg: `path: ${pathPts.length} points, ${markers.length} markers` });
  if (!pathPts.length || !main) return;
  for (let j = 0; j < pathPts.length && j < markers.length; j++) {
    const R = markers[j], [x, y] = pathPts[j];
    R.add(0x84).writeFloat(x); R.add(0x88).writeFloat(y);
    R.add(0x824).writeFloat(x); R.add(0x828).writeFloat(y);
    R.add(0xb4).writeFloat(PATH_SCALE); R.add(0xb8).writeFloat(PATH_SCALE); R.add(0xbc).writeFloat(0.0);
    R.add(0xa40).writeU8(PATH_FADED);
    drawChar(drawer, R, ptr(0), 1);
  }
}

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
