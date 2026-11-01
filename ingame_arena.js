// Frida: "arena" mod for the real Super Meat Boy (personal use). Trained agents play every level as
// ghosts; there is no controllable main Meat Boy.
//
// Python (ingame.py) simulates N agents in the calibrated clone at 60 Hz and posts their positions; this
// script draws each one with the game's own character renderer, points the camera at the leading agent,
// shows the best finishing time on the game's timer sign, and on S (game window focused) asks Python to
// save the best agent, then finishes the level through the game's own flow so it loads the next one.
//
// Reverse-engineered hooks (image VAs, base 0x400000; ASLR resolved at runtime):
//   FUN_004e9020  level load, arg0 -> {char *name}
//   FUN_00519d40  character update (vtable slot 7: FUN_0051f560, FUN_00520d30, FUN_00517150), ecx = char
//   FUN_0051fae0  character draw, thiscall(drawer, char, sprite = 0, skipPost): the sprite is queued with a
//                 *pointer* to char+0x824 (render position, via FUN_00458f60), read at frame flush, so each
//                 ghost needs its own persistent char copy. char+0xa40 != 0 draws at 0.35 alpha.
//   char +0x84/+0x88 position (centre), +0x81c/+0x824 previous/render position, +0x7b0 state (16 = intro)
//   FUN_004a1810  HUD draw, ecx = hud; hud+0x38c state (11 playing, 13 level complete), hud+0x384 time
//   FUN_0045c220  set formatted text(textObject, wchar fmt, ...); [0x702d0c] = the timer sign's text
//   FUN_004c1910 / FUN_004a5c50  level end (goal reached)
// Camera: follows the (hidden) character's position, clamped to the level by the game.
// Level change: teleport the hidden character onto the goal and run its real update -> the game's level
// end -> level-complete screen (hud state 13) -> one "continue" press -> the game loads the next level.
const CFG = __CONFIG__;   // {levels, finish_frames}
const base = Process.getModuleByName('SuperMeatBoy.exe').base;
const img = a => base.add(a - 0x400000);
const rd = fn => { try { return fn(); } catch (e) { return null; } };
const LV = JSON.parse(File.readAllText(CFG.levels)).levels;

const UPD = img(0x519d40), DRAW = img(0x51fae0);
const updateChar = new NativeFunction(UPD, 'void', ['pointer'], 'thiscall');
const drawChar = new NativeFunction(DRAW, 'void', ['pointer', 'pointer', 'pointer', 'int'], 'thiscall');

let curName = null, cur = null, real = null, frames = 0;
let active = true;   // false after shutdown(): the replaced functions behave exactly like the originals
rpc.exports = { shutdown() { active = false; ghosts = []; return true; } };
let ghosts = [], ghostLevel = null, cam = null, best = null, leader = -1, rail = [], railMem = [];   // from Python
let settle = 0;   // live frames of the hidden character after the intro, before it is frozen
let finishing = 0, finishRequested = false, hud = null, hudState = -1, completeFrames = 0, pressA = 0;

function setLevel(name) {
  curName = name; cur = LV[name] || null; real = null; finishing = 0; finishRequested = false; settle = 0;
  ghosts = []; cam = null; best = null; leader = -1;
  send({ ev: 'level', name, known: !!cur });
}
Interceptor.attach(img(0x4e9020), { onEnter(a) {
  const n = rd(() => a[0].readPointer().readCString());
  if (n) setLevel(n);
} });
// attached mid-level: identify it by its tilemap (size + origin) via FUN_004657e0 / FUN_00465b00
const tmHooks = [];
function tilemapProbe() {
  if (curName) return;
  const G = this.context.ecx;
  const W = rd(() => G.add(0x20).readU16()), H = rd(() => G.add(0x22).readU16());
  const ox = rd(() => G.add(0x18).readFloat()), oy = rd(() => G.add(0x1c).readFloat());
  for (const [name, l] of Object.entries(LV))
    if (l.cols === W && l.rows === H && l.origin[0] === ox && l.origin[1] === oy) { setLevel(name); break; }
}
for (const a of [0x4657e0, 0x465b00]) tmHooks.push(Interceptor.attach(img(a), { onEnter: tilemapProbe }));
setTimeout(() => tmHooks.forEach(h => h.detach()), 10000);

for (const a of [0x4c1910, 0x4a5c50]) Interceptor.attach(img(a), { onEnter() {
  if (finishRequested) send({ ev: 'level_end', level: curName });
} });

// messages from Python
function onGhosts(m) { const p = m.payload; ghosts = p.g; ghostLevel = p.level; cam = p.cam; best = p.best; leader = p.leader; rail = p.rail || []; recv('ghosts', onGhosts); }
recv('ghosts', onGhosts);
function onFinish() { if (cur && real) { finishRequested = true; finishing = CFG.finish_frames; } recv('finish', onFinish); }
recv('finish', onFinish);

// the hidden "camera" character: frozen (no physics, no deaths, no finish), parked on the leading agent
Interceptor.replace(UPD, new NativeCallback(function (self) {
  if (!active) { updateChar(self); return; }
  if (real === null && cur) real = self;
  if (!real || !self.equals(real)) { updateChar(self); return; }
  frames++;
  const st = rd(() => self.add(0x7b0).readU32());
  if (finishing > 0) {                      // walk into the goal through the game's own update
    finishing--;
    // the game records the level timer (hud+0x384) as this finish's time: keep it out of the player's
    // records and leaderboards by making it slower than any real run
    if (hud) hud.add(0x384).writeFloat(999.99);
    self.add(0x84).writeFloat(cur.goal_xy[0]); self.add(0x88).writeFloat(cur.goal_xy[1]);
    updateChar(self);
  } else if (st === 16 || settle < 45) {
    // let the level intro and the spawn animation finish (standing at the spawn, no input) before
    // freezing: the agents are drawn from this character's current animation frame
    if (st !== 16) settle++;
    updateChar(self);
  } else if (cam && ghostLevel === curName) {
    for (const o of [0x84, 0x81c, 0x824]) { self.add(o).writeFloat(cam[0]); self.add(o + 4).writeFloat(cam[1]); }
  }
  if (frames % 30 === 0) updateTitle();
}, 'void', ['pointer'], 'thiscall'));

// draw: skip the hidden character, draw every agent from a persistent copy of it instead
const CHAR_SIZE = 0xc00, ghostMem = [];
const dbg = { drawAll: 0, drawReal: 0, ghostBatches: 0 };
if (CFG.debug) setInterval(() => {
  send({ ev: 'debug', ...dbg, cur: curName, ghostLevel, ghosts: ghosts.length, settle, finishing,
         real: real && real.toString(), st: real && rd(() => real.add(0x7b0).readU32()),
         f9e0: real && rd(() => real.add(0x9e0).readU32()) });
  dbg.drawAll = dbg.drawReal = dbg.ghostBatches = 0;
}, 2000);
Interceptor.replace(DRAW, new NativeCallback(function (drawer, ch, sprite, flag) {
  dbg.drawAll++; if (real && ch.equals(real)) dbg.drawReal++;
  if (!active || !real || !ch.equals(real) || finishing > 0) { drawChar(drawer, ch, sprite, flag); return; }
  if (!sprite.isNull() || ghostLevel !== curName) return;
  dbg.ghostBatches++;
  try {
    for (let i = 0; i < ghosts.length; i++) {
      if (!ghostMem[i]) ghostMem[i] = Memory.alloc(CHAR_SIZE);
      const G = ghostMem[i], [gx, gy, right] = ghosts[i];
      Memory.copy(G, ch, CHAR_SIZE);
      for (const o of [0x824, 0x81c]) { G.add(o).writeFloat(gx); G.add(o + 4).writeFloat(gy); }
      // the hidden character may be frozen in its spawn animation or in the "ready" state (16) that
      // lasts until the first button press and is not drawn: draw the copies standing, full scale
      G.add(0xb4).writeFloat(1.0); G.add(0xb8).writeFloat(1.0); G.add(0xbc).writeFloat(0.0);
      if (G.add(0x7b0).readU32() === 16) G.add(0x7b0).writeU32(0);
      const fl = G.add(0x7b4).readU32();
      G.add(0x7b4).writeU32(right ? (fl & ~1) : (fl | 1));       // bit0 = facing left
      G.add(0xa40).writeU8(i === leader ? 0 : 1);                  // leader solid, others translucent
      drawChar(drawer, G, ptr(0), 1);
    }
    for (let j = 0; j < rail.length; j++) {
      if (!railMem[j]) railMem[j] = Memory.alloc(CHAR_SIZE);
      const R = railMem[j], [rx, ry] = rail[j];
      Memory.copy(R, ch, CHAR_SIZE);
      for (const o of [0x824, 0x81c]) { R.add(o).writeFloat(rx); R.add(o + 4).writeFloat(ry); }
      R.add(0xb4).writeFloat(0.35); R.add(0xb8).writeFloat(0.35); R.add(0xbc).writeFloat(0.0);
      if (R.add(0x7b0).readU32() === 16) R.add(0x7b0).writeU32(0);
      R.add(0xa40).writeU8(1);
      drawChar(drawer, R, ptr(0), 1);
    }
  } catch (e) { send({ ev: 'ghost_error', msg: e.message }); ghosts = []; }
}, 'void', ['pointer', 'pointer', 'pointer', 'int'], 'thiscall'));

// HUD: the timer sign shows the best finishing time; after a requested finish, continue past the
// level-complete screen once so the game loads the next level
const TIMER_TEXT = img(0x702d0c);
const fmtBest = Memory.allocUtf16String('BEST %.02f'), fmtNone = Memory.allocUtf16String('BEST --.--');
Interceptor.attach(img(0x45c220), { onEnter(a) {
  if (!cur || !a[0].equals(TIMER_TEXT.readPointer())) return;
  if (best === null) { a[1] = fmtNone; return; }
  a[1] = fmtBest;
  this.context.esp.add(12).writeDouble(best);   // cdecl varargs: the double after (text, fmt)
} });
Interceptor.attach(img(0x4a1810), { onEnter() {
  hud = this.context.ecx;
  const st = rd(() => hud.add(0x38c).readS32());
  if (st !== hudState) hudState = st;
  if (st === 13 && finishRequested) {
    if (++completeFrames === 60) { pressA = 8; send({ ev: 'continue', level: curName }); }
  } else completeFrames = 0;
} });
const xinput = ['XINPUT9_1_0.dll', 'xinput1_3.dll', 'XInput1_4.dll'].map(m => rd(() => Process.getModuleByName(m))).find(m => m);
let packet = 1;
Interceptor.attach(xinput.findExportByName('XInputGetState'), {
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

// S (game window focused): save the best agent and move on
const user32 = Process.getModuleByName('user32.dll');
const GetAsyncKeyState = new NativeFunction(user32.findExportByName('GetAsyncKeyState'), 'int16', ['int']);
const GetForegroundWindow = new NativeFunction(user32.findExportByName('GetForegroundWindow'), 'pointer', []);
const GetWindowThreadProcessId = new NativeFunction(user32.findExportByName('GetWindowThreadProcessId'), 'uint32', ['pointer', 'pointer']);
const SetWindowTextW = new NativeFunction(user32.findExportByName('SetWindowTextW'), 'int', ['pointer', 'pointer']);
const GetCurrentThreadId = new NativeFunction(Process.getModuleByName('kernel32.dll').findExportByName('GetCurrentThreadId'), 'uint32', []);
const pidBuf = Memory.alloc(4);
function gameWindow() {
  const h = GetForegroundWindow();
  if (h.isNull()) return null;
  const tid = GetWindowThreadProcessId(h, pidBuf);
  return pidBuf.readU32() === Process.id ? { h, tid } : null;
}
let sDown = false;
setInterval(() => {
  const down = (GetAsyncKeyState(0x53) & 0x8000) !== 0;
  if (down && !sDown && gameWindow() && cur && !finishRequested) send({ ev: 'save', level: curName });
  sDown = down;
}, 40);

let lastTitle = '';
function updateTitle() {
  const w = gameWindow();   // only from the window's own thread (a cross-thread WM_SETTEXT could deadlock)
  if (!w || GetCurrentThreadId() !== w.tid) return;
  const t = `Super Meat Boy | ARENA ${curName || ''} | ${ghosts.length} agents | ` +
            (best === null ? 'no finish yet' : `best ${best.toFixed(2)}s`) + ' | S: save best agent + next level';
  if (t !== lastTitle) { lastTitle = t; SetWindowTextW(w.h, Memory.allocUtf16String(t)); }
}
send({ ev: 'ready' });
