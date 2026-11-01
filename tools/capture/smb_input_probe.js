// Frida: scripted pad-input experiments in the running game, frame-synchronous with the player update.
// Measures what the clone cannot infer from free play: input latency and jump-hold behaviour.
// Each trial waits until Meat Boy stands still on the ground, then plays a fixed per-frame button script
// and logs every player frame (buttons applied that frame + state) to OUT as JSON lines.
const OUT = __OUT__;
const base = Process.getModuleByName('SuperMeatBoy.exe').base;
const img = a => base.add(a - 0x400000);

// trial = {name, seq: [button mask per frame], approach?: [{hold: mask, until: "js condition on s"}]}
// approach phases run first (holding `hold` until the condition on the live state s holds), then seq;
// afterwards idle until landed and settled
const trials = __TRIALS__;
for (const t of trials) for (const ph of (t.approach || [])) ph.test = new Function('s', 'return ' + ph.until + ';');
let phase = 0;

let ti = 0, k = -1, settle = 0, buttons = 0, frame = 0, polls = 0, log = [];
Interceptor.attach(img(0x520d30), {
  onEnter() { this.p = this.context.ecx; },
  onLeave() {
    frame++;
    const P = this.p;
    const s = { x: P.add(0x84).readFloat(), y: P.add(0x88).readFloat(), vx: P.add(0x9c).readFloat(), vy: P.add(0xa0).readFloat(), st: P.add(0x7b0).readU32(), fl: P.add(0x7b4).readU32() };
    if (ti >= trials.length) return;
    const t = trials[ti];
    if (k >= 0 || phase > 0) log.push(JSON.stringify({ trial: t.name, k, phase, applied: buttons, polls, ...s }));
    const still = s.st !== 16 && s.vx === 0 && s.vy === 0 && !(s.fl & 2);
    if (k < 0) {                        // waiting to start
      settle = still ? settle + 1 : 0;
      if (settle >= 20) { settle = 0; phase = 1; }
      const ap = t.approach || [];
      while (phase > 0 && phase <= ap.length && ap[phase - 1].test(s)) phase++;
      if (phase > ap.length) { k = 0; phase = 0; }
    } else {
      k++;
      if (k > t.seq.length + 5) {       // after the script: wait until settled, then next trial
        settle = still ? settle + 1 : 0;
        if (settle >= 10 || k > 400) { ti++; k = -1; settle = 0; flush(); }
      }
    }
    const ap = t.approach || [];
    buttons = (k >= 0 && k < t.seq.length) ? t.seq[k]
            : (k < 0 && phase > 0 && phase <= ap.length) ? ap[phase - 1].hold : 0;   // what the next poll reports
  },
});
const xi = Process.getModuleByName('XINPUT9_1_0.dll');
let packet = 1;
Interceptor.attach(xi.findExportByName('XInputGetState'), {
  onEnter(a) { this.i = a[0].toInt32(); this.o = a[1]; },
  onLeave(r) {
    if (this.i !== 0 || ti >= trials.length) return;
    polls++;
    const o = this.o;
    o.writeU32(packet++); o.add(4).writeU16(buttons);
    for (let q = 6; q < 16; q++) o.add(q).writeU8(0);
    if (trials[ti].agentpad) {   // same encoding as ingame_agent.js: left stick for direction, RT with run (X)
      o.add(7).writeU8(buttons & 0x4000 ? 255 : 0);
      o.add(8).writeS16(buttons & 0x4 ? -32767 : buttons & 0x8 ? 32767 : 0);
    }
    r.replace(ptr(0));
  },
});
function flush() {
  if (!log.length) return;
  const f = new File(OUT, 'a'); f.write(log.join('\n') + '\n'); f.close(); log = [];
  send({ done: ti, of: trials.length });
}
send({ ready: trials.length });
