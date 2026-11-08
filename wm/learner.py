"""Train mode's background learner: fine-tunes the world model on the data as it grows and publishes better models.

    python -m wm.learner --init runs/wm_step4/wm_model.pt --out runs/wm_live [--min-shards 20] [--once]

agent.py --learn starts it as a child process (and stops it on exit). Data: every runs/*/wm_data folder with the
time-step lock (the model's own training folders first, in its order, then new ones by name; a folder is locked
when its first shard's world frames all step 16.667 ms), as wm.train: held-out levels never (eval_levels.json),
the user-excluded demos never (plan.demos.USER_EXCLUDED: their replay folders and records), every 5th shard held
out. Demos at --demo-share (0.2): runs/demos/recorded.json, the replay folders' "demo" agents and the agent's own
new best finishes (runs/demos/self.json, written by agent.py --learn).

A cycle runs when at least --min-shards new shards or a new self-demo exist since the last one (--once: one cycle
now, then exit): it fine-tunes the published model (else --init) for --epochs on a --sample-frac share of the
data (at most --max-samples transitions), then rolls it out on a fixed set of held-out waves (wm.compare_steps
metric, chosen once from --init's held-out waves and kept in learner.json) and publishes it as
<out>/wm_model.pt (atomic replace, version number in the model's info "learner") only when its median error at
+16 and +48 frames is not worse than the published model's. Every cycle adds a line to <out>/learner.log.
Below-normal process priority (Windows) and --threads 2, so the game keeps its speed; Ctrl+C exits at once.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

RUNS = os.path.join(HERE, "runs")
LIVE = os.path.join(RUNS, "wm_live")
RECORDED = os.path.join(RUNS, "demos", "recorded.json")
SELF_DEMOS = os.path.join(RUNS, "demos", "self.json")
HORIZONS = (16, 32, 48)
JUDGED = (16, 48)                     # publish only when the median error here is not worse
EVAL_WAVES = 40                       # the fixed evaluation set: this many of --init's held-out waves
EVAL_STARTS = 10                      # start frames per evaluation wave
BELOW_NORMAL = 0x4000                 # Windows BELOW_NORMAL_PRIORITY_CLASS


def _rel(p: str) -> str:
    p = os.path.abspath(p)
    try:
        return os.path.relpath(p, HERE).replace("\\", "/")
    except ValueError:
        return p


def _norm(d: str) -> str:
    return os.path.normcase(os.path.normpath(d if os.path.isabs(d) else os.path.join(HERE, d)))


def atomic_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_json(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def add_self_demo(entry: dict, path: str = SELF_DEMOS) -> int:
    """Appends one self-demo (recorded.json's form: wm_data, shard, agent, frames, level, seconds, run) to the
    list in path (atomic write); returns the list's length."""
    recs = read_json(path, [])
    recs.append(entry)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    atomic_json(path, recs)
    return len(recs)


def locked(d: str) -> bool:
    """The folder's first shard was recorded with the time-step lock (every world frame 16.667 ms)."""
    from plan.demos import world_dts, DT_LOCK, DT_TOL
    from wm.buffer import shard_files
    fs = shard_files(d)
    if not fs:
        return False
    dts = world_dts(fs[0])
    return bool(dts) and all(abs(float(x) - DT_LOCK) <= DT_TOL for x in dts)


def discover(first: list, runs: str = RUNS) -> list:
    """The training folders: `first` (the model's own, in its order, existing ones), then every other locked
    runs/*/wm_data with shards by name; replay folders of user-excluded demos never."""
    from plan.demos import excluded_demo_folder
    from wm.buffer import shard_files
    out, seen, own = [], set(), {_norm(d) for d in first}
    for d in first + sorted(glob.glob(os.path.join(runs, "*", "wm_data"))):
        k = _norm(d)
        if k in seen or not os.path.isdir(k) or excluded_demo_folder(k) or not shard_files(k):
            continue
        seen.add(k)
        if k in own or locked(k):
            out.append(_rel(k))
    return out


def shard_counts(folders: list) -> dict:
    from wm.buffer import shard_files
    return {d: len(shard_files(_norm(d))) for d in folders}


def due(state: dict, counts: dict, n_self: int, min_shards: int) -> tuple[bool, int, int]:
    """(a cycle is due, new shards, new self-demos) since the last cycle."""
    seen = state.get("shards", {})
    new = sum(max(0, n - seen.get(d, 0)) for d, n in counts.items())
    new_self = max(0, n_self - int(state.get("self_demos", 0)))
    return (new >= min_shards or new_self > 0), new, new_self


def eval_waves(model, n: int = EVAL_WAVES) -> list:
    """n of the model's held-out waves (wm.compare_steps.held_out), evenly spread, held-out levels left out."""
    from wm.buffer import eval_levels, shard_files
    from wm.compare_steps import held_out
    held_lv = eval_levels()
    _, held = held_out(model)
    waves = []
    for d, k in sorted(held):
        fs = shard_files(d)
        if k < len(fs):
            meta = json.loads(str(np.load(fs[k])["meta"]))
            if meta.get("level") not in held_lv:
                waves.append([_rel(d), int(k)])
    if len(waves) > n:
        waves = [waves[i] for i in np.linspace(0, len(waves) - 1, n).round().astype(int)]
    return waves


def errors(model, waves: list) -> dict:
    """{horizon: [median, 90%, n]} of the open-loop position error on the waves (wm.compare_steps.wave_errors)."""
    from wm.compare_steps import wave_errors
    err = wave_errors([model], [(_norm(d), k) for d, k in waves], HORIZONS, EVAL_STARTS)[0]
    return {str(h): [round(float(np.median(e)), 3), round(float(np.percentile(e, 90)), 3), len(e)] if len(e)
            else [None, None, 0] for h, e in err.items()}


def not_worse(cand: dict, pub: dict) -> bool:
    """The candidate's median error at every JUDGED horizon is not above the published model's."""
    for h in JUDGED:
        c, p = cand.get(str(h), [None])[0], pub.get(str(h), [None])[0]
        if c is None or (p is not None and c > p):
            return False
    return True


def fine_tune(model, folders: list, demo_paths: list, args, log=print):
    """One fine-tuning pass of `model` (in place) on the folders' training shards; returns transitions used."""
    from wm.buffer import eval_levels
    from wm.features import build, RES_SCALE
    from wm.train import all_demos, demo_oversample, fit
    t0 = time.time()
    d = build(folders, max_samples=args.max_samples, log=lambda *a: None, seed=args.seed, frac=args.sample_frac,
              exclude=eval_levels(), step=model.step, demos=all_demos(demo_paths, folders))
    if model.step > 1:                       # the targets in the model's own scale
        d["yc"] = (d["yc"] * RES_SCALE / model.res_scale).astype(np.float32)
        d["wc"] = d["full"].astype(np.float32)
    val = d["shard"] % 5 == 0
    tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    fit_idx, real = demo_oversample(tr_idx, d["demo"], args.demo_share)
    log(f"[learner] {len(d['X'])} transitions ({len(tr_idx)} train), demos {real:.1%} of the samples "
        f"[{time.time() - t0:.0f}s]")
    ts = {k: d[k][fit_idx] for k in ("X", "yc", "yb", "ys", "yr") + (("wc",) if "wc" in d else ())}
    model.train()
    fit(model, ts, d, va_idx if len(va_idx) else tr_idx[:1000], args.epochs, args.lr,
        max(1, min(args.batch, len(fit_idx))), t0)
    model.eval()
    return len(d["X"])


def publish(model, info: dict, out: str) -> str:
    """Writes the model to <out>/wm_model.pt by an atomic replace."""
    path = os.path.join(out, "wm_model.pt")
    tmp = path + ".tmp"
    model.save(tmp, info)
    os.replace(tmp, path)
    return path


def log_line(out: str, text: str):
    with open(os.path.join(out, "learner.log"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
    print(text, flush=True)


def fmt(e: dict) -> str:
    return ", ".join(f"+{h} {e[str(h)][0]}/{e[str(h)][1]}" for h in HORIZONS if str(h) in e)


def cycle(args, state: dict, folders: list, counts: dict, n_self: int) -> dict:
    """Fine-tune, evaluate, publish when not worse; returns the new state (written by the caller)."""
    from wm.model import WorldModel
    t0 = time.time()
    base = _norm(state["base"])
    model = WorldModel.load(base)
    demo_paths = [p for p in (args.demos, args.self_demos) if p and os.path.exists(p)]
    n = fine_tune(model, folders, demo_paths, args)
    cand = errors(model, state["eval_waves"])
    pub = state["errors"]
    ok = not_worse(cand, pub)
    v = int(state.get("version", 0))
    new = dict(state, shards=counts, self_demos=n_self, cycles=int(state.get("cycles", 0)) + 1)
    if ok:
        v += 1
        info = dict(model.info or {}, data=folders,
                    learner={"version": v, "parent": _rel(base), "errors": cand, "parent_errors": pub,
                             "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        new.update(version=v, errors=cand, base=_rel(publish(model, info, args.out)))
    log_line(args.out, f"cycle {new['cycles']}: {'published v' + str(v) if ok else 'kept v' + str(v)} "
                       f"({time.time() - t0:.0f} s, {n} transitions, {len(folders)} folders, {n_self} self-demos); "
                       f"error median/90% candidate {fmt(cand)}; published {fmt(pub)}")
    return new


def set_low_priority():
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), BELOW_NORMAL)
        except (OSError, AttributeError):
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="the model to start from (when --out has no newer published one)")
    ap.add_argument("--out", default=LIVE)
    ap.add_argument("--min-shards", type=int, default=20, help="new shards that start a cycle (or a new self-demo)")
    ap.add_argument("--poll", type=float, default=30.0, help="seconds between checks for new data")
    ap.add_argument("--once", action="store_true", help="one cycle now, then exit")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--sample-frac", type=float, default=0.07)
    ap.add_argument("--max-samples", type=int, default=600000)
    ap.add_argument("--demos", default=RECORDED, help="recorded.json (plan.demos recorded)")
    ap.add_argument("--self-demos", default=SELF_DEMOS, help="the agent's own new bests (agent.py --learn)")
    ap.add_argument("--demo-share", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    from shutdown import immediate
    immediate()
    set_low_priority()
    import torch
    torch.set_num_threads(max(1, args.threads))
    from wm.model import WorldModel
    os.makedirs(args.out, exist_ok=True)
    spath = os.path.join(args.out, "learner.json")
    state = read_json(spath, {})
    pub_path = os.path.join(args.out, "wm_model.pt")
    same_init = state.get("init") == _rel(args.init) and state.get("base") == state.get("init")
    newer_pub = os.path.exists(pub_path) and os.path.getmtime(pub_path) >= os.path.getmtime(args.init)
    keep = bool(state.get("base") and state.get("eval_waves") and state.get("errors")
                and os.path.exists(_norm(state["base"])) and (same_init or newer_pub))
    if not keep:                             # start (again) from --init: its held-out waves and errors; the
        init = WorldModel.load(args.init).eval()     # version number keeps counting
        state = {"init": _rel(args.init), "base": _rel(args.init), "version": int(state.get("version", 0)),
                 "eval_waves": eval_waves(init),
                 "shards": shard_counts([_rel(_norm(d)) for d in init.info.get("data", [])])}
        state["errors"] = errors(init, state["eval_waves"])
        atomic_json(spath, state)
        log_line(args.out, f"start from {_rel(args.init)}: {len(state['eval_waves'])} evaluation waves, error "
                           f"median/90% {fmt(state['errors'])}")
    first = [_rel(_norm(d)) for d in WorldModel.load(_norm(state["base"])).info.get("data", [])]
    while True:
        folders = discover(first)
        counts = shard_counts(folders)
        n_self = len(read_json(args.self_demos, []))
        go, new, new_self = due(state, counts, n_self, args.min_shards)
        if go or args.once:
            print(f"[learner] cycle: {new} new shards, {new_self} new self-demos", flush=True)
            state = cycle(args, state, folders, counts, n_self)
            atomic_json(spath, state)
            first = folders
        if args.once:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
