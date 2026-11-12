"""Train mode's background learner: fine-tunes the world model on the data as it grows and publishes better models.

    python -m wm.learner --help

Choose --manifest FILE or explicitly --legacy-split. Strict mode requires a checkpoint from the SAME
trace-split-v1 manifest. It runs one initial cycle, then waits; --once explicitly requests one cycle.
New files remain pending outside the snapshot and never trigger training. A new/extended manifest is a new
experiment requiring fresh training; extension fine-tuning is not supported. Validation entry IDs are frozen
in learner.json (at most 40 entries), with per-agent membership supplied by the manifest.

agent.py --learn requires --dataset-manifest FILE or --legacy-split and forwards the choice. The following
automatic data-discovery description applies ONLY to explicit legacy mode: every runs/*/wm_data folder with the
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


def eval_waves(model, n: int = EVAL_WAVES, dataset=None) -> list:
    """n of the model's held-out waves (wm.compare_steps.held_out), evenly spread, held-out levels left out."""
    if dataset is not None:
        dataset.require_model(model)
        waves = [e["id"] for e in dataset.entries if any(a["split"] == "validation" for a in e["agents"].values())]
        if not waves or not any(a["split"] == "train" for e in dataset.entries for a in e["agents"].values()):
            raise ValueError("manifest needs both training and validation traces")
        if len(waves) > n:
            waves = [waves[i] for i in np.linspace(0, len(waves) - 1, n).round().astype(int)]
        return waves
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


def errors(model, waves: list, dataset=None) -> dict:
    """{horizon: [median, 90%, n]} of the open-loop position error on the waves (wm.compare_steps.wave_errors)."""
    from wm.compare_steps import wave_errors
    err = wave_errors([model], waves, HORIZONS, EVAL_STARTS, dataset=dataset)[0] if dataset is not None else \
        wave_errors([model], [(_norm(d), k) for d, k in waves], HORIZONS, EVAL_STARTS)[0]
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
    from wm.features import build, model_layout, model_prior, RES_SCALE
    from wm.train import all_demos, demo_oversample, fit
    t0 = time.time()
    from wm.dataset import sample_validation
    dataset = getattr(args, "dataset", None)
    d = build(folders, max_samples=args.max_samples, log=lambda *a: None, seed=args.seed, frac=args.sample_frac,
              exclude=eval_levels(), step=model.step, demos=all_demos(demo_paths, folders), dataset=dataset,
              protocol=model_layout(model).protocol, prior_protocol=model_prior(model))
    if model.step > 1:                       # the targets in the model's own scale
        d["yc"] = (d["yc"] * RES_SCALE / model.res_scale).astype(np.float32)
        d["wc"] = d["full"].astype(np.float32)
    val = sample_validation(d, strict=True) if dataset is not None else d["shard"] % 5 == 0
    tr_idx, va_idx = np.nonzero(~val)[0], np.nonzero(val)[0]
    fit_idx, real = demo_oversample(tr_idx, d["demo"], args.demo_share)
    log(f"[learner] {len(d['X'])} transitions ({len(tr_idx)} train), demos {real:.1%} of the samples "
        f"[{time.time() - t0:.0f}s]")
    ts = {k: d[k][fit_idx] for k in ("X", "yc", "yb", "ys", "yr") + (("wc",) if "wc" in d else ())}
    model.train()
    fit(model, ts, d, va_idx if len(va_idx) else tr_idx[:1000], args.epochs, args.lr,
        max(1, min(args.batch, len(fit_idx))), t0)
    model.eval()
    model.cycle_stats = {"transitions": len(d["X"]), "train_transitions": len(tr_idx),
                         "validation_transitions": len(va_idx), "fit_samples": len(fit_idx),
                         "demo_samples": int(d["demo"][fit_idx].sum()), "epochs": args.epochs,
                         "sample_frac": args.sample_frac, "max_samples": args.max_samples,
                         "batch": args.batch, "seed": args.seed, "lr": args.lr,
                         "levels": sorted(set(d["level"]))}
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
    from wm.dataset import load_model, LEGACY
    model, parent = load_model(base)
    dataset = getattr(args, "dataset", None)
    if dataset is not None:
        dataset.require_model(model)
    demo_paths = [p for p in (args.demos, args.self_demos) if p and os.path.exists(p)]
    n = fine_tune(model, folders, demo_paths, args)
    cand = errors(model, state["eval_waves"], dataset=dataset) if dataset is not None else errors(model, state["eval_waves"])
    pub = state["errors"]
    ok = not_worse(cand, pub)
    v = int(state.get("version", 0))
    new = dict(state, shards=counts, self_demos=n_self, cycles=int(state.get("cycles", 0)) + 1,
               dataset=dataset.reference() if dataset is not None else {"protocol": LEGACY})
    if ok:
        v += 1
        stats = getattr(model, "cycle_stats", {})
        from wm.protocol import feature_usage
        features = feature_usage(model.info)                 # the encoding fine-tuning kept (the parent's)
        parent_features = features["checkpoint_trained"]
        info = dict(model.info or {}, data=folders,
                    active_feature_protocol=features["active"], parent_feature_protocol=parent_features,
                    feature_protocol=features["active"] if stats.get("epochs", 0) > 0 and
                                     stats.get("fit_samples", 0) > 0 else parent_features,
                    dataset=dataset.reference() if dataset is not None else {"protocol": LEGACY},
                    parent_checkpoint=parent,
                    train=stats.get("train_transitions"), held_out=stats.get("validation_transitions"),
                    levels=stats.get("levels", []), sample_frac=stats.get("sample_frac"),
                    demos=demo_paths, demo_share=getattr(args, "demo_share", None), training=stats,
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
    from wm.dataset import add_cli, cli_dataset, load_model, LEGACY
    add_cli(ap)
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
    initial, _ = load_model(args.init)
    args.dataset = dataset = cli_dataset(args, initial)
    os.makedirs(args.out, exist_ok=True)
    spath = os.path.join(args.out, "learner.json")
    state = read_json(spath, {})
    if dataset is not None and not state and set(os.listdir(args.out)) - {"learner_output.txt"}:
        raise ValueError("strict learner needs a new empty output folder or compatible snapshot state")
    if dataset is not None and state and state.get("dataset", {}).get("sha256") != dataset.hash:
        raise ValueError("learner state belongs to a different or legacy snapshot; use a new output folder")
    pub_path = os.path.join(args.out, "wm_model.pt")
    same_init = state.get("init") == _rel(args.init) and state.get("base") == state.get("init")
    newer_pub = os.path.exists(pub_path) and os.path.getmtime(pub_path) >= os.path.getmtime(args.init)
    keep = bool(state.get("base") and state.get("eval_waves") and state.get("errors")
                and os.path.exists(_norm(state["base"])) and (same_init or newer_pub))
    if not keep:                             # start (again) from --init: its held-out waves and errors; the
        init = initial.eval()                    # version number keeps counting
        state = {"init": _rel(args.init), "base": _rel(args.init), "version": int(state.get("version", 0)),
                 "eval_waves": eval_waves(init, dataset=dataset),
                 "shards": shard_counts([_rel(_norm(d)) for d in init.info.get("data", [])]),
                 "dataset": dataset.reference() if dataset is not None else {"protocol": LEGACY}}
        state["errors"] = errors(init, state["eval_waves"], dataset=dataset) if dataset is not None else errors(init, state["eval_waves"])
        atomic_json(spath, state)
        log_line(args.out, f"start from {_rel(args.init)}: {len(state['eval_waves'])} evaluation waves, error "
                           f"median/90% {fmt(state['errors'])}")
    resumed = WorldModel.load(_norm(state["base"]))
    if dataset is not None:
        dataset.require_model(resumed)
    first = [_rel(_norm(d)) for d in resumed.info.get("data", [])]
    while True:
        folders = dataset.sources if dataset is not None else discover(first)
        counts = shard_counts(folders)
        n_self = 0 if dataset is not None else len(read_json(args.self_demos, []))
        if dataset is not None:
            pending = sum(counts.values()) - len(dataset.entries)
            go, new, new_self = state.get("cycles", 0) == 0, 0, 0
            print(f"[learner] immutable snapshot: {max(0, pending)} files pending outside it; no automatic growth",
                  flush=True)
        else:
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
