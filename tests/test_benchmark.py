"""plan/benchmark.py on synthetic run folders (no game; agent.py only imported for its defaults)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import benchmark as B          # noqa: E402

SPEC = B.load_spec()


def _plan(level, wave, frame, act, chosen=("run R 16f",), used=None):
    return {"frame": frame, "agent": 0, "level": level, "wave": wave, "actions": [act] * 48,
            "best_route": None if used is None else {"frame": 0, "used": used, "finish": 11},
            "path": [[frame + h, 0.0] for h in range(49)], "state": {"supported": True}, "end_supported": True,
            "chosen": list(chosen), "score": {"estimate": 100.0, "total": 1.0, "t_start": 50.0, "t_end": 40.0},
            "alternatives": [], "v_end": 3.0, "timing": {"total": 60.0}, "trigger": ["interval"]}


def _run(d, waves, still=(), events=()):
    """waves: [(level, wave, 'finish' | 'death' | 'time' | 'cut', seconds or cause [, options])]: a plan every 4
    frames; 'time' ends without a finish or a death, 'cut' has plans and states but no wave event (interrupted).
    options: frames (default 12), replay (decisions with best_route.used true: 'all', 'one' or none), x / y (the
    death place, default 11, 0; a time-out's last state x), demo (a "demo check" event). still: (level, wave) pairs
    whose plans choose only "wait 8f". events: extra event lines first (e.g. a level event with a model)."""
    os.makedirs(d)
    with open(os.path.join(d, "plans.jsonl"), "w") as fp, open(os.path.join(d, "states.jsonl"), "w") as fs, \
            open(os.path.join(d, "events.jsonl"), "w") as fe, open(os.path.join(d, "times.jsonl"), "w") as ft:
        fe.write(json.dumps({"kind": "lock", "level": None, "wave": 0, "ok": True}) + "\n")
        for e in events:
            fe.write(json.dumps(e) + "\n")
        for lv, w, end, x, *opt in waves:
            o = opt[0] if opt else {}
            n = o.get("frames", 12)
            for i, fr in enumerate(range(0, n, 4)):
                used = {"all": True, "one": i == 0}.get(o.get("replay"))
                fp.write(json.dumps(_plan(lv, w, fr, 1, ("wait 8f",) if (lv, w) in still else ("run R 16f",), used))
                         + "\n")
            for fr in range(n):
                px = float(fr) if end != "time" else float(o.get("x", fr))
                fs.write(json.dumps({"frame": fr, "wave": w, "level": lv,
                                     "agents": [[0, px, 0.0, 0.0, 0.0, 0]]}) + "\n")
            if end == "cut":
                continue
            if end == "death":
                fe.write(json.dumps({"kind": "death", "level": lv, "wave": w, "frame": n, "agent": 0,
                                     "x": o.get("x", 11.0), "y": o.get("y", 0.0), "cause": x}) + "\n")
            elif end == "finish":
                fr = int(round(x * 60))
                ft.write(json.dumps({"level": lv, "wave": w, "agent": 0, "frames": fr, "seconds": x,
                                     "source": "planned", "lock": True}) + "\n")
                fe.write(json.dumps({"kind": "finish", "level": lv, "wave": w, "frame": fr - 1, "agent": 0,
                                     "frames": fr, "seconds": x}) + "\n")
            if o.get("demo"):
                fe.write(json.dumps({"kind": "demo check", "level": lv, "wave": w, "frame": n, "ok": True}) + "\n")
            fe.write(json.dumps({"kind": "wave", "level": lv, "wave": w, "frame": n, "frames": n, "agent_frames": n,
                                 "why": {"finish": "quota", "death": "done"}.get(end, "time"),
                                 "finished": int(end == "finish")}) + "\n")


def _two_runs(tmp_path):
    runs = tmp_path / "runs"
    _run(str(runs / "bench_a"), [("1-5", 1, "finish", 4.0), ("1-5", 2, "death", "hazard saw"),
                                 ("1-8", 3, "death", "hazard saw"), ("1-8", 4, "finish", 6.5),
                                 ("1-1", 5, "finish", 1.0)])                   # not a benchmark level
    _run(str(runs / "bench_b"), [("1-5", 1, "finish", 3.5), ("1-8", 2, "death", "kill box")], still={("1-5", 1)})
    return runs


def test_record_aggregates_benchmark_levels(tmp_path):
    runs = _two_runs(tmp_path)
    rec = B.report(["bench_a", "bench_b"], runs_dir=str(runs), out_dir=str(tmp_path / "bench"), spec=SPEC)
    L5, L8 = rec["levels"]["1-5"], rec["levels"]["1-8"]
    assert (L5["waves"], L5["finishes"], L5["completion"], L5["best_s"]) == (3, 2, pytest.approx(2 / 3, abs=1e-4), 3.5)
    assert L5["deaths"] == {"hazard saw": 1} and L5["deaths_per_wave"] == pytest.approx(1 / 3, abs=1e-4)
    assert (L8["waves"], L8["finishes"], L8["completion"], L8["best_s"]) == (3, 1, pytest.approx(1 / 3, abs=1e-4), 6.5)
    assert L8["deaths"] == {"hazard saw": 1, "kill box": 1}
    assert L5["median_s"] == pytest.approx(3.75) and L8["median_s"] == 6.5 and rec["levels"]["1-11"]["median_s"] is None
    assert L5["decisions"] == 9 and L5["ms_per_decision"] == 60.0 and L5["replan_interval"] == 4.0
    assert L5["approx_medians"] and [r["waves"] for r in L5["runs"]] == [2, 1]
    # still (plan.analyze_run.is_still): bench_b's 1-5 wave, 3 of the 9 decisions, summed over the runs exactly
    assert L5["still_decisions"] == 3 and L5["still"] == pytest.approx(1 / 3, abs=1e-4) and L8["still"] == 0.0
    assert rec["overall"]["still_decisions"] == 3 and rec["overall"]["still"] == pytest.approx(3 / 18, abs=1e-4)
    assert set(L5["position_error"]) == {str(h) for h in B.HORIZONS}
    assert L5["position_error"]["4"]["followed"][2] > 0
    assert rec["levels"]["1-11"]["waves"] == 0 and rec["levels"]["1-11"]["completion"] is None
    assert "1-1" not in rec["levels"] and rec["overall"]["waves"] == 6 and rec["overall"]["finishes"] == 3
    assert rec["wave_counts"]["1-5"] == 3 and rec["config"] == SPEC["config"] and rec["label"] == "bench_a+bench_b"
    assert set(rec["model_spec"]) == {"path", "exists", "mtime", "size"} and "git_commit" in rec
    assert rec["model_used"] == [{"run": "bench_a", "models": []}, {"run": "bench_b", "models": []}]
    assert os.path.exists(tmp_path / "bench" / "bench_a+bench_b.json")
    # one run: its own medians, exact
    one = B.build_record(["bench_b"], SPEC, runs_dir=str(runs))
    assert not one["levels"]["1-5"]["approx_medians"] and one["levels"]["1-5"]["position_error"]["1"]["followed"][2] == 3


def test_baseline_deltas(tmp_path, capsys):
    runs = _two_runs(tmp_path)
    out = str(tmp_path / "bench")
    B.report(["bench_a"], set_baseline=True, runs_dir=str(runs), out_dir=out, spec=SPEC)
    assert os.path.exists(os.path.join(out, "baseline.json"))
    capsys.readouterr()
    rec = B.report(["bench_b"], runs_dir=str(runs), out_dir=out, spec=SPEC)
    d = rec["vs_baseline"]["deltas"]
    assert rec["vs_baseline"]["label"] == "bench_a"
    # the K metrics: bench_a 1-5 = finish, death (fresh 1/2), bench_b 1-5 = finish (fresh 1/1)
    assert d["1-5"]["fresh_completion"] == pytest.approx(1.0 - 0.5) and d["1-5"]["best_s_k"] == pytest.approx(-0.5)
    assert d["1-5"]["first_try"] == 0.0 and d["1-5"]["waves_to_first"] == 0.0 and d["1-5"]["ms_per_decision"] == 0.0
    assert d["1-5"]["frames_to_first"] == 0.0 and d["1-5"]["replay_completion"] is None
    # 1-8: bench_a death then finish (first finish at wave 2), bench_b death only (none within K)
    assert d["1-8"]["fresh_completion"] == pytest.approx(-0.5) and d["1-8"]["best_s_k"] is None
    assert d["1-8"]["solved"] == -1.0 and d["1-8"]["waves_to_first"] is None
    assert d["1-5"]["median_s_k"] == pytest.approx(-0.5) and d["1-8"]["median_s_k"] is None
    assert d["1-11"]["fresh_completion"] is None and "completion" not in d["1-5"]
    assert d["1-5"]["still"] == pytest.approx(1.0) and d["1-8"]["still"] == 0.0
    assert d["overall"]["solved"] == -1.0 and d["overall"]["distinct_outcomes"] == -2.0
    out = capsys.readouterr().out
    assert "d fresh" in out and "d still" in out and "med_s" in out and "d med" in out
    assert "still: decisions whose chosen plan" in out and "vs baseline bench_a" in out and "all logged waves" in out
    # an older baseline without the K metrics: no K deltas, the others still
    old = {**rec, "levels": {lv: {a: b for a, b in v.items() if a != "k"} for lv, v in rec["levels"].items()},
           "overall": {a: b for a, b in rec["overall"].items() if a != "k"}}
    assert B.deltas(rec, old)["1-5"]["fresh_completion"] is None and B.deltas(rec, old)["1-5"]["still"] == 0.0


def test_cmd_has_every_frozen_flag_and_no_placeholder():
    text = B.command(SPEC, now=0)
    for k, fl in B.FLAGS:
        assert f"{fl} {SPEC['config'][k]}" in text
    assert "--no-overlay" not in text and "--name bench_" in text and "python smbrl/agent.py" in text
    assert SPEC["switches"] == ["--wm-collect"] and " --wm-collect --name bench_" in text
    assert "do not add --learn" in text and "changing the level with S is fine" in text
    name = text.split("--name ")[1].split()[0]
    assert (f"--death-memory smbrl/runs/{name}/death_memory.json --move-memory smbrl/runs/{name}/move_memory.json"
            in text)                                                    # the run's own memories, empty at start
    assert f"python smbrl/plan/benchmark.py report {name}" in text and "python -m plan.benchmark" not in text
    assert "<" not in text and ">" not in text
    for lv in SPEC["levels"]:
        assert f"{lv}: {SPEC['waves_per_level']} waves" in text


def test_runs_from_either_folder(tmp_path):
    """`python smbrl/plan/benchmark.py` from the folder above smbrl/ and `python -m plan.benchmark` from smbrl/;
    a run folder name resolves under smbrl/runs whatever the current folder."""
    import subprocess
    root = os.path.dirname(HERE)
    for cwd, argv in ((root, [os.path.join("smbrl", "plan", "benchmark.py")]), (HERE, ["-m", "plan.benchmark"])):
        r = subprocess.run([sys.executable, *argv, "cmd"], cwd=cwd, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        assert "python smbrl/plan/benchmark.py report bench_" in r.stdout
    runs = _two_runs(tmp_path)
    os.makedirs(tmp_path / "elsewhere" / "bench_a")                 # a same-named folder in the current one
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path / "elsewhere")
        assert B.resolve_run("bench_a", str(runs)) == str(runs / "bench_a")
    finally:
        os.chdir(cwd)


def test_levels_disjoint_from_held_out(tmp_path):
    held = json.load(open(B.EVAL))["held_out"]
    assert not set(SPEC["levels"]) & set(held)
    bad = tmp_path / "spec.json"
    bad.write_text(json.dumps({**SPEC, "levels": ["1-5", held[0]]}))
    with pytest.raises(ValueError, match="held out"):
        B.load_spec(str(bad))


def test_frozen_config_is_agent_defaults():
    assert SPEC["config"] == B.resolved_config()


def test_held_out_levels_are_reported_apart(tmp_path, capsys):
    """The spec's held-out levels (a subset of eval_levels.json's held_out) are played in the same run and reported
    in their own block: never in the benchmark's levels or overall; a non-held-out level there is refused."""
    held = json.load(open(B.EVAL))["held_out"]
    assert SPEC["held_out_levels"] and set(SPEC["held_out_levels"]) <= set(held)
    h1 = SPEC["held_out_levels"][0]
    runs = tmp_path / "runs"
    _run(str(runs / "bench_h"), [("1-5", 1, "finish", 4.0), (h1, 2, "finish", 9.0), (h1, 3, "death", "hazard saw")])
    rec = B.report(["bench_h"], runs_dir=str(runs), out_dir=str(tmp_path / "bench"), spec=SPEC)
    assert h1 not in rec["levels"] and rec["overall"]["waves"] == 1 and rec["overall"]["finishes"] == 1
    H = rec["held_out"]
    assert H["levels_spec"] == SPEC["held_out_levels"] and H["levels"][h1]["waves"] == 2
    assert H["levels"][h1]["finishes"] == 1 and H["levels"][h1]["best_s"] == 9.0
    assert H["overall"]["waves"] == 2 and H["overall"]["levels"] == len(SPEC["held_out_levels"])
    out = capsys.readouterr().out
    assert "held-out levels (generalization" in out and f"{h1}: 1 hazard saw" in out
    hk = H["levels"][h1]["k"]
    assert hk["first_try"] is True and hk["waves_to_first"] == 1 and hk["fresh_completion"] == 0.5
    assert H["overall"]["k"]["first_try"] == 1 and H["overall"]["k"]["levels_played"] == 1
    assert rec["overall"]["k"]["waves_k"] == 1                   # the held-out waves are not in the trained block
    text = B.command(SPEC, now=0)
    for lv in SPEC["held_out_levels"]:
        assert f"{lv}: {SPEC['waves_per_level']} waves (held out)" in text
    d = B.deltas(rec, {"levels": {}, "overall": {}})                       # an older baseline without the block
    assert d["held_out"][h1]["fresh_completion"] is None
    bad = tmp_path / "spec.json"
    bad.write_text(json.dumps({**SPEC, "held_out_levels": ["1-2"]}))
    with pytest.raises(ValueError, match="not held out"):
        B.load_spec(str(bad))


def test_k_metrics_fresh_replay_and_distinct_outcomes(tmp_path, capsys):
    """PLAN_GAME_SEARCH.md section 4 over the first K waves: first-try, waves and game frames to the first finish,
    fresh and replay completion apart, best / median over K, distinct outcomes; waves beyond K ignored."""
    runs = tmp_path / "runs"
    _run(str(runs / "bench_k"), [
        ("1-5", 1, "death", "hazard saw", {"frames": 20}),                       # fresh death
        ("1-5", 2, "death", "hazard saw", {"frames": 20}),                       # the same place again
        ("1-5", 3, "finish", 4.0, {"frames": 240}),                              # first finish, fresh
        ("1-5", 4, "finish", 4.0, {"frames": 240, "replay": "all"}),             # replays: one finish frame
        ("1-5", 5, "finish", 4.0, {"frames": 240, "replay": "one"}),             # one replayed decision: replay
        ("1-5", 6, "death", "kill box", {"frames": 16, "x": 200.0, "replay": "all"}),   # a replay that died
        ("1-5", 7, "finish", 3.0, {"frames": 180}),                              # beyond K = 6: ignored
        ("1-8", 1, "time", None, {"frames": 24, "x": 41.0}),                     # time-out: its last place
        ("1-8", 2, "time", None, {"frames": 24, "x": 45.0}),                     # the same tile
        ("1-8", 3, "finish", 5.0, {"frames": 300, "demo": True}),                # a demo wave: replay
    ])
    rec = B.report(["bench_k"], runs_dir=str(runs), out_dir=str(tmp_path / "bench"), spec=SPEC, k=6)
    k5, k8 = rec["levels"]["1-5"]["k"], rec["levels"]["1-8"]["k"]
    assert (k5["k"], k5["waves_k"], k5["waves_logged"]) == (6, 6, 7)
    assert k5["first_try"] is False and k5["solved"] and k5["waves_to_first"] == 3
    assert k5["frames_to_first"] == 20 + 20 + 240                                 # agent frames up to the finish
    assert (k5["fresh_waves"], k5["fresh_finishes"], k5["fresh_completion"]) == (3, 1, pytest.approx(1 / 3, abs=1e-4))
    assert (k5["replay_waves"], k5["replay_finishes"], k5["replay_completion"]) == (3, 2, pytest.approx(2 / 3, abs=1e-4))
    assert k5["best_s_k"] == 4.0 and k5["median_s_k"] == 4.0                     # the 3.0 s finish is wave 7 > K
    assert (k5["distinct_finish_frames"], k5["distinct_death_places"], k5["distinct_stuck_places"]) == (1, 2, 0)
    assert k5["distinct_outcomes"] == 3
    # replayed decisions: wave 4 all 60, wave 5 one, wave 6 all 4; over every logged decision and over K
    L5 = rec["levels"]["1-5"]
    assert L5["replayed_decisions"] == 60 + 1 + 4 and L5["decisions"] == 5 + 5 + 60 * 3 + 4 + 45
    assert k5["replayed_decisions_k"] == 65 and k5["decisions_k"] == L5["decisions"] - 45
    assert k8["first_try"] is False and k8["waves_to_first"] == 3 and k8["frames_to_first"] == 24 + 24 + 300
    assert (k8["fresh_waves"], k8["fresh_finishes"], k8["replay_waves"], k8["replay_finishes"]) == (2, 0, 1, 1)
    assert (k8["distinct_finish_frames"], k8["distinct_death_places"], k8["distinct_stuck_places"]) == (1, 0, 1)
    assert rec["levels"]["1-11"]["k"]["waves_k"] == 0 and rec["levels"]["1-11"]["k"]["solved"] is None
    o = rec["overall"]["k"]
    assert (o["levels_played"], o["first_try"], o["solved"], o["waves_k"]) == (2, 0, 2, 9)
    assert (o["fresh_waves"], o["fresh_finishes"], o["replay_waves"], o["replay_finishes"]) == (5, 1, 4, 3)
    assert o["distinct_outcomes"] == 5
    out = capsys.readouterr().out
    assert "trained levels, first 6 waves:" in out and "model used (from the run's logs): bench_k: not recorded" in out
    row = next(ln for ln in out.splitlines() if ln.startswith("1-5 "))
    assert "6/6" in row and "1/3 33%" in row and "2/3 67%" in row and "(1/2/0)" in row and "65/239 27%" in row
    row = next(ln for ln in out.splitlines() if ln.startswith("1-11"))
    assert " fail" not in row                                                     # not played is not a failure
    # the same run with the default K = 20: every wave counts
    k20 = B.build_record(["bench_k"], SPEC, runs_dir=str(runs))["levels"]["1-5"]["k"]
    assert k20["k"] == B.K_DEFAULT and k20["waves_k"] == 7 and k20["best_s_k"] == 3.0
    assert B.build_record(["bench_k"], SPEC, runs_dir=str(runs), k=2)["levels"]["1-5"]["k"]["solved"] is False


def test_interrupted_waves_and_a_restarted_level_align(tmp_path):
    """A wave without a wave event (a level switch or a stop cut it) is not a wave; its decisions still count in
    the logged decisions. A level played again restarts its wave numbers; plans still match their waves."""
    runs = tmp_path / "runs"
    _run(str(runs / "bench_r"), [
        ("1-13", 1, "finish", 4.0, {"replay": None}),
        ("1-13", 2, "cut", None, {"frames": 40, "replay": "all"}),               # interrupted, 10 replayed decisions
        ("1-14", 1, "death", "hazard saw"),
        ("1-13", 1, "death", "hazard saw"),                                      # 1-13 again: waves from 1
        ("1-13", 2, "finish", 4.5, {"frames": 280, "replay": "all"}),
    ])
    rec = B.build_record(["bench_r"], SPEC, runs_dir=str(runs))
    L = rec["levels"]["1-13"]
    assert L["waves"] == 3 and L["decisions"] == 3 + 10 + 3 + 70 and L["replayed_decisions"] == 10 + 70
    k = L["k"]
    assert k["waves_k"] == 3 and k["first_try"] is True and k["waves_to_first"] == 1
    assert (k["fresh_waves"], k["fresh_finishes"], k["replay_waves"], k["replay_finishes"]) == (2, 1, 1, 1)
    assert k["replayed_decisions_k"] == 70 and k["decisions_k"] == 3 + 3 + 70
    assert k["distinct_finish_frames"] == 2 and k["distinct_death_places"] == 1


def test_model_used_is_read_from_the_runs_logs(tmp_path, capsys):
    """The model the run logged (a level or start event's "model", --learn's "learn model" path), never the
    spec's; a run that logs none says "not recorded"."""
    runs = tmp_path / "runs"
    _run(str(runs / "bench_m"), [("1-5", 1, "finish", 4.0)],
         events=[{"kind": "level", "level": "1-5", "wave": 0, "name": "1-5", "model": "runs/wm_x/wm_model.pt"},
                 {"kind": "learn model", "level": "1-5", "wave": 0, "path": "runs/wm_live/wm_model.pt"}])
    _run(str(runs / "bench_n"), [("1-5", 1, "finish", 4.0)])
    rec = B.report(["bench_m", "bench_n"], runs_dir=str(runs), out_dir=str(tmp_path / "bench"), spec=SPEC)
    assert rec["model_used"] == [{"run": "bench_m", "models": ["runs/wm_x/wm_model.pt", "runs/wm_live/wm_model.pt"]},
                                 {"run": "bench_n", "models": []}]
    out = capsys.readouterr().out
    assert ("model used (from the run's logs): bench_m: runs/wm_x/wm_model.pt, runs/wm_live/wm_model.pt; "
            "bench_n: not recorded") in out
    assert "model the spec asks for (not verified as the one used)" in out


def test_cli_k(monkeypatch):
    got = {}
    monkeypatch.setattr(B, "report", lambda runs, label, sb, spec, k: got.update(runs=runs, k=k))
    B.main(["report", "bench_x", "--k", "7"])
    assert got == {"runs": ["bench_x"], "k": 7}
    B.main(["report", "bench_x"])
    assert got["k"] == B.K_DEFAULT == 20
    with pytest.raises(SystemExit):
        B.main(["report", "bench_x", "--k", "0"])
