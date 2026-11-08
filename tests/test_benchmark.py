"""plan/benchmark.py on synthetic run folders (no game; agent.py only imported for its defaults)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import benchmark as B          # noqa: E402

SPEC = B.load_spec()


def _plan(level, wave, frame, act, chosen=("run R 16f",)):
    return {"frame": frame, "agent": 0, "level": level, "wave": wave, "actions": [act] * 48,
            "path": [[frame + h, 0.0] for h in range(49)], "state": {"supported": True}, "end_supported": True,
            "chosen": list(chosen), "score": {"estimate": 100.0, "total": 1.0, "t_start": 50.0, "t_end": 40.0},
            "alternatives": [], "v_end": 3.0, "timing": {"total": 60.0}, "trigger": ["interval"]}


def _run(d, waves, still=()):
    """waves: [(level, wave, 'finish' | 'death', seconds or cause)]: 12 frames each, a plan every 4 frames.
    still: (level, wave) pairs whose plans choose only "wait 8f"."""
    os.makedirs(d)
    with open(os.path.join(d, "plans.jsonl"), "w") as fp, open(os.path.join(d, "states.jsonl"), "w") as fs, \
            open(os.path.join(d, "events.jsonl"), "w") as fe, open(os.path.join(d, "times.jsonl"), "w") as ft:
        fe.write(json.dumps({"kind": "lock", "level": None, "wave": 0, "ok": True}) + "\n")
        for lv, w, end, x in waves:
            for fr in range(0, 12, 4):
                fp.write(json.dumps(_plan(lv, w, fr, 1, ("wait 8f",) if (lv, w) in still else ("run R 16f",)))
                         + "\n")
            for fr in range(12):
                fs.write(json.dumps({"frame": fr, "wave": w, "level": lv,
                                     "agents": [[0, float(fr), 0.0, 0.0, 0.0, 0]]}) + "\n")
            if end == "death":
                fe.write(json.dumps({"kind": "death", "level": lv, "wave": w, "frame": 12, "cause": x}) + "\n")
            else:
                ft.write(json.dumps({"level": lv, "wave": w, "agent": 0, "frames": int(x * 60), "seconds": x,
                                     "source": "planned", "lock": True}) + "\n")
            fe.write(json.dumps({"kind": "wave", "level": lv, "wave": w, "frame": 12, "frames": 12,
                                 "why": "quota" if end == "finish" else "done",
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
    assert set(rec["model"]) == {"path", "exists", "mtime", "size"} and "git_commit" in rec
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
    assert d["1-5"]["completion"] == pytest.approx(1.0 - 0.5) and d["1-5"]["best_s"] == pytest.approx(-0.5)
    assert d["1-5"]["deaths_per_wave"] == pytest.approx(-0.5) and d["1-5"]["ms_per_decision"] == 0.0
    assert d["1-8"]["completion"] == pytest.approx(-0.5) and d["1-8"]["best_s"] is None
    assert d["1-11"]["completion"] is None
    assert d["1-5"]["still"] == pytest.approx(1.0) and d["1-8"]["still"] == 0.0
    out = capsys.readouterr().out
    assert "d compl" in out and "d still" in out and "still: decisions whose chosen plan" in out


def test_cmd_has_every_frozen_flag_and_no_placeholder():
    text = B.command(SPEC, now=0)
    for k, fl in B.FLAGS:
        assert f"{fl} {SPEC['config'][k]}" in text
    assert "--no-overlay" in text and "--name bench_" in text and "python smbrl/agent.py" in text
    assert "<" not in text and ">" not in text
    for lv in SPEC["levels"]:
        assert f"{lv}: {SPEC['waves_per_level']} waves" in text


def test_levels_disjoint_from_held_out(tmp_path):
    held = json.load(open(B.EVAL))["held_out"]
    assert not set(SPEC["levels"]) & set(held)
    bad = tmp_path / "spec.json"
    bad.write_text(json.dumps({**SPEC, "levels": ["1-5", held[0]]}))
    with pytest.raises(ValueError, match="held out"):
        B.load_spec(str(bad))


def test_frozen_config_is_agent_defaults():
    assert SPEC["config"] == B.resolved_config()
