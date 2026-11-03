"""plan/analyze_run.py on a small synthetic run folder (no game, no model)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from plan import analyze_run as A          # noqa: E402


def _plan(frame, act, dx=0.0, supported=False, total=1.0, alt=0.95, chosen=("run R 16f",)):
    return {"frame": frame, "agent": 0, "level": "1-1", "wave": 1, "actions": [act] * 48,
            "path": [[frame + h + (dx if h >= 4 else 0.0), 0.0] for h in range(49)],
            "state": {"supported": supported}, "end_supported": True, "chosen": list(chosen),
            "score": {"total": total, "progress": 0.0, "uncertainty": 0.8},
            "alternatives": [{"moves": ["wait 8f"], "total": alt}, {"moves": ["x"], "total": None}],
            "v_end": 3.0, "timing": {"stage1": 10.0, "stage2": 20.0, "final": 30.0, "total": 60.0},
            "counts": {"stage2": {"steps": 8, "of_steps": 16}}}


def _write(d):
    os.makedirs(d)
    plans = [_plan(0, 1, dx=10.0), _plan(2, 1, alt=0.5), _plan(4, 2, supported=True, chosen=("run-jump R hold 10",))]
    with open(os.path.join(d, "plans.jsonl"), "w", encoding="utf-8") as f:
        for p in plans:
            f.write(json.dumps(p) + "\n")
        f.write('{"frame": 6, "half wri')                        # still being written
    with open(os.path.join(d, "states.jsonl"), "w", encoding="utf-8") as f:
        for fr in range(12):
            f.write(json.dumps({"frame": fr, "wave": 1, "level": "1-1",
                                "agents": [[0, float(fr), 0.0, 0.0, 0.0, 1 if fr >= 10 else 0]]}) + "\n")
    events = [{"kind": "lock", "level": None, "wave": 0, "ok": True},
              {"kind": "death", "level": "1-1", "wave": 1, "frame": 10, "cause": "hazard h3"},
              {"kind": "wave", "level": "1-1", "wave": 1, "frame": 10, "why": "done", "finished": 0, "frames": 10},
              {"kind": "wave", "level": "1-1", "wave": 2, "frame": 300, "why": "quota", "finished": 1,
               "frames": 300},
              {"kind": "finish", "level": "1-1", "wave": 3, "frame": 239, "frames": 240, "seconds": 4.0, "lock": True},
              {"kind": "wave", "level": "1-1", "wave": 3, "frame": 240, "why": "quota", "finished": 1,
               "frames": 240}]
    with open(os.path.join(d, "events.jsonl"), "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_analyze_a_synthetic_run(tmp_path, capsys):
    run = tmp_path / "runs" / "plan_20261103-000000"
    _write(str(run))
    out = tmp_path / "out.json"
    r = A.main([str(run), "--json", str(out)])
    assert json.load(open(out))["decisions"] == 3 and "decisions 3" in capsys.readouterr().out
    assert r["skipped_lines"] == {"plans.jsonl": 1}
    assert r["replan_interval"] == [2.0, 2.0]
    # played: frames 0-3 action 1 (plans 0 and 1), 4-9 action 2 (plan 2, until the last alive frame)
    assert sorted(A.q([4, 2, 6], (10, 50, 90))) == r["followed_frames"]
    assert r["plan_changed"] == {"all": 0.5, "airborne": 0.0, "n": 2, "n_airborne": 1}
    assert r["airborne"] == pytest.approx(2 / 3, abs=1e-4) and r["ends_supported"] == 1.0
    assert r["margin"]["n"] == 3 and r["margin"]["near_tie"] == pytest.approx(2 / 3, abs=1e-4)    # None totals skipped
    assert r["stage2_steps_share"][0] == 0.5 and r["timing_ms"]["total"] == [60.0, 60.0]
    # +4: plan 0 (followed 4 frames, 10 off) and plan 2 (exact) followed; plan 1 (followed 2) not
    e4, e8 = r["position_error"]["4"], r["position_error"]["8"]
    assert e4["followed"] == A.q([10.0, 0.0]) + [2] and e4["not_followed"] == [0.0, 0.0, 1]
    assert e8["followed"][2] == 0 and e8["not_followed"] == [10.0, 10.0, 1]  # frames 10 / 12: dead / not logged
    L = r["levels"]["1-1"]
    assert L["decisions"] == 3 and L["first_moves"][0] == ["run R", pytest.approx(2 / 3, abs=1e-4)]
    assert L["deaths"] == {"hazard h3": 1} and L["waves"] == 3
    # wave 2: a planned finish without an event -> the quota wave's length, an estimate; wave 3: the event
    fin = {f["wave"]: f for f in L["finishes"]}
    assert fin[2]["from"].startswith("estimate") and fin[2]["frames"] == 300 and fin[2]["seconds"] == 5.0
    assert fin[3]["from"] == "finish event" and fin[3]["seconds"] == 4.0 and L["best_s"] == 4.0
    # latest; one level only
    assert A.resolve_run("latest", str(tmp_path / "runs")) == str(run)
    assert A.analyze(str(run), "1-2")["decisions"] == 0


def test_move_family():
    assert A.family("run-jump R hold 10 after 4f") == "run-jump R"
    assert A.family("wait 8f") == "wait" and A.family("jump L hold 6 turn") == "jump L turn"
    assert A.family("fall drift L 12f") == "fall drift L" and A.family("jump up hold 22") == "jump up"
