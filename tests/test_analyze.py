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
    # an older run without "trigger" / "switch": none reported, every plan a committed one
    assert r["triggers"] == {"decisions": 0, "by_kind": {}} and r["switch"]["rate"] is None
    assert r["committed_followed_frames"][3] == 3 and r["success_margin"]["n"] == 0
    assert "none logged (an older run)" in A.report(r)
    assert r["clipped"] == {"share": None, "n": 0} and "not logged: an older run" in A.report(r)


def test_clipped_share(tmp_path):
    run = tmp_path / "plan_clip"
    _write(str(run))
    with open(run / "plans.jsonl", "w", encoding="utf-8") as f:
        for k, c in enumerate((0, 5, 0, 12)):
            f.write(json.dumps({**_plan(2 * k, 1), "model_clipped": c}) + "\n")
    r = A.analyze(str(run))
    assert r["clipped"] == {"share": 0.5, "n": 4}
    assert "clipped (the model's path went into solid tiles): 50%" in A.report(r)


def test_still_share(tmp_path):
    """Still: the chosen plan has no moves other than wait / pad (empty chosen: pad only); kept: from_prev too."""
    assert A.is_still([]) and A.is_still(["wait 8f", "wait 16f"])
    assert not A.is_still(["wait 8f", "run R 16f"]) and not A.is_still(["run R 16f"])
    run = tmp_path / "plan_still"
    _write(str(run))
    with open(run / "plans.jsonl", "w", encoding="utf-8") as f:
        for k, (ch, prev) in enumerate(((["wait 8f"], False), ([], True), (["wait 8f", "run R 16f"], True),
                                        (["run R 16f"], False))):
            f.write(json.dumps({**_plan(2 * k, 1, chosen=ch), "from_prev": prev}) + "\n")
    r = A.analyze(str(run))
    assert r["still"] == {"share": 0.5, "kept": 0.25, "n": 2}
    L = r["levels"]["1-1"]
    assert L["still"] == 0.5 and L["still_kept"] == 0.25 and L["still_n"] == 2
    out = A.report(r)
    assert "still (chosen plan only wait / pad): 50% of decisions (n 2), kept from the previous plan 25%" in out
    assert "still 50% (kept 25%)" in out


def test_move_family():
    assert A.family("run-jump R hold 10 after 4f") == "run-jump R"
    assert A.family("wait 8f") == "wait" and A.family("jump L hold 6 turn") == "jump L turn"
    assert A.family("fall drift L 12f") == "fall drift L" and A.family("jump up hold 22") == "jump up"


def test_triggers_switches_committed_plans_and_success_margins(tmp_path):
    """A run with plans.jsonl "trigger" / "switch" / from_prev: triggers by kind, switch rate, kept gains, frames
    each committed plan was followed (through the decisions that kept it), and goal touches compared by frames."""
    run = tmp_path / "plan_x"
    os.makedirs(run)
    p0 = _plan(0, 1, total=2.0, alt=1.95)
    p0.update(trigger=["start"], switch=None, from_prev=False)
    p1 = _plan(4, 1, total=1.0, alt=1.3)                          # kept against a better one (+0.3 < 0.5)
    p1.update(trigger=["interval"], switch={"kept": True, "gain": 0.3, "margin": 0.5, "unit": "V", "reason": "kept"},
              from_prev=True)
    p2 = _plan(8, 2, total=1e5 - 20, alt=1e5 - 24)                # a success: 4 frames before the best rival success
    p2["alternatives"].append({"moves": ["y"], "total": 3.0})       # a non-success: not its rival
    p2.update(trigger=["move end", "surprise +3 err 4.0 > 3.0"],
              switch={"kept": False, "gain": 2.1, "margin": 0.5, "unit": "V", "reason": "switched: +2.10"},
              from_prev=False)
    with open(run / "plans.jsonl", "w", encoding="utf-8") as f:
        for p in (p0, p1, p2):
            f.write(json.dumps(p) + "\n")
    with open(run / "states.jsonl", "w", encoding="utf-8") as f:
        for fr in range(14):
            f.write(json.dumps({"frame": fr, "wave": 1, "level": "1-1", "agents": [[0, float(fr), 0.0, 0.0, 0.0, 0]]})
                    + "\n")
    r = A.analyze(str(run))
    assert r["triggers"] == {"decisions": 3, "by_kind": {"start": 1, "interval": 1, "move end": 1, "surprise": 1}}
    assert r["switch"]["switched"] == 1 and r["switch"]["kept"] == 1 and r["switch"]["rate"] == pytest.approx(1 / 3,
                                                                                                             abs=1e-4)
    assert r["switch"]["kept_gain"] == {"V": [0.3, 0.3, 1]}
    # plan 0 (committed) played frames 0-7 (itself, then plan 1 keeping it): followed 8; plan 2: frames 8-13
    assert r["committed_followed_frames"][3] == 2 and sorted(A.q([8, 6], (10, 50, 90))) == r["committed_followed_frames"][:3]
    # the V margins ignore the success; the success is compared by touch frames (4 earlier)
    assert r["margin"]["n"] == 2 and r["success_margin"] == {"median": 4.0, "near_tie": 0.0, "n": 1}
    out = A.report(r)
    assert "switch rate" in out and "interval 1" in out and "goal touches" in out


def test_frame_scored_runs_no_progress_and_completion(tmp_path):
    """Runs scored in frames (score.estimate): no progress = T(end) >= T(start) (the time-to-go did not shrink),
    margins in frames, completed / landing shares."""
    run = tmp_path / "plan_f"
    os.makedirs(run)
    ps = []
    for k, (t0, t1, comp) in enumerate([(300.0, 250.0, None), (300.0, 300.0, 6), (300.0, 310.0, None),
                                        (None, 9e3, None)]):
        est = 48 + (comp or 0) + t1                                       # waiting / slow plans still shrink T
        p = _plan(2 * k, 1, total=-est, alt=-est - 2.0)
        p["score"] = {"total": -est, "estimate": est, "t_start": t0, "t_end": t1, "frames_used": 48 + (comp or 0),
                      "landing": comp is not None or k == 0}
        p["completion"] = None if comp is None else {"frames": comp, "on": "ground", "continuation": "keep"}
        ps.append(p)
    with open(run / "plans.jsonl", "w", encoding="utf-8") as f:
        for p in ps:
            f.write(json.dumps(p) + "\n")
    with open(run / "states.jsonl", "w", encoding="utf-8") as f:
        for fr in range(10):
            f.write(json.dumps({"frame": fr, "wave": 1, "level": "1-1", "agents": [[0, float(fr), 0.0, 0.0, 0.0, 0]]})
                    + "\n")
    r = A.analyze(str(run))
    assert r["levels"]["1-1"]["no_progress"] == pytest.approx(2 / 4)       # 300 -> 300 and 300 -> 310; no T: unknown
    assert r["margin"]["unit"] == "frames" and r["margin"]["n"] == 4 and r["margin"]["near_tie"] == 1.0
    assert r["completed"] == 0.25 and r["landing"] == 0.5
    assert "frames" in A.report(r)
