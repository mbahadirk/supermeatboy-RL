"""Wave search (plan/search.py, agent.py --search) against a fake Frida script, no game: exact prefix replay, the
archive's soonest-per-cell rule, the crumble guard, the verifier, the wave end, held-out levels, the benchmark
report on a search run, and agent.py unchanged with the search off (tests/data/search_off_golden.json: the replies,
decisions and shard labels recorded before the search existed).

    python tests/test_search.py --write      # re-record the search-off golden (only when the planner is meant to change)
"""
import glob
import gzip
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

DATA = os.path.join(HERE, "runs", "real_20261102-165351", "wm_data")
MODEL = os.path.join(HERE, "runs", "wm_locked", "wm_model.pt")
GOLDEN = os.path.join(HERE, "tests", "data", "search_off_golden.json")
LEVEL = "1-13"
need_data = pytest.mark.skipif(not os.path.isdir(DATA), reason="recorded data not here")
need_model = pytest.mark.skipif(not (os.path.isdir(DATA) and os.path.exists(MODEL)), reason="recorded data not here")
OFF_FRAMES = 30
# the search-off scenarios: agent.py arguments and agent count
OFF_CASES = {"planner": (["--agents", "2", "--replan", "2", "--budget-ms", "60"], 2),
             "explore_random": (["--agents", "1", "--explore", "2", "--modes", "off", "--seed", "5"], 3),
             "explore_modes": (["--agents", "1", "--explore", "1"], 2)}


class _Script:
    def __init__(self):
        self.posts = []

    def post(self, msg):
        self.posts.append(msg)


def _recorded():
    from wm.buffer import load_level
    words, meta = load_level(DATA, LEVEL)
    for p in sorted(glob.glob(os.path.join(DATA, "wave_*.npz"))):
        z = np.load(p)
        if json.loads(str(z["meta"])).get("level") == LEVEL:
            world = json.load(gzip.open(p[:-4] + "_world.json.gz", "rt", encoding="utf-8"))
            return words, meta, z["rows"].copy(), world
    raise AssertionError("no recorded 1-13 wave")


def _row(rows, a, f):
    q = rows[(rows["agent"] == a) & (rows["frame"] == f)]
    q = q[0] if len(q) else rows[rows["agent"] == a][-1]
    return [float(q["x"]), float(q["y"]), float(q["vx"]), float(q["vy"]), int(q["flags"]), int(q["state"]), 0,
            int(q["keys"])]


def play_off(tmp_path, case: str) -> dict:
    """One recorded wave through agent.py with the search off: every reply, every decision (agent, frame,
    actions) and the shard's labels."""
    import agent
    argv, n = OFF_CASES[case]
    run = tmp_path / case
    args = agent.build_parser().parse_args(
        argv + ["--model", MODEL, "--name", str(run), "--wm-collect", "--telemetry", "off",
                "--death-memory", str(tmp_path / f"{case}_deaths.json"),
                "--move-memory", str(tmp_path / f"{case}_moves.json"),
                "--best-times", str(tmp_path / f"{case}_best.json")])
    r = agent.Runner(args, telemetry=None)
    r.script = _Script()
    words, meta, rows, world = _recorded()
    ags = sorted(set(rows["agent"].tolist()))[:n]
    r.on_message({"type": "send", "payload": dict(meta, name=LEVEL, agents=n, ev="info")},
                 words.astype("<u4").tobytes())
    replies = []
    st = None
    for w in world[:OFF_FRAMES]:
        st = [_row(rows, a, w["f"]) for a in ags]
        msg = dict(w, ev="frame", frame=w["f"], wave=1, s=st)
        msg.pop("f")
        k = len(r.script.posts)
        r.on_message({"type": "send", "payload": msg}, None)
        replies += [p["payload"] for p in r.script.posts[k:] if p["type"] == "act"]
    final = [list(s[:6]) + [1, s[7]] for s in st]
    r.on_message({"type": "send", "payload": {"ev": "wave_end", "wave": 1, "status": [1] * n, "finish": [-1] * n,
                                              "frame": world[OFF_FRAMES - 1]["f"] + 1, "why": "done", "s": final,
                                              "hz": [], "sh": []}}, None)
    r.close()
    plans = [json.loads(x) for x in open(run / "plans.jsonl", encoding="utf-8")]
    meta_s = json.loads(str(np.load(next((run / "wm_data").glob("wave_*.npz")))["meta"]))
    return {"replies": replies, "plans": [[p["agent"], p["frame"], p["actions"]] for p in plans],
            "shard": {"source": meta_s["source"], "agent_sources": meta_s["agent_sources"]}}


@need_model
@pytest.mark.parametrize("case", sorted(OFF_CASES))
def test_search_off_is_unchanged(tmp_path, case):
    """With --search absent agent.py plays exactly as before the search existed: the same replies to every frame
    (actions, leader), the same decisions and the same shard labels as the record made at af19e82."""
    if not os.path.exists(GOLDEN):
        pytest.skip("no search-off golden")
    want = json.load(open(GOLDEN, encoding="utf-8"))[case]
    got = play_off(tmp_path, case)
    assert got["replies"] == want["replies"]
    assert got["plans"] == want["plans"]
    assert got["shard"] == want["shard"]


# ---- the search through agent.py: a fake game ------------------------------------------------------------------
GOAL_DX, DEATH_DX = 120.0, -60.0          # the fake game: finish / die this far right / left of the spawn


class FakeGame:
    """realgame.js's protocol with a deterministic 1-D world: each update moves an agent by its action (2 units,
    3 with run, per frame, right minus left), at x >= spawn + GOAL_DX it finishes, at x <= spawn + DEATH_DX it
    dies. A wave runs until the runner's reply says "end" or every agent is done; deaths and finishes of the last
    update reach the runner only through the wave end's final states (as in the game). glitch: added to agent 0's
    reported x (a world that does not replay)."""

    def __init__(self, runner, n, meta, words):
        self.r, self.n, self.meta, self.words = runner, n, meta, words
        self.sx, self.sy = (float(v) for v in meta["spawn"])

    def info(self):
        self.r.on_message({"type": "send", "payload": {"ev": "debug", "msg": "time step locked to the fixed step"}},
                          None)
        self.r.on_message({"type": "send", "payload": dict(self.meta, name=LEVEL, agents=self.n, ev="info")},
                          self.words.astype("<u4").tobytes())

    def wave(self, no: int, glitch: float = 0.0, max_frames: int = 3000) -> dict:
        from mpc.candidates import ACTIONS
        n, sx, sy = self.n, self.sx, self.sy
        x, vx, status, fin = [sx] * n, [0.0] * n, [0] * n, [-1] * n
        acts = [[] for _ in range(n)]
        xs = [[sx] for _ in range(n)]
        posts = self.r.script.posts

        def states():
            return [[x[i] + (glitch if i == 0 and x[i] != sx else 0.0), sy, vx[i], 0.0, 0, 0, status[i], 0]
                    for i in range(n)]

        for f in range(max_frames):
            k = len(posts)
            self.r.on_message({"type": "send", "payload": {"ev": "frame", "frame": f, "wave": no, "dt": 16.667,
                                                          "s": states()}}, None)
            rep = [p["payload"] for p in posts[k:] if p["type"] == "act"]
            assert len(rep) == 1
            rep = rep[0]
            if "end" in rep:
                why, frame = rep["end"], f
                break
            for i in range(n):
                if status[i]:
                    continue
                a = ACTIONS.index(tuple(rep["a"][i]))
                acts[i].append(a)
                L, R, _, run = ACTIONS[a]
                dx = (int(R) - int(L)) * (3.0 if run else 2.0)
                x[i] += dx
                vx[i] = dx * 60.0
                xs[i].append(x[i])
                if x[i] >= sx + GOAL_DX:
                    status[i], fin[i] = 2, f
                elif x[i] <= sx + DEATH_DX:
                    status[i] = 1
            if all(status):
                why, frame = "done", f + 1
                break
        else:
            raise AssertionError("the wave did not end")
        self.r.on_message({"type": "send", "payload": {"ev": "wave_end", "wave": no, "status": list(status),
                                                      "finish": list(fin), "frame": frame, "why": why,
                                                      "s": states(), "hz": [], "sh": []}}, None)
        return {"acts": acts, "xs": xs, "frame": frame, "why": why, "status": status, "fin": fin}


def _search_runner(tmp_path, n=8, *extra):
    import agent
    args = agent.build_parser().parse_args(
        ["--search", "--search-agents", str(n), "--name", str(tmp_path / "run"), "--wm-collect", "--telemetry", "off",
         "--death-memory", str(tmp_path / "deaths.json"), "--move-memory", str(tmp_path / "moves.json"),
         "--best-times", str(tmp_path / "best_times.json"), "--seed", "1", *extra])
    r = agent.Runner(args, telemetry=None)
    r.script = _Script()
    return r


def _goal_t(monkeypatch, sx):
    """T in the fake world: frames to the goal at 3 units / frame (the real field is checked on its own)."""
    from plan import search as SR
    monkeypatch.setattr(SR.Search, "node_t",
                        lambda self, nodes, field: np.array([(sx + GOAL_DX - nd.state[0]) / 3.0 for nd in nodes]))


def _lines(path):
    return [json.loads(x) for x in open(path, encoding="utf-8")] if os.path.exists(path) else []


@need_data
def test_search_replays_prefixes_exactly_and_ends_its_waves(tmp_path, monkeypatch):
    """Every agent plays its node's actions exactly from the wave start (the game then shows the node's recorded
    state at its end), then its branch; the wave ends SETTLE frames after the longest sequence ran out, with
    "end": "search done"; the verifier (agent 0) replays the best node and reproduces it."""
    from plan import search as SR
    words, meta, _, _ = _recorded()
    _goal_t(monkeypatch, float(meta["spawn"][0]))
    r = _search_runner(tmp_path)
    assert r.n_total == 8 and r.n_plan == 0 and r.model is None and r.args.end_after == 0
    assert r.args.wave_seconds >= 60.0
    g = FakeGame(r, 8, meta, words)
    g.info()
    assert r.script.posts[-1] == {"type": "go", "payload": {"agents": 8}}
    S = r.search
    g.wave(1)
    ends = []
    for no in range(2, 5):
        before = {k: (list(v.actions), v.end, v.state) for k, v in S.arch.nodes.items()}
        out = g.wave(no)
        plan = S.plan
        assert len(plan) == 8
        replayed = 0
        for i, p in enumerate(plan):
            node = p["node"]
            if node is None or p["kind"] == "finish copy":
                continue
            acts, end, state = before[node]
            got = out["acts"][i]
            if len(got) >= end:                                     # alive through its prefix
                assert got[:end] == acts                            # the node's actions, exactly
                if S.arch.nodes[node].finish:                       # the same finish frame
                    assert out["fin"][i] + 1 == end
                else:                                               # the same state at its end
                    assert out["xs"][i][end] == state[0]
                replayed += 1
            n_branch = min(len(got), len(p["seq"])) - end
            assert got[end:end + n_branch] == p["seq"][end:end + n_branch]   # then its branch
        assert replayed >= 1
        if out["why"] == "search done":                             # the wave end: SETTLE after the longest sequence
            assert out["frame"] == max(len(p["seq"]) for p in plan) + SR.SETTLE == S.end_frame
            ends.append(no)
        else:                                                       # every agent done before that
            assert out["why"] == "done" and all(out["status"]) and out["frame"] <= S.end_frame
    assert ends
    r.close()
    rec = _lines(tmp_path / "run" / "search.jsonl")
    assert [x["wave"] for x in rec] == [1, 2, 3, 4]
    assert all({"picked", "branches", "inserted", "verification", "best_T", "archive", "replay"} <= set(x) for x in rec)
    assert rec[0]["verification"] is None and rec[0]["inserted"]["new"] > 0
    assert all(x["verification"] is not None and x["verification"]["ok"] for x in rec[1:])
    ev = _lines(tmp_path / "run" / "events.jsonl")
    assert [e.get("model") for e in ev if e["kind"] == "level"] == ["none (search)"]
    waves = [e for e in ev if e["kind"] == "wave"]
    assert all(w["all_agent_frames"] > w["frames"] for w in waves)          # every agent's frames, summed
    assert {w["why"] for w in waves} <= {"search done", "done"}
    states = _lines(tmp_path / "run" / "states.jsonl")
    assert all(len(s["agents"]) == 8 for s in states)                       # every search agent's state
    shard = json.loads(str(np.load(sorted((tmp_path / "run" / "wm_data").glob("wave_*.npz"))[0])["meta"]))
    assert shard["source"] == "search" and shard["agent_sources"] == {"search": list(range(8))}


@need_data
def test_search_finishes_logs_and_benchmark_report(tmp_path, monkeypatch, capsys):
    """Search finishes reach times.jsonl (source search), the finish events and best_times.json (search, lock on);
    after the first trusted finish the waves replay it (the wave event's replay_route) and copies of it are
    played; the benchmark report reads the run: first-try, waves and game frames (every agent) to the first
    finish, fresh / replay apart, distinct outcomes, the model 'none (search)'."""
    from plan import benchmark as B
    words, meta, _, _ = _recorded()
    _goal_t(monkeypatch, float(meta["spawn"][0]))
    r = _search_runner(tmp_path, 10)
    g = FakeGame(r, 10, meta, words)
    g.info()
    outs = [g.wave(no) for no in range(1, 7)]
    r.close()
    run = tmp_path / "run"
    times = _lines(run / "times.jsonl")
    assert times and all(t["source"] == "search" and t["lock"] is True for t in times)
    ev = _lines(run / "events.jsonl")
    fin = [e for e in ev if e["kind"] == "finish"]
    assert len(fin) == len(times) and all(e["source"] == "search" for e in fin)
    assert json.load(open(tmp_path / "best_times.json"))[LEVEL]["search"]["frames"] == min(t["frames"] for t in times)
    deaths = [e for e in ev if e["kind"] == "death"]
    assert deaths and all(e["cause"] == "search" and not e["memory_added"] for e in deaths)
    assert not (tmp_path / "deaths.json").exists() and not (tmp_path / "moves.json").exists()
    rec = _lines(run / "search.jsonl")
    first = next(k for k, x in enumerate(rec) if x["finishes"]["trusted"])
    assert all(not x["replay"] for x in rec[:first + 1]) and all(x["replay"] for x in rec[first + 1:])
    assert all(x["branches"].get("finish copy", 0) > 0 for x in rec[first + 1:])
    assert all(x["best_finish_frames"] is not None for x in rec[first:])
    best = [x["best_finish_frames"] for x in rec[first:]]
    assert best == sorted(best, reverse=True)                                # the fastest finish is kept
    waves = [e for e in ev if e["kind"] == "wave"]
    assert [w["replay_route"] for w in waves] == [x["replay"] for x in rec]
    out = capsys.readouterr().out
    assert "[search] 1-13 wave 1:" in out and f"finished" in out.split("[search] 1-13 wave 1:")[0]
    spec = {"levels": [LEVEL], "held_out_levels": [], "waves_per_level": 6, "config": {"model": "none"},
            "switches": []}
    rb = B.build_record([str(run)], spec, label="t", k=6)
    k = rb["levels"][LEVEL]["k"]
    fw = next(j for j, o in enumerate(outs) if any(s == 2 for s in o["status"]))
    assert k["waves_k"] == 6 and k["first_try"] == (fw == 0) and k["waves_to_first"] == fw + 1
    assert k["frames_to_first"] == sum(w["all_agent_frames"] for w in waves[:fw + 1])
    assert k["fresh_waves"] == first + 1 and k["replay_waves"] == 6 - first - 1
    assert k["distinct_finish_frames"] == len({t["frames"] for t in times})
    assert k["distinct_death_places"] >= 1 and k["distinct_outcomes"] >= 2
    assert rb["model_used"][0]["models"] == ["none (search)"]
    assert "none (search)" in B.table(rb)


@need_data
def test_verifier_mismatch_is_logged_and_the_node_dropped(tmp_path, monkeypatch):
    """A world that does not replay (agent 0's reported position off by 5 units): the verifier's state differs
    from the best node's record; search.jsonl logs the mismatch (no other agent touched a special tile first:
    "mismatch") and the node leaves the archive."""
    words, meta, _, _ = _recorded()
    _goal_t(monkeypatch, float(meta["spawn"][0]))
    r = _search_runner(tmp_path)
    g = FakeGame(r, 8, meta, words)
    g.info()
    g.wave(1)
    S = r.search
    g.wave(2, glitch=5.0)
    r.close()
    rec = _lines(tmp_path / "run" / "search.jsonl")
    v = rec[1]["verification"]
    assert v is not None and not v["ok"] and v["label"] == "mismatch" and v["cells"] == []
    node = S.arch.nodes[v["node"]]
    assert node.invalid == "mismatch" and S.arch.cells.get(node.key) != node.id
    if not v["finish"]:
        assert v["got"]["x"] == pytest.approx(v["want"]["x"] + 5.0)


@need_data
def test_search_on_a_held_out_level(tmp_path, monkeypatch):
    """The search may play a held-out level: it searches as anywhere (search.jsonl, nodes), but writes no
    world-model shard and no death or move memory."""
    import wm.buffer
    monkeypatch.setattr(wm.buffer, "eval_levels", lambda: {LEVEL})
    words, meta, _, _ = _recorded()
    _goal_t(monkeypatch, float(meta["spawn"][0]))
    r = _search_runner(tmp_path)
    g = FakeGame(r, 8, meta, words)
    g.info()
    for no in (1, 2, 3):
        g.wave(no)
    r.close()
    run = tmp_path / "run"
    assert r.protected
    assert [e["protected"] for e in _lines(run / "events.jsonl") if e["kind"] == "level"] == [True]
    assert len(_lines(run / "search.jsonl")) == 3 and len(r.search.arch.nodes) > 1
    assert not list((run / "wm_data").glob("wave_*.npz"))
    assert not (tmp_path / "deaths.json").exists() and not (tmp_path / "moves.json").exists()


@need_data
def test_node_t_on_the_real_field(tmp_path):
    """The search's T: the progress field agent.py builds (finite at the spawn of 1-13, unreachable inf mapped)."""
    from plan import search as SR
    words, meta, _, _ = _recorded()
    r = _search_runner(tmp_path, 2)
    g = FakeGame(r, 2, meta, words)
    g.info()
    g.wave(1)
    r.close()
    S = r.search
    nodes = list(S.arch.nodes.values())
    t = S.node_t(nodes, r.field)
    assert len(t) == len(nodes) and np.isfinite(t).all() and t[0] < SR.T_UNREACHABLE
    assert t[0] == pytest.approx(float(r.field.times(np.array([nodes[0].state[0]]), np.array([nodes[0].state[1]]),
                                                     0, nodes[0].supported, np.array([0.0]))[0]))


def test_search_refusals(tmp_path):
    import agent
    base = ["--name", str(tmp_path / "run"), "--telemetry", "off", "--death-memory", str(tmp_path / "d.json"),
            "--move-memory", "none", "--best-times", str(tmp_path / "b.json")]
    with pytest.raises(SystemExit, match="search-agents"):
        agent.Runner(agent.build_parser().parse_args(base + ["--search", "--search-agents", "201"]))
    assert agent.build_parser().parse_args(base + ["--search"]).search_agents == 50
    r = agent.Runner(agent.build_parser().parse_args(base + ["--search", "--agents", "3", "--explore", "4"]))
    assert (r.n_plan, r.n_explore, r.n_total) == (0, 0, 50) and r.search is not None
    r.close()


# ---- the search's own rules, with a small level ------------------------------------------------------------------
CRUMBLE, SWITCH, FULL = 0x164, 0x160, 0x1        # tile words: crumble (id 100), switch block (id 96), plain wall


def _small_level(special=True):
    from types import SimpleNamespace
    w = np.zeros((6, 12), np.uint32)
    w[0, :] = FULL
    if special:
        w[0, 5] = CRUMBLE                           # cell box x 100..120, y 0..20
        w[0, 8] = SWITCH                            # x 160..180
    return SimpleNamespace(name="t", W=12, H=6, ox=0.0, oy=0.0, words0=w, words=w.copy(), plats=[])


def _search(n, lv=None, wave_frames=3600):
    from plan.search import Search
    s = Search(n, seed=0, wave_frames=wave_frames)
    s.set_level(lv or _small_level())
    return s


def _st(x, y=26.0, status=0, vx=0.0):
    return [x, y, vx, 0.0, 0, 0, status, 0]


def _scripted(s, wave, seqs, points, positions, frames, tc=None):
    """A wave with given sequences / insertion points per agent; positions(i, f) -> state row; tc: {frame: tc}."""
    s.begin_wave(wave, 0, [positions(i, 0) for i in range(len(seqs))], None)
    for i in range(len(seqs)):
        s.plan[i] = {"node": s.arch.root.id, "prefix": 0, "seq": seqs[i], "points": points[i], "kind": "library"}
    s.points = [set(p["points"]) for p in s.plan]
    s.end_frame = max(len(q) for q in seqs) + 10
    for f in range(frames + 1):
        s.observe(f, [positions(i, f) for i in range(len(seqs))], (tc or {}).get(f))
    return s.wave_end(frames)


def test_crumble_guard_trusts_only_self_contained_states():
    """Crumble breaks are shared: a state is trusted only when every special tile that changed by its frame and that
    its agent touched changed at that agent's own first touch + 29-30 frames. Agent 1 breaks the block itself
    (trusted); agent 2 reaches it after agent 1 started it (broken by another agent); agent 3 falls into it after it
    broke; agent 4 touches the switch block, which changed with no clear toucher; agent 0 never comes near. The
    open-neighbour bits alone are no change."""
    from plan import search as SR
    s = _search(5)
    crumble_idx, switch_idx = 0 * 12 + 5, 0 * 12 + 8
    far, on_c, on_s = 30.0, 110.0, 170.0

    def pos(i, f):
        if i == 1:
            return _st(on_c if f >= 10 else far)
        if i == 2:
            return _st(on_c if f >= 30 else far)
        if i == 3:
            return _st(on_c, 10.0) if f >= 45 else _st(far)
        if i == 4:
            return _st(on_s if f >= 5 else far)
        return _st(50.0 if f else far)                                       # agent 0: another cell, never near
    tc = {20: [[crumble_idx, CRUMBLE | 0x400000]],        # a neighbour broke: open-neighbour bit only
          39: [[crumble_idx, 0x100]], 25: [[switch_idx, 0x100]]}
    seqs = [[0] * 60 for _ in range(5)]
    rec = _scripted(s, 1, seqs, [[50]] * 5, pos, 60, tc)
    assert int(s.change[s.cell_index[crumble_idx]]) == 39 and int(s.change[s.cell_index[switch_idx]]) == 25
    assert s.trusted(1, 50) == (True, None)                                  # its own touch (10) + 29
    assert s.trusted(2, 50) == (False, "crumble broken by another agent")    # touched at 30, broke at 39
    assert s.trusted(3, 50)[0] is False                                      # entered after the break
    assert s.trusted(4, 50) == (False, "special tile changed (no clear toucher)")
    assert s.trusted(0, 50) == (True, None)
    assert s.trusted(2, 35) == (True, None)                                  # before the break: nothing changed yet
    assert rec["inserted"]["untrusted"] == 3 and rec["guard"] and rec["breaks"] == 2
    assert rec["untrusted_why"] == {"crumble broken by another agent": 1, "crumble broken before its touch": 1,
                                    "special tile changed (no clear toucher)": 1}
    keys = {n.agent for n in s.arch.nodes.values() if n.end == 50}
    assert keys == {0, 1}                                                    # inserted: agents 0 and 1 only
    # a lag of 30 is the agent's own break too; 28 is not (another agent's earlier touch would explain it)
    s2 = _search(1)
    rec2 = _scripted(s2, 1, [[0] * 60], [[50]], lambda i, f: _st(on_c if f >= 10 else far), 60,
                     {40: [[crumble_idx, 0x100]]})
    assert rec2["inserted"]["new"] == 1 and s2.trusted(0, 50) == (True, None)
    s3 = _search(1)
    _scripted(s3, 1, [[0] * 60], [[50]], lambda i, f: _st(on_c if f >= 10 else far), 60, {38: [[crumble_idx, 0x100]]})
    assert s3.trusted(0, 50)[0] is False
    # a level without special tiles: no guard
    s4 = _search(1, _small_level(special=False))
    assert not s4.guard and SR.CRUMBLE_L == (29, 30)


def test_insertion_keeps_the_soonest_node_per_cell_and_skips_near_deaths():
    """States at insertion points become nodes when their cell key is new or reached sooner; a state NEAR_DEATH
    frames or less before its agent's death is not inserted; a later, slower visit of the same key is not."""
    from plan import search as SR
    s = _search(3, _small_level(special=False))

    def pos(i, f):
        if i == 2 and f >= 55:
            return _st(70.0, status=1)                                       # agent 2 dies at frame 55
        return _st(70.0 if f else 30.0)                     # from the spawn's cell to cell (3, 1), standing
    rec = _scripted(s, 1, [[0] * 60] * 3, [[30], [20, 40], [44, 50]], pos, 60)
    assert rec["inserted"]["new"] == 1 and rec["inserted"]["near_death"] == 1   # 50: 5 frames before the death
    ns = [n for n in s.arch.nodes.values() if n.end > 0]
    key = ns[0].key
    assert s.arch.nodes[s.arch.cells[key]].end == 20                         # the soonest of 20, 30, 40, 44
    assert key == ((3, 1), True, 0, 0, 0)
    # a later wave reaches the key sooner: replaced; later: kept
    rec = _scripted(s, 2, [[0] * 30], [[12]], lambda i, f: _st(70.0), 30)
    assert rec["inserted"]["sooner"] == 1 and s.arch.nodes[s.arch.cells[key]].end == 12
    rec = _scripted(s, 3, [[0] * 30], [[15]], lambda i, f: _st(70.0), 30)
    assert rec["inserted"] == {"new": 0, "sooner": 0, "untrusted": 0, "near_death": 0, "unseen": 0}
    assert s.arch.nodes[s.arch.cells[key]].actions == [0] * 12
    # moving right vs standing: another key (sign of vx beyond KEY_SPEED, 1 unit / frame = 60 units / s)
    rec = _scripted(s, 4, [[0] * 30], [[18]], lambda i, f: _st(70.0, vx=120.0), 30)
    assert rec["inserted"]["new"] == 1 and SR.NEAR_DEATH == 10


def test_verifier_labels_interference():
    """The verifier's mismatch is labelled interference when another agent touched a special tile on the node's
    recorded path before the node's agent had; the node is dropped and the next best is verified next."""
    s = _search(3)
    crumble_idx = 5
    s.begin_wave(1, 0, [_st(30.0)] * 3, None)
    root = s.arch.root
    b = s.arch.new(actions=[0] * 40, end=40, state=(110.0, 26.0, 0.0, 0.0, 0), supported=True,
                   key=((5, 1), True, 0, 0, 0), parent=root.id, wave=1, agent=1, path={0: 12})
    s.arch.cells[b.key] = b.id
    s.abort_wave()
    s.node_t = lambda nodes, field: np.array([0.0 if nd is b else 50.0 for nd in nodes])
    s.begin_wave(2, 0, [_st(30.0)] * 3, None)
    assert s.verify is b and s.plan[0]["kind"] == "verify" and s.plan[0]["seq"][:40] == b.actions

    def pos(i, f):
        if i == 0:
            return _st(110.0, 5.0 if f >= 30 else 26.0) if f >= 12 else _st(30.0)   # fell through at 30
        if i == 2:
            return _st(110.0) if f >= 5 else _st(30.0)                       # touched the block first (5 < 12)
        return _st(30.0)
    for f in range(61):
        s.observe(f, [pos(i, f) for i in range(3)], {34: [[crumble_idx, 0x100]]}.get(f))
    rec = s.wave_end(60)
    v = rec["verification"]
    assert not v["ok"] and v["label"] == "interference" and v["cells"] == [[5, 0]]
    assert b.invalid == "interference" and b.key not in s.arch.cells
    s.begin_wave(3, 0, [_st(30.0)] * 3, None)
    assert s.verify is not b


def test_selection_by_t_with_expansion_and_death_penalties():
    """Frontier nodes are picked by T, plus EXPAND_W per branch already played and DEATH_W x their death share."""
    from plan import search as SR
    s = _search(1 + SR.AGENTS_PER_NODE, _small_level(special=False))
    s.begin_wave(1, 0, [_st(30.0)] * (1 + SR.AGENTS_PER_NODE), None)
    s.abort_wave()
    root = s.arch.root
    mk = lambda x, end: s.arch.insert(actions=[0] * end, end=end, state=(x, 26.0, 0.0, 0.0, 0), supported=True,
                                      key=((int(x // 20), 1), True, 0, 0, 0), parent=root.id, wave=1, agent=0)
    mk(70.0, 10), mk(130.0, 20)
    a = s.arch.nodes[s.arch.cells[((3, 1), True, 0, 0, 0)]]
    b = s.arch.nodes[s.arch.cells[((6, 1), True, 0, 0, 0)]]
    T = {a.id: 50.0, b.id: 80.0, root.id: 200.0}
    s.node_t = lambda nodes, field: np.array([T[nd.id] for nd in nodes])
    s.begin_wave(2, 0, [_st(30.0)] * (1 + SR.AGENTS_PER_NODE), None)
    assert s.verify is a and {p["node"] for p in s.plan[1:]} == {a.id}       # the lowest T
    s.abort_wave()
    a.expanded, a.died = 10, 10                                               # 50 + 10 + 60 > 80
    s.begin_wave(3, 0, [_st(30.0)] * (1 + SR.AGENTS_PER_NODE), None)
    assert {p["node"] for p in s.plan[1:]} == {b.id}
    for p in s.plan[1:]:
        assert p["seq"][:b.end] == b.actions and p["points"][0] > b.end and p["kind"] in ("library", "perturbed")


if __name__ == "__main__" and "--write" in sys.argv:
    import tempfile
    out = {}
    with tempfile.TemporaryDirectory() as d:
        import pathlib
        for c in sorted(OFF_CASES):
            out[c] = play_off(pathlib.Path(d), c)
            print(c, len(out[c]["replies"]), "replies,", len(out[c]["plans"]), "decisions")
    with open(GOLDEN, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print("wrote", GOLDEN)
