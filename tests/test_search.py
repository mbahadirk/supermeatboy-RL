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
