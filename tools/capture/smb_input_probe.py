"""Run scripted pad-input experiments in the running Super Meat Boy (personal use; Frida).

    python smbrl/tools/capture/smb_input_probe.py jump|move [out.jsonl]

Each trial starts once Meat Boy stands still on the ground; per-frame rows go to imported/capture/.
"""
import json
import os
import sys
import threading

import frida

HERE = os.path.dirname(os.path.abspath(__file__))
A, RIGHT, LEFT, X = 0x1000, 0x0008, 0x0004, 0x4000
SETS = {
    "jump": [{"name": f"jump{n}", "seq": [A] * n} for n in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30)]
            + [{"name": "right1", "seq": [RIGHT]}, {"name": "right3", "seq": [RIGHT] * 3},
               {"name": "run10_release", "seq": [RIGHT | X] * 10}, {"name": "left10_release", "seq": [LEFT | X] * 10}],
    "move": [{"name": "setup_run30", "seq": [RIGHT | X] * 30},
             {"name": "run40_release", "seq": [RIGHT | X] * 40},
             {"name": "runleft40_release", "seq": [LEFT | X] * 40},
             {"name": "run40_turn40", "seq": [RIGHT | X] * 40 + [LEFT | X] * 40},
             {"name": "walk40_release", "seq": [RIGHT] * 40},
             {"name": "runleft25_jump_release_air", "seq": [LEFT | X] * 25 + [LEFT | X | A] * 8 + [A] * 12},
             {"name": "run25_jump_turn_air", "seq": [RIGHT | X] * 25 + [RIGHT | X | A] * 8 + [LEFT | X | A] * 12}],
    # 1-1 right wall (x = 160): run there, jump onto it, then try wall-jump inputs while sliding
    "wall": [{"name": name, "seq": seq,
              "approach": [{"hold": RIGHT | X, "until": "s.x > 100"}, {"hold": RIGHT | X | A, "until": "s.st === 9"},
                           {"hold": RIGHT, "until": "s.vy < 0"}]}
             for name, seq in (("toward_tap", [RIGHT] * 3 + [RIGHT | A] + [RIGHT] * 30),
                               ("neutral_tap", [0] * 3 + [A] + [0] * 30),
                               ("away_tap", [LEFT] * 3 + [LEFT | A] + [LEFT] * 30),
                               ("away_then_tap", [LEFT] * 3 + [LEFT] + [LEFT | A] + [LEFT] * 30),
                               ("toward_hold_jump", [RIGHT] * 3 + [RIGHT | A] * 15 + [RIGHT] * 20),
                               ("neutral_tap_late", [0] * 12 + [A] + [0] * 30))],
    # what the trained agent did on that wall: run held toward it, jump toggled every frame
    "wall2": [{"name": name, "seq": seq,
               "approach": [{"hold": RIGHT | X, "until": "s.x > 100"}, {"hold": RIGHT | X | A, "until": "s.st === 9"},
                            {"hold": RIGHT | X, "until": "s.vy < 0"}]}
              for name, seq in (("run_toward_tap", [RIGHT | X] * 3 + [RIGHT | X | A] + [RIGHT | X] * 30),
                                ("alt_run_toward", [RIGHT | X, RIGHT | X | A] * 12 + [RIGHT | X] * 20),
                                ("alt_toward", [RIGHT, RIGHT | A] * 12 + [RIGHT] * 20),
                                ("gap2_run_toward", [RIGHT | X] * 2 + [RIGHT | X | A] + [RIGHT | X] * 2 + [RIGHT | X | A] + [RIGHT | X] * 30))],
    # the trained agent's exact pad encoding (stick + RT); see ingame_agent.js
    "wall3": [{"name": name, "seq": seq, "agentpad": True,
               "approach": [{"hold": RIGHT | X, "until": "s.x > 100"}, {"hold": RIGHT | X | A, "until": "s.st === 9"},
                            {"hold": RIGHT | X, "until": "s.vy < 0"}]}
              for name, seq in (("agentpad_alt_run_toward", [RIGHT | X, RIGHT | X | A] * 12 + [RIGHT | X] * 20),
                                ("agentpad_run_toward_tap", [RIGHT | X] * 3 + [RIGHT | X | A] + [RIGHT | X] * 30),
                                ("agentpad_toward_tap", [RIGHT] * 3 + [RIGHT | A] + [RIGHT] * 30))],
    # jump re-press after exactly g released frames while wall sliding (jump held until falling)
    "wallgap": [{"name": f"gap{g}", "agentpad": True, "seq": [RIGHT | X] * g + [RIGHT | X | A] + [RIGHT | X] * 25,
                 "approach": [{"hold": RIGHT | X, "until": "s.x > 100"}, {"hold": RIGHT | X | A, "until": "s.st === 9"},
                              {"hold": RIGHT | X | A, "until": "s.vy < 0"}]}
                for g in (1, 2, 3, 1, 2)],
    # replay of the trained agent's recorded actions from 1-1 (rows 113-159 of its in-game trace),
    # started at the same x and speed (390 to the right at x = 134.9)
    "agentreplay": [{"name": "agent_rows113", "agentpad": True, "seq": [20488, 20488, 20488, 16392, 20488, 16392, 20488, 16392, 20488, 16392, 20488, 16392, 20488, 16392, 20488, 16392, 20488, 16392, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488, 20488],
                     "approach": [{"hold": LEFT | X, "until": "s.x < -60"}, {"hold": RIGHT | X, "until": "s.x >= 134.8"}]}],
}


def main():
    which = sys.argv[1]
    out = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else
                          os.path.join(HERE, "..", "..", "imported", "capture", f"input_probe_{which}.jsonl"))
    if os.path.exists(out):
        os.remove(out)
    with open(os.path.join(HERE, "smb_input_probe.js"), encoding="utf-8") as f:
        src = f.read().replace("__OUT__", json.dumps(out.replace("\\", "/"))).replace("__TRIALS__", json.dumps(SETS[which]))
    session = frida.attach("SuperMeatBoy.exe")
    script = session.create_script(src)
    done = threading.Event()

    def on(msg, _data):
        p = msg.get("payload", msg)
        print(p, flush=True)
        if isinstance(p, dict) and "done" in p and p["done"] >= p["of"]:
            done.set()

    script.on("message", on)
    script.load()
    done.wait(300)
    script.unload()
    session.detach()
    print("wrote", out)


if __name__ == "__main__":
    main()
