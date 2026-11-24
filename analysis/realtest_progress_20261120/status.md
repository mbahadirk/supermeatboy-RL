
2026-11-20 — IN PROGRESS: realtest native transition and five extra waves.
The first A+ now starts a non-resetting five-completed-wave countdown, retaining the best native A+.
23 tests passed, including historical winner validation, duplicate waves, native finish ownership and
continue retries. Live run live_aplus_fix_a completed A+ wave1 + waves2..6 then loaded 1-2, but emitted
repeated terminal wave6 events during transition. Current code prevents that re-entry and retries continue.
Final live run live_aplus_fix_b reached A+ + five extra waves and native HUD13/main state22, but repeated
continue pulses have NOT loaded the next level. The final native fix is therefore NOT VERIFIED/COMPLETE.
Computer Use was stopped by the user's physical Escape key. Test B was asked to save/detach via STOP.
Original ppo_aplus_v1 checkpoint and native save were backed up under analysis/realtest_progress_20261120;
separate live test runs preserve the original checkpoint. Resume investigation at the native result-screen
input path; do not claim the current final version reliably advances. Logs: live_a.log and live_b.log.
