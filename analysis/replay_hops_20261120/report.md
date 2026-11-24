# ONLYPPO replay continuation and repeated-hop shaping — 2026-11-20

## Replay continuation

The old implementation only fabricated XInput A presses while HUD state was 13. It did not reliably
activate the game's replay continuation. The result-screen Next Level callback is native VA 0x4aa050;
its transition schedules VA 0x4a61e0. Read-only disassembly of the current Super Meat Boy v1.2.5 executable
confirmed the callback accepts a null input state, uses caller-cleaned x86 arguments, clears replay
bindings and queues the normal fade/load. It is now invoked once after 60 result-screen frames, on HUD
onLeave (after rendering), only while finishRequested and HUD state13 still match. New-level load clears
the latch. Existing native transition ownership and first-A+ plus five-completed-wave countdown remain.
No fake controller polling is required. Chapter-final transitions keep the native game's own behaviour.

A first direct callback test failed with Frida's default exception handling. A short non-handling exception
observer identified a continuable Windows debug-print exception: address 0x40010006 on the trace through
RaiseException -> OutputDebugStringA -> native fade scheduling. This was being stolen as 'system error',
interrupting the callback before the level load was queued. The NativeFunction now explicitly uses
{abi:'mscdecl', exceptions:'propagate'} so Windows handles that diagnostic normally.

Live evidence: runs/live_native_next_20261120_b earned A+ first at wave3, completed waves4..8, and requested
its best native A+ 1.975s. The pre-fix direct call stalled. Hot-attaching the corrected callback to that
same completed test's HUD returned normally and loaded 1-2; the trainer also received level/info for 1-2,
then automatically saved/detached. See native_next_probe.json and ../realtest_progress_20261120/native_next_b.log.
This was a hot-corrected native callback test, not a second cold full training run of the final source.
The user manually opened 1-1 for the test. The original ppo_aplus_v1 model was not trained or overwritten
by the test; separate test runs loaded the saved pre-test copy. The game remains open on 1-2.

## 1-9 behaviour and small cost

Computer Use observed the game/monitor on 1-9 with agents clustered in the same lower corridor and high
jump-action probabilities. No successful 1-9 best/replay exists in the inspected run; therefore analysis
used the live session, not an invented stored replay. A 4-second, 20Hz read-only sample of 40 agents yielded
80 samples. Eleven agents made 2-4 physical takeoffs while moving less than 40 units horizontally. Sample:
agent0 x-range33.02 / takeoffs3; agent6 x-range26.96 / takeoffs4. Source: sample_1_9.json.

Added a separate small event cost, without increasing standing/stall costs: actual grounded->airborne
upward takeoffs with jump requested, from launch positions within 20 units, after existing progress grace
expires. First three launches are free; later launches cost0.03 each, capped at0.6 total per life/wave.
New route/exploration progress resets the local count, and genuinely different launch positions (including
moving platforms or climbing) restart it. Standing, waiting, holding jump without a takeoff, dropping off
a ledge, and terminal/dead states do not trigger this cost. The cap survives local progress resets.
The measurement demonstrates the loop; no claim of improved learned 1-9 completion is made yet.

Validation: full test count is recorded in PROJECT_STATE.md. Tests include native continuation without
XInput, duplicate/stale/early guards and reset, correct Windows ABI/exception handling, wait/ledge/climb/
platform exclusions, bounded cost and real trainer held-jump physical takeoffs feeding PPO trajectories.
Existing perception and realtest tests remain. No world model/action planner was added.
