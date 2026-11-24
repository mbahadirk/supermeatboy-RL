# 1-20 early-death / learning feedback audit — 2026-11-24

Offline only: no game control, runtime changes, policy optimization, model overwrite or
live eval. The original realtest run continued while files were read. Checkpoint snapshot,
SHA256s, selected input probes, native info and metrics are retained here. `audit.py`
reproduces a fresh snapshot, not necessarily this historical instant of the growing run.

## Recorded failure

Run `aplus_v2_1_14_continue`,40 agents,30s wave limit,batch16384,shape inputs,realtest.
Initial quick read:203 completed1-20 waves,8120 deaths,zero finishes/timeouts. The subsequent
full audit boundary includes207 completed waves:8279 deaths,one timeout,zero finishes/8280
trials. The source was still growing; these are separate snapshot boundaries.

First20 waves: median last-agent/wave duration28.5frames=0.475 simulated seconds.
Most individuals die even earlier. Representative waves1,50,100,207 have a median observed
individual death frame5 (~0.083s); wave200 median16. Top death bin is x~-320,y~-220.
The spawn is(-330,-250). The large overhead saw's native circle is centered near
(-319.88,-173.50),radius47.48. Early upward movement reaches it almost immediately.
Some deaths on the final terminal frame are reported only by wave_end, not an additional
frame; death-position histograms are therefore samples, not exactly all8280 endpoints.

The first reconstructed PPO patch already contains overhead hazard cells: the saw is not
simply omitted from perception. Reconstructed initial route lookahead points left
(first target offset~-50,0; farther target~-100,+41.7). The learned policy often chooses
early jump/right rather than consistently opening space to the left before climbing.
The running trainer's original route/logits were not recorded. Reconstruction uses the
native initial geometry and recomputes routes, so this direction is diagnostic rather
than a direct dump of the online route object.

## Why 200 waves do not mean 200 PPO feedback rounds

Recorded live action samples across207 completed waves total100589. Many waves contain
only~250..800 actions across all40 agents because lifetimes are so short. A full30s,
40-agent wave with everyone alive would contain72000: the entire207-wave set has only
about1.4 of those waves' action volume.

`realgame.py:on_wave_end` invokes PPO.update only when accumulated steps>=16384,
then consumes the entire batch and resets the counter to zero. Simulating this boundary
with the recorded per-wave action counts gives **6 PPO.update feedback rounds** for both
zero and maximum possible prior-level carry. This is derived, not a logged update count;
the initial carry itself is not recorded. Each PPO.update includes4 epochs and multiple
minibatch optimizer steps:6 does **not** mean only6 gradient steps.

The early stdout audit log initially printed a simple total/batch6..7 estimate. The
authoritative results.json and current audit.py use the actual end-of-wave/full-batch
reset semantics, giving6 for this snapshot.

At batch2048 the same *fixed recorded* wave lengths would trigger41 feedback rounds;
batch4096 would trigger22. Those are counterfactual scheduling calculations, **not live
improvement measurements**. Smaller batches change variance and minibatch counts; they
do not automatically multiply useful learning or guarantee success.

## What rewards and weights show

Selected native waves1,50,100,200,207 were replayed offline through the current producer
with recorded actions and learning disabled. Reconstructed median trajectory returns
are approximately-14.74..-14.78, dominated by R_DEATH=-15. The sampled trajectories are
negative, not an observed large positive payoff for jumping into the saw. Initial routes
were recomputed without prior-wave hazard sweep memory; these are reconstructed rewards,
not original logged training rewards. No successful1-20 counterfactual was demonstrated.

Current actor/critic parameters remain finite,maximum absolute parameter~3.345: no
NaN/Inf or numerical explosion was found. On the same reconstructed spawn-neighborhood
probes (frames0 and3), jump probability:

| Checkpoint | Jump probability |
| --- | --- |
|1-19 best before1-20 |79.54% |
|Current1-20 snapshot |61.24% |
|Much earlier1-13 best |28.64% |

Thus policy behavior has changed and early jumping has been partially suppressed. This
does not establish that the older model can pass1-20; the earlier checkpoint also strongly
favors moving right, which does not establish the required safe initial left opening.
Across all selected state probes, the current model still requests jump with85.93%
mean probability and action entropy~0.942nats: no global constant-output collapse.
Online requested-jump fractions in waves1/50/100/200 were89.5/84.2/82.3/79.8%; wave207
was97.1%, partly because its few long-lived traces dominate its action sample weighting.
These online fractions and fixed-input model probabilities are different measures.

The supported explanation is a strong inherited start-action bias combined with rapid,
highly similar deaths, little coverage of safer action sequences, and delayed feedback
between policy updates. Bad terminal outcomes alone do not supply demonstrations of the
safe alternative; PPO compares trajectory advantages under its current value estimate.
This is not proof that one reward coefficient or batch size is the sole cause.

## Smallest next controlled experiment

First test the existing --batch2048 option in a separate realtest run from a preserved
checkpoint, keeping rewards/architecture/geometry unchanged. Measure *individual survival,
left-opening attempts, successful passage beyond the first saw and completed levels*, not
wave count alone. This makes feedback more frequent for very short lives; improvement
is unverified. If the policy still almost never attempts the safe start, isolate its
state-conditioned action exploration/route-following at the initial obstacle. Blanket
jump or waiting penalties would also affect necessary later moves; none were added here.

All audit outputs are offline. The user's training/model files were not modified by this
investigation; analysis-created dummy trainer folders and current_snapshot.pt are separate.
