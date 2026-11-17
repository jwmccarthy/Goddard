# FS4: control cadence, CARL and occupancy matching

Follow-up to [FS4_RECOVERY_AUDIT.md](FS4_RECOVERY_AUDIT.md), 2026-11-15.

## What is established

The learned FS8 policy works well when executed at FS4, so a four-tick action
interval does not inherently prevent aerial control. Learning that policy at
FS4 can nevertheless be harder: action persistence, exploration and the effect
of each individual decision change with physical duration. Multi-frame AIL
adds a separate issue: the discriminator can detect replay/simulator dynamics
differences rather than useful behavioral differences.

The new native audit also found a **specific CARL contact-selection
reproducibility defect**. A temporary deterministic tie-break removes the
observed cross-cadence discrepancies in two seeds. This does not establish that
the defect causes the long-run aerial-learning gap.

## 1. A direct native cadence test

The active trainer was verified to load:

`/home/jack/.local/share/goddard/carl/reset-consistency-20261115-112936/stage/carl/_carl.cpython-311-x86_64-linux-gnu.so`

SHA256: `fc3ed0a27536bd3cc37f4bf0cacb1ca1ae153b19b51a883da2651c6abb2a7b92`.
Its saved source matches the inspected stepping, action decoding, control,
observation and reset source in `/home/jack/CARL/`.

Hold the same decoded action for eight physics ticks and compare:

- eight native FS1 calls;
- two native FS4 calls;
- one native FS8 call.

The scripted kickoff/jump/boost/air-control test was **bit-identical** in all
observations, controller fields, action masks, physical state and OR-aggregated
car event flags. It covered 7,680 shared simulation endpoints, 426 jump-event
windows and 170 dodge-event windows. Packed touch flags were excluded from raw
state equality because their intentional aggregation interval differs.

Replay-started random-control and policy trajectories exposed occasional
divergences. A second harness included two FS8 and two FS4 replicas to test
same-cadence repeatability. Each seed used 128 simulations for 20 seconds;
comparisons stop including a simulation after its first episode boundary in
any replica, since reset times are legitimately quantized to decision boundaries.

| Seed | Original FS8/FS8 divergent simulations | Original FS4/FS4 | Original FS8/FS4 | Temporary fix: all five comparisons |
| --- | ---: | ---: | ---: | ---: |
| 212 | 0 | 1 | 2 | **0** |
| 213 | 0 | 1 | 1 | **0** |

Small initial floating-point discrepancies can grow into large trajectory
differences over subsequent contacts. The maximum difference late in a run is
therefore not a measure of an individual integration-step error.

### Contact tie ordering

In `include/Physics/CarArena/SAT.cuh`, `atomicAdd(carHitCount, 1)` determines
the storage order of overlapping car/triangle candidates. In
`include/Physics/CarArena/Manifold.cuh`, `insertBestContact` keeps the first
candidate satisfying the strict comparison `dist < bestDist`. Equal-depth
candidates therefore inherit GPU execution ordering.

The isolated probe adds a tie-break on the stable triangle index:

```cpp
if (dist < bestDist || (dist == bestDist && triangle < bestTriangle))
```

With this change, FS1/FS4/FS8 and same-cadence replicas were bit-identical at
**46,374 shared simulation endpoints** across the two replay-started seeds.
This strongly implicates contact tie ordering in the measured discrepancies.
The patched reference trajectory can itself differ from the original build;
this is a reproducibility test within each build, not a cross-build behavioral
equivalence or aerial-learning test.

The experiment lives in `/tmp/opencode/carl-contact-probe/`, with the built
package at `/tmp/opencode/carl-contact-probe-stage/`. The installed trainer
continues using the original binary. Broader physics-regression validation and
a learning comparison of the temporary change have not been run.

### Reset and terminal observations

Native stepping integrates the configured tick block, packs the transition,
then resets completed simulations. A deliberately induced goal crossing at
tick 2 was reported at ticks 2/4/8 for FS1/FS4/FS8. All returned the correct
±1 native goal reward and preserved terminal observations separately from the
reset observation. A six-tick no-touch timeout was reported at ticks 6/8/8,
correctly marked truncated rather than goal-terminated.

Thus decision-boundary rounding is real; the tests did not reveal a missed
goal or a terminal/reset observation swap. At FS8, physics continues through
the remainder of a block after a goal crossing before the reset is delivered.

## 2. Temporal commitment changes exploration exponentially

FS8 supplies 66.7 ms of commitment per action; FS4 supplies 33.3 ms. More frequent
independent draws can interrupt a useful control sequence before its effect is
visible. A richer set of simultaneous action combinations does not automatically
make those combinations persist across time.

For an explicit example, force the first jump, then independently continue
holding it with probability .5 at each subsequent decision. Stop permanently
at the first release to isolate the first jump from later double-jumps/flips.
The probability of maintaining the first jump for 200 ms is:

- FS8: two more successful draws, `.5² = 25%`.
- FS4: five more successful draws, `.5⁵ = 3.125%`.

In a native 2,048-car experiment, the corresponding measured fractions were
**25.54% versus 3.37%**. Mean heights after .4 seconds were 130.4 versus 108.1 uu.
This is a deliberately simplified exploration example, not an estimate of the
trained policies' probabilities or proof that jump holding explains their gap.
Earlier frozen-policy hold interventions did not recover learned aerial play.

CARL persists jump/boost controller state across API calls. The code uses
`jump && !lastJump` for a new press, a 25 ms minimum/200 ms maximum initial jump
window, and a 100 ms minimum boost burst. Those durations use 120 Hz physics
ticks, not decision counts. Holding identical inputs across two FS4 calls does
not restart the jump or boost timer.

For continuous signed controls there is a related effect: with independent
zero-mean samples held for duration h, integrated control variance over fixed
time T scales as `T * h * variance(action)`. Halving h can make exploratory
torques cancel more effectively, despite doubling the number of random choices.
State-dependent feedback and native controller latches complicate this simple
model, but faster sampling is not automatically broader trajectory exploration.

## 3. Shorter decisions also change credit and optimization

Under smooth continuous-time dynamics and a physical-time-scaled reward, the
advantage of changing a control for only one short interval tends to shrink
with that interval. Approximation and return-estimation errors need not shrink
at the same rate. Jump presses and contacts are discontinuous exceptions, so
this is not a universal factor-of-two law for Rocket League actions.

[Tallec, Blier and Ollivier (ICML 2019)](https://proceedings.mlr.press/v97/tallec19a.html)
analyze discretization-sensitive value differences and exploration. Their
Q-learning/DDPG collapse result is **not a proof that PPO must fail at FS4**;
the exploration argument and the need to match physical-time scales are the
relevant connections here.

The repository already time-scales gamma/lambda. Its existing production runs
still differ in dense reward rate, rollout duration and optimizer updates per
physical second, as documented in the recovery audit. The short continuation
experiments did not establish reward scaling or any tested architecture change
as an aerial-learning fix.

## 4. Multi-frame AIL can expose an unmatchable dynamics signature

A state-window discriminator sees both where the policy goes and how consecutive
states fit together. If reconstructed expert positions/velocities obey different
short-time relationships from CARL, the optimal classifier can use that
difference even when behavior is similar. A policy cannot freely change its
simulator's transition law to erase such a cue.

The earlier matched quiet-flight test found position/velocity residual medians
of 8.16 uu in replays versus .38 uu in native trajectories; that scalar alone
classified source with AUROC .991 in the small matched subset. This establishes
an available domain cue, not that it is the production discriminator's decisive
feature. Noise reduced this isolated cue, but did not establish an online fix.

More observations provide more consistency constraints. Shorter intervals can
also magnify reconstruction noise in inferred acceleration. This is a
fundamental AIL identifiability concern distinct from the contact-order defect.
A separate **single-state occupancy objective** is a reasonable next experiment:
it targets coverage without directly comparing short-time physics histories.

## Artifacts

JSONs under `/tmp/opencode/fs4-recovery/`:

- `native-cadence-equivalence.json`: scripted, random-replay and policy-replay checks.
- `native-cadence-repeatability.json`, `native-cadence-repeatability-213.json`:
  original native build, including same-cadence controls.
- `native-cadence-contact-tiebreak.json`, `native-cadence-contact-tiebreak-213.json`:
  isolated deterministic-tie build.
- `jump-discretization.json`: first-jump commitment experiment.
- `terminal-cadence.json`: goal/timeout boundary checks.

Harnesses: `/tmp/opencode/carl-cadence-equivalence.py`,
`carl-cadence-repeatability.py`, `carl-jump-discretization.py`, and
`carl-terminal-cadence.py`. The repeatability harness records the loaded binary's
path and SHA256. Ordinary probes use the production staged CARL import path;
tie-break probes use the isolated staged package.
