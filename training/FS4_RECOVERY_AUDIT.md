# FS4 recovery and multi-observation discrimination — 2026-11-15

Follow-up to [FS4_AERIAL_FOLLOWUP.md](FS4_AERIAL_FOLLOWUP.md). The question here
is successful aerial behavior, rather than airborne occupancy alone.

The later [foundational audit](FS4_FOUNDATIONS.md) checks CARL directly, identifies
a contact-order reproducibility issue, and separates that finding from temporal
exploration and replay/simulator dynamics mismatch.

## Findings so far

1. **There is a later aerial-specific FS4 gap in the existing runs.** At matched
   simulated experience, FS4 makes more overall contacts but fewer aerial ones.
2. **Four-tick control can execute the FS8 policy's aerial behavior.** Across
   two evaluation seeds, that policy produces 1.676 aerial contact onsets/car/min
   with four-tick actions versus 1.615 with eight-tick actions. Longer holds applied
   to the FS4-trained policy do not recover the FS8-trained policy's behavior.
3. **The production comparisons still change the learning problem.** Dense
   imitation rewards, collection duration, optimizer cadence, and temporal
   window duration need separate attention. Gamma/lambda are already
   time-scaled; multiplying them by another frameskip correction is incorrect.
4. **Endpoint scalar erasure is not exercised when all those fields are hidden.**
   Combining `--discriminator-endpoint-scalars` with both
   `--discriminator-no-flags` and `--discriminator-no-boost` has no extra effect.

The repeated failure to develop aerial maneuvers remains the important result.
The earlier jump-entropy experiment demonstrated a short exploration increase,
not a solution to that failure.

## Existing runs at equal physical-time budgets

TensorBoard steps count agent decisions. Compare `step * frameskip`, rather than
the same decision count. Means below use the same agent-physics-tick bins.

| Budget | Run | FS / window / decision stride | Aerial contacts / 1,000 ticks | All contacts / 1,000 ticks |
| --- | --- | --- | ---: | ---: |
| 7.5–8B | `184227-1` | 4 / 3 / 4 | .07692 | 4.15721 |
| 7.5–8B | `171925-1` | 8 / 2 / 4 | **.12578** | 2.94523 |
| 1.5–2B | `205816-1`, no flags/boost | 4 / 2 / 4 | .06037 | 2.27497 |
| 1.5–2B | `211728-1`, no flags/boost | 8 / 2 / 4 | .05984 | 2.34878 |

All run names begin `gaifo-20261115-`. These are observational comparisons,
not independently replicated controlled runs. The last FS8 run was still
training when the snapshot was taken. Similar early contact rates do not
invalidate the user's observation of a later aerial regression.

At 1.5–2B ticks, the last two runs' imitation rewards per tick are −.36828 and
−.15714; their jump entropies are .02951 and .05040. Different discriminators
have different logits, so these reward values cannot isolate cadence by
themselves. The implementation nevertheless adds one full imitation reward per
decision: identical logits produce twice the physical-time reward rate at FS4.
Goals remain ±50. Log reward is negative, so changing its rate also changes
the cost of delaying a goal or termination relative to the fixed goal reward.

### Exact temporal matching

- FS8, window 2, decision stride 4: `[t−32, t]`.
- FS4, window 2, decision stride **8**: `[t−32, t]`.
- FS4, window 3, decision stride 4: `[t−32, t−16, t]`.
- FS4, window 2, decision stride 4: `[t−16, t]`.

Thus two versus three observations can be tested at an equal 32-tick span.
Changing FS8 to FS4 while retaining window 2/stride 4 halves the span.

### Matching rollout length alone does not match optimization

The production trainer iterates over **all** collected decisions in both PPO
and discriminator training. Doubling FS4's rollout to 64 while keeping batch
sizes fixed doubles optimizer steps per iteration. To match an FS8/rollout-32
run's physical-time collection and optimizer counts at fixed simulation count:

| Quantity | FS8 reference | FS4 matched counterpart |
| --- | ---: | ---: |
| Rollout decisions | 32 | 64 |
| PPO minibatch decisions | 16,384 | 32,768 |
| D batch windows per source | 65,536 | 131,072 |
| PPO/D epochs | unchanged | unchanged |
| D target tau per optimizer update | unchanged | unchanged |
| Two-frame D decision stride | 4 | 8 |
| Dense imitation reward | 1× | approximately .5× |
| Goal reward | ±50 | ±50 |
| Decision-based training/annealing budget | 1× | 2× |

Microbatch sizes can stay fixed: they control memory/gradient accumulation, not
optimizer-step counts. This table is a controlled-comparison recipe, not an
empirically established optimal configuration. Dense imitation scaling is only
implemented in the isolated experiment at present. Even this comparison does
not equate trajectory-level PPO divergence: identical per-decision KL permits
more cumulative policy change per simulated second at FS4.

## Frozen control-cadence transfer

Frozen checkpoints: FS4 `184227-1` at 1,900,019,712 decisions and FS8 `171925-1`
at 2,600,468,480 decisions. These checkpoints have **different training ages**;
the useful causal comparison is changing execution cadence within each frozen
policy, not attributing the entire between-policy gap to frameskip.

Evaluation uses 128 self-play simulations for 60 seconds, native one-tick
physics, and 75% bounded uniform replay resets. Contact metrics count the onset
of a contiguous contact event, not all contact-containing ticks. Do not compare
these numbers directly with the earlier exploration audit's touch-tick counts.
Age-conditioned occupancy excludes the first two seconds following resets.
Completed floor departures exclude episode/end-of-recording censoring; they
include flips/incidental departures and are not named-skill success labels.

Modes:

- `fresh4`: resample all controls every four ticks.
- `hold8`: hold all controls for eight ticks.
- `jump8`: hold only jump for eight ticks; refresh other controls every four.
- `aerial_controls8`: hold horizontal/vertical/boost/air-roll heads for eight.
- `correlated`: reuse sampling noise for two decisions while recomputing logits.

Means across evaluation seeds 151 and 152:

| Trained policy | Execution | Aerial contact onsets/car/min | Mean jump hold | Completed floor departures with aerial contact |
| --- | --- | ---: | ---: | ---: |
| FS4 | fresh4 | .416 | 101 ms | .79% |
| FS4 | hold8 | .373 | 122 ms | .97% |
| FS4 | jump8 | .432 | 131 ms | .98% |
| FS8 | fresh4 | **1.676** | 76 ms | 5.91% |
| FS8 | hold8 | **1.615** | 135 ms | 6.74% |
| FS8 | jump8 | 1.861 | 142 ms | 8.12% |

Four-tick execution shortens jump holds, yet preserves aerial contacts for
the FS8-trained policy. Selective jump persistence helps the launch statistics
without explaining the much larger learned-policy gap. Its aerial-contact gain
for the FS4-trained policy is only 3.8% on average and changes sign across seeds.
Holding the other aerial controls reduced the FS8 policy's contact rate to
1.211 in seed 151; correlated sampling produced 1.641. Those last two modes
were only screened on one seed. Neither is a justified general fix from this
screen. The two seeds are evaluation replicates, not independent trained actors.

## Matched physical-time continuation experiment

Status: completed for continuation seeds 161/162, with evaluation seeds
1161/1162 and same-seed frozen baselines. All arms begin from the same FS8 actor,
critic, online discriminator and target discriminator.

| Arm | FS | Rollout | D frames / decision stride | Imitation reward multiplier |
| --- | ---: | ---: | --- | ---: |
| FS8 control | 8 | 32 | 2 / 4 | 1 |
| FS4 unscaled | 4 | 64 | 2 / 8 | 1 |
| FS4 scaled | 4 | 64 | 2 / 8 | .5 |
| FS4 scaled, three frames | 4 | 64 | 3 / 4 | .5 |
| Previous arm + endpoint scalars | 4 | 64 | 3 / 4 | .5 |

Each arm uses 128 self-play simulations, 128 iterations, and **8,388,608
agent-physics ticks**. Each iteration collects 256 physics ticks per car,
performs one discriminator update with 8,192 windows per source, and uses two
PPO epochs. PPO minibatches are 2,048 at FS8 and 4,096 at FS4: eight optimizer
steps per iteration in both cases. Learning rates are 3e-4; target tau is .2.
Adam states restart identically. Gamma inherits the source checkpoint's current
physical-time discount; lambda remains time-scaled from .98 at FS4.

Only dense imitation reward is multiplied by .5; goals retain ±50. This is a
physical-time integration approximation (the exact constant-reward discounted
factor is about .5025 for these discounts). Policy and critic observations,
corpus eligibility, and replay reset sampling retain their production semantics.

The three-frame discriminator inserts a zero-weight middle temporal block into
each first MLP layer, preserving the original endpoint blocks and all other
weights. The harness asserts initial target-logit parity on complete expert
windows. Padding/first-valid relative references can differ near resets. The
endpoint-scalar arm explicitly changes visible scalar semantics and retains
flags/boost, making this a real test of the option.

Expert data use the previously cached 100,000 complete windows from 1,418
training matches, with 160 matches reserved. All arms use this same cache and
the existing reset reservoir. The smaller batches, cache, restarted optimizers,
and short horizon make this a local transfer diagnostic, not a production-scale
or from-scratch aerial-learning benchmark. The FS8 control measures how much
the diagnostic itself changes the established policy.

### Results and interpretation

Means across the two continuation/evaluation seeds:

| Condition | Aerial contact onsets/car/min | All contact onsets/car/min | Completed floor departures with aerial contact |
| --- | ---: | ---: | ---: |
| Frozen baseline, FS8 execution | 1.619 | 23.30 | 6.94% |
| Frozen baseline, FS4 execution | 1.680 | 23.55 | 5.90% |
| FS8 continuation control | 1.094 | 17.01 | 4.33% |
| FS4 unscaled, two frames | 1.162 | 22.50 | 3.81% |
| FS4 scaled, two frames | 1.094 | 18.27 | 3.57% |
| FS4 scaled, three frames | 1.090 | 20.11 | 3.46% |
| FS4 scaled, three frames + endpoint scalars | 1.037 | 18.53 | 3.35% |

**No tested arm preserves the original aerial-contact behavior.** The FS8
control itself loses 32% of aerial contacts; FS4's loss is therefore not evidence
of a unique four-tick control failure. Halving imitation reward does not improve
aerial contacts in either seed (1.227→1.109 and 1.098→1.078). Adding the middle
frame produces essentially the same mean contact rate as scaled two-frame
training, with mixed seed-level changes. Endpoint-scalar erasure has no
demonstrated benefit in this short comparison. These results do not select a
long-run FS4 recipe or explain the user's from-scratch multi-frame regression.

The small-batch harness has
four times the production PPO optimizer steps and eight times its D optimizer
steps per physical-time budget. Those rates are matched **between experimental
arms**, but are too different from production to treat a winner as a validated
production recipe. The substantial FS8-control degradation makes that concern
concrete. Reward-rate matching remains a controlled-comparison requirement,
not an empirically established aerial-learning improvement.

The harness also supports `--batch-regime production-rate`: 1,024 simulations,
65,536 D windows per source per update, and PPO minibatches of 8,192 at FS8 /
16,384 at FS4. With two PPO epochs this matches the reference production
optimizer counts per physical tick; 16 iterations gives the same 8,388,608-tick
budget. This control remains to be run. It still uses a smaller collection
batch than production and the same cached experts/restarted optimizers.

The production `--resume` path intentionally rejects changing frameskip and
discriminator stride/window length. The experiment imports weights into a new
isolated run; it is not an ordinary compatible checkpoint resume command.

## Coordinated action exploration

`--joint-actions` is now an opt-in production policy architecture: a single
categorical over all 648 existing control combinations. See
[GAIFO.md](GAIFO.md#joint-action-combinations) for use and checkpoint semantics.

This is a plausible exploration intervention because independent categorical
draws cannot express stochastic correlation between pitch, boost, jump and the
other controls **conditional on the same state**. The actor's shared network
already correlates their preferences across states and can output deterministic
combinations. Thus this is not evidence that the independent actor lacks the
control vocabulary for aerial play. The frozen FS8→FS4 transfer also shows that
an independent actor can execute good aerial contacts at FS4.

A clean warm-start comparison is possible: for each combination, sum the
independent actor's corresponding final-layer weight rows and biases. Its joint
logit is then the sum of component logits; after the same product legality mask,
softmax exactly reproduces the factorized probability distribution. Subsequent
updates allow those combinations to diverge from the product distribution.
The diagnostic harness supports this as `fs4_scaled_three_joint`, asserting
initial masked log-probability and entropy parity with the original actor.
This import is isolated to the experiment, not an ordinary checkpoint resume.

Verification so far: seven CPU tests cover probability/entropy/gradient parity,
correlated choices, legality, PPO, compilation, checkpoint modes, historical
opponents and CALM conditioning. A native CUDA-compiled collection/PPO update
with historical opponents and a viewer checkpoint roundtrip passed. A
one-iteration production-rate three-frame AIL warm-start smoke also passed.
These are correctness checks; the behavioral comparison below is now complete.

The joint-action comparison started from the **aerial-poor FS4 checkpoint**
`184227-1`, step 1,900,019,712, retaining its three-frame/stride-four
discriminator and its dense-reward/goal scale. Seeds 171/172 compare independent
actions against the probability-matched joint lift. Each arm has 1,024
simulations, rollout 64, 16 iterations, and 8,388,608 agent-physics ticks.
It uses the production-rate batch regime above: 256 PPO steps and 16 D steps per
arm. That matches the FS8 reference's optimizer rate, not the faster original
FS4 production rate. Both action-distribution arms use the same regime.
Evaluation uses seeds 1171/1172, 128 simulations × 60 seconds, ordinary FS4
execution, contact onsets and completed floor departures. This remains a short
recovery experiment, not from-scratch acquisition of aerial maneuvers.

The expected benefit is coordinated exploration. The opposing tradeoff is
sample efficiency: 648 separately adjustable action logits can need more data
than reusable independent component preferences. Joint sampling also acts one
decision at a time; aerial follow-through still depends on learned temporal
control and the discriminator's reward signal.

### Joint-action results

Means across continuation seeds 171/172, evaluated at seeds 1171/1172:

| Condition | Aerial contact onsets/car/min | High-aerial contact onsets/car/min | All contact onsets/car/min | Floor departures with ≥.5 s aerial | Floor departures with aerial contact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen FS4 baseline | .4121 | .1523 | 22.04 | 6.82% | .78% |
| Independent continuation | **.4707** | .1367 | 19.34 | 8.76% | 1.14% |
| Joint continuation | **.4707** | .1230 | 19.76 | 8.21% | 1.16% |

**No demonstrated joint-action advantage in this short recovery trial.** The
mean aerial contact rates are exactly equal. Joint actions lose .02344
contacts/car/min in seed 171 and gain .02344 in seed 172. Joint high-aerial
contact rate is lower on average, and the fraction of sustained floor
departures is lower in both seeds. Mean jump holds are nearly unchanged
(103 ms independent / 104 ms joint).

Both continuations improve aerial contacts relative to the frozen baseline,
while reducing overall contacts; that shared change cannot be credited to the
joint representation. Mean PPO KL is similar: .00633/.00637 independent versus
.00629/.00613 joint, so there is no obvious gross update-size mismatch in that
diagnostic. Initial probabilities/entropy were matched, but the joint actor has
more parameters and its sampled trajectories naturally diverge.

The option is usable and remains a plausible from-scratch exploration
experiment. These two short continuations do **not** justify recommending it as
an FS4 or multi-frame aerial fix, nor do they rule out a benefit after longer
training. The tested policies already have concentrated action preferences;
this was not a from-scratch comparison.

## Forward-KL state occupancy

The next controlled recovery experiment adds a separate single-state density
ratio reward to encourage expert coverage. See [FS4_OCCUPANCY.md](FS4_OCCUPANCY.md)
for the forward-KL derivation, clipping/centering, two-seed protocol and independent
held-out coverage diagnostics. The two-seed comparison is complete: weight .3
improves all-age coarse coverage in both seeds, but aerial-contact gains are
seed-dependent and overall contacts fall. This does not establish a reliable
aerial fix. Production's optional forward-KL **main reward** is a distinct
variant using the existing discriminator, without an auxiliary coefficient.

## Artifacts

Under `/tmp/opencode/fs4-recovery/`:

- `recent-runs.json`: equal-budget production-run summaries.
- `cadence-151.json` and `cadence-152.json`: frozen execution sweeps.
- `matched-161.json` and `matched-162.json`: continuation metrics and baselines,
  written as arms complete; matching isolated checkpoints have mode suffixes.
- `smoke-160*`: one-iteration execution/parity checks, **not behavioral evidence**.
- `joint-fs4-{171,172}.json`: independent-versus-joint FS4 continuations and
  same-seed frozen baselines; matching checkpoints use the same prefixes.

Harnesses: `/tmp/opencode/fs4-recent-runs.py`,
`/tmp/opencode/fs4-cadence-transfer.py`, and
`/tmp/opencode/fs4-matched-continuations.py`. Native commands use
`PYTHONPATH=/home/jack/.local/share/goddard/carl/stage:/home/jack/Goddard`,
`OMP_NUM_THREADS=2`, and the repository `.venv/bin/python`.
