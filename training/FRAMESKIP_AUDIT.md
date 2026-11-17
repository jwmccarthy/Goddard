# Frameskip and aerial-touch audit — 2026-11-15

**Later follow-up:** [FS4_RECOVERY_AUDIT.md](FS4_RECOVERY_AUDIT.md) includes
newer equal-budget comparisons showing a later aerial-specific FS4 deficit,
plus frozen-policy cadence-transfer and matched-time continuation experiments.
The early-run comparison below is a historical snapshot.

## Finding

Frameskip-4 and frameskip-8 training are not equivalent experiments, even when
their discriminator windows span the same number of physics ticks. The current
frameskip-4 run is slightly ahead at equal simulated experience. An older
frameskip-4 run does lag later; the available runs do not isolate a single cause.

The fractional observation rate (for example, 7.5 Hz) does not create fractional
ticks, rounding, or interpolation. Replay sampling uses integer row strides.

## Recorded runs and matched budgets

TensorBoard global step counts **agent decisions**, not physics ticks. Multiply
it by frameskip to compare simulated experience. The following are means over
the indicated **agent-physics-tick** ranges, not comparisons at the same plotted
step or the same wall-clock time:

| Run suffix | Frameskip | D window / stride | Tick range | Aerial contacts / 1,000 ticks |
| --- | ---: | --- | --- | ---: |
| `171925-1` | 8 | 2 / 4 | 2.5–3.0 billion | 0.04184 |
| `184227-1` (current) | 4 | 3 / 4 | 2.5–3.0 billion | 0.04694 |
| `161646-1` (earlier) | 4 | 4 / 2 | 2.5–3.0 billion | 0.04648 |
| `171925-1` | 8 | 2 / 4 | 7.5–8.0 billion | 0.12578 |
| `161646-1` | 4 | 4 / 2 | 7.5–8.0 billion | 0.07661 |

The full directory prefix is `runs/gaifo-20261115-`. The current run had not yet
reached the final comparison range at audit time. These are single-seed training
runs, not replicated causal ablations.

## What the touch metric actually counts

`rewards/diagnostics.py` accumulates one Boolean per car and **decision interval**:

```python
touches & ~current.car_on_ground & (current.car_position[..., 2] > 182.5)
```

It divides the sum by `agent_decisions * frameskip / 1000`. The denominator is
correctly time-normalized. The numerator is not a distinct-touch event count:

- CARL's `car_ball_touches` means there was contact somewhere in the last action
  hold, based on `ballContactTick > tickCount - touchWindow` in
  `include/Physics/Observations.cuh`.
- Sustained contact can contribute to several short intervals but fewer long
  ones; multiple contacts within one interval collapse into a single Boolean.
- Height and grounded status are checked at the interval endpoint, not the
  actual contact tick. Crossing the height threshold or landing can change
  classification as the interval length changes.

A native one-tick simulation of the **same trajectory** (frozen frameskip-8
policy, actions held eight ticks, 256 simulations for 7,200 ticks each) produced:

| Counting cadence | Existing aerial-window metric | Distinct contact onsets classified at touch time |
| --- | ---: | ---: |
| 4 ticks | 0.18646 | 0.14433 |
| 8 ticks | 0.17457 | 0.14462 |

The small difference in onset rates comes from excluding episode-ending
intervals separately at each cadence. Counting all ticks directly gives 0.14540.
Thus the tested counting bias **raises** the frameskip-4 value by about 6.8%; it
does not explain a depressed frameskip-4 training curve. The metric is useful
within a fixed cadence but is only approximately comparable across cadences.

## Training differences that remain

For `171925-1` versus `184227-1`, saved configurations differ only in frameskip,
discriminator window length, and quantities derived from those or schedule age.
Both use factorized relative MLP discrimination, goal weight 50, noise 0.002,
8,192 simulations, rollout 32, and a 75% replay-reset mixture with a 3M-state bank.
Nevertheless:

1. **Different discriminator input:** both span 32 ticks, but frameskip 8 sees
   `[t-32, t]`, whereas frameskip 4 sees `[t-32, t-16, t]`. The latter is a larger
   MLP input with an additional motion constraint. Equal span is not equal data.
2. **Different reward rate:** `training/imitation.py::imitation_rewards` adds one
   discriminator reward per decision without multiplying by elapsed time.
   Frameskip 4 receives twice as many such rewards per physical second for the
   same per-decision score. Goal rewards remain ±50 per goal. At 2.5–3B ticks,
   mean imitation reward per tick was **−0.4453** for frameskip 4 and **−0.1847**
   for frameskip 8 (2.41× magnitude). Different logits contribute in addition to
   the 2× decision-rate difference. These negative imitation costs change the
   balance against goals and episode duration; they are not an aerial-specific
   penalty. Time-normalizing the logged metric does not normalize the objective.
3. **Different rollout and update cadence:** 32 decisions cover 128 ticks
   (1.067 s) at frameskip 4 versus 256 ticks (2.133 s) at frameskip 8. This changes
   bootstrap boundaries and the number of policy/discriminator/target updates
   per unit of simulated experience.
4. **Different control/exploration cadence:** actions, including jump, boost,
   and air roll, are reselected at 30 Hz rather than 15 Hz. Identical entropy
   settings and per-decision action distributions do not imply identical
   trajectories or temporally correlated exploration.

Gamma and GAE lambda **already are time-adjusted**, using exponents
`frameskip / 4` in `training/arguments.py` and `gaifo.py`. They are not simply
left unscaled. Annealing schedules and TensorBoard's global step still use
agent decisions.

## Additional frozen-policy check

Using the same frameskip-8 checkpoint, seed 93, sampled actions, 256 simulations,
native kickoff resets, and 60 seconds per simulation:

| Action hold | Exact per-tick aerial-contact onsets / 1,000 ticks | Airborne fraction |
| --- | ---: | ---: |
| 8 ticks | 0.14540 | 5.625% |
| 4 ticks | 0.16927 | 5.657% |

This one short evaluation found no inherent loss of aerial contacts from
four-tick action holds. It changes the action sampling rate of a frozen policy,
not the training process, and uses kickoff-only resets rather than the training
replay-reset mixture. It cannot identify which training difference causes the
older run's gap.

## Controlled next comparison

Before attributing a gap to frameskip itself:

- Compare at equal agent-physics ticks.
- Use exactly matching discriminator offsets: the frameskip-4 counterpart of
  frameskip 8 / stride 4 / length 2 is **frameskip 4 / stride 8 / length 2**.
- Match rollout duration (64 decisions at frameskip 4 versus 32 at frameskip 8),
  accounting explicitly for the resulting batch/update-frequency changes.
- Time-normalize dense imitation reward relative to event rewards, with an
  explicit checkpoint-compatible option rather than silently changing existing
  runs. For an eight-tick reference scale, multiply imitation reward by
  `frameskip / 8`; this is a proposed experiment, not an implemented flag.
- Evaluate with contact-onset counts and aerial classification at the contact
  tick, on common reset distributions and multiple seeds.

Audit scripts, full configuration snapshots, matched-budget statistics, and
native evaluation outputs are under `/tmp/opencode/aerial-cadence-audit/`, with
scripts `/tmp/opencode/aerial-cadence-audit.py` and
`/tmp/opencode/aerial-cadence-native.py`.

This investigation does not change training behavior or restart the active run.
