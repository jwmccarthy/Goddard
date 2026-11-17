# Aerial imitation-reward audit — 2026-11-15

Follow-up measurements of real tick phases, whole-match-held-out shortcut
mitigations, and FS4 exploration interventions:
[FS4_AERIAL_FOLLOWUP.md](FS4_AERIAL_FOLLOWUP.md).

## Conclusion

The checkpoints do **not** show a general discriminator penalty for being
airborne. Human aerials score substantially better than human ground driving;
the same ordering holds for policy states. The evidence instead points to a
combination of:

1. A learned penalty for some **policy ground-to-air transition patterns**,
   especially in the older four-frame discriminator. This is not a universal
   jump penalty: a controlled jump intervention initially improves reward.
2. Very limited jump exploration and a large gap between expert and policy
   aerial occupancy/quality.
3. An always-negative, per-decision imitation objective whose costs disappear
   at goals. This creates a duration incentive and is considerably stronger at
   frameskip 4 relative to the fixed ±50 goal rewards.
4. Short local discriminator context and finite-rollout credit assignment,
   which do not explicitly identify a complete aerial maneuver or its outcome.

These are measured mechanisms and implementation properties, not a completed
multi-seed training ablation establishing one root cause. In particular,
negative rewards alone do not establish suppression: PPO optimizes advantages,
and a less-negative aerial reward can encourage aerials.

## Scope and method

Frozen checkpoints (directory prefix `runs/gaifo-20261115-`):

| Suffix | Decisions | Frameskip | Discriminator length / decision stride |
| --- | ---: | ---: | --- |
| `184227-1` | 1,900,019,712 | 4 | 3 / 4 |
| `171925-1` | 2,600,468,480 | 8 | 2 / 4 |
| `161646-1` | 2,100,297,728 | 4 | 4 / 2 |

- Audited the full 38,080-file, 1,578-match compatible replay corpus.
- Scored 131,072 uniformly sampled eligible expert windows for each layout
  through the saved target and online discriminators.
- Evaluated each frozen policy in 256 native simulations for approximately
  60 seconds, using the training 75% replay / 25% kickoff reset mixture.
  The audit uses a bounded 65,536-state uniform reset reservoir, rather than
  training's 3M-state reservoir, with the same four-second edge exclusions.
- Current FS4 was evaluated with seeds 93, 94, and 95; comparators with seed 93.
  These are evaluation seeds, not independently trained policies.
- Used production `TransitionRollout`, reward composition, and GAE. Retained
  every fourth decision for scoring summaries: 233,472 samples per FS4 seed and
  118,784 for FS8. Rollouts round up to 60.8 / 61.87 seconds respectively.
- No optimizer updates were performed. The frozen-reward GAE diagnostics differ
  from live training, which updates its discriminator before scoring rollouts.
- Aerial means ego not on a surface and height >182.5 uu; high aerial means
  >500 uu. Ground-to-air means a complete window starts on a surface and ends
  airborne; this includes hops, flips, and wall departures, not just aerial
  attacks. Floor driving means on a surface below 100 uu.

## 1. The reward sign is correct; aerial states are preferred

`training/imitation.py:249` labels generated windows 1 and expert windows 0.
`training/discriminator.py:267–274` returns, for this configuration:

```python
reward = log_sigmoid(-generated_logit).mean(heads)
```

More expert-like windows receive higher (less-negative) reward. There is no
label/reward sign inversion. Ego and global heads have equal reward weight.

Mean **imitation-only reward per decision**, current FS4 target, seed 93:

| Behavior | Human replay windows | Policy windows |
| --- | ---: | ---: |
| Floor driving | −0.594 | −1.716 |
| Ground-to-air transition | −0.332 | −1.819 |
| Aerial | −0.204 | −1.344 |
| High aerial | −0.135 | −1.171 |
| Aerial within 350 uu of ball | −0.151 | −1.094 |

Excluding incomplete histories and terminal transitions preserves the ordering
(floor −1.722, aerial −1.366, ground-to-air −1.818). Across all three FS4 seeds,
floor reward ranges −1.716 to −1.685, aerial −1.344 to −1.270, and ground-to-air
−1.819 to −1.784. Both discriminator heads prefer established aerial states;
the global head is generally more punitive than the ego head.

### The learned transition penalty differs between discriminators

Score the **same FS4 policy windows** with the FS8 target, selecting the first
and last frames so its required `[t−32,t]` input is exact:

| Scoring model | Floor driving | Ground-to-air | Aerial |
| --- | ---: | ---: | ---: |
| Current FS4 target | −1.716 | −1.819 | −1.344 |
| FS8 target, same trajectories | −1.371 | −1.154 | −1.222 |

Thus the same ground-to-air examples are worse than floor driving under the FS4
target but better under the FS8 target. This isolates a difference in learned
reward functions from a difference in evaluated trajectories. It does **not**
isolate window length, training age, or the distributions on which those
discriminators were trained.

The older FS4 checkpoint scores its own ground-to-air windows −2.298 versus
floor −1.705 and established aerial −1.356. Its window spans 24 ticks, compared
with 32 ticks in the current FS4/FS8 pair, so transition-category averages are
not identical event selections across all three runs.

In current FS4 seed 95, floor-to-air windows with the flipped flag set average
−2.008, versus −1.713 without it; wall-to-air windows average −1.278. The weak
region includes ordinary low-height flips, not just attempted high aerials.

## 2. Paired intervention: jumping itself is not immediately punished

From 2,048 sampled floor replay states, duplicate the entire simulation. Suppress
ego jump for 0.2 seconds in one branch and hold it for 0.2 seconds in the other;
then follow the same frozen policy. Other action heads and the opponent use
common random uniforms, with each branch adapting to its own observations.
Histories start empty in both branches. Compare paired imitation-only sums,
excluding pairs where either original episode ends within the comparison.

| Horizon | Surviving grounded pairs | Jump minus no-jump imitation return | Paired standard error |
| --- | ---: | ---: | ---: |
| 0.267 s | 2,046 | +0.080 | 0.022 |
| 0.533 s | 2,046 | +0.470 | 0.082 |
| 1.067 s | 2,031 | +0.272 | 0.213 |
| 2.133 s | 1,946 | −4.074 | 0.578 |

At 0.533 seconds, jump-branch mean height is 187 uu versus 39 uu in the control.
For the subset with the ball above 182.5 uu and within 2,000 uu, the 1.067-second
return difference is +0.864; it turns negative by 2.133 seconds (−1.818).

This is evidence against an unavoidable immediate jump penalty. The later
negative difference includes follow-through, landing, position changes, and
policy responses. It is not an intervention demonstrating a competent aerial;
survival conditioning and replay-state initialization also limit generalization.
The standard errors describe these sampled state pairs, not training-seed
uncertainty. Observational ground-to-air penalties must not be interpreted as
the causal effect of pressing jump in otherwise identical situations.

## 3. Aerial examples are present, but the policy visits them much less

For the current layout, the full eligibility audit finds:

- All candidates: **30,381,696**; eligible: **25,777,370**.
- Aerial candidates: **4,720,625**; eligible: **3,980,521**.
- Aerial share changes only **15.54% → 15.44%** after filtering.
- Current FS4 and FS8 retain exactly the same number of aerial endpoints.

Therefore correction/terminal filtering is not removing most aerial examples or
creating a large aerial-frequency disparity between these two layouts. Sampling
is uniform over eligible endpoints, rather than balanced over maneuvers. No
maneuver enrichment is enabled in these checkpoints.

Expert aerial occupancy is about **15.4%**, versus **5.56–5.86%** in current FS4
evaluation and **8.66%** in FS8. Human high-aerial occupancy is about **7.9%**,
versus roughly **1.8–2.0%** in current FS4.

Current FS4 seed 95, conditioned on pre-action floor states where jump is legal:

- Mean jump probability: **2.15%** per decision.
- **86.6%** of states have jump probability below **0.1%**.
- Mean jump entropy: **0.0206 nats**, versus a binary maximum of 0.693.

This confirms a narrow jump policy even after removing action-mask effects.
It does not establish whether low exploration caused or followed the learning
gap. Ground-to-air GAE is not consistently strongly negative across all runs,
and raw GAE signs alone do not establish suppression after PPO normalization.

Replay resets also provide some aerial exposure without learning the launch:
in seed 95, the first two seconds occupy 13.7% of samples but contain 24.3% of
aerial samples. Aerial occupancy drops from 10.2% in that interval to 5.0%
afterward. Most aerial samples still occur later; they are not all reset artifacts.

## 4. Negative episodic costs create a real objective mismatch

`training/imitation.py:434–461` adds negative imitation reward once per decision,
zeros it on goal-terminal transitions, and adds ±50 for the goal.
`training/rollout.py:137–146` ends bootstrapping at episode boundaries. There is
no absorbing-state imitation continuation after a goal. Scoring **or conceding**
therefore ends the future stream of imitation costs.

Recent logged means:

| Run | Imitation / physical second | Mean undiscounted imitation cost / episode |
| --- | ---: | ---: |
| Current FS4 | −48.99 | −838.7 |
| FS8 | −18.95 | −316.8 |
| Older FS4 | −51.24 | −781.9 |

These are measurements at each run's latest logging age, not an equal-budget
training comparison. Dividing logged rewards by ticks does not time-normalize
the optimized reward. Goal rewards stay fixed at ±50.

As a scale diagnostic, a hypothetical constant stream at those mean rates has
discounted value approximately **−164.9 FS4 versus −64.3 FS8**. Those are not
actual action values; state-dependent rewards and goal prospects matter. They
show why duration and goal incentives deserve an explicit objective-level test,
rather than interpreting negative reward as an aerial-specific penalty.

On sampled current-FS4 concession transitions, mean critic value is −53.10 and
terminal reward −50, giving raw one-step advantage +3.10. This is compatible
with the duration pressure, but also with ordinary critic error near an already
inevitable loss; it is **not proof that the agent deliberately concedes**. The
older FS4 checkpoint does not show the same positive mean concession advantage.

## 5. Credit and discriminator context are local

- Current FS4's discriminator sees only **0.267 seconds**, and reward heads are
  unconditional on a maneuver identity or intended future outcome. It can favor
  expert-like midair states without teaching how to reach them from the floor.
- Rollout 32 spans **1.067 seconds** at FS4 versus **2.133 seconds** at FS8. Beyond
  a rollout boundary, credit depends on critic bootstrapping.
- Gamma/lambda are correctly time-scaled. Both runs have roughly a **3.35-second
  discount e-folding time** and **1.11-second GAE-trace e-folding time**. A TD
  residual two seconds later has trace weight about **0.164**, when it is within
  the same rollout. FS4 does not have an unscaled-gamma bug.
- These checkpoints use a current-state critic even though imitation reward
  depends on history. That can add prediction noise; overall logged explained
  variance is nevertheless around **0.915** for current FS4. The evidence does
  not support blaming a globally broken critic.
- Flags and orientations retain clean discrete/directional signals; the 0.002
  motion noise covers position and velocity fields, not flags or orientation.
  Zeroing all flags at inference improves policy ground-to-air scores but also
  changes ordinary floor scores dramatically. This out-of-distribution probe
  does **not** establish a flag bug or justify removing flags without retraining.

## 6. Follow-up: jump holding across frameskip boundaries

CARL **does preserve a held jump** between calls to `step`. In the staged native
source, `include/Physics/Controls/Context.cuh:82–83` computes:

```cpp
ctx.jumpPressed = c.jump && !s.lastJump;
s.lastJump = c.jump;
```

`lastJump` is persistent car state. `Controls/Aerial.cuh:19–25,50–74` continues
first-jump acceleration while jump remains held, up to the 0.2-second hold
window. Consecutive true actions do not create consecutive new jump presses.
A release followed by a press can trigger a double jump or dodge when eligible.

Native event-capture verification used the same seven-head discrete action path
as training. Hold for 24 ticks, release for 8, press again for 8:

- FS4: six consecutive hold decisions produce **one** initial jump event.
- FS8: three consecutive hold decisions produce **one** initial jump event.
- Neither hold produces a double jump or dodge. The later release/repress
  produces exactly one double jump.
- Heights match exactly at shared times: **88.68756 uu at tick 24**, **116.00888
  at tick 32**, and **159.88469 at tick 40**.

The policy's physical hold duration does differ. A separate frozen evaluation
(seed 96, 256 simulations × 60 seconds, same reset mixture) measured complete
true-action bursts initiated from the floor, excluding episode-censored bursts:

| Policy | Bursts | Mean hold | Held at least 0.2 s |
| --- | ---: | ---: | ---: |
| Current FS4 | 6,852 | **97.4 ms** | **14.9%** |
| FS8 | 7,271 | **134.6 ms** | **23.9%** |
| Older FS4 | 6,452 | **113.3 ms** | **22.3%** |

A 0.2-second hold requires six consecutive decisions at FS4 versus three at
FS8. The current FS4 policy actually releases sooner in physical time, which can
reduce first-jump lift. Short holds are also legitimate for flips and other
maneuvers, so this does not prove a control bug or explain the entire learning
gap. It is a concrete exploration/control difference worth tracking alongside
the discriminator's ground-to-air scores.

## Recommended controlled tests, in priority order

### Follow-up: onset when introducing multi-frame discrimination

The user's observation that the regression begins with multi-frame discrimination
prioritizes temporal reward learning over a jump-hold implementation defect.
No single-frame production checkpoint remains in the available run directories.

Temporal diagnostics on complete current-FS4 windows show a source-domain gap:
the median ego displacement-minus-trapezoidal-velocity residual over 16 ticks is
**11.31 uu for expert aerials versus 0.92 uu for policy aerials**. The parser
resamples continuous fields and repairs frozen motion by interpolation, while
flags use nearest-source sampling; native trajectories have different numerical
consistency. This residual also includes genuine within-interval acceleration
and contacts, so its difference alone does not prove a discriminator shortcut.

Frozen-input sensitivity probes narrow the suspect region. Replacing only ego
position deltas with trapezoidal velocity integration changes policy
ground-to-air reward from **−1.819 to −1.673**, while ordinary floor reward changes
by only −0.0012. Freezing ego flag transitions improves policy ground-to-air
reward by **+0.104**, but worsens expert ground-to-air reward by **−0.278**.
Temporal boost, flags, and position/velocity relationships are plausible cues;
these synthetic windows are not physically replayed counterfactuals.

I also trained diagnostic discriminators on fixed expert/policy data, holding
the endpoint examples and minibatch selections fixed: one endpoint, one endpoint
repeated into three slots, two endpoints, three frames, and three frames without
flags. Each used two initialization seeds, 2,000 updates, and 2,048 examples per
source per update. Expert evaluation holds out windows, not whole matches;
policy evaluation uses a different simulation seed. These are offline classifier
tests, not additional RL runs.

The offline tests **did not reproduce a universal temporal takeoff penalty**:
in every variant, policy ground-to-air examples still scored above floor
examples. Three-frame takeoff-minus-floor reward was +0.200 to +0.230, versus
+0.213 to +0.251 with one endpoint. Thus adding temporal input alone to this
fixed dataset is insufficient to reproduce the trained checkpoint's inversion.
The evolving policy/discriminator interaction, training age, and potentially
event-phase fidelity remain unresolved. A matched online one-frame/multi-frame
ablation is the decisive next test; flags removal is not yet justified as a fix.

Additional artifacts: `temporal-probe.json`, `discriminator-ablation-101.json`,
and `discriminator-ablation-102.json` in the audit directory. Scripts are
`/tmp/opencode/aerial-temporal-probe.py` and
`/tmp/opencode/aerial-discriminator-ablation.py`.

### Experiments

1. **Match the objective's physical time scale:** test an explicit imitation
   multiplier `frameskip / 8`, keeping event rewards event-based, and match
   discriminator offsets, rollout duration, schedules, and evaluation budgets.
2. **Test terminal-duration handling within AIL:** compare a principled
   absorbing-state or fixed-horizon formulation. Simply adding a positive
   constant can replace termination pressure with survival pressure; it is not
   a neutral fix in variable-length episodes.
3. **Log the bottleneck directly:** reward/advantage by launch, established
   aerial, and recovery; legal-state jump entropy; contact-time aerial onsets;
   reset age; separate ego/global scores. Track complete launch-to-outcome clips.
4. **Ablate temporal discrimination and exploration separately:** same data and
   time budget, matched two-frame offsets first; then train a flags ablation or
   adjust jump exploration in separate runs. Inference-only flag removal is not
   a valid training ablation.
5. **Test complete-maneuver conditioning / longer context:** use intact expert
   launch→flight→contact/recovery sequences within AIL, while retaining uniform
   bounded replay resets. Better scores for isolated airborne states are not
   sufficient evidence of learning the complete maneuver.

## Artifacts and reproducibility

Artifacts: `/tmp/opencode/aerial-reward-audit/` contains immutable model snapshots,
expert/reset samples, policy samples, per-category online/target scores, source
census, logs, paired intervention results, and return/exploration summaries.

Scripts:

- `/tmp/opencode/aerial-reward-audit.py`
- `/tmp/opencode/aerial-source-audit.py`
- `/tmp/opencode/aerial-jump-branches.py`
- `/tmp/opencode/aerial-jump-holds.py`

Example invocation from the repository root:

```bash
PYTHONPATH="$HOME/.local/share/goddard/carl/stage:$PWD" OMP_NUM_THREADS=2 \
  .venv/bin/python /tmp/opencode/aerial-reward-audit.py gaifo-20261115-184227-1 --seed 95
```

No training defaults, rewards, observation layouts, replay files, or checkpoints
were modified by this audit.
