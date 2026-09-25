# July 31 comparisons

The two launchers provide separate kickoff-only comparisons. Neither replaces
the installed packages or modifies an existing run.

## Dependency-only comparison

Run the **current `basic.py` network and kickoff-only training pipeline** with
either the current dependencies or historical CARL and JARL. Both stacks
select the same explicitly pinned reward variant.

- Historical CARL: `965cba1` (July 31, 15:37 EDT). Built for CUDA SM 75 and 89;
  binary SHA-256: `bdff1d5d2ec6995e8a46d8dd708ede58923c7efe94426f2e541e2d645e52acf5`.
- Historical JARL: `0c6ed3a` (July 30, latest commit before August).
- The later July 31 CARL commit `4fc3940` **must not be used for training**:
  `f863e30` inverted the environment reset guard, silently resetting to
  kickoff on *every action* while reporting no done flags. `965cba1` predates
  this regression. The launcher checks the binary hash and the no-touch reset.

On RunPod, from any working directory:

```sh
/Goddard/.venv/bin/python -B /Goddard/experiments/july31_stack/train.py --stack july31 --verify-only
/Goddard/.venv/bin/python -B /Goddard/experiments/july31_stack/train.py \
  --stack july31 --n-sim 8192 --rollout 32 --epochs 16 --lr 3e-5 \
  --run-name july31-kickoff-compare-20260924
```

For a matched current-stack run, use the **same flags** with `--stack current`
and a distinct `--run-name`. The launcher enforces zero replay reset probability
and skips loading replays. All other BASIC arguments retain their defaults or
can be passed directly to `basic.py` through the launcher. Each run writes its
own TensorBoard logs to `/Goddard/runs/<run-name>/` and checkpoints to
`/Goddard/checkpoints/<run-name>/`.

Watch `Gameplay/timeout_fraction`, `Gameplay/touches_per_1000_steps` (learner
touches per 1,000 learner action steps), and `Gameplay/touch_episode_fraction`
(fraction of completed learner episodes in which the learner touched the ball).
Both progress bars show the last two touch metrics; only completed episodes
contribute to the episode fraction. Historical JARL names its episode reward
metric `Episode/reward`, while current JARL uses `episode/current_reward`.
The launcher also prints a per-update `UPDATE learner_steps=` line to its
standard output (useful when running it in the background).

To isolate the same one-weight kickoff intervention in **current BASIC**, use
the same `train.py --stack july31` or `--stack current` command with
`--reward-variant kickoff-velocity` and a different run name. The experiment
changes only `SeerRewardWeights.kickoff: 0.0 -> 0.1`. Direct current `basic.py`
training now defaults to the proven `0.1` weight; the wrapper explicitly sets
the old `0.0` value with `--reward-variant historical` for a reproducible
baseline.

Compatibility changes are confined to `train.py`: July 31 CARL's discrete-action
constructor did not accept the later `discrete_actions=True` keyword; July 31
JARL reversed its `head`/`foot` constructor names and used the earlier PPO update
API. The launcher preserves BASIC's PPO KL cutoff and uses the exact same
`basic.py` and `rewards.py` for both runs. Historical CARL cannot restore newer
replay-only internal controls, so this comparison deliberately uses kickoffs.

## Full July 31 Goddard baseline

`full_july31.py` runs Goddard commit `a817186` (July 31, 22:19 EDT): its
original `ppo.py`, `rewards.py`, replay-reset plumbing, training checkpoint,
and network. It uses the same pinned CARL `965cba1` and JARL `0c6ed3a` above.
The historical checkout lives at `vendor/goddard_a817186/`, independent of
the current `/Goddard` code. The launcher enforces kickoff-only resets; the
old training script still invokes its replay loader even at reset probability
zero, so the launcher supplies an unused one-row dataset in its place.

```sh
/Goddard/.venv/bin/python -B /Goddard/experiments/july31_stack/full_july31.py --verify-only
/Goddard/.venv/bin/python -u -B /Goddard/experiments/july31_stack/full_july31.py \
  --num-simulations 8192 --rollout-steps 32 --epochs 16 --frameskip 8 \
  --run-name full-july31-kickoff-8192-20260924
```

The full-size command retains the historical defaults for the remaining
flags: 16-step sequences, 256-wide independent GRUs, BF16, learning rate
`1e-5`, 30-second no-touch timeout, and 10-billion learner timesteps.
TensorBoard logs go to `/Goddard/runs/<run-name>/`; checkpoints go to
`/Goddard/checkpoints/<run-name>/`. Each update prints a `learner_steps=` line
with learner touch and timeout rates; TensorBoard logs the same `Gameplay/*`
metrics as the dependency comparison. Touch-episode and timeout fractions are
reported only after completed episodes.

This is a **whole-training-stack baseline**. Its `SeerReward` applies full
zero-sum reward, and its original PPO update does not have the current BASIC
KL early-stop guard, so differences from `train.py` cannot be attributed to
CARL/JARL alone. The CARL pairing intentionally predates the later July 31
reset regression described above.

For a one-parameter reward ablation, add `--reward-variant kickoff-velocity`
and use a different `--run-name`. This changes only the July 31 Seer `kickoff`
weight from its historical `0.0` to Nexto's `0.1`: it rewards moving toward
the ball while the ball is still at center. The default `historical` variant
continues to use the exact archived reward weights.
The separate `--reward-variant ball-progress-15` changes only
`player_ball_progress` from `0.75` to `15.0` to test whether stronger
distance-to-ball progress improves learning from kickoffs.

## Controlled kickoff-learning result

All training runs used 8,192 1v1 simulations, 32-step rollouts, 16 PPO epochs,
frameskip 8, BF16, learning rate `1e-5`, seed 0, the same July 31 CARL/JARL,
and **only kickoff resets**. Within each training-code pair, the *only*
deliberate change was `SeerRewardWeights.kickoff: 0.0 -> 0.1`; initial policy
snapshot files have identical SHA-256 hashes. At approximately 31 million
learner steps, frozen policies played 1,024 kickoff games each against legal
random opponents with evaluation seeds 9210 and 9211, half on each side and
at most 1,500 actions per first episode:

| Training code | Kickoff weight | Games with a touch / 2,048 | Goals for / against |
| --- | ---: | ---: | ---: |
| Goddard `a817186` | 0.0 | 116 | 15 / 21 |
| Goddard `a817186` | 0.1 | 1,140 | 130 / 60 |
| Current `basic.py` | 0.0 | 138 | 13 / 17 |
| Current `basic.py` | 0.1 | **1,333** | **146 / 76** |

The reward intervention increases both ball contact and goal-scoring, not
merely training reward. The stronger ball-progress-only ablation improved
contact, but less than kickoff velocity: at ~31 million steps, in 1,024 games
with seed 9210 it touched in 322 games and scored 39 goals versus 577 games
and 55 goals for the kickoff variant of the same historical training code.

Reproduce the current-BASIC evaluation with either seed:

```sh
/Goddard/.venv/bin/python -B /Goddard/experiments/july31_stack/evaluate_kickoff.py \
  --seed 9210 --num-simulations 1024 --max-steps 1500 \
  /Goddard/checkpoints/basic-july31deps-kickoff-baseline-8192-20260924/policy_000030685184.pt \
  /Goddard/checkpoints/basic-july31deps-kickoff-reward-8192-20260924/policy_000031139071.pt
```

The current `rewards.py` now enables the confirmed `0.1` kickoff weight by
default. The experiment launcher continues to pin `0.0` explicitly for its
`historical` reward variant so the original comparison remains reproducible.
