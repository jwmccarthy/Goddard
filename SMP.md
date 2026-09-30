# Score-Matching Motion Priors (SMP)

`smp.py` adapts [SMP](https://arxiv.org/abs/2512.03028) to 1v1 Rocket League
and adds the learner-score correction from [SMILING](https://arxiv.org/abs/2410.13855).
It learns from replay *states*, without action labels or a binary discriminator.

## Run

```sh
# Pretrain the expert prior and then train PPO in the same invocation.
uv run python smp.py --replay-dir parsed_replays/ --n-sim 8192 \
  --rollout 64 --trajectory-length 8 --ppo-epochs 8

# Or pretrain once, then reuse the prior with other policies.
uv run python smp.py --replay-dir parsed_replays/ --prior-only
uv run python smp.py --prior-checkpoint checkpoints/smp/priors/prior_smp-<run-id>.pt \
  --replay-dir parsed_replays/ --n-sim 8192 --rollout 64 --ppo-epochs 8

# Inspect completed policy checkpoints (MLP and GRU are supported).
uv run python watch_checkpoints.py --checkpoint-dir checkpoints
```

`--prior-updates` (default 20,000) controls offline score pretraining; an EMA
of the expert weights (`--prior-ema`, default 0.995) becomes the frozen prior. It
periodically writes `prior_*.training.pt`; use `--resume-prior PATH` and a
**total** `--prior-updates` target to continue an interrupted pretraining run.
Policy training writes `smp_*.pt` under `checkpoints/smp/<run-id>/` and resumes
with `--resume-checkpoint PATH --timesteps TOTAL`. A saved expert prior can also
be used without the replay corpus by passing `--replay-reset-fraction 0` and
omitting `--replay-dir`.

## Reward and replay representation

The frozen expert model `g_e` denoises a short joint-scene window containing
the ball **and both cars**. An independent `g_pi` is trained by regression on a
reservoir of earlier policy rollouts, preserving samples from older behavior
modes. Both models see the same diffused clip, timestep, and Gaussian noise.
SMP uses a small fixed ensemble of noise levels (default 8, 15, 22); its
expert denoising errors are normalized by per-level reference errors measured
after pretraining. The SMILING-inspired policy cost is the *unnormalized*
expert-minus-learner denoising-error difference, averaged over noise levels.
Squared errors are reported as mean error per scored feature to keep the PPO
reward scale independent of the clip length; this is a fixed factor relative
to the squared norm in the paper's equation. The imitation reward combines
`exp(-smp_scale * normalized_expert_error)` with `-contrast_weight * cost`,
clipping only the contrast bonus. Goal and physical touch rewards are added
separately.

Car flags stay clean as discrete conditioning inputs; Gaussian diffusion and
the score loss apply to continuous scene features. Relative ball-to-car and
car-to-car motion features help the model learn contact and contest dynamics.
Both player viewpoints are averaged into **one shared scene-prior reward**;
goal, aerial-touch, and flip-reset rewards remain signed per team. Windows
that cross a reset are excluded from imitation scoring; terminal goals still
receive their goal reward.

The expert loader deduplicates paired replay POV files, splits held-out data
by replay, resamples to the simulator frameskip, excludes parser-flagged
physical corrections/discontinuities, and retains genuine touch events. Half
of each pretraining batch follows natural replay frequencies; the other half
balances ordinary play, ball proximity, aerial play, contests, and real
contact/bump clips when present. For physics-valid starts, CARL samples safe
replay states with paired-car control state when available; its hidden control
timers and boost-pad cooldowns cannot be recovered from a scene-only diffusion
sample.
