# Diffusion Imitation from Observation

`difo.py` runs DIFO alongside `gaifo.py`, sharing the native environment,
actor/critic, PPO loop, expert sampler, generated-history replay, bounded uniform
replay resets (four-second period-edge buffers), self-play, logging and saves.
It does not change the running GAIfO trainer, replay files or native binaries.

```bash
python difo.py --replay-dir replays/rlcs/parsed/ \
  --n-sim 8192 --rollout 32 --frameskip 8 \
  --factorize --mlp-discriminator --discriminator-relative \
  --discriminator-window-length 2 --discriminator-decision-stride 3 \
  --discriminator-no-flags --discriminator-no-boost --compile
```

Use the same staged CARL build/PYTHONPATH required by GAIfO's ego-controller
observation schema. The example is a configuration, not a validated aerial
training recipe. Diffusion costs more memory/compute than the scalar GAIfO head;
reduce discriminator update/inference microbatch sizes if necessary.

## Objective and architecture

Following [DIFO (NeurIPS 2024)](https://arxiv.org/abs/2410.05429) and its
[reference implementation](https://github.com/NTURobotLearningLab/DIFO), predict
Gaussian noise in the final scene conditioned on preceding scenes, diffusion
timestep and expert/agent class. Each class's energy is its mean squared
noise-prediction error. With this repository's expert=0 convention:

```
agent_logit = scale * (expert_energy - agent_energy)
loss = bce_weight * BCE(agent_logit, agent_label)
     + mse_weight * mean(expert_energy * (1 - agent_label))
reward = log(sigmoid(-agent_logit))
```

Only expert examples contribute to auxiliary denoising. As in the reference,
the auxiliary mean includes zeros for agent examples (half the expert-only
mean for balanced batches). BCE is not multiplied by GAIfO's factor of two.
Classification still backpropagates through both class energies on both sources.
Training samples timesteps uniformly across the full cosine DDPM schedule.
Reward samples default to the middle half of the schedule. There is no reverse
diffusion rollout and no diffusion action policy.

This is a scene-window adaptation, not an exact reproduction of the paper's
task-specific U-Nets. Scene encoders/history architectures are transferred from
GAIfO; a conditional MLP predicts noise in **fixed observed scene features**,
not a freely learned denoising target. Ego heads predict the ego car and relative
ball/opponent positions; global heads predict the full scene. Preceding frames
alone condition the prediction; the clean successor is never passed as a
condition. An empty reset history uses an explicitly marked zero sentinel.
Hidden flags/fuel are masked in both conditions and noise/loss dimensions.

Unlike the reference's independent class noise draws, both class predictions
use the same timestep/noise to reduce variance of the energy difference. Their
individual energy expectations are unchanged, but stochastic BCE/reward estimates
need not be identical. Phase-paired training also shares draws across phases.

## Parameters

All PPO/GAIfO command-line options remain available and meaningful, except
`--amp`: least-squares AMP imitation is incompatible with DIFO's source BCE and
is rejected. `--bf16` remains supported and means mixed precision.

- `--frame-embedding`, `--discriminator-hidden`, `--mlp-discriminator`, `--gru`:
  scene/history encoder sizes and architecture; hidden also sizes the denoiser.
- Window length/stride, relative windows, factorization/ego-only, flags/fuel
  hiding, rounding, mirroring and training motion noise retain their roles.
  New runs default to a two-frame short window and decision stride 3; omitted
  length/stride inherit their saved values on resume.
- `--discriminator-hierarchical` adds the same short/medium/long layout described
  in [GAIFO.md](GAIFO.md#optional-multi-timescale-adversarial-rewards). Each scale
  has its own diffusion heads: the same successor is diffused, with a different
  trailing history conditioning it. Medium/long conditions use temporal
  transformers. Denoising/classification losses and transformed rewards are
  averaged equally across scales; diffusion samples are still averaged within
  each head before the reward transform. Batch defaults, shared boundary-safe
  sampling, per-scale metrics and fresh-run/resume rules match GAIfO.
- Target EMA, historical generated windows, microbatching, optimizer settings,
  maneuver sampling and experimental real-tick phase consistency transfer.
- Goal reward and optional/default-off RewardPlus transfer unchanged. They can
  compete with distribution matching; previous GAIfO reward magnitudes are not
  calibrated to DIFO. Automatic goal calibration searches **DIFO** logs.
- DIFO defaults to `--discriminator-log-reward`. The explicit extension
  `--no-discriminator-log-reward` selects GAIfO's softplus reward instead.

| DIFO option | New-run default |
|---|---:|
| `--diffusion-steps` | 1000 |
| `--diffusion-mse-weight` | 1 |
| `--diffusion-bce-weight` | 1 |
| `--diffusion-logit-scale` | 10 |
| `--diffusion-reward-samples` | 1 |
| `--diffusion-reward-t-min` | steps / 4, inclusive |
| `--diffusion-reward-t-max` | 3 × steps / 4, exclusive |

Multiple reward samples average energies **before** the reward nonlinearity.
Reward inference is stochastic even in eval mode. Increasing samples reduces
Monte Carlo noise but adds compute. DIFO-specific omitted options inherit from
the checkpoint; all are checked strictly on resume. As with GAIfO, other
non-inherited options must be repeated when resuming. Run directories are
`runs/difo-*`; checkpoints declare algorithm `difo` and reward objective
`difo-scene-window-v1`. Cross-algorithm resume is rejected, including GAIfO
discriminator weights. Online/target diffusion state, optimizers, expert/history
state and RNG are saved through the existing checkpoint machinery.

The implementation encodes history once per head even for multi-sample rewards,
shares the first denoiser projection between class queries, and batches both
class branches into one network call. DDPM signal/noise scales and time features
are precomputed. Training and rewards retain GAIfO's independent microbatch
limits and compiled callables; GRUs use the existing eager fallback.

This implementation does not provide stronger policy-distribution convergence
guarantees than GAIfO. Gameplay benefit, including aerial behavior, requires a
controlled training comparison; the shared defaults are not a claim of parity
with the paper's tuned experiments.

Correctness tests: `python -m unittest training.test_diffusion`.
