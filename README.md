# Goddard

GPU-resident CARL training with three entrypoints:

- `ppo.py`: PPO with gameplay rewards.
- `gaifo.py`: PPO with a learned imitation reward.
- `dac.py`: GAIfO with DAC-style absorbing states and terminal rewards.

Shared runtime code lives in `training/`, `imitation/`, and `rewards/`.
`replays/` provides Ballchasing download, dense replay parsing, and reset states.
Replay data and training runs are separate from source code.
`vis/` provides the retained checkpoint viewer (`python -m vis --help`).

## 10 Hz discriminator observations

Use the existing dense `dac_parsed` data without reparsing. Policy decisions
remain 30 Hz; only spacing inside discriminator windows changes:

```bash
uv run --no-sync python dac.py \
  --n-sim 8192 --rollout 32 --frameskip 4 \
  --timesteps 100_000_000_000 --ppo-lr 3e-4 --ppo-lr-end 1e-5 \
  --discriminator-log-reward --factorize --mlp-discriminator --discriminator-relative \
  --discriminator-window-length 8 --discriminator-decision-stride 3 \
  --discriminator-history-interval 8 \
  --replay-dir replays/rlcs_2026_1v1_playoffs/dac_parsed
```

Eight observations at 10 Hz span 0.7 seconds. Use length 2 for two bookends
100 ms apart. Stride defaults to 1, preserving the original decision spacing.

See [10 Hz details](imitation/TEN_HZ.md), [GAIfO](imitation/README.md), and
[DAC](DAC.md).

`--amp` optionally selects an Adversarial Motion Priors-style least-squares
discriminator and bounded [0, 1] imitation reward for both GAIfO and DAC. It is
off by default and replaces, rather than combines with, log/odds reward flags.
See [AMP semantics](imitation/README.md#amp-least-squares-objective-and-bounded-reward).

`--target-discriminator` optionally disables online accuracy gating and uses a
frozen Polyak-averaged copy for all imitation rewards. Its default eight-update
EMA span gives tau = 2/9; `--target-discriminator-tau` overrides it. Fresh-rollout
`discriminator/exploitation_gap` logs averaged-minus-online mapped reward:
persistent positive gaps suggest raising tau manually. See
[target smoothing, diagnostics and resume behavior](imitation/README.md#polyak-target-discriminator-and-exploitation-gap).

Discriminator training/validation match expert/generated history masks to avoid
a reset-padding shortcut; stored data and reward inference remain unchanged.
An explicit `--discriminator-learning-rate` on resume now overrides the saved
optimizer LR; omitting it preserves that LR. See
[padding](imitation/README.md#matched-reset-history-padding) and
[LR resume semantics](imitation/README.md#discriminator-learning-rate-on-resume).

`--discriminator-motion-noise-scale 0.001` (the default for new invocations)
adds small Gaussian noise to normalized position and linear/angular velocity
inputs from **both** agent and expert windows during discriminator training.
Set it to `0` to disable. Policy observations, physics and reward inference stay
clean. See [noise semantics](imitation/README.md#discriminator-motion-noise).

`--discriminator-contrastive` optionally adds temporally correlated, gated car
position augmentation and clean/perturbed logit consistency on both sources
(requires at least three discriminator frames). It defaults off; reward inference
stays clean. See [contrastive semantics](imitation/README.md#temporal-contrastive-consistency).

For the separate 5M-row, parse-time aerial-stratified corpus, see
[aerial dataset generation](replays/AERIAL.md). This does not replace existing
replay data or automatically switch a training run.

For success-only goal touches, saves and completed air carries, see
[successful expert selection](replays/SUCCESSFUL.md). The separate exporter keeps
long observation clips and selected causal windows, preserves the original
splits/reset bank, and excludes failed or unresolved events without a fallback.
The authorized full export is ready at `.autoresearch/successful-v2/dataset`:
**129.02 MiB**, **2,155 successful clips** and the unchanged **500,000-state reset
bank**. See its [distribution report](.autoresearch/successful-v2/distribution_report.md).
It does not switch any running trainer; labels remain rule-qualified, not
manually verified skill annotations.

## Independent actor/critic history

For a windowed MLP actor and critic, add:

```bash
--mlp-policy --mlp-critic --policy-window-length 16 \
  --discriminator-window-length 2 --discriminator-decision-stride 3
```

At frameskip 4, both actor and critic use consecutive **30 Hz** observations
spanning 0.5 s. The discriminator independently samples its **10 Hz** window
spanning 0.1 s. Both MLPs use the flattened, masked window, with hidden width 384
(`--hidden-size`) and 3 hidden layers (`--layers`). `--critic-hidden 512 --critic-layers 4`
overrides only the critic; `--critic-hidden-size` remains a width alias.
Use `--gru-policy`/`--gru-critic` instead for GRUs (always single-layer). Without window flags,
the original single-state MLP defaults remain unchanged.
See [history and checkpoint compatibility](imitation/README.md#dense-actorcritic-history).

## Value-learning diagnostics

GAIfO/DAC now default to unclipped critic fitting, loss conditioning by rollout
return standard deviation, and a separate critic gradient limit of 1.0. Raw
value predictions, rewards, GAE and actor clipping are unchanged. TensorBoard
adds `explained_variance`, raw `value/rmse`, `value/bias` and target-scale metrics
without extra GRU passes. `value_loss` is now normalized, so it is not directly
comparable to an older raw-loss curve. See
[critic fitting and resume compatibility](imitation/README.md#critic-fitting-and-explained-variance).

## Dense replay preparation

`python -m replays.download` downloads Ballchasing replays using
`BALLCHASING_TOKEN`. Never put the token in source or commands saved to version
control. To prepare goal-aware dense replays for DAC:

```bash
uv run --no-sync python -m replays.parse \
  --replay-dir /path/to/raw-replays --output-dir /path/to/dac-parsed \
  --frameskip 4 --team-size 1 --include-goals
```

`python -m replays.collect_duels --limit 10000` collects close ranked duels
with explicitly recorded SSL ranks for both players, prioritizing
Ballchasing-identified pros. See [collection filters and limitations](replays/DUELS.md).

## Tests

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run --no-sync pytest -q
```
