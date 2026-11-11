# PPO GAIfO with DAC-style absorbing states

`dac.py` is a separate PPO experiment. It implements the absorbing-state portion
of Discriminator-Actor-Critic (DAC), not DAC's off-policy actor-critic algorithm.
`gaifo.py` retains its original goal-transition handling.

For a goal, each player's final discriminator window ends with a zero-valued
absorbing frame and an explicit indicator. Each recorded goal also contributes an
absorbing-to-absorbing self-loop to the discriminator's training or held-out
partition. The discriminator learns an absorbing embedding separately for each
head. Synthetic self-loops have no policy action and are never passed to PPO's
actor loss. Real goal transitions retain their gameplay goal reward and receive
the discriminator's terminal-transition reward.

An independent scalar value is trained with the recursive Bellman target
`r_abs + gamma * V_abs`, using the current discriminator reward. It is used to
bootstrap genuine goal terminations; normal timeouts remain truncations and use
the actual final observation. By default, `--absorbing-value-learning-rate` is
0.03 and `--absorbing-value-updates` is 16 per rollout. TensorBoard logs
`absorbing/reward`, `absorbing/value`, `absorbing/value_loss`, and
`absorbing/goal_transitions`. Validation also reports class counts and per-head
accuracy separately for live, terminal, and absorbing examples. These groups can
be rare in a validation batch, so their measurements may be noisy.
Additional diagnostics log the instantaneous fixed point `r_abs / (1 - gamma)`,
the signed learned-value error relative to it, per-head absorbing logits, and
expert and generated self-loop fractions in the train and held-out pools. Pool
fractions do not account for generated history-buffer mixing.

The experimental corpus is `replays/rlcs_2026_1v1_playoffs/dac_parsed/`. It was
re-parsed with `--include-goals` from the replay IDs present in the existing
corpus, excluding the previously quarantined 120 Hz replay IDs. Its
`pro_files.json` preserves the existing parsed-file allowlist. Each segment has
a JSON sidecar indicating whether it ended in a recorded goal and, for goals,
whether the ego team scored or conceded. Non-goal segment ends are marked
unknown, not assumed to be terminals. A corpus-level `dac_metadata.json` records
the format, frameskip, and team size; `dac.py` rejects mismatched settings.
Goal frames themselves are not valid reset states.

Critic-fitting defaults now disable value clipping, condition the PPO value loss
by rollout return standard deviation, and use a separate critic gradient limit
of 1.0. This does not rescale raw values or change the absorbing reward/value
Bellman target. `explained_variance`, raw RMSE/bias and target-scale diagnostics
use stored pre-PPO rollout predictions, without extra recurrent inference. See
[critic fitting](imitation/README.md#critic-fitting-and-explained-variance) for
metric interpretation, optional normalized clipping and legacy resume flags.

The last goal frame is used to identify the terminal event and is replaced by
the absorbing frame for discriminator scoring. If the preceding replay window
contains flagged unreliable frames, that goal window and its self-loop are not
used. This representation avoids exposing post-goal physics in expert windows,
but doesn't establish that all replay and generated terminal contexts match.

Example:

```bash
uv run python dac.py --n-sim 4096 --factorize --rollout 64 \
  --timesteps 50000000000 --replay-dir replays/rlcs_2026_1v1_playoffs/dac_parsed \
  --discriminator-window-length 8 --gru --gae-lambda 0.98 \
  --discriminator-log-reward --goal-reward
```

For the AMP-style least-squares discriminator and bounded **[0, 1]** imitation
reward, replace `--discriminator-log-reward` with **`--amp`**. It uses expert
target -1, policy target +1, and `max(0, 1 - 0.25 * (score + 1)^2)` for live,
terminal and absorbing rewards. The absorbing-value target uses the same mapping.
No gradient penalty or mixed-precision change is implied. Log/odds reward flags
are incompatible, and switching AMP mode requires a fresh run. Optional gameplay
or shaping terms can take the total reward outside that bound. See
[AMP semantics](imitation/README.md#amp-least-squares-objective-and-bounded-reward).

**`--target-discriminator`** optionally trains every online head without accuracy
gating and scores rewards with a frozen Polyak copy. The default
`--target-discriminator-span 8` gives tau = 2/9, updated after each discriminator
optimizer step, not each microbatch/rollout. This is an EMA with a decaying tail,
not an eight-step cutoff. All live/terminal rewards, both shaping windows,
absorbing self-loop rewards, value targets and absorbing diagnostics use that
same averaged copy.

Fresh-live-rollout `discriminator/exploitation_gap` compares **averaged minus
online mapped reward**, including real timeout observations but excluding
history and goal/absorbing transitions/loops. A persistent positive gap suggests
raising `--target-discriminator-tau` manually; no adaptive rule is applied.
Checkpoints preserve averaged history; changing tau on resume warns, while
switching target mode requires a fresh run. See
[target smoothing and metrics](imitation/README.md#polyak-target-discriminator-and-exploitation-gap).

Expert/generated training and validation windows use matched missing-history
masks, including short post-reset windows; no observations are fabricated and
live/terminal reward inference is unchanged. See
[reset-history padding](imitation/README.md#matched-reset-history-padding).
On resume, an explicit `--discriminator-learning-rate` now takes effect after
optimizer restoration; omitting it retains the saved rate. Adam moments and
averaged weights are preserved, and `discriminator/learning_rate` logs the
effective value. See [LR resume behavior](imitation/README.md#discriminator-learning-rate-on-resume).

Use `--resume` only with a `dac.py` checkpoint; GAIfO checkpoints use a different
discriminator and are rejected. History starts empty on resume. The absorbing
value and optimizer are included in DAC checkpoints. The learned continuation
value is experimental and may initially be far from its fixed point; monitor it
and evaluate kickoff goals and concessions separately before a long run.
Select `--mlp-discriminator` to use an MLP over the ordered discriminator
window instead of the default transformer; it cannot be combined with `--gru`.
This changes the discriminator architecture and requires a fresh run.
`--discriminator-relative` optionally represents the first valid scene in each
window absolutely and later scenes as differences from that reference. It also
requires a fresh run and disables per-frame reward-inference caching.

Use `--discriminator-potential-shaping` to add a discounted difference of
discriminator scores to the base imitation reward. The optional
`--discriminator-potential-weight` controls its magnitude (default 1.0). Goal
terminations have zero terminal potential; the usual learned absorbing reward
and continuation value remain unchanged. See `imitation/README.md` for details.

Use `--ego-only` to train and score only the ego discriminator, including its
own absorbing embedding and scalar absorbing reward. It may be combined with
`--factorize`, but not `--multiplicative-reward`. Ego-only checkpoints are not
compatible with global or two-head discriminator checkpoints.

## 10 Hz discriminator observations

The actor/critic history is independent: `--policy-window-length 16` with
`--mlp-policy --mlp-critic` (windowed MLPs) or `--gru-policy --gru-critic` gives
both heads consecutive decision-step
observations (0.5 s at 30 Hz), even with a two-frame, stride-3 discriminator
(0.1 s at 10 Hz). Hidden widths both default to 384; `--critic-hidden-size`
optionally overrides only the critic. See
[dense history and legacy checkpoints](imitation/README.md#dense-actorcritic-history).

Use `--discriminator-decision-stride 3` with frameskip 4 to subsample the existing
dense replay and agent histories. No reparse is needed. Length 2 spans 100 ms;
length 8 spans 0.7 seconds. Windows slide and rewards are scored at every 30 Hz
policy decision. See [the scale command](README.md#10-hz-discriminator-observations)
and [temporal semantics](imitation/TEN_HZ.md).

`--discriminator-motion-noise-scale` adds symmetric Gaussian motion augmentation
to policy/expert discriminator-training windows, independently of the stride.
New invocations default to 0.001 normalized units; 0 disables it. DAC absorbing
frames and indicators are not perturbed, and reward/absorbing-value scoring stay
clean. Legacy checkpoint resumes warn if this setting changes. See
[noise semantics](imitation/README.md#discriminator-motion-noise).

`--discriminator-contrastive` optionally adds gated temporal car-position views
and a clean/perturbed logit-consistency penalty for both policy and expert
windows. It requires at least three discriminator frames and defaults off.
Classification keeps its accuracy gate unless `--target-discriminator` is enabled;
consistency can still update accurate
heads. Terminal/absorbing windows are excluded from this augmentation, and
policy rewards and absorbing-value scoring stay clean. The flag, weight and
radius are checkpointed, with warnings on changed settings. This does not add a
signed goal/concede reward or otherwise change DAC's terminal objective. See
[contrastive semantics](imitation/README.md#temporal-contrastive-consistency).

`--info` optionally adds episode-persistent categorical actor/critic codes and a
state-window posterior MI reward. Defaults are 8 codes and weight 0.1. Expert data
and the unconditional discriminator are unchanged; goal/absorbing transitions
receive no MI reward. It requires a fresh latent-policy run and saves the posterior
and optimizer in checkpoints. See [InfoGAIL semantics](imitation/README.md#infogail-style-latent-modes).
