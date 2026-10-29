# AMP-style GAIFO

## Expert corpus

The local `replays/parsed_replays/` corpus contains selected pro POVs and
unlabelled pro private RLCS matches confirmed by the user. Legacy rank-labelled
files are excluded.
Its `pro_files.json` allowlist is applied before loading any replay frames,
including discriminator validation and replay-reset data. Stray `.npy` files
outside the allowlist are ignored; missing listed files raise an error. The
allowlist records selected-POV parse markers matched to the verified player
roster and the user-confirmed RLCS replay IDs. Unlabelled filenames alone are
not automatically accepted without inclusion in the allowlist.
Custom/synthetic dataset directories without an allowlist retain normal loading.

The previous mixed-corpus checkpoints already learned from non-pro examples.
Removing those files does not undo that training; start a new run for a clean
pro-only experiment. Replay schemas, parsing, and the 70% reset fraction are
unchanged.

The default discriminator regresses expert windows to **+1** and policy
windows to **−1**, using the sum of their mean squared errors. The expert
input-gradient penalty is `0.5 * weight * mean(||gradient||²)`, with weight
**10** by default (`--discriminator-gradient-penalty`). It is computed in
FP32 over normalized input windows, separately for each head so gradients
cannot cancel between heads. Microbatching preserves the full-batch weighting.
For the sequence penalty, one trained timestep is sampled uniformly per expert
window. This estimates the mean per-step squared gradient norm without
building a full temporal Jacobian. Its GRU pass disables cuDNN because the
penalty requires second derivatives.

Penalty graphs are kept separate per head, and zero-weight specialist
samples are skipped with the original full-batch normalization preserved.
Classification gradients are accumulated before constructing penalty graphs,
so their saved activations do not overlap in memory. Both gradient contributions
still accumulate before the same single clipping and optimizer step.

Each head uses a causal GRU and emits a verdict at every step. Single-head
outputs have shape `(batch, length)`; factorized outputs have shape
`(batch, length, 3)`. The far head retains its frame-0 context. AMP loss
broadcasts the expert/policy targets across time and skips the first two
steps (or retains the last step for shorter windows). Held-out loss follows
the same convention; held-out accuracy still reports the final verdict.
`--temporal-hidden` controls the GRU width.

Discriminator updates run every iteration by default. There is no accuracy
gate; held-out accuracy remains a diagnostic. `--discriminator-every` and
`--discriminator-epochs` still allow explicit scheduling. Historical policy
samples, held-out splits, and replay resets are unchanged.

`--discriminator-batch-size` controls optimizer steps: the defaults are
16,384 / 4,096 / 2,048 policy windows for 1v1 / 2v2 / 3v3, respectively,
each paired with the same number of expert windows. Compute chunks are separate:

- `--discriminator-update-microbatch-size`: **8,192** windows per source per
  backward pass, accumulating gradients before each optimizer step.
- `--discriminator-inference-microbatch-size`: **65,536** windows per forward
  pass for both reward generation and held-out evaluation.

Neither microbatch size changes the number of optimizer steps. The previous
`--discriminator-microbatch-size` spelling remains an alias for update microbatching.

## Speed options

`--bf16` now covers discriminator classification as well as policy training
and reward inference. Parameters, squared-loss arithmetic, and the expert
gradient penalty remain FP32; classification activations use mixed precision.
FP32 remains the default.

`--discriminator-gradient-penalty-samples 1024` optionally samples at most
1,024 expert windows **per optimizer batch** for the penalty. Classification
still uses every window. Uniform sampling, original band weights, and sample
count normalization preserve the same penalty objective in expectation, but
introduce additional gradient noise. The default still regularizes all expert
windows. Neither option removes AMP, GRUs, or differential rewards.

```bash
uv run --no-sync python gaifo.py --factorize --differential \
  --multiply-discriminator-reward --bf16 \
  --discriminator-gradient-penalty-samples 1024
```

Each head's style reward is `max(0, 1 - 0.25 * (D - 1)²)`, without z-scoring.
With `--factorize`, exactly one near/far head is active per car, using the
existing window proximity test with a **1000-unit** threshold. Rewards combine
as either:

- Default: `0.5 * active_specialist_reward + 0.5 * global_reward`.
- `--multiply-discriminator-reward`: `active_specialist_reward * global_reward`.

Both combinations are bounded in `[0, 1]` before the existing frameskip
adjustment, `--discriminator-reward-scale`, and `--imitation-reward-coef`.
Without `--reward-plus`, PPO receives only the scaled discriminator reward.
With `--reward-plus`, goal, flip-reset, and shaping rewards are added, using
the configured weights and physical-reward cap. Gameplay diagnostics remain
active in both modes.
Multiplication requires factorized AMP rewards.

## Differential rewards

Add `--differential` to compare final verdicts for **consecutive full windows**.
Within a rollout, each window is evaluated once and its verdict is reused
for the next transition. At rollout boundaries, the preceding full window is
re-evaluated under the same updated discriminator weights, rather than reusing
a stale potential. One extra history frame is retained for this purpose.
Rewards start only when both full windows exist and never bridge episode resets.
Ordinary differential mode
uses `0.5 * (clamp(D_t, -1, 1) - clamp(D_previous, -1, 1))` per head, then
blends the active specialist and global heads. This is bounded before reward
scaling, but still flat when both verdicts exceed the same clamp boundary.

With both `--differential` and `--multiply-discriminator-reward`, each head uses:

```text
potential(D) = logsigmoid(2 * atanh(clamp(D, -0.95, 0.95)))
reward = 0.5 * delta_potential(active_specialist) + 0.5 * delta_potential(global)
```

This is the difference of a weighted log-product, not a product of signed
differences. Each window's potential uses its own near/far gate, so its value
matches when reused in the next reward, including when the gate switches.
For unchanged routing, if both active heads worsen, the reward is negative.
Differential rewards are not additionally scaled
by frameskip: they already measure a change across the actual policy step.
The discriminator reward scale and imitation coefficient still apply.

Start a **new run**, for example:

```bash
uv run --no-sync python gaifo.py --factorize --differential --multiply-discriminator-reward
```

AMP is the only objective. The legacy accuracy-target, objective-selector,
log-odds, and reward-max-magnitude flags have been removed. Differential mode
is opt-in; standard bounded AMP rewards remain the default. Older MLP and
last-step-only GRU discriminator checkpoints cannot resume into this sequence
architecture; their saved policies still work in the browser viewer.

## Experimental episode memory

Add `--episode-memory` to the existing command to enable persistent GRU hidden
states. The default remains the windowed baseline. Start a **new run**: checkpoint
metadata rejects mixing windowed and episode-recurrent discriminators.

```bash
uv run python gaifo.py --n-sim 8192 --rollout 32 --max-ticks 36_000 \
  --factorize --ppo-lr 3e-4 --ppo-lr-end 1e-5 \
  --replay-dir replays/parsed_replays/ --frameskip 4 --trajectory-length 8 \
  --multiply-discriminator-reward --timesteps 100_000_000_000 \
  --differential --bf16 --discriminator-gradient-penalty-samples 1024 \
  --episode-memory
```

In this mode:

- `--trajectory-length 8` is the **TBPTT chunk length**, not the memory horizon.
  Hidden states persist across chunks and rollouts until episode reset. Each
  chronological pass processes each frame once rather than rebuilding overlapping
  windows. Pre-update seed construction and post-update reward inference are
  separate passes.
- Policy chunks retain detached starting hidden states and reset masks. Expert
  streams advance chronologically through contiguous valid replay segments,
  resetting at file boundaries and the existing invalid/discontinuity flags.
  Training and held-out streams remain separate. No replay schema or reset
  eligibility changes are made.
- Both classification and specialist routing are per timestep. Routing uses
  **current** distance, and the far encoder uses current ball-relative context;
  it no longer has a window's frame-0 context. The first two episode observations
  are warm-up; rewards are zero for the first two actions, not seven actions
  or two actions per chunk. Padding is excluded from loss and metrics.
- Batch, update-microbatch, history-capacity, and history-add sizes are frame
  budgets, divided by chunk length (rounded down, with at least one chunk for
  nonzero update/add budgets). This keeps the optimizer update count roughly
  comparable without multiplying the work by overlapping windows. Inference
  microbatching also uses a frame budget.
- History replay stores complete chronological chunks, their starting hidden
  states, and reset/supervision masks. The mix fraction is unchanged.
- AMP targets, least-squares arithmetic, head balancing, BF16, and FP32 expert
  gradient penalties remain. Penalties uniformly sample valid expert positions
  with replacement, using one sample per expert chunk by default, capped by
  `--discriminator-gradient-penalty-samples`. Input gradients cover the current
  chunk only, holding cached memory fixed; this is a **truncated** penalty, not
  the full-episode input Jacobian.
- Differential reward remains `scale * (Phi_next - Phi_current)`, with the same
  per-head log-potentials and weights. Both potentials use the updated model in
  one chronological pass. Terminal observations are evaluated before reset,
  never compared against the next episode. The next rollout's initial scene
  is re-scored rather than reusing a stale scalar potential.
- Without `--reward-plus`, PPO still receives only discriminator reward.
  The actor and critic remain feed-forward.

**Cached-state approximation:** policy chunk seeds are constructed before the
discriminator update; expert states are carried between optimizer steps; history
seeds can be older still. These hidden states are **not** reconstructed exactly
under the latest weights, and there is currently no burn-in. Reward pairs share
one current-weight pass from a common cached seed, but that seed is approximate.
Backpropagation stops at chunk boundaries, so long-lived memory does not imply
full-episode credit assignment. Monitor this experiment against the windowed
baseline rather than assuming longer memory improves learning.

Environment, replay buffer, stream positions, and live hidden states are reset
on checkpoint resume, as the environment is restarted; model and optimizer state
are restored. This is not bit-exact continuation of an episode.

Tests: `uv run --no-sync pytest -q`.
