# CALM on replay windows

`calm.py` implements the **low-level pretraining stage** of
[CALM](https://arxiv.org/html/2305.02195v1) adapted to Rocket League: a motion
encoder, latent-conditioned policy/value function, and conditional adversarial
discriminator. Manual skill labels and replay actions are unnecessary.

```bash
PYTHONPATH="$HOME/.local/share/goddard/carl/stage" \
  .venv/bin/python calm.py \
  --replay-dir replays/rlcs --frameskip 8 --n-sim 1024 \
  --rollout 32 --minibatch-size 4096 --bf16 --compile \
  --run-name calm-2s
```

The staged CARL path above supplies the current ego-controller observation
schema. Use the same native build as other current trainers. Defaults are
imitation-only; `--goal-reward-weight 15`, for example, adds the existing signed
goal reward. The magnitude is a tunable choice, not a validated CALM setting.
`--reward-plus` and `--gameplay-reward-weight` retain their usual meanings.

## Two different window lengths

The **encoder reference** spans two seconds by default. Both endpoints are
included: 61 observations at frameskip 4 (30 Hz), or 31 at frameskip 8 (15 Hz).
All source phases are eligible. References are full,
clean, overlapping windows in the canonical parsed dataset. They cannot cross
file/period boundaries, goal-terminal rows, nonfinite states, or correction flags,
including flagged rows skipped by downsampling. Partial references are excluded.
Parsed files and existing manual maneuver labels remain full-length on disk.

New runs default to **state-matched reference sampling with a precomputed index**.
At startup, extract the **starting ego and ball positions** of every eligible
window into a six-dimensional SciPy KD-tree. At reset or a scheduled reference
switch, query the exact nearest 16 windows across the entire loaded corpus and
pick randomly among them. `--calm-match-topk 1` chooses the single closest start.
Positions are compared in physical units after undoing axis-specific normalization;
ego and ball receive equal squared-distance weight. Matching remains position-only:
velocity, orientation, boost and opponents do not enter the distance.

The index is built once per run and reused across all switches. It stores static
positions and endpoint IDs on CPU, not learned embeddings or whole clips. Host
arrays cost roughly 64 bytes per eligible window, plus tree nodes; transient GPU
gathers are chunked at 65,536 starts. Batched queries use four CPU workers by
default (`--calm-match-workers`). Only query positions and returned endpoint IDs
cross the CPU/GPU boundary. There is no fixed-radius cutoff: if all demonstrations
are far away, the nearest ones can still be far away. Equal-distance ties may
return any valid nearest windows.

`--calm-match-backend sampled` retains the previous GPU search over a fresh bank
of up to `--calm-match-candidates 4096` randomly sampled windows per switch call.
That candidate limit is used only by the sampled backend. Indexed search uses
every eligible loaded window; an explicit expert frame limit still limits the
loaded corpus.

A full-corpus check indexed 14,953,669 eligible starts in 7.63 seconds, using
913 MiB for host arrays plus tree nodes. Batched exact queries took a median
13.8 ms for 512 actors or 382 ms for 16,384 actors with four CPU workers.
On replay-state queries perturbed in position, the average combined ego+ball
distance across the returned top 16 was 229 uu, versus 981 uu for the sampled
4,096-candidate search. These measure lookup quality/cost, not learned skill or
match performance; CPU queries also synchronize their small GPU position input.

The chosen reference remains fixed during its hold: it does not continually
change to reward whatever the policy happens to do. This sampling prior is a
Rocket League extension beyond the paper's uniform reference sampling. Use
`--calm-reference-sampling uniform` for that baseline. TensorBoard logs the actual
ego/ball start distances in uu and the number of reference switches.

The **discriminator** observes shorter, causal sequences. Its default is five
observations at decision stride one: 0.267 seconds at frameskip 8. Positive
examples are randomly positioned subwindows **inside the exact reference that
conditioned the generated example**, not unrelated gameplay. The discriminator
window must fit inside the reference. Reset/reference-switch padding is matched
between real and generated inputs, preventing a padding shortcut.

## Learning and control

1. An MLP encodes a reference into a unit-length 64-dimensional latent.
2. The policy and value function observe the original simulator observation plus
   this latent internally. Action legality still uses only the original
   simulator observation. Replay and simulator observation layouts are unchanged.
3. Each car holds a reference for a randomly sampled 1–3 seconds, or until an
   episode reset. Switching references restarts discriminator history without
   terminating PPO's return/advantage calculation. Timeout bootstrap uses the
   final observation and the outgoing latent.
4. Discriminator training compares matched real/reference pairs against generated
   pairs and real motion paired with a random unit-sphere latent. The two negative
   losses receive half weight each, balancing the positive class. An input-gradient
   penalty acts on the expert motion and its latent. Encoder gradients are stopped.
5. PPO recomputes encodings from saved **reference indices**. The encoder learns
   through policy gradients, plus overlapping-window alignment and independent-
   window uniformity losses. Critic gradients do not train the encoder. The actual
   collection-time latents are retained for discriminator fitting and rewards.

This is a Rocket League adaptation, not a reproduction of the paper's humanoid
results. It uses canonical normalized ball/car scenes, CARL's discrete controls,
the existing PPO objective, optional target-discriminator EMA, and the existing
bounded uniform replay resets with four-second edge buffers. The default reward
is `softplus(-generated_logit)`, equivalent to negative log non-expert probability.

All cars on both teams train through **current-policy self-play**, with independent
reference draws. The reference is a desired behavior, not a script controlling
the other cars or ball. References involving particular interactions may be
unachievable from the current state. Useful latent control, skill coverage, and
competitive match performance require empirical evaluation; a successful training
smoke test does not demonstrate those outcomes.

The paper's downstream high-level controller/FSM is a later training stage. For
direct reference-conditioned inference, use
`z = agent.actor.encoder(reference_scenes)` followed by
`agent.get_action(observations, latent=z)`. The reference tensor has shape
`[batch, calm_window_length, team_scene_size]`, with the same canonical POV and
cadence as training. Latents can be held or changed independently of the camera.
The ordinary checkpoint viewer/exporter currently constructs unconditional agents;
CALM needs this explicit reference-conditioned inference path.

## Options and checkpoints

| Option | Default |
| --- | --- |
| `--calm-window-seconds` | 2 |
| `--calm-latent-size` | 64 |
| `--calm-encoder-hidden` | 256 |
| `--calm-min-hold-seconds` / `--calm-max-hold-seconds` | 1 / 3 |
| `--calm-regularization-weight` | 0.1 |
| `--calm-regularization-batch-size` | 256 |
| `--calm-gradient-penalty` | 5 |
| `--calm-reference-sampling` | state-matched |
| `--calm-match-backend` | indexed |
| `--calm-match-workers` | 4 |
| `--calm-match-candidates` (sampled backend only) | 4096 |
| `--calm-match-topk` | 16 |
| `--discriminator-window-length` | 5 |
| `--discriminator-decision-stride` | 1 |
| `--discriminator-batch-size` | 65536 |
| `--discriminator-update-microbatch-size` | 32768 |
| `--discriminator-inference-microbatch-size` | 32768 |

`--bf16` applies to collection, reward inference, and PPO. The discriminator's
input-gradient penalty and second derivatives stay in FP32. `--compile` compiles
policy/reference-encoder inference, reference preprocessing, PPO including encoder
gradients, rewards, GAE, and the second-order discriminator loss/backward. Functional
input gradients expose the penalty's complete graph to the compiler; it is the
same penalty, not a first-order approximation. `--no-compile-discriminator-loss`
retains eager discriminator updates. CUDA graph replay remains unsupported here.

PPO caches deterministic features for distinct references within a rollout (up to
512 MiB), reusing them across epochs. It never caches learned embeddings across
optimizer updates: every PPO minibatch still differentiates through the encoder.
Over-budget caches fall back to per-minibatch preprocessing. Alignment/uniformity
samples share one batched encoder forward. TensorBoard `timing/*_seconds` separates
collection, discriminator updates, rewards/GAE, and PPO using CUDA events without
adding per-stage synchronizations.

At 8192 two-car simulations and rollout 32, the defaults now make 8 discriminator
optimizer updates and 16 backward microbatches per discriminator epoch, rather
than the original 64 updates and 512 microbatches. Increasing the outer batch
changes update frequency; the microbatch controls accumulation/memory. Both can
be overridden independently. Keep a larger outer batch when requesting a larger
microbatch, otherwise the outer batch caps its effective size. New source defaults
take effect after restarting/resuming the training process.

For an explicit throughput-oriented launch, use:

```bash
# Append to the existing calm.py launch arguments:
--compile --bf16 \
--discriminator-batch-size 65536 \
--discriminator-update-microbatch-size 32768 \
--discriminator-inference-microbatch-size 32768
```

On the development RTX 4090, with a concurrent training workload, compiled
FP32 gradient-penalty forward/backward for 65,536 source examples took 8.85 ms
at microbatch 16,384, 7.24 ms at 32,768, and 8.96 ms at 65,536. Peak temporary
allocations were approximately 161, 325, and 643 MiB respectively. These were
two-frame discriminator component measurements, not end-to-end throughput;
use the timing metrics to assess the actual run.

The initial implementation supports a single conditional MLP discriminator and a
current-state conditioned critic. Hierarchical/factorized/ego-only discriminators,
jump-phase sampling, scene mirroring, maneuver-biased reference sampling, history
mixing, and frozen historical-opponent pools require conditional extensions and
are rejected rather than silently applying unconditional behavior.

Runs use `runs/calm-*` and algorithm-tagged CALM checkpoints. Resume with the same
ordinary architecture/data arguments and `--resume runs/<run>/latest.pt`. Omitted
CALM-specific arguments and discriminator window/stride inherit saved values.
Checkpoints predating state matching retain uniform sampling on resume. Changing
the sampling mode or top-k/candidate parameters requires a new run. Existing
state-matched checkpoints retain their sampled backend when the backend flag is
omitted. To upgrade those runs, explicitly add **`--calm-match-backend indexed`**:
weights and optimizer states are restored, and a warning identifies the change
in reference selection/RNG consumption. New runs use the indexed backend by default.
Encoder/policy/value weights, both optimizers, online/target discriminators,
expert RNG, replay-reset RNG, and global RNG are restored. The corpus file-index
signature is checked to avoid silently changing the reference pool. As in the
other trainers, simulator episodes restart on resume; this is not a bitwise
continuation of simulator state. PPO/GAIfO/DIFO checkpoints are not interchangeable.

Tests:

```bash
.venv/bin/python -m unittest training.test_calm
```

Coverage includes exact reference duration and boundary exclusion, matched local
subwindows, unit latents, encoder gradient ownership, unchanged action masking,
reference changes vs. episode endings, timeout bootstrap, complete discriminator
and PPO updates, and checkpoint restoration/mismatch rejection.
Indexed neighbors are checked against global brute-force distances, including
single-neighbor queries, duplicate positions, out-of-range positions, and pools
smaller than top-k. Selection tests verify full eligible-pool use and RNG restoration.
