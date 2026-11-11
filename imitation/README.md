# Factorized GAIfO

This implementation adapts the low-dimensional [GAIfO algorithm](https://arxiv.org/abs/1807.06158):
PPO replaces TRPO, discriminators use temporal windows, and factorization is optional.

The discriminator consumes a chronological window ending at the successor state.
`--discriminator-window-length` sets the maximum number of frames (default 16).
Shorter windows at episode or replay-segment starts are masked; windows
never cross episode resets, files, or invalid replay frames.

`--discriminator-decision-stride 3` optionally spaces the ordinary dense replay
and agent discriminator observations at 10 Hz (requires frameskip 4). With
`--discriminator-window-length 2`, it selects the first and fourth underlying
30 Hz observations, spanning 100 ms. The existing `dac_parsed` files are sampled
on demand; no reparse is needed. Policy actions,
policy/critic observations and reward scoring remain 30 Hz. The stride defaults
to 1; checkpoints must use the same stride when resuming. All intervening
decisions are checked for resets/invalid replay frames, even if not selected.
For a scale command and verification details, see
[TEN_HZ.md](TEN_HZ.md).

### Matched reset-history padding

Discriminator training and held-out classification now give each paired
generated/expert window the **union of their missing-history masks**. Only
existing observations are hidden; no pre-reset history is fabricated. This also
covers historical policy samples and experts with short segment prefixes. Both
sources therefore have identical mask distributions within each batch, rather
than letting an explicit validity flag identify generated reset windows.

For two observations at stride 3, the first two generated windows after a reset
lack the older bookend; the paired expert windows receive that same mask. An
adjacent two-observation transition (stride 1) already has both bookends after
the first action and needs no reset-prefix padding. Matching happens before
noise, relative encoding and microbatching, and is shared by all heads and
BCE/AMP objectives. DAC retains its terminal/absorbing indicators for diagnostic
bookkeeping; masked tokens are ignored by its heads.

Stored clips, resets, rollout/history buffers and **reward inference are not
modified**. In particular, this does not zero post-reset imitation rewards or
change terminal incentives. Held-out `discriminator/generated_padded_fraction`
and `discriminator/expert_padded_fraction` report the matched fractions. Resuming
a checkpoint predating matching is allowed but warns: old weights can retain a
previously learned shortcut, and this fix does not establish its original impact.

By default, it outputs logits for the probability that the transition came from the policy.
Expert labels are **0**, policy labels are **1**, and classification uses the
sum of the two class-mean binary cross-entropies. Each rollout evaluates the
discriminator against held-out policy and expert windows. Only heads below the
accuracy target receive classification updates, unless `--target-discriminator`
disables accuracy gating. Optional contrastive consistency
can still update a head above that target. The target increases linearly from
`--discriminator-accuracy-target` (default 0.85) to
`--discriminator-accuracy-target-end` (default 1.0) over `--timesteps`, using
total training progress, including when resuming. Set both targets equal for a
fixed threshold. TensorBoard logs it as `discriminator/accuracy_target`. Generated training
windows mix current samples with earlier policy windows by default; there is no
gradient penalty.

## AMP least-squares objective and bounded reward

Add **`--amp`** to `gaifo.py` or `dac.py` to replace BCE and its reward with an
Adversarial Motion Priors-style least-squares objective/reward pair. It is **off
by default**, and is unrelated to automatic mixed precision (`--bf16` is unchanged).

Keeping this implementation's policy-positive score convention, each head uses
raw score `s`, **expert target -1** and **policy target +1**:

```text
loss   = 0.5 * mean_expert((s + 1)^2) + 0.5 * mean_policy((s - 1)^2)
reward = max(0, 1 - 0.25 * (s + 1)^2)
```

The targets are bounded; the raw discriminator output is not clipped or passed
through a sigmoid/tanh. The reward is in **[0, 1]**: 1 at the expert target, 0 at
the policy target, and 0 for scores outside [-3, 1]. This is canonical AMP's
mapping with the score sign reversed, **without a negative offset or rescaling**.
Factorized heads retain their loss weights and independent zero-threshold accuracy
gates by default; `--target-discriminator` bypasses the gates. Their bounded rewards are averaged, or multiplied with
`--multiplicative-reward`; either combination stays in [0, 1]. Ego-only mode and
all discriminator architectures are supported. No gradient penalty is added.

`--amp` cannot be combined with `--discriminator-log-reward` or
`--exp-log-odds-reward`. Gameplay rewards, InfoGAIL and potential shaping remain
optional; **the combined PPO reward need not stay bounded**. Potential shaping
uses the AMP mapping for both windows. Ordinary GAIfO still zeroes imitation
rewards on goals; DAC instead uses this mapping for its terminal transitions and
absorbing self-loop reward, including the absorbing-value Bellman target.

For example, append `--amp` to a new training command, removing any log/odds
reward flag. Checkpoints record a distinct objective and require the same AMP
mode on resume. Legacy checkpoints without the flag mean **AMP disabled**;
switching between BCE and AMP requires a fresh run. Existing runs are unchanged.

## Polyak target discriminator and exploitation gap

Add **`--target-discriminator`** to `gaifo.py` or `dac.py` to train every online
discriminator head on each scheduled minibatch, **without accuracy gating**, and
compute imitation rewards with a separate frozen, exponentially averaged copy.
It is **off by default** and works with BCE or AMP, every discriminator
architecture, global/ego/factorized heads, and relative encoding. Held-out accuracy
is still logged, but its targets no longer decide whether a head updates.

```bash
# Append to a fresh GAIfO or DAC run:
--target-discriminator --target-discriminator-span 8
```

The reward copy starts from the online weights and follows:

```text
target <- (1 - tau) * target + tau * online
tau = 2 / (span + 1)
```

The default **span 8** gives **tau = 2/9 ≈ 0.222**. This is a nominal EMA span
with a decaying tail, **not a hard eight-update cutoff**. Averaging happens once
after each successful discriminator **optimizer step**, never per backward
microbatch or once per rollout. Floating parameters/buffers are averaged;
integer/bool buffers are copied. The target receives no gradients. It supplies
ordinary rewards, both potential-shaping windows, and DAC's terminal/self-loop
rewards, absorbing-value target and absorbing diagnostics. Other optional
gameplay/InfoGAIL rewards are unchanged.

Alternatively, use **`--target-discriminator-tau VALUE`** with `0 < VALUE <= 1`;
it is mutually exclusive with the span flag. Larger tau means less lag; tau 1
copies the online model after every step. Neither smoothing flag alone enables
target mode.

### Fresh-rollout exploitation diagnostics

In target mode, every rollout is scored by both models on the **same clean,
fresh nonterminal windows**, including real timeout observations. Historical
policy samples, expert samples, goal/absorbing transitions and synthetic
absorbing loops are excluded. The comparison is read-only and does not change
PPO rewards, online buffers, module modes or RNG state. TensorBoard logs:

- `discriminator/exploitation_gap`: mean **target-minus-online mapped imitation
  reward**, before gameplay, MI or potential-shaping terms. Positive means the
  averaged model rewards the current policy more than the online model does.
- `discriminator/exploitation_gap_abs_mean`, `exploitation_gap_rms` and
  `exploitation_gap_positive_fraction`: magnitude and prevalence of that gap.
- `discriminator/target_rollout_reward`, `online_rollout_reward`,
  `target_rollout_score`, `online_rollout_score` and `exploitation_logit_gap`:
  mapped reward means and raw score means. The raw gap uses the opposite order,
  **online minus target**; raw scores/gaps also have `ego_`/`global_` variants.
- `discriminator/exploitation_samples`, `exploitation_live_fraction` and
  `exploitation_valid`: eligible sample count/share and validity. With no live
  windows, validity is 0 and gap values are 0, not evidence of agreement.
- `discriminator/target_tau`, `target_span`, `target_updates` (optimizer steps
  this rollout) and `accuracy_gating_enabled` (0 in target mode).

A large, persistent **positive reward gap** suggests lag exploitation: consider
raising tau manually (or shortening the span). There is **no automatic tau
adaptation**. Prefer the reward gap for AMP: its bounded mapping peaks at score
-1, so more-negative raw scores do not always mean better rewards.

Checkpoints preserve both online and averaged weights. Resume restores the
averaged history rather than copying online weights over it. Explicitly changing
tau on resume is allowed and warns; switching target mode requires a fresh run.
Legacy checkpoints without target fields mean disabled. No existing trainer is
changed or restarted automatically.

### Discriminator learning rate on resume

Fresh runs default to **`--discriminator-learning-rate 3e-4`**. On resume, omitting
that flag preserves the saved discriminator optimizer LR. Explicitly supplying
it overrides the saved LR **after** restoring optimizer state, preserving Adam
moments and target-discriminator history; a changed rate warns. For example,
append `--discriminator-learning-rate 1e-4` to the compatible resume command.
The effective rate is recorded in configuration and logged as
`discriminator/learning_rate`. Previously, optimizer restoration silently
overwrote this CLI setting. The discriminator rate remains fixed, independent of
the PPO learning-rate schedule.

## Discriminator motion noise

`--discriminator-motion-noise-scale` sets the standard deviation of zero-mean
Gaussian training augmentation in **normalized observation units**. The default
for new invocations is **0.001**; use `0` for the previous noise-free behavior:

```bash
# Append to a dac.py or gaifo.py command:
--discriminator-motion-noise-scale 0.001
```

The same distribution is applied to **both policy and expert windows**, including
historical policy samples, with independent draws per window, selected frame,
body and coordinate. It perturbs ball/car positions and linear/angular velocities
only. At scale 0.001, position standard deviations are approximately 4.1, 6.0 and
2.1 world units along X/Y/Z; velocity standard deviations are 2.3 uu/s for cars
and 6 uu/s for the ball; angular-velocity standard deviations are 0.0055 and
0.006 rad/s respectively. These are noisy discriminator measurements, not
perturbed physical simulator states.

Noise is applied before relative-window encoding and shared by factorized heads.
Car orientation, boost, discrete flags, padding and absorbing frames/indicators
are unchanged. Replay arrays, rollout/history storage, reset states, policy/critic
inputs, and physics are not modified. Draws are sampled before splitting an
update batch into microbatches, so the same RNG state produces the same augmented
batch regardless of microbatch size.

Update gating and the existing held-out accuracy metrics use a fixed augmented
validation batch, reused before/after each update. Additional metrics
`discriminator/clean_heldout_accuracy`, `ego_clean_heldout_accuracy` and
`global_clean_heldout_accuracy` report clean-window accuracy when noise is enabled.
`discriminator/motion_noise_scale` logs the configured scale. **Reward inference,
potential shaping and absorbing-value scoring stay clean and deterministic**;
frame caches retain their original behavior.

The scale is saved in checkpoint configuration. Legacy checkpoints mean scale 0;
resuming with a different scale is allowed but warns that augmentation changed.
Use an explicit `--discriminator-motion-noise-scale 0` for a legacy continuation
without this change. No running process is automatically modified or restarted.
This is an experimental regularizer, not an established fix for motion artifacts
or behavioral mode collapse.

## Temporal contrastive consistency

`--discriminator-contrastive` optionally combines temporal observation corruption
with a **clean-versus-perturbed per-head logit consistency penalty**. It works with
both `gaifo.py` and `dac.py`, all discriminator architectures, factorization,
ego-only heads and relative encoding. It requires a window length of at least 3
and is **off by default**. Here "contrastive" means paired nuisance consistency,
not InfoNCE or representation learning with negative pairs.

```bash
# Add to a run with at least three discriminator frames:
--discriminator-contrastive \
  --discriminator-contrastive-weight 1.0 \
  --discriminator-contrastive-radius 40
```

The radius is a **maximum horizontal car-position displacement in world units**,
not the normalized standard deviation used by `--discriminator-motion-noise-scale`.
For each window/car, a zero-mean-direction, temporally correlated random bridge
has zero displacement at both endpoint frames and a random peak magnitude in
`[0, radius]`. Only eligible interior car X/Y positions change. All ball channels,
car Z positions, velocities, orientations, boost, flags and absorbing indicators
are preserved. Bridges are independently drawn for both sources and every car;
no held-out replay examples or donor statistics are used to generate them.

Eligibility is computed on **clean absolute scenes**, before white noise or
relative encoding. An interior frame needs two adjacent smooth, contact-clear
intervals: moving floor/aerial motion, stable car flags, low turn/acceleration,
distance from walls and radius-aware swept separation from the ball/other cars.
Large ball-velocity impulses also disable the adjacent intervals. These are
conservative observation-based contact heuristics, not authoritative contact
labels. Event neighborhoods stay unchanged while other safe regions of a longer
window can still be perturbed. Incomplete histories and any terminal/absorbing
windows are excluded entirely. Watch eligibility metrics: some windows or runs
may have few eligible examples.

Classification trains on temporal views with unchanged source labels, followed
by the existing symmetric white motion noise. Separately, the consistency term
is the squared difference between **clean** and temporal-view logits, with
gradients through both views. It uses the existing head weights and balances the
eligible means of policy and expert sources independently; an empty source adds
zero rather than changing the other's weight. The weight defaults to 1.0;
setting it to 0 disables both temporal augmentation and consistency training.
All draws and source normalizers are fixed before microbatching.

Classification retains its existing accuracy gate unless `--target-discriminator`
is enabled. Consistency remains active
for every head during scheduled discriminator updates even above that gate;
otherwise an accurate shortcut-using head could never be corrected. A
consistency-only batch with no eligible windows does not step the optimizer.
There are separate per-head `*_classification_update_enabled` and
`*_contrastive_update_enabled` metrics. `*_update_enabled` includes either loss.

TensorBoard additionally logs `discriminator/classification_loss`,
`contrastive_loss` (before multiplying by its weight), `contrastive_weight`,
`contrastive_radius`, and `contrastive_{generated,expert}_fraction`.
Paired held-out views are drawn once and reused before/after the update, with
per-head `*_contrastive_{generated,expert}_heldout_mse` and
`contrastive_{generated,expert}_heldout_reward_abs_change` metrics. Matching
`pre_*` metrics show the before-update values. The original classification
validation/gating windows are unchanged.

**Policy/reward inference, potential shaping, absorbing-value scoring, stored
replays/history/rollouts and native physics stay clean and unchanged.** The
corrupted views represent measurement uncertainty, not repaired expert physics.
Checkpoint configuration records the flag/weight/radius; compatible resumes
warn if they change, and legacy checkpoints mean disabled. This is an experimental
mitigation: validate clean shortcut sensitivity and native gameplay ranking
before treating it as a successful fix. It does not change DAC's goal objective.

## InfoGAIL-style latent modes

Add **`--info`** to `gaifo.py` or `dac.py` for an episode-persistent categorical
policy code and a state-only mutual-information objective. It is off by default:

```bash
--info --info-codes 8 --info-weight 0.1
```

Each player's code is sampled independently from a uniform categorical prior at
every episode reset (goal, timeout, or manual viewer reset). Its one-hot encoding
is appended to **actor and critic** observations. It persists across rollout
boundaries; PPO reuses the original code when evaluating stored actions. Final
timeout observations retain the finished episode's code for value bootstrapping.
Action legality masks still receive only the original physical observation.

A separate MLP-window posterior `Q(z | states)` sees only normalized scene states,
with the same causal window length, decision stride, relative-input setting and
reset masking as the discriminator. It never sees actions, latent codes, player
IDs or expert labels. It trains by categorical maximum likelihood on **current
training-simulation rollouts only**; held-out simulations are evaluated but not
used to train Q. Expert windows and the discriminator are unchanged. Posterior
learning is independent of the discriminator's accuracy gate and history buffer.
`--info-learning-rate` (default `3e-4`), `--info-epochs` (default `1`) and
`--info-batch-size` (default `16384`) control its updates.

The added reward is exactly **`info_weight * log Q(z | states)`**, evaluated with
the frozen posterior **before** fitting the current rollout, so fitting its labels
cannot immediately earn a memorization bonus on those same samples. This term is
non-positive; the uniform-prior entropy constant is omitted from the reward.
Windows with fewer than two valid sampled states receive no MI reward or posterior
update. Goals and DAC's synthetic absorbing transitions/self-loops also receive
none; timeout windows use real final states. DAC's absorbing reward/value objective
is otherwise unchanged. The negative MI term can affect preferences over episode
length, so monitor goals/concessions as well as aggregate reward.

TensorBoard reports `info/posterior_loss`, `info/log_q`, `info/mi_lower_bound`
(`log(number_of_codes) + mean(log Q)` over eligible windows), `info/reward`,
`info/reward_rms`, `info/valid_fraction`, `info/code_entropy`, and `info/codes_used`.
`info/heldout_accuracy`, `info/heldout_log_q` and `info/heldout_mi_lower_bound` help
distinguish generalizable mode decoding from fitting rollout-specific states.
Held-out simulations are still controlled by and used to train the shared policy;
these are posterior-generalization diagnostics, not an independent gameplay test.

The posterior and optimizer are checkpointed. Enabling/disabling `--info`, changing
the code count, or changing the reward weight requires a **fresh run**. Old
non-latent checkpoints remain compatible when the feature is disabled. The viewer
automatically reconstructs latent-policy inputs from the checkpoint and samples
a persistent code per player/episode. No existing run is restarted or modified.

This is categorical InfoGAIL applied to state windows, **not** continuous
hyperspherical ASE or CALM's expert-clip-conditioned discriminator. MI encourages
decodable differences between learned behaviors; it does **not** guarantee useful
skills or coverage of every expert mode. Long-run effectiveness is unproven.

## Dense actor/critic history

`--policy-window-length` controls **one shared observation history** for windowed
actor and critic heads, independently of discriminator history. It defaults to **16
consecutive policy decisions**; it is never downsampled by
`--discriminator-decision-stride`:

```bash
--mlp-policy --mlp-critic --policy-window-length 16 \
  --discriminator-window-length 2 --discriminator-decision-stride 3
```

At frameskip 4, the actor and critic both see up to 16 observations at **30 Hz**
(0.5-second span), while the discriminator sees two observations at **10 Hz**
(0.1-second span). Both histories slide at every policy decision. Expert-history
retention limits constrain only the discriminator window, not the policy window.
InfoGAIL's state-window posterior still follows the discriminator window/stride.

`--mlp-policy` and `--mlp-critic` enable **feed-forward MLPs over the entire
ordered observation window**. Padding is zeroed and each frame has an explicit
valid-frame indicator before flattening. The first layer consumes the raw window,
not a recurrent unroll or a stack of wide per-frame projections. Hidden width
and depth use `--hidden-size` and `--layers` (defaults 384 and 3).

`--gru-policy` and `--gru-critic` select GRUs instead. MLP-window and GRU flags are
mutually exclusive per head, but different head types can be combined. Without
either window flag for a head, its original single-state MLP remains unchanged.
Setting a history length alone does not enable windowed inference. Switching a
head between single-state MLP, windowed MLP and GRU requires a fresh run.

The actor and critic share input frames, not network parameters. Their hidden
widths both default to **384**, controlled by `--hidden-size`. For GAIfO/DAC,
`--critic-hidden` (alias `--critic-hidden-size`) overrides only the critic width;
`--critic-layers` overrides only its MLP hidden-layer count. Omitted overrides
inherit `--hidden-size` and `--layers`, respectively. Both ordinary and windowed
MLP critics support these controls, without changing actor/discriminator settings
or either history. The final value-output layer is not counted; 0 hidden layers
gives a linear critic. For example:

```bash
--mlp-policy --mlp-critic --policy-window-length 8 \
  --hidden-size 384 --layers 3 --critic-hidden 512 --critic-layers 4
```

This keeps a 3-hidden-layer, 384-wide actor and uses a 4-hidden-layer, 512-wide
critic over the same eight decisions. GRUs remain **single-layer**:
`--critic-hidden` controls their hidden width, but `--critic-layers` (like `--layers`)
does not change recurrent depth. The old fixed 64-unit GRU critic is no longer
the default. Existing single-state MLP architecture defaults are unchanged.

Acting, PPO action re-evaluation and value targets use the same dense, causal
history. Histories carry across rollout boundaries, are left-masked at episode
starts, and never cross a goal/timeout/manual reset. Timeout bootstraps append
the real final observation before clearing history for the next episode. Windowed
MLPs have no recurrent hidden state; GRU hidden states are recomputed per window,
not carried indefinitely. Length 1 is
also supported. TensorBoard logs `policy/observation_hz` and
`policy/window_span_ticks`; with neither head windowed the effective span is zero.

Checkpoints record policy history and critic width/depth. A history, critic-width
or MLP critic-depth change requires a fresh run rather than silently changing a resumed model.
To resume an older GRU checkpoint, explicitly set `--policy-window-length` to its
old discriminator window length and `--critic-hidden-size 64`. The viewer
automatically reconstructs these legacy settings; new checkpoints use their
independent policy history. Existing MLP checkpoints retain their input behavior.
Checkpoints predating `--critic-layers` inherit their recorded shared `layers`,
and the viewer reconstructs both new independent depths and legacy shared depths.

## Critic fitting and explained variance

New `gaifo.py` and `dac.py` invocations use these critic-fitting defaults:

| Setting | Default |
|---|---:|
| Value clipping | **Off** (`--no-clip-value-loss`) |
| Critic loss conditioning | **On** (`--normalize-value-loss`) |
| Critic gradient-norm limit | **1.0** (`--critic-max-grad-norm`) |
| Actor gradient-norm limit | **0.5** (`--max-grad-norm`, unchanged) |

For each rollout, the loss scale is `max(std(GAE returns), 1)` and stays fixed
across all its PPO minibatches and epochs. The critic minimizes
`0.5 * mean(((V - return) / scale)**2)`. This conditions gradients as learned
reward magnitudes change; it does **not** normalize stored rewards, change the
actor objective, rescale the critic's predictions, or change GAE/timeout/absorbing
bootstraps. Predictions and targets always remain in raw return units. There is
no new running-normalizer state or PopArt head adjustment. These are starting
defaults, not a demonstrated improvement in long-run learning.

Value clipping is opt-in. Its independent `--value-clip-coef` defaults to `0.2`,
meaning a raw radius of `0.2 * scale` with loss normalization, or `0.2` raw return
units without normalization. It no longer inherits the actor's clipping setting.
Disabling it avoids the old clipped-loss plateau after a small useful value
change within an update. Finite-loss/gradient checks and separate actor/critic
gradient clipping remain active. Learning rates, epoch count, gamma, lambda and
discriminator training are unchanged.

**Do not directly compare the new `value_loss` magnitude to an old raw-loss
curve:** normalization changes its units. Multiplying the new `value_loss` by
`value/loss_scale**2` expresses that same PPO fitting loss in raw squared-return
units; it does not restore the old clipping behavior. `0.5 * value/rmse**2` is
instead the raw **pre-PPO** loss, not the minibatch/epoch-averaged fitting loss.
For tracking, monitor:

- `explained_variance`: `1 - Var(return - V) / Var(return)`. Near 1 explains most
  variation, 0 resembles a constant predictor, and negative values explain less
  variation than that baseline. It does not measure mean-value calibration.
- `value/bias` (prediction minus target), `value/rmse`: raw-unit prediction error.
  A constant offset can give excellent explained variance and still large bias.
- `value/return_mean`, `value/return_std`, `value/prediction_mean`,
  `value/prediction_std`: changing target scale versus prediction tracking.
- `value/normalized_rmse`: raw RMSE divided by `max(return_std, 1)`.
- `value/loss_scale`, `value/loss_normalized`, `value/clip_range`: the effective
  fitting scale/mode and clipping radius (zero when clipping is off).
- `value/grad_norm`, `value/grad_clip_fraction`, `policy/grad_norm`: pre-clipping
  norms and how often the critic limit is reached during PPO minibatches.

Diagnostics use **stored pre-PPO rollout values** against that rollout's fixed
GAE targets, over all actors/decisions. They add no actor/critic inference pass
and consume no random samples. They are not post-update/held-out errors or proof
of accurate multi-second Monte Carlo returns. For nearly constant targets,
explained variance is undefined: it logs 0 with
`value/explained_variance_valid = 0` (otherwise 1), rather than NaN.

Checkpoints record the fitting configuration. Compatible resumes warn if it
changes; weights/optimizer state and raw-unit inference remain loadable. To
preserve legacy fitting with the usual old limits, explicitly use:

```bash
--clip-value-loss --value-clip-coef 0.2 --no-normalize-value-loss \
  --critic-max-grad-norm 0.5
```

Use the old run's actual clipping coefficients if they differed. Standalone
`ppo.py`/training-pack PPO retain their previous defaults (raw value loss,
clipping enabled, critic gradient limit inherited from the actor), but gain the
same diagnostics and optional controls. No active run is automatically changed.

## Discriminator heads and rewards

With factorization, one ego head sees the ego car's features, the ball position
relative to the ego, and each opponent's position relative to the ego. It does not
see ball velocity, opponent velocity, or teammate features. The global head sees
all cars and the ball. Both heads score every window and are trained independently
with class-balanced losses (BCE by default, least-squares with `--amp`); there is
no distance-based routing.
`--ego-only` instead builds and trains only the ego discriminator, with a single
reward and, by default, a single held-out accuracy gate. It works on its own or with
`--factorize`; `--multiplicative-reward` is incompatible because there is no
second head. Its objective differs from existing global and factorized runs, so
their checkpoints cannot be resumed in ego-only mode.

Each head's default GAIfO reward is `-log(sigmoid(logit))`, computed stably as
`softplus(-logit)`. In factorized mode, the ego reward and global
reward are **averaged** by default. `--multiplicative-reward` instead multiplies
them. Both combinations and the factorized loss are
extensions rather than claims about the paper.
`--discriminator-log-reward` instead maximizes the log expert probability,
`log(1 - sigmoid(logit))`, computed stably as `logsigmoid(-logit)`. This reward
is non-positive and unbounded below. In factorized mode the two log rewards are
averaged. It cannot be combined with `--exp-log-odds-reward` or, in factorized
mode, `--multiplicative-reward`: multiplying negative log rewards would reverse
the intended incentive. As with the other modes, goal-terminating policy
transitions receive zero imitation reward; with negative rewards, shorter
episodes can therefore be preferable even without a goal reward.

`--discriminator-potential-shaping` optionally adds a discounted difference
to the ordinary imitation reward: `weight * (gamma * phi(next) - phi(current))`,
where `phi` is the current discriminator reward scored on a causal window.
Set `--discriminator-potential-weight` to adjust the weight (default 1.0).
The discriminator is evaluated with the same parameters on both windows;
rollout boundaries retain history, and episode resets do not share history.
True goal terminations use zero terminal potential, while timeouts use their
final observation. This also works with DAC's absorbing-state reward, which is
not changed. TensorBoard logs the weighted term as
`discriminator_potential/mean` and `discriminator_potential/rms`.
Checkpoint resumes require the same shaping mode and weight. Since the
discriminator changes during training, the standard fixed-potential policy
invariance guarantee does not directly cover the entire learning process.
By default, the feed-forward actor and MLP critic are trained by
PPO on discriminator rewards only. `--reward-plus` adds the standalone PPO
reward-plus components, including goals, flip resets, and demolitions, to the
discriminator reward. The default multiplier is 1.0 for `--reward-plus`. For
`--goal-reward`, it is 0.25, or 2.4 with `--exp-log-odds-reward`. Override
either default with `--reward-plus-weight`. These are
starting estimates from short 1v1 calibration runs with multiplicative rewards,
not guarantees of balance under the new additive default.
TensorBoard logs unweighted and weighted reward-plus RMS as `reward_plus/total_rms`
and `reward_plus/weighted_rms`, and discriminator RMS as `rms/imitation`.
Discriminator reward inference caches frame encodings and projections; attention is
recomputed for each window. A GRU critic also caches projections during rollout.
`--gru` replaces each discriminator transformer with a single-layer GRU of the
same width. `--gru-critic` independently selects a GRU critic; otherwise the
critic is a single-state MLP unless `--mlp-critic` enables a windowed MLP.
GRU hidden states are recomputed per window,
so they never carry information across the window boundary or an episode reset.
Alternatively, `--mlp-discriminator` uses an MLP over the ordered, flattened
window of per-frame embeddings. It zeroes padded frames and provides an explicit
valid-frame indicator for each position; shorter histories are right-aligned.
It supports global, ego-only, and factorized discriminators, including DAC
absorbing states. It is mutually exclusive with `--gru`, and changing the
discriminator architecture requires a new run rather than resuming a checkpoint.
`--discriminator-relative` is an independent input option for any of those
architectures. Within each window, the first valid normalized scene stays
absolute and every later scene is its elementwise difference from that first
scene. This includes velocities, orientation vectors, boost, and car flags, not
only positions. Padding is excluded when choosing the reference, and DAC's
absorbing indicator is kept separate. This is a distinct input mode and needs
a fresh run. Since the same frame has a different representation in different
windows, per-frame encoding caches cannot be reused; reward inference can be
slower and use more memory, especially for long windows.
With `--gru-critic` or `--gru-policy` and `--compile`, the PPO loss remains uncompiled because PyTorch does not
support full-graph GRU compilation.
`--gru-policy` separately replaces the feed-forward actor with a GRU over the
shared **policy** history, not the discriminator window. See
[dense actor/critic history](#dense-actorcritic-history).

By default, discriminator updates use batches of 65,536 windows per source for 1v1, with
16,384 generated and 16,384 expert windows per backward microbatch. Reward inference
uses microbatches of 65,536 windows. These settings can be adjusted with
`--discriminator-batch-size`, `--discriminator-update-microbatch-size`, and
`--discriminator-inference-microbatch-size`.

`--exp-log-odds-reward` instead uses `exp(-logit)`, the expert-to-policy
odds, capped at 10 per head. Factorized rewards average the ego and
global values by default (maximum 10), or multiply them with
`--multiplicative-reward` (maximum 100). This optional reward is
a further extension and is not the paper's `-log(D)` reward.

Expert windows never cross invalid replay frames or files. Validation splits
whole replay IDs so all segments and POVs from one match remain together. At
least 20 replay IDs are held out by default when available, while retaining at
least one for training. `--discriminator-heldout-replays` controls this minimum;
`--discriminator-heldout-size` controls the number of windows evaluated.
The historical-window capacity, addition size, rollout interval, and mixing fraction are
controlled by `--discriminator-history-capacity`, `--discriminator-history-add-size`,
`--discriminator-history-interval`, and `--discriminator-history-mix-fraction`.
By default, 16,384 windows are added every four rollouts (starting with the first),
so the full 262,144-window buffer contains samples from 16 rollouts spread across
roughly 64 rollouts. TensorBoard logs the number added as `discriminator/history_added`.
History starts empty on a new run or resume and never includes held-out simulations.
The local `pro_files.json`
allowlist is honored when present. By default 75% of resets
sample replay states and the rest use CARL's normal resets; `--replay-reset-fraction`
changes the fraction. Reset sampling excludes correction/discontinuity flags,
but does not filter jump or flip phases. CARL resets all boost pads to available, even
when the recorded reset frame had unavailable pads.
Held-out expert replay IDs are excluded from reset sampling. Goal-terminating
policy transitions are excluded from discriminator training and receive zero
imitation reward because parsed expert segments end before the goal frame;
reward-plus goal rewards are unchanged.
Checkpoints from previous discriminator architectures or loss
objectives cannot resume; start a new GAIfO run.

Example:

```bash
uv run python gaifo.py --n-sim 8192 --rollout 32 --max-ticks 36_000 \
  --factorize --ppo-lr 3e-4 --ppo-lr-end 1e-5 \
  --replay-dir replays/parsed_replays/ --frameskip 4 \
  --timesteps 100_000_000_000 --bf16
```

Previous differential-reward and gradient-penalty flags remain removed. `--amp`
selects only the least-squares objective and bounded reward documented above.
