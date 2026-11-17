# GAIfO

`gaifo.py` uses the existing PPO actor/critic, rollout, optimizer, replay-reset
loader, checkpoint and browser-viewer paths. The actor and critic use the expanded
ego-controller observation schema; the discriminator only sees the shared
replay-observable ball/car scene prefix. It never sees controller internals,
replay correction/event labels, boost pads or goal-relative shortcuts.

## Throughput, compilation and checkpoints

PPO, GAIfO and DIFO now default to **BF16 mixed precision, compilation,
matrix-kernel autotuning, CUDA graphs for supported callables, and broader PPO
update compilation**. FP32 model parameters and Adam state are retained; BF16
autocast applies to model compute, while PPO probabilities, values and loss
reductions are FP32. These defaults also apply when resuming with no explicit
runtime overrides, and take effect on the next process launch. They do not
change model sizes, minibatch sizes, update counts or reward definitions.

Individual opt-outs are `--no-bf16`, `--no-compile-autotune`,
`--no-cuda-graphs` and `--no-compile-ppo-update`. `--no-compile` disables all
compiler paths. Autotuning increases initial compilation time; changing BF16
changes floating-point rounding. Native RocketSim and CALM retain their existing
backend-specific runtime defaults.

The broader PPO path compiles minibatch gathering together with loss/backward,
plus separate gradient-clipping, Adam and metric-accumulation regions. Adam
remains fused. A persistent tensor learning rate avoids recompiling the optimizer
on each annealing update. CUDA graph inputs mark frozen rollout buffers as
static-address inputs to avoid copying the full rollout every minibatch. The
gradient/optimizer/metric regions use compilation without graph replay, retaining
the host finite check **before every Adam update**. This is not a single captured
graph for the entire optimizer step. Compiler caches are not checkpoint state.

There is no diagnostic discriminator inference: no held-out accuracy passes,
maneuver validation passes, or online/target reward-gap comparison. All eligible
expert matches and generated learner scenarios train. Legacy holdout CLI flags
accept only zero. Offline dataset tooling can still create explicit splits, but
the trainer does not use them. Without a target, accuracy gating is also removed;
heads train every configured discriminator epoch.

On resume, old holdout state is ignored and previously excluded matches/rows
become eligible for discriminator training and bounded uniform replay resets.
A warning identifies this intentional distribution/update-schedule change;
old actor, critic, online/target weights, optimizers, expert RNG and generated
history are retained. Exact old-run RNG continuation is not promised because
validation draws are gone and the candidate pools change. Boundary guards,
goal exclusions, padding matching and the expert maneuver mixture remain.

`--compile` now compiles discriminator reward inference and loss/backward as
well as policy, PPO loss and GAE. `--no-compile-discriminator-loss` keeps
discriminator training eager (useful if a workload/precision benchmark shows
compiled backward is slower). Discriminator compilation uses dynamic batch sizes for partial
microbatches and leaves checkpoint-owned models/parameter names unwrapped.
GRU discriminators remain eager because Dynamo does not trace `nn.GRU`.
Compiled transformers disable PyTorch's unsupported eval-only fused MHA
fastpath and compile the regular attention/FFN path instead.
Discriminator loss/backward uses Inductor without CUDA graph replay: replay
can overwrite saved tensors across accumulated microbatches. `--cuda-graphs`
still applies to supported policy/PPO and discriminator reward callables.
Finiteness checks are combined into a single safety barrier before each
optimizer update; nonfinite losses/gradients still abort before that update.

`--matmul-precision high` enables TF32 for remaining float32 matrix operations
on supported GPUs. Choose `--no-bf16 --matmul-precision highest` for full FP32
model compute. This changes floating-point rounding,
not reward definitions. The harmless Inductor online-softmax split-reduction
warning is narrowly filtered; graph-break, recompilation and numerical warnings
remain visible. TF32's disabled-acceleration warning is addressed by the new
default, not globally hidden.

Both trainers update `latest.pt` every **100,000,000 agent transitions** by
default (`--save-every-steps`). Saves happen after the first completed rollout
update crossing each global boundary, including after resume, and on normal
completion. This is not a retained archive at every boundary. Explicit legacy
`--save-every N` overrides the step interval with N rollout iterations.
Already-running trainers keep their loaded code/settings until relaunched.

## Cadence

Keep the canonical parsed replays at frameskip 4 (30 Hz). Nothing needs reparsing
for a frameskip-8 run. Expert windows are downsampled in memory, and all source
phases are eligible:

| Run frameskip | Policy rate | Default discriminator rate | Expert row spacing |
| --- | --- | --- | --- |
| 4 | 30 Hz | 10 Hz | 3 |
| 8 (default) | 15 Hz | 5 Hz | 6 |
| 12 | 10 Hz | 3⅓ Hz | 9 |
| 16 | 7.5 Hz | 2.5 Hz | 12 |

Expert sampling excludes frozen kickoff/countdown states: the ball must be at
center and every car near a kickoff spawn and stationary. Small reconstruction
noise is tolerated (5 uu position, 10 uu/s car speed, 1 uu/s ball speed and
0.05 rad/s angular speed). Active kickoff approaches remain eligible as soon as
any car moves beyond that tolerance. Windows crossing excluded source rows are
rejected even when those rows fall between sampled frames. This also applies to
phase pairs, maneuver sampling and CALM references. Filtering happens at load
time; restart training to use it. Parsed replay files do not need rebuilding.

The run frameskip must be an integer multiple of `--replay-frameskip` (default 4).
`--discriminator-decision-stride` spaces discriminator observations by that many
run decisions; new runs default to **3**, with **2 frames** per short window
(`--discriminator-window-length`). Both omitted options inherit their saved
values on resume. Changing these explicitly on resume is rejected by the
existing checkpoint layout checks.
Rewards are scored at every policy decision using a causal window ending at the
real successor observation. The default two-frame window spans **0.2 seconds**
at frameskip 8. Spacing and window length remain configurable.

Small Gaussian noise defaults to standard deviation **0.001** in normalized
position, linear-velocity and angular-velocity channels, independently per body,
sample and selected frame for both sources. Noise is drawn **after cadence
selection**, not on skipped 30 Hz frames. Orientations, boost, flags and padding
are unchanged. Stored data and reward inference remain clean. The standard
deviation itself does not scale with the decision duration.

### Optional multi-timescale adversarial rewards

Add `--discriminator-hierarchical` to enable three independent temporal scales.
This is hierarchical **discrimination**, with the existing PPO policy:

```bash
--discriminator-hierarchical --mlp-discriminator --factorize \
--discriminator-window-length 2 --discriminator-decision-stride 3 --frameskip 8
```

For this example the scales are **2 / 11 / 21 frames**, spanning **0.2 / 2 / 4
seconds**. The short scale retains the configured window and architecture.
Medium and long use separate one-layer temporal transformers with the existing
frame embedding/hidden widths, even with `--mlp-discriminator` or `--gru`.
All scales use the same observation spacing, rounded up to whole intervals.
Medium spans at least 2 seconds and twice the short span; long spans at least
4 seconds and twice the medium span. Thus unusually long short windows remain
supported. With an explicit 16-frame, frameskip-8, stride-1 configuration, the
scales span 1 / 2 / 4 seconds (16 / 31 / 61 frames).

Hierarchical transformers compute only the final query and its feed-forward
output, while retaining keys/values from every valid history frame. For the
single-layer encoder this is mathematically the same final-token function and
gradient as computing every query and then discarding all but the last output.
It reduces attention work from quadratic to linear in window length and avoids
the unused per-frame feed-forward work. Original parameter names/shapes and
checkpoint compatibility are retained; floating-point rounding can differ.
The math attention backend remains in use for stable masked BF16 backward.

Every scale receives its own ego/global heads according to `--factorize` or
`--ego-only`. Source losses and transformed rewards are averaged equally across
spatial heads, then equally across scales: **one third per scale**, rather than
summing rewards or averaging logits before the reward transform. Softplus, log
expert probability and AMP retain their existing definitions. Target EMA covers
all heads. Relative encoding is anchored independently within each scale.

One shared longest-window history carries across PPO rollout boundaries, even
when the rollout is shorter than a window. Each head slices a trailing window
ending at the same true successor. Reset boundaries (including skipped decisions)
mask earlier frames; startup/short-episode histories are left-padded, with no
future observations or expert trajectory continuation supplied to the policy.
Expert validity checks cover the entire longest span, including skipped source
rows, rejecting crossings of files, corrections and goals. Training matches
padding between sources before slicing scales. All scales share a window's
mirroring/noise draw; reward inputs remain clean. Experimental jump-phase pairs
also share their real sampled endpoints across scales.

Hierarchical-only defaults are **8,192 source windows per discriminator update**
and **4,096 per update/inference microbatch**, configurable with the existing
batch flags. All latest eligible generated transitions still train for four
epochs by default. With the default zero history-mixing fraction, the hierarchical
path does not allocate the materialized historical-window pool. Enabling history
mixing uses the existing capacity/add-size/interval settings and stores longest
windows (paired in jump-phase mode). Memory/compute grow with window lengths,
number of heads and simulations.

A bounded compiled BF16 comparison on the RTX 4090, using 16/31/61-frame
factorized windows and 2,048 total input windows, measured final-token evaluation
at 11.59 ms versus 15.69 ms for full-sequence forward/backward (1.35×), and
3.82 ms versus 5.65 ms for inference (1.48×). Temporary training memory fell
from 1,512 MiB to 807 MiB. A separate 8,192-source-window update comparison
measured 108.70 ms with 4,096-window microbatches versus 129.30 ms with 1,024
(1.19×), at approximately 3,489 versus 1,022 MiB extra allocated memory. These
are interleaved component measurements with another trainer running, not
end-to-end throughput guarantees or total process VRAM. Smaller microbatches
remain available through the existing flags when needed.

New runs default off. Resume inherits the saved hierarchical setting when the
flag is omitted; old checkpoints default off. Switching between ordinary and
hierarchical architectures requires a **fresh run**. Scale layout, online/target
weights, optimizer, expert RNG and enabled generated history are checkpointed.
Window length and decision stride also inherit on resume. Other architecture
flags still need to be repeated as usual.

TensorBoard records `discriminator/{short,medium,long}_span_ticks` and
`imitation/{short,medium,long}_reward_per_tick` using the existing reward pass.
These per-scale values are before their one-third mixture weight; their mean is
the combined imitation reward. This supplies longer adversarial context; aerial
learning benefit still needs a training comparison.

### Experimental jump-phase robustness

`--experimental-jump-phase` enables an **experimental, default-off** discriminator
path. It does not change the policy's action interval, discriminator frame
spacing/window duration, actor/critic observations, gamma, goal magnitude,
RewardPlus selection or expert maneuver fraction. Example addition:

```bash
--experimental-jump-phase
```

CARL is stepped one physics tick at a time while each policy action is held for
the configured `--frameskip`. Generated discriminator windows choose a uniform
offset over all ticks in that interval (0–7 for frameskip 8), shifting the whole
window together, not its individual frames. Reward inference also uses a clean,
randomly phased causal window; it does not average logits or rewards across
phases. Prefixes near a reset have only the offsets available in that episode.

The discriminator receives two independently drawn real phases of each training
window. Its loss adds a **weight-1 mean squared logit disagreement** to ordinary
source classification, averaged across heads. Motion noise is shared between
the paired views, and padding is matched across both views and sources. Both
ego/global heads and the normal target-EMA path are retained. The historical
generated pool stores paired views so old generated data also trains the
consistency objective. Ordinary expert sampling remains present according to
the existing maneuver mixture; the option does not select jumps exclusively.

Experts use only real **canonical replay rows**, never fabricated one-tick human
observations. At frameskip 8/source cadence 4 the available offsets are 0/4;
at frameskip 4/source cadence 4 adjacent canonical endpoints provide the
nondegenerate consistency pair. This latter expert perturbation is four ticks,
not an exact match to native offsets 0–3. Replay parsing already involves
reconstruction/resampling, so these are not exact measured human button times.
Pairs cannot cross files, goals, corrections or discontinuities; partial
histories are consistently padded. The source corpus is not changed.

Gameplay rewards and diagnostics run **once per policy decision**, from its
true previous/current physical states with touches accumulated across its
ticks. Nonterminal endpoints, default Nexto shaping and optional RewardPlus
match ordinary native stepping. On a mid-hold goal/timeout, the actual terminal
observation is retained for bootstrap/reward bookkeeping; remaining native
kickoff ticks use neutral controls and are excluded from the ending episode.
The replay reset is drawn once at the decision boundary, after discarding that
temporary kickoff. Thus boundary detection is physics-tick-resolved rather
than rounded to the end of a native multi-tick step. Histories never splice
those ending/temporary/new episodes. Extremely short episode limits below one
hold are rejected, as are repeated episode boundaries within one hold.

This mode has **substantial extra runtime and GPU memory cost**: per-tick
observations/history, paired historical windows, and an extra discriminator
forward/backward view. Reduce simulations/rollout or history capacity if needed.
The tick adapter bypasses redundant per-tick Gym reward/episode bookkeeping;
only the decision boundary transfers completion flags to the host. Tick history
is stored contiguously, and a cached cumulative last-reset index replaces dense
reset scans for every sampled window. Reward inference gathers only one phase;
training still draws both independent views with the unchanged objective.
On an RTX 4090 (frameskip 8, 2,048 simulations), a decision step measured 9.49 ms
before versus 6.01 ms after this optimization (1.58x). With 16-frame windows,
decision stride 2 and a 16,384-window batch, paired sampling measured 1.72x faster
and reward-window sampling 3.32x faster. These are component timings, not an
end-to-end training speedup; model/update cost and hardware affect total gains.
It does not add diagnostic-only inference passes. Existing compile/BF16/GRU
settings still apply. It requires the existing CARL Python/native interfaces,
not a CARL rebuild or replacement of a loaded binary.

New runs default off. Resume inherits the saved mode when neither flag is
specified; explicitly use `--experimental-jump-phase` or
`--no-experimental-jump-phase` to switch. Switching warns that discriminator
and critic/policy need adaptation, retains their weights/optimizers, and clears
only incompatible generated-window history. The new mode is saved in the
checkpoint. Omitting the flag on an old checkpoint keeps the old default path.

Logs include `discriminator/experimental_jump_phase` and
`discriminator/phase_consistency_weight`; event/reward densities remain per
physics tick. Offline tests found reduced phase sensitivity, but also reduced
source accuracy. They do **not** establish improved gameplay or a conclusive
fix. Test separately from RewardPlus or other objective changes.

### Scene mirroring

Scene mirroring is disabled by default, including on resume. Use
`--discriminator-mirror` to enable it or `--no-discriminator-mirror` to explicitly
disable it. A change on resume
warns about the training-distribution change and retains model/optimizer/history
state. The setting is saved in the checkpoint and logged as
`discriminator/scene_mirroring`.

When enabled, each expert/generated training window has a 50% chance of reflection across the
arena's x=0 plane (left/right). All frames, ball and cars share that reflection;
both views of an experimental jump-phase pair share it too. Positions, linear
velocities and forward/up vectors negate x; angular velocities negate y and z
because they are axial vectors. Car ordering, teams, scalar flags and padding
are unchanged. Mirroring is applied to sampled windows (including historical
generated windows), not stored replay/history data, policy inputs, reset states
or discriminator reward inference. No extra model inference passes are added.

### Discriminator-only flag ablation

Add `--discriminator-no-flags` (also accepted as `--distriminator-no-flags`) to a new training command to hide `on_ground`,
`is_demoed`, `has_flipped`, `has_double_jumped`, and `is_boosting` for **every car**
from both discriminator heads. Ball/car physics, orientation and boost amount
remain available (see `--discriminator-no-boost` below to hide the amount). This is an input-information ablation: the five slots are
replaced with constants inside the model, including their relative-frame deltas,
without changing tensor layouts or parameter shapes.

The same masking applies to expert/generated data, online/target models,
classification/phase-consistency training, and imitation reward scoring.
Actor/critic observations, replay files, reset state, stored histories, and
simulation flags are unchanged. It does not fix reconstructed physics artifacts
(including demolished-car position repair), nor does it automatically disable
experimental jump-phase sampling or consistency.

Flags remain enabled by default. Resume inherits the checkpoint's saved setting;
older checkpoints without the option retain flags. An explicit change on resume
warns and retains model/optimizer/history state: masking old inputs does **not**
instantly retrain their learned rewards, and the discriminator, critic and policy
need adaptation. Prefer a fresh run for a controlled comparison. The setting is
saved in checkpoints and logged as `discriminator/flags`; use
`--discriminator-flags` to explicitly re-enable it.

### Discriminator-only boost-amount ablation

Add `--discriminator-no-boost` to a new training command to hide the boost amount
of **every car** from both discriminator heads. In relative-window mode this
removes both the first frame's fuel level and the later frames' fuel change, so
the discriminator sees neither how much boost a car holds nor how much it spent
inside the window. The `is_boosting` flag is a separate input: it stays visible
unless `--discriminator-no-flags` is also given. Boosting remains observable
through its effect on motion either way.

Masking is applied inside the model, after relative encoding and boost rounding,
to expert/generated data, online/target models, training and reward scoring.
Tensor layouts and parameter shapes are unchanged. Actor/critic observations,
replay files, reset states, stored histories and simulation are unaffected, and
boost rounding has no effect while the amount is hidden.

Motivation: frozen discriminators scored lower ego fuel as less expert-like, so
spending boost lowered imitation reward for seconds afterward (see
`/tmp/opencode/gaifo/expert-scenarios/REPORT.md`). That evidence is a sensitivity
probe; no run has shown this option improves boosting or aerial play.

Boost amount stays visible by default. Resume inherits the checkpoint's saved
setting; older checkpoints keep it visible. An explicit change on resume warns
and retains weights, which then need adaptation; prefer a fresh run. The setting
is saved in checkpoints and logged as `discriminator/boost_amount`; use
`--discriminator-boost` to explicitly re-enable it.

### Endpoint-only flags and boost (temporal-history erasure)

Enable with:

```bash
--discriminator-endpoint-scalars
```

Every car retains its **endpoint** boost amount and five flags (`on_ground`,
`is_demoed`, `has_flipped`, `has_double_jumped`, `is_boosting`), but their
within-window history is erased. For example, `[60, 55, 50]` boost becomes
`[50, 50, 50]`. Position, velocity, angular velocity, and orientation keep their
full history. This removes direct scalar event-timing and consumption cues;
motion can still reveal jumps, contacts, and boosting.

Erasure happens inside each discriminator head **before relative encoding and
boost rounding**, using that window's endpoint and preserving reset padding.
In relative mode the first valid slot contains the endpoint scalar values and
subsequent scalar deltas are zero. The same preprocessing applies to expert and
generated windows, online and target models, training and reward scoring,
phase pairs, and all hierarchical scales. Existing `--discriminator-no-flags`
and `--discriminator-no-boost` hide their fields completely, including endpoint
values, when selected. Either switch can be used alone or both together.

This is discriminator-only: stored replay/generated windows, reset states,
actor/critic observations and parameter shapes retain their existing layouts.
DIFO applies it once before constructing its condition/target; CALM applies it
to its conditional discriminator, not its actor's reference-motion encoder.

Default: **off** for new runs and older checkpoints. Resume inherits the saved
setting unless explicitly overridden with `--discriminator-endpoint-scalars`
or `--no-discriminator-endpoint-scalars`. Changing it on resume retains weights,
optimizer states and raw histories, and reports the changed input semantics.
The configuration is checkpointed and logged as `discriminator/endpoint_scalars`.

The retrained offline audit reduced takeoff-phase sensitivity by about **48%**
while retaining the takeoff/floor reward contrast. Removing scalar history also
removes legitimate temporal information; online aerial-learning benefit remains
unverified. See [FS4_AERIAL_FOLLOWUP.md](FS4_AERIAL_FOLLOWUP.md).

### Whole-point boost precision

New runs enable `--discriminator-round-boost` by default. Both expert and
generated discriminator inputs round boost fields to whole boost points, then
retain the normalized scale (one point = 0.01). In relative-window mode this
happens **after** relative encoding: the first valid frame's absolute boost is
rounded, and later frames' signed boost deltas are rounded directly. Thus
consuming 6.660 or 6.667 points produces the same seven-point delta. Rounding
absolute endpoints before subtraction would not guarantee this equivalence.
In non-relative mode, the absolute boost levels are rounded instead.

This transform is shared by both heads, online/target discriminators, training
and reward scoring. It does not alter physics, policy observations,
stored replay data, stored generated histories, padding or flags. Motion-noise
augmentation remains unchanged; no boost noise is added on top of the rounding.
Use `--no-discriminator-round-boost` to disable it for an ablation.

Older checkpoints without this setting retain their original unrounded encoding
when constructed from saved configurations. Resume defaults to the checkpoint's
saved setting. To change an existing run explicitly, pass
`--discriminator-round-boost` alongside its resume arguments. A warning notes
that the old online/target weights need adaptation; rounding does not instantly
repair their learned rewards. The new setting is saved in subsequent checkpoints.

New parse outputs record their frameskip in sidecar metadata. Old canonical
arrays without this key are treated as frameskip 4. A metadata mismatch is
rejected rather than silently using a different cadence. Existing replay files
and completed parse reports are not rewritten.

## Discriminator

The restored scene encoder and temporal head support the default transformer,
`--gru`, or `--mlp-discriminator`; optional `--factorize` ego/global heads,
`--ego-only`, and `--discriminator-relative` are available. Default classification
is source-balanced BCE (generated=1, expert=0), with `softplus(-score)` rewards.
`--amp` selects least-squares targets and bounded AMP-style rewards; it is
unrelated to `--bf16`.

`--discriminator-log-reward` restores the alternative `log P(expert)` reward
(`logsigmoid(-score)`, nonpositive). It applies to reward scoring and cannot be
combined with `--amp`. Resuming between the original softplus/log reward
modes is rejected. `--ppo-lr` and `--ppo-lr-end` are aliases for `--learning-rate`
and `--learning-rate-end`; they control PPO only, not the discriminator optimizer.

### Forward-KL reward

Use `--discriminator-forward-kl-reward` **instead of**
`--discriminator-log-reward` to make the expert/policy density ratio the main
imitation reward. There is no additional occupancy weight or discriminator:

```bash
# Add to your gaifo.py arguments; remove --discriminator-log-reward if present.
--discriminator-forward-kl-reward --discriminator-ratio-cap 10
```

For each source-balanced BCE head, generated=1 and expert=0 imply ideal logit
`z = log(rho_policy / rho_expert)`. The reward is:

```text
r_imitation = mean_over_heads(min(exp(-z), ratio_cap)) - 1
r_total = r_imitation + goal_reward + gameplay_reward_weight * gameplay_reward
```

Exponentiation uses float32 and clamps the log-ratio before `exp`. The default
cap is 10. The **fixed baseline of 1** removes the uniform survival bonus:
equal expert/policy densities give reward zero, underrepresented expert regions
give positive rewards, and overrepresented regions give negative rewards.
This is not rollout-mean centering. Rewards lie in `[-1, cap-1]`.
The offset is applied before terminal masking. Goal-terminal imitation stays zero;
time-limit successors use their pre-reset observations. The existing goal
weight remains separate and is not automatically recalibrated.

This is a **capped forward-KL local-gradient surrogate**, not an exact scalar
KL loss or the log-ratio (reverse-KL) reward. The discriminator is held fixed
during PPO. Density estimation, clipping and target lag make the optimization
approximate. With generated-history mixing, the ratio's denominator is the
training mixture rather than solely the current policy; use zero history mix
for the current-policy interpretation (the current CLI default is zero).

`--factorize` is supported: transform the **ego and global logits separately**,
then average their capped ratios. The global head sees the full scene, including
cross-car relationships; the ego head provides an additional projected-distribution
objective. Their average is a composite objective, not one joint-density KL.
Hierarchical mode likewise averages transformed rewards equally across scales.
With temporal windows this matches window distributions, not single-state
occupancy. It does not add an explicit altitude or maneuver reward.

The mode and cap inherit from checkpoints; old checkpoints default off/cap 10.
You can explicitly enable this on an existing checkpoint or disable it with
`--no-discriminator-forward-kl-reward`. Weights and optimizers are retained;
a reward-change warning records the switch. It is mutually exclusive with
`--amp` and `--discriminator-log-reward`.
Checkpoints from the original uncentered forward-KL implementation automatically
use the fixed baseline on resume, with a migration warning. New checkpoints
record `discriminator_forward_kl_baseline=1`. Restart/resume the trainer to apply
this code change to an already running process; no extra CLI flag is needed.
The offset changes the variable-length episodic objective intentionally; it
removes the constant survival bonus but does not guarantee the absence of stalling.
TensorBoard logs `imitation/forward_kl_reward`, `imitation/forward_kl_baseline`, `imitation/ratio_cap`, and
`imitation/ratio_capped_fraction` over live head predictions.

The earlier [state-occupancy experiment](FS4_OCCUPANCY.md) instead used a separate
single-state discriminator and a weighted, centered bonus. Its results do not
validate this main-reward variant.

### Target discriminator

The frozen **target discriminator is enabled by default**. It starts as an exact
copy and receives a Polyak update after each online optimizer step, including
when gradients accumulate over microbatches. Default EMA span is 8 optimizer
steps (`tau=2/9`); set `--target-discriminator-span` or `--target-discriminator-tau`.
Rewards use only the target. With `--no-target-discriminator`, rewards use the
online model. All heads train every discriminator epoch in either mode.
Diagnostic-only online/target comparison and held-out inference are removed.

Other restored protections:

- All eligible expert matches/POVs/periods and simulator groups contribute to
  training; no holdout scenarios are reserved. Previously held-out expert
  matches also become eligible for bounded uniform replay resets.
- Frozen-opponent transitions are excluded if historical self-play is explicitly
  enabled (it defaults off for GAIfO).
- Generated-window history is mixed into discriminator updates (default 50%,
  262,144-window capacity, up to 16,384 additions every four rollouts).
- Paired source padding is matched before noise and microbatching, so episode
  age cannot be used as a source label.
- Expert windows never cross replay periods, goals, invalid rows or flagged
  corrections, including skipped intermediate rows. Filtering is loader-only:
  the unfiltered corpus remains unchanged.
- Generated histories persist across rollouts but are masked at resets. Timeout
  successors use terminal observations, not the next episode's reset frame.
  Goal-terminal transitions receive zero imitation reward and are not used to
  fit the discriminator, matching the previous no-goal GAIfO objective.

The default reward is imitation only. Nexto diagnostics still run; use
`--gameplay-reward-weight` to explicitly add a scaled Nexto reward. The historical
GAIfO discount convention is retained: `--gamma` and `--gae-lambda` refer to
frameskip 4 and are exponentiated by `frameskip / 4` for GAE. PPO-only defaults
and discount semantics are unchanged.

### Optional gamma annealing

Gamma stays fixed unless `--gamma-end` is supplied. The annealing duration
defaults to `--timesteps`; use `--gamma-anneal-steps` to override it. For example,
add these flags to a **new** training command:

```bash
--gamma .99 --gamma-end .995 --gamma-anneal-steps 1000000000
```

This linearly interpolates the reference gamma from .99 at transition zero to
.995 at one billion environment agent transitions, then holds .995. Progress
uses the same global transition count as the training step, including both
teams and historical-opponent slots, not physics ticks or optimizer updates.
Gamma is selected at the beginning of each rollout and held constant throughout
its return/GAE calculation, including timeout bootstraps. GAE lambda is unchanged.
At frameskip 8, the effective decision discount is the square of the scheduled
reference gamma: .9801 initially and .990025 at the endpoint.

The same options work with `ppo.py`, where gamma retains its existing
**per-decision** meaning without frameskip conversion. TensorBoard logs
`ppo/gamma` (configured/reference scale) and `ppo/gamma_per_tick` (the equivalent
physics-tick discount, not a change to GAE). Scheduled discounts use a scalar tensor to avoid recompiling GAE
for every new gamma value when `--compile` is enabled.

Checkpoints save the start, end and duration. Resume with the same schedule
start/end flags; omitting `--gamma-anneal-steps` reuses the saved duration.
Progress continues from the checkpoint's saved global step, never from zero,
and does not depend on a changed `--timesteps` budget. Mismatched schedules
are rejected, including accidentally omitting `--gamma-end`. Older fixed-gamma
checkpoints remain resumable without the new flags. Changing an existing
fixed-gamma checkpoint into a scheduled run is not supported by strict resume.
Lowering gamma changes the objective and requires critic adaptation; this option
does not reset the critic or change any active training process.

## Full-sequence aerial expert mixture

An optional `--expert-maneuver-index replays/rlcs/maneuvers/aerial-v1.json`
adds a default 25% stratified maneuver component to expert training batches;
`--expert-maneuver-fraction` overrides that fraction. Remaining examples still
come from the unfiltered corpus. Every eligible observation throughout each
indexed setup/flight/recovery interval can end a sampled window, not only touches.
Strata, matches and sequences are balanced hierarchically, and all windows use
original contiguous source rows. The full match pool is available to both
ordinary and maneuver sampling. Replay-reset sampling remains uniform rather
than maneuver-enriched. Held-out inference/accuracies are no longer computed.
Resume warns when the expert mixture changes.

See [`replays/MANEUVERS.md`](../replays/MANEUVERS.md) for the built dataset,
detector thresholds, limitations, source-integrity checks and rebuilding commands.
This is a sampling-only experiment; discriminator cadence and receptive field
do not change, and already-running trainers are unaffected.

## Joint action combinations

Add to a **new run**:

```bash
--joint-actions --ent-coef 0.01
```

This replaces the seven independent categorical draws with one categorical
distribution over all **648** combinations of CARL's existing control values
(`3 × 3 × 3 × 2 × 2 × 3 × 2`). For example, pitch-back + boost + jump can receive
probability as a coordinated unit. Each sampled combination decodes to the
ordinary seven-control action vector. A combination is masked whenever any of
its controls is illegal under the current CARL mask.

The motivation is **within-decision exploration correlation**. Independent heads
already share a state-conditioned network and can produce coordinated
deterministic controls. A joint distribution additionally permits correlated
stochastic choices, such as either committing to a boost/jump/pitch combination
or doing none of those, rather than sampling all mixtures independently.
This option uses the complete Cartesian product, not a hand-pruned action table
or a multi-step maneuver. It acts at the configured frameskip.
`--factorize` independently selects the discriminator's ego/global heads.

PPO uses the sampled combination's joint log probability and joint categorical
entropy. Use `--ent-coef`; per-head entropy overrides are rejected in this mode.
TensorBoard records `entropy/joint`, `entropy_coef/joint`,
`entropy_bonus/joint`, and the existing total entropy/bonus metrics. Joint entropy
is not the sum of marginal entropies once the controls become correlated.

Supported by the shared PPO/GAIfO/DIFO actor, CALM's conditional actor, historical
opponent snapshots, compilation, and PPO/GAIfO viewer playback. The actor's final
layer grows from 18 to 648 logits, increasing compute/memory and introducing
more separately learned output parameters. Aerial-learning benefit is still an
experimental question; this does not by itself establish a fix for FS4.
The first two-seed FS4 warm-start comparison found equal mean aerial-contact
rates for independent and joint actions; see
[FS4_RECOVERY_AUDIT.md](FS4_RECOVERY_AUDIT.md#joint-action-results).

Default: **off** for new runs and older checkpoints. Resume inherits the saved
mode when the flag is omitted. Independent and joint actor architectures cannot
be interchanged with ordinary `--resume`; use a new run. `--no-joint-actions`
explicitly selects independent heads for a new run.

## Per-action entropy exploration

With independent actions, the trainers support a separate entropy coefficient for each CARL action head:
`--ent-coef-horizontal`, `--ent-coef-vertical`, `--ent-coef-throttle`,
`--ent-coef-powerslide`, `--ent-coef-boost`, `--ent-coef-air-roll`, and
`--ent-coef-jump`. Horizontal jointly controls steer/yaw; vertical controls pitch.
Unspecified heads inherit `--ent-coef` (default .01), preserving the previous
summed-entropy objective when no overrides are supplied. Coefficients must be
finite and nonnegative; zero disables that head's entropy bonus.

For a boost-focused exploration experiment, add:

```bash
--ent-coef .01 --ent-coef-boost .03
```

This weights boost entropy three times as strongly without changing other
heads' coefficients. It encourages uncertainty, not boost-on specifically, and
is not a boost-usage reward or a probability floor. CARL's action masks still
apply: a head with only one legal choice has zero entropy and receives no bonus.
Only learner transitions contribute when historical self-play is enabled.

With independent actions, named TensorBoard metrics are logged for all seven heads:

- `entropy/<head>`: mean masked categorical entropy in nats during PPO updates;
- `entropy_coef/<head>`: effective coefficient, including inherited defaults;
- `entropy_bonus/<head>`: coefficient times entropy;
- `entropy_bonus/total`: the total entropy bonus subtracted from PPO loss.

The existing `entropy` metric remains the sum of the seven head entropies.
These are training-batch averages, not aerial-only or boost-initiation rates.
Three-choice heads have a larger maximum entropy than binary heads; values are
not normalized by the number of choices.

Coefficients are saved in checkpoint configuration. As with other PPO update
settings, resume uses the supplied CLI settings, so repeat the override flags
to preserve them. A changed effective coefficient emits a warning but is allowed
for controlled exploration ablations. Older checkpoints remain compatible.
Already-running trainers retain their loaded code and settings; applying this
option requires launching or explicitly resuming a trainer.

## Optional RewardPlus shaping

`--reward-plus` selects the former custom RewardPlus mechanics instead of Nexto
for the gameplay component. It is **off by default**; default GAIfO remains
imitation-only apart from an explicitly enabled goal reward. Example:

```bash
--reward-plus --reward-plus-weight .1 --goal-reward-weight 100
```

`--reward-plus-weight` aliases `--gameplay-reward-weight`. Its default is 1 when
`--reward-plus` is enabled, otherwise 0. Explicit zero disables the gameplay
contribution while still allowing its diagnostics. These historical weights
can produce large sparse rewards; select a mixing weight deliberately rather
than treating RewardPlus as a verified fix for the discriminator timing issue.

Restored terms include acceleration/boost-free speed progress, boost pickup,
aerial lift/shot improvement, ground pop and a bounded 1.5-second air-dribble
setup window, kickoff approach/touch, wall touch/control, inverted-ground
penalty, flip resets and demos. Old conditions/weights are retained; this
does not add the previously removed wasted-air/boost or angular-velocity terms.
Stateful windows clear on episode boundaries, other touches or lost-validity
conditions as before. Component magnitudes log as
`reward_plus/<component>_abs_per_tick`, never per decision.

GAIfO excludes RewardPlus's built-in goal term and retains its separate
`--goal-reward-weight`/calibrated-goal path, so goals are not counted twice and
existing goal magnitudes do not change. Without explicit goal reward, GAIfO
RewardPlus adds **non-goal shaping only**. PPO's `--reward-plus` instead selects
the complete custom reward, including its historical 200-point goals; it does
not mix Nexto rewards into it.

GAIfO resume permits changing the flag/mixing weight with an explicit adaptation
warning; repeat the intended flags when resuming. PPO requires the checkpoint's
reward-objective ID to match, preventing accidental Nexto/RewardPlus switches.
No active trainer is changed by adding this option.

## Optional calibrated goal reward

Gameplay event rates (`touches`, `aerial_touches`, `demos`, `goals_for`,
`goals_against`) use `gameplay/<event>_per_1000_ticks`: the denominator is
agent decisions times frameskip, so rates are comparable across frameskips.
These are aggregate agent-physics ticks, not wall-clock time. Historical
`*_per_1000_steps` logs are unchanged and still supported by goal calibration;
new fallback calibration uses `gameplay/ticks_per_step` to recover per-decision
event rates. Exact completed-episode calibration remains preferred.

Other time-normalized diagnostics also use tick units:

- `reward_per_tick`, `environment_reward_per_tick`, `imitation/reward_per_tick`,
  `goal_reward/abs_per_tick`, and `reward_nexto/<component>_abs_per_tick` divide
  actual accumulated reward by agent-physics ticks rather than decisions.
- `imitation/episode_length_ticks_mean` measures duration in physics ticks.
- `physics_ticks_per_second` and the progress rate use aggregate agent-ticks/s.
- `self_play/learner_ticks` and `self_play/environment_ticks` count agent-ticks
  represented in each update; configured snapshot/global step counters retain
  their training-transition meaning.

Fractions (demoed time, timeouts, historical opponents, replay resets), per-episode
reward totals, losses, entropy and value-fit statistics are not per-step rates
and are unchanged. TensorBoard's global progress axis, checkpoint/annealing
schedules and actual training rewards also remain unchanged. Renamed metrics
do not mix new tick units into historical per-decision series. Tick-normalized
reward logging does not itself make the training objective frameskip-invariant.

`--goal-reward` adds a team-signed goal reward: **+W for scoring, -W for
conceding**, including the goal-terminal transition where imitation reward is
zero. This does not enable Nexto shaping. The default remains imitation only.

With no explicit weight, calibration reads the latest GAIfO run under
`--output-dir`, or the run supplied by `--goal-reward-log-dir`. It prefers the
completed-episode mean of `abs(sum(imitation_rewards))`, weighted by completed
episode counts over the latest 100 aligned logging intervals. The value is
frozen for the new run and saved with its source/provenance in the checkpoint.
Resuming a goal-enabled run reuses its saved calibration rather than recalculating.

Older runs did not log imitation-only episode totals. Their fallback is explicitly
labeled `legacy-rate-estimate`: absolute mean imitation reward per decision
divided by the inferred episode-ending rate from goals and no-touch timeout
fraction. It assumes approximately stationary recent rewards/episode flow and
negligible max-tick truncations. It is **not** an exact measurement of completed
episode returns. On `gaifo-20261111-225802-1`, this estimate was approximately
190–192 reward units during the investigation; recalibration may differ as the
run progresses.

All new runs log exact `imitation/episode_abs_sum_mean`, `imitation/episode_sum_mean`,
`imitation/episode_length_ticks_mean` and `imitation/episode_count`, excluding the goal
bonus and gameplay shaping. Accumulators span rollout boundaries and reset at
episode ends. `goal_reward/weight` and `goal_reward/abs_per_tick` identify the separate
goal contribution. Explicit `--goal-reward-weight 192` also enables the goal
reward without reading any logs. Changing the goal magnitude on resume is
rejected rather than silently changing the objective.

```bash
PYTHONPATH=/tmp/opencode/ego-controller/stage uv run --no-sync python gaifo.py \
  --frameskip 8 --replay-dir replays/rlcs \
  --ppo-lr 3e-4 --ppo-lr-end 1e-5 --discriminator-log-reward \
  --goal-reward --goal-reward-log-dir runs/gaifo-20261111-225802-1
```

## Running and resuming

Use the staged CARL build described in `training/OBSERVATIONS.md`:

```bash
PYTHONPATH=/tmp/opencode/ego-controller/stage uv run --no-sync python gaifo.py \
  --frameskip 8 --replay-dir replays/rlcs

PYTHONPATH=/tmp/opencode/ego-controller/stage uv run --no-sync python -m vis \
  runs/<gaifo-run>/latest.pt
```

GAIfO defaults to FP32/uncompiled execution, as before. `--bf16 --compile`
enables the shared accelerated PPO path. Expert scenes/indices are loaded into
memory; `--expert-frame-limit` optionally caps that loader for small experiments
without changing the corpus. The replay reset bank is separately capped by
`--replay-reset-size` and remains a uniform in-memory subset of eligible rows.

Replay resets now require **more than four seconds of separation from both
period boundaries**, in both the training bank and viewer sampler. Legacy
periods lack exact previous-goal timestamps, so every boundary is conservatively
treated as a possible goal: this also excludes four seconds at non-goal edges.
Periods too short to contain an eligible interior contribute no resets. The
buffer uses parsed-cadence metadata (legacy canonical data defaults to frameskip
4), not the policy frameskip. An extra endpoint margin covers irregular appended
goal frames. Replay files are never trimmed or rewritten, and reservoir sampling
remains uniform over the eligible interior rows. This changes future bank loads,
not already loaded banks in active trainers.

A full read-only scan after adding this guard found **20,861,943 eligible 1v1
rows**. A bounded 2,048-state reservoir load completed in 34.36 s; synthetic
checks also covered the viewer path, both parsed cadences, short periods, exact
boundary exclusion, reproducibility and source-file hashes.

Checkpoints include online/target discriminator weights, discriminator optimizer,
expert sampling RNG, generated history, validation simulator assignment, replay
reset RNG, and ordinary PPO/RNG state. Resume validates schema, cadence, split and
architecture compatibility. Simulator episodes themselves start fresh. These
checkpoints are not compatible with the old removed GAIfO implementation.
