# Learning and aerial-exploration sweep — 2026-11-09

Read-only audit; no reward/configuration changes, optimizer steps, trainer
restarts or checkpoint replacements. Reproduction artifacts and the frozen
checkpoint are under `/tmp/opencode/learning-sweep-20261109/`.

The newest saved run inspected was `runs/dac-20261109-094154-0`, checkpoint
**6,973,030,400 transitions**: two discriminator frames at stride 3, reward-plus
enabled at weight 1, signed goal weight 200, gamma .99, lambda .95, rollout 32.
The older four-frame run `dac-20261108-224043-0` used **no reward-plus or signed
goal reward**. They must not be described as the same objective/experiment.

## Highest-priority findings

### 1. Low jump exploration is hidden by aggregate entropy — measured

A frozen-policy native probe generated **196,608 transitions**, using 256
simulations and a bounded, representative training-match reset bank. On
**44,499 grounded aerial-opportunity observations** (ball above 300 uu and within
2,200 uu), mean jump-head entropy was **0.0225 nats**, versus a binary maximum
of 0.693. Jump was legal in these grounded states; this is not a masking artifact.
The mean jump probability was 2.70%, with the low entropy indicating mostly
state-deterministic choices rather than diffuse exploration.

Grounded pitch and roll head entropies were 1.014 and 1.010 nats (maximum
1.099 each). These controls usually have limited effect until takeoff/flip, yet
they contribute heavily to the healthy-looking aggregate entropy (~3.17 in
training). **Aggregate entropy is not evidence that jump/boost exploration is
healthy.** There is no fixed entropy budget being literally consumed by those
axes: the loss is additive. Do not simply mask pitch on the takeoff step.

This is a credible exploration bottleneck, not proof that each no-jump decision
was wrong. Next experiments should measure legal per-head/state-conditioned
entropy and trial targeted exploration alongside setup-heavy resets.

### 2. Reward-plus favors accessible speed/boost income before aerial success — verified structure, causal effect unproven

In `rewards/shaping.py`:

- There is **no explicit airtime, wasted-air or boost-use penalty**. The negative
  mechanical term is inverted-*ground* time, not being upside down in flight.
- Aerial lift/shot rewards require a qualifying solo touch. Merely approaching,
  taking off or making a promising but unsuccessful attempt earns neither.
- Air-dribble setup requires a qualifying ground pop first, then close controlled
  follow-up within 1.5 seconds. It does not cover arbitrary floor/wall launches.
- Lift needs an upward ball-velocity change; shot shaping needs an improved,
  strongly goal-aligned trajectory. Useful downward/sideways defensive contacts
  can receive zero from both terms.
- Speed and boost gains are positive-clipped, not potential differences. Speed
  loss and boost spending do not undo previously awarded gains. A synthetic
  600→900→600 uu/s unboosted cycle earns **6.52** speed reward despite zero net
  speed gain. The same positive acceleration while boosting earns **3.26**.
  These are incentive-algebra tests, not a demonstration of an optimal native
  exploit or evidence that the policy intentionally farms them.

The newest run's last 100 logged updates averaged **0.239 speed reward per actor
decision**, versus about **0.026 combined aerial lift/shot/pop/setup/reset**.
Boost gain added 0.0318. This is observed reward availability, not a causal
ablation. Gameplay reward RMS was 6.90 versus imitation RMS 2.06; the sparse
±200 goals contribute strongly to that RMS, so it must not be attributed solely
to dense speed shaping.

Recommended test: reduce/correct repeatable dense income versus potential-based
opportunity progress, preserve grounded declines, and compare *useful* aerial
contacts/recoveries and concessions. Avoid adding unconditional altitude/airtime
reward, which can teach jumping without purpose.

### 3. Discriminator shortcuts and terminal incentives remain separate risks

The earlier four-frame audit found reward sensitivity to replay midpoint
irregularity, including ~0.149 reward gain from replay-sized positional noise on
otherwise unchanged native windows. Parse-time selection does **not** repair
that domain difference. Neither inspected checkpoint enabled the new opt-in
contrastive regularizer; the two-frame architecture cannot use it without a
window/configuration change. Long-run mitigation effectiveness is unproven.

Earlier paired DAC outcome failures were on the **pure-imitation** objective.
The newest reward-plus run does have signed goals, so those exact rankings must
be re-tested rather than assumed unchanged. Absorbing bootstrapping itself
passed the earlier wiring audit; a shared absorbing value is not automatically
a goal-aligned task reward.

The new expert mixture can also introduce a height/source shortcut if generated
opportunities remain poorly covered. The unconditional 40% component and separate
setup-heavy resets help coverage, but **generated/expert opportunity-conditional
balancing has not been added**. Monitor this before treating the dataset as a fix.

## Additional learning constraints and misleading diagnostics

- **Credit assignment:** gamma .99 at 30 Hz has a 2.30-second discount half-life.
  The direct GAE trace (.99×.95) has a **0.377-second half-life**; a reward one
  second later has only ~0.159 direct trace weight. A rollout spans 1.067 seconds.
  Value bootstrapping can propagate longer-term reward, so these are not hard
  horizons. Multi-second aerial setup is nevertheless demanding for a sparse
  touch reward. Test longer traces/rollouts or suitable shaping, not a blind
  discount increase.
- **Reset mismatch:** `ReplayResetProvider` restores scene physics and internal
  car state, but `CARLTorchVectorEnv._apply_typed_reset_state` resets boost pads
  to available rather than restoring replay cooldowns. This can create overly
  generous pickup opportunities. The new bank retains the existing simulator
  behavior; it does not silently claim pad-state fidelity.
- **Policy memory:** the inspected actor/critic are feed-forward. Some jump
  release/repress timing and partially observed internal state may benefit from
  history; that is a hypothesis requiring a matched experiment, not a verified
  architecture bug.
- **Self-play metrics:** goals-for and goals-against are necessarily equal when
  pooled over both teams. They do not establish strength or win-rate improvement
  against an independent opponent. `episode_return` is accumulated by the
  environment **before imitation rewards are added**, so it is not the optimized
  mixed return. The pure-imitation run logging zero episode return is not proof
  of zero PPO rewards. Add fixed-opponent/native task evaluation before tuning
  from these aggregate charts alone.
- **Source quality/coverage:** only 1,569/10,000 downloaded duels passed the
  current strict raw admission policy, mostly because older files lack required
  header fields. This is an evidence limitation, not proof those games are bad.
  Missing ranks/completion evidence were not converted into training approval.

## Checks that did not indicate a general learning freeze

- Last-100-update PPO approximate KL: **0.00970**, clip fraction **0.0865**;
  discriminator updates occurred on every logged update. No accuracy-gate freeze
  was evident in this run.
- Frozen native probe critic explained variance: **0.933** on its bootstrapped
  32-step targets, with return std ~47.6 and TD-return RMSE ~12.3. This is not
  evidence of perfect long-horizon values, but contradicts a completely failed
  critic diagnosis based on raw value loss alone.
- Sampled actor gradient norms were 0.455–0.608. Critic norms were much larger
  (216–281), but actor and critic are clipped **separately** in `training/ppo.py`,
  so the critic cannot shrink the actor gradient through a shared norm clip.
- All 198,131 eligible rows across a 256-file reset audit had known counterpart
  internal states. No missing-opponent-state problem appeared in that sample.
- Existing native action-controller/masking checks previously passed; this sweep
  did not find an action-space inability to execute aerial controls.

Evidence: `metrics.json`, `native_probe.json`, `native_probe.py`, `snapshot.pt`
under the artifact directory. Incentive characterization tests are in
`training/test_aerial_exploration.py`. Findings above distinguish code facts,
observations, and proposed causal experiments; none is a claimed cure.
