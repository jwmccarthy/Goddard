# Forward-KL state occupancy experiment

Follow-up to [FS4_RECOVERY_AUDIT.md](FS4_RECOVERY_AUDIT.md) and
[FS4_FOUNDATIONS.md](FS4_FOUNDATIONS.md), 2026-11-15.

## Objective and direction

Encourage the policy to cover expert game-state regions while retaining the
temporal imitation reward and ±50 goal reward. Coverage is an exploration
hypothesis; improved occupancy alone does not establish successful aerial play.

Let p = expert occupancy and q = current policy occupancy. For normalized
occupancies, maximizing negative forward KL has functional derivative

```text
F(q) = -KL(p || q)
δF/δq(s) = p(s) / q(s).
```

A balanced binary state discriminator trained with expert=0, generated=1 has
ideal logit `z(s) = log(q(s)/p(s))`. Therefore the local policy-gradient reward
is `exp(-z(s))`, with the discriminator held fixed during the policy update.
Using `-z(s)` instead gives the reverse-KL log-density-ratio reward. The existing
temporal log reward is `logsigmoid(-z)`, a third, distinct choice.

The ratio reward is a **local gradient surrogate**. It is not a global identity
equating forward KL with expected ratio: under exact densities,
`E_q[p/q] = 1` wherever q covers p. Nor would adding a detached occupancy KL
statistic directly to PPO loss provide gradients through simulator states.

The experiment uses:

```text
ratio(s) = min(exp(-z_target(s)), 10)
bonus(s) = ratio(s) - mean_generated(ratio)
reward = temporal_imitation + goal_reward + coefficient * bonus
```

Clipping occurs before exponentiation for numerical stability. This is a
**bounded forward-KL-inspired local-gradient surrogate**, not exact global KL
minimization. Learned density error, target-network lag, clipping, GAE, PPO
clipping and finite variable-length episodes further separate the practical
update from the ideal normalized-occupancy gradient. Centering removes the
sampled constant component; it is not a proof of exact episodic-objective
equivalence. Truly unvisited states still need sampled paths leading toward
them; a density estimator cannot supply those trajectories by itself.

## Separate single-state discriminator

- Global 51-feature scene: ball and both cars in the observer's team-normalized
  coordinates. One frame, no temporal relative encoding, no factorization.
- Existing MLP discriminator implementation, flags and boost visible, boost
  rounding enabled, motion noise .002, balanced binary training.
- Adam LR .0003; 2,048/source microbatches; target EMA tau .2.
- Bootstrap from 100,000 expert endpoints and 100,000 cached native endpoints
  (native seed 97), with 300 updates of 8,192 samples/source.
- Validation: 30,000 expert endpoints from held-out matches and 30,000 native
  endpoints from seed 98. Expert training/validation split: 1,418/160 matches.
- Online: one update of up to 65,536/source per rollout, using current-policy
  generated endpoints and the same sampled expert endpoints as temporal AIL.
  Goal-terminated transitions are excluded, matching production discriminator
  sampling; time-limit truncations retain their final, pre-reset observations.
- The baseline also trains this discriminator and computes its bonus, with
  coefficient zero, to match extra initialization, RNG and update work.

The initial held-out balanced accuracy was **68.05%**. On native validation
states, mean capped ratios were 1.013 overall, .799 for floor states and 2.436
for aerial states; .590% hit the cap. These classifications are diagnostic
subsets only: they do not define the reward. Higher ratios for aerial states
are compatible with missing coverage, but accuracy does not establish density
calibration or guarantee a useful policy gradient.

## Controlled recovery comparison

- Source: aerial-poor FS4 checkpoint `gaifo-20261115-184227-1`, step 1,900,019,712.
- Independent categorical actor, temporal discriminator frames `[t−32,t−16,t]`,
  original unscaled imitation reward, goal weight 50.
- Seeds 231/232; coefficients 0, .1 and .3.
- 1,024 simulations, 64 decisions/rollout, FS4, 16 iterations:
  **8,388,608 agent-physics ticks per arm**.
- PPO: two epochs, minibatch 16,384, LR .0003, 256 optimizer updates/arm.
  Temporal D: 16 updates/arm, up to 65,536 windows/source/update.
- Uniform bounded replay-reset bank, .75 replay-reset probability, existing
  four-second replay edge buffers; cached experts retain complete source windows.
- All arms use the production staged CARL binary, SHA256
  `fc3ed0a27536bd3cc37f4bf0cacb1ca1ae153b19b51a883da2651c6abb2a7b92`.
- Evaluation: seeds 1231/1232, 128 simulations × 60 seconds, native per-tick
  contact onsets. Frozen source evaluation uses these same seeds.

This matches optimizer counts per physics tick to the reference production
regime, but uses smaller collection batches, cached expert data and restarted
actor/temporal-D optimizers. It tests short recovery, not from-scratch learning.
Earlier continuation harness versions included terminal transitions in D
sampling; compare against the contemporary zero-weight arm rather than treating
older baselines as exact controls.

## Independent coverage diagnostics

Learn 256 k-means clusters from 100,000 expert states, after featurewise
standardization with scale floor .05. Estimate expert test occupancy using the
30,000 held-out states. During evaluation, assign policy states at 30 Hz and
report histogram forward KL, smoothing the policy distribution with 1% expert
mass. This is **coarse, smoothed histogram KL**, not full continuous-state KL.

Report both all-age and age≥2-second distributions to expose how much matching
comes from replay resets. The age-filtered distribution is a useful policy
diagnostic, not the same occupancy measure as unfiltered expert sampling.
Clustering uses all global scene features and supplies no training reward in
the state-discriminator experiment.

Also report aerial/high-aerial contact onsets, overall contacts, completed floor
departures with ≥.5 seconds above the aerial threshold, and departures with
aerial contact. These are behavioral proxies, not direct labels of completed
air dribbles, flip resets or other maneuvers.

## Results

Means across the two seeds, evaluated for 60 seconds in 128 simulations/seed:

| Condition | Held-out coarse KL | Age≥2s coarse KL | Aerial contacts/car/min | High-aerial contacts/car/min | All contacts/car/min | Sustained departures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen source | .2301 | .3187 | .3691 | .1035 | 21.57 | 7.11% |
| Zero-weight continuation | .2527 | .3398 | .3848 | .0938 | 21.79 | 8.63% |
| Weight .1 | .2423 | .3334 | .4102 | .1094 | 19.91 | 7.81% |
| Weight .3 | .2248 | .3364 | .4414 | .1309 | 18.67 | 8.66% |

Weight .3 improves all-age coarse coverage relative to the contemporary baseline
in both seeds. The age≥2s result is mixed (.3619 versus .3243 in seed 231;
.3108 versus .3554 in seed 232). Aerial contacts also improve only in seed 231
(.4961 versus .3672); seed 232 slightly declines (.3867 versus .4023).
Overall contacts decline in both seeds at both nonzero weights.

Thus the mean aerial-contact increase is promising but **not a reliable
two-seed aerial-learning improvement**. There is no evidence here of completed
named maneuvers. Clipping averages .62–.65% of generated states (maximum 3.10%
in an individual rollout); mean PPO approximate KL stays near .0062–.0063.

## Verification and status

- Exact two-state calculation: density-ratio surrogate gradient equals the
  negative forward-KL gradient when unclipped, and favors the neglected mode.
- Extreme logits: capped ratio remains finite.
- Native one-iteration smoke: zero/.1 arms both complete PPO and evaluation;
  finite actor weights, near-zero centered bonus mean, finite held-out metrics.
- Reloaded state discriminator and optimizer; checkpoint round-trip gives
  bit-identical bonus predictions on 2,048 held-out native endpoints.
- Two-seed online comparison completed; results above.

## Artifacts and reproduction

Experimental harnesses are `/tmp/opencode/state_occupancy.py`,
`/tmp/opencode/occupancy_kl.py`, `/tmp/opencode/fs4-matched-continuations.py`, and
`/tmp/opencode/fs4-cadence-transfer.py`. Artifacts under
`/tmp/opencode/fs4-recovery/` include `state-occupancy-initial.pt`,
`state-occupancy-pretrain.json`, `occupancy-clusters.pt`, `occupancy-smoke-230*`,
and `forward-occupancy-{231,232}*`.

```bash
env PYTHONPATH=/home/jack/.local/share/goddard/carl/stage:/home/jack/Goddard \
  OMP_NUM_THREADS=2 .venv/bin/python /tmp/opencode/fs4-matched-continuations.py \
  --seed 231 --iterations 16 --eval-seconds 60 --tag forward-occupancy \
  --batch-regime production-rate --source gaifo-20261115-184227-1 \
  --occupancy --state-occupancy --baseline \
  --modes fs4_unscaled_three fs4_unscaled_three_occ01 fs4_unscaled_three_occ03
```

Repeat with seed 232. This separate-state auxiliary variant lives in the
isolated experiment harness. Production now offers a different
`--discriminator-forward-kl-reward` mode, documented in [GAIFO.md](GAIFO.md#forward-kl-reward):
it replaces the main imitation reward using the existing discriminator's
per-head density ratios, with no auxiliary weight. Since 2026-11-16 this main
reward subtracts a fixed baseline of 1 (rather than the experiment's rollout mean).
