# FS4 temporal shortcuts and exploration — 2026-11-15

Follow-up to [the aerial reward audit](AERIAL_REWARD_AUDIT.md), specifically the
reported regression when adding multi-frame discrimination.

**Latest investigation:** [FS4_RECOVERY_AUDIT.md](FS4_RECOVERY_AUDIT.md).
The entropy results below establish short-term exploration gains, not recovered
aerial-maneuver learning. The newer investigation tests control transfer and
physical-time-matched learning directly.

## Practical conclusion

- **Exploration-focused trial:** raise only `--ent-coef-jump` from `.01` to
  **`.10`**; `.03` is the lower-intensity tested setting. Keep other heads at
  `.01`, GAE lambda `.98`, and initially keep the existing 32-step rollout.
  Both jump coefficients increased aerial occupancy and aerial contact rate in
  both short continuation seeds; `.10` had the larger gain. This is a candidate
  for a longer matched run, not a demonstrated long-term optimum.
- **Temporal mitigation:** the strongest feature-restriction candidate retains
  current flags/boost but removes their history. It cuts offline phase
  sensitivity 48% while retaining takeoff preference. Full flag removal loses
  useful discrimination. It is now available as the default-off production option
  `--discriminator-endpoint-scalars`. The later short online comparison in
  [FS4_RECOVERY_AUDIT.md](FS4_RECOVERY_AUDIT.md) found no aerial-contact benefit.
- The existing `--experimental-jump-phase` with weight-1 consistency improves
  offline robustness, but did **not** improve aerial contact rate in the short
  online comparison. Increasing noise alone also failed to fix phase sensitivity.
- Longer traces, lambda `.99`, indiscriminate jump-temperature increases, and
  longer holds alone are not supported as reliable fixes by these measurements.

The exploration-only parameter change can be expressed as:

```bash
--frameskip 4 --rollout 32 --ent-coef 0.01 --ent-coef-jump 0.10 --gae-lambda 0.98
```

Keep the other run settings matched for that comparison, including reward
scaling, discriminator layout, and gamma schedule. The trials below used the
audited three-frame/stride-four checkpoint. They do not establish identical
effects for the newer running two-frame/stride-four configuration. In particular,
two frames/stride four at FS4 span **16 ticks**, whereas the audited three-frame
FS4 and two-frame FS8 windows span **32 ticks**. A two-frame FS4 control with
matched span uses decision stride **8**.

## Temporal findings

### A replay/simulator fingerprint survives behavior matching

The position–velocity discrepancy is not confined to collisions or different
proportions of boosted aerials. On held-out expert matches and an independent
native rollout, restrict to all three frames airborne at 182.5–1,500 uu, away
from walls and the ball, not boosting or flagged as flipped. Further restrict
both intervals' mean acceleration to within 200 uu/s² of gravity. Match source
counts in height, speed, and vertical-velocity bins.

For the resulting **176 windows per source**, mean-interval trapezoidal
position–velocity residual has median **8.16 uu expert / 0.38 uu native**.
The residual alone distinguishes sources with **AUROC 0.991** (higher means
expert). The broader quiet-flight subset gives 0.986 on 199 per source.
This is strong evidence of an available low-level domain cue, not proof that
the trained discriminator uses this particular scalar or that it caused the
online aerial regression. The small matched subset and approximate physics
matching also limit attribution to the parser versus simulator physics.

Adding the existing discriminator motion noise to both sources reduces this
simple residual classifier's AUROC to about **0.635 at sigma .002**, and **0.525
at .006** (20 independent noise draws, same preselected ballistic windows).
This measures obscuring a cue, not improving learned reward or control.

### Real tick-phase sensitivity is concentrated around takeoff

Run native physics at one tick, holding each policy action for four ticks.
Compare actual three-frame windows at `[t−32,t−16,t]`, shifted together by
0, 1, 2, or 3 ticks. Exclude every window crossing an episode boundary.
Two evaluation seeds each use 256 simulations × 30 seconds, approximately
450,000 complete windows. Category membership is fixed from phase zero.

For the saved FS4 target, shifting **two ticks** changes mean reward by:

| Phase-zero category | Seed 97 | Seed 98 |
| --- | ---: | ---: |
| Floor | +.0159 | +.0153 |
| Ground-to-air | **+.2460** | **+.2425** |
| Established aerial | +.0092 | +.0075 |

These are real trajectories, not interpolated or edited discriminator inputs.
They still advance physical state by up to 25 ms: some sensitivity is legitimate
near an event. The result is not an instruction to always select the
highest-reward phase. Among takeoffs whose **ego flags remain identical across
all phases**, the two-tick change instead averages −.163 / −.155. Flag transitions
alone therefore do not account for the effect; motion/event alignment matters.

### Retrained mitigations: consistency works better than noise alone

Nine matched CPU discriminator variants, two initialization seeds, 1,000 Adam
updates at 3e-4, 512 windows per source per update, existing factorized
ego/global MLP and log reward. Every variant sees the same sampled base windows.
Expert split: **1,418 training / 160 held-out matches**, with all POVs and periods
of a match assigned together; 100,000 / 30,000 cached training/test pairs.
Native training/test trajectories use separate seeds 97 / 98; 100,000 / 30,000
windows. All source windows are complete and retain normal corpus eligibility.

Native phase augmentation chooses real offsets 0–3 ticks. Expert pairs use
adjacent canonical endpoints, four ticks apart: no invented sub-tick human data.
Consistency is mean squared paired-logit disagreement, with the same motion
noise on both views, added to the existing balanced source loss. The weight-1
variant matches the objective of the existing `--experimental-jump-phase` option.

Means across the two initialization seeds:

| Variant | Takeoff mean absolute reward change, +2 ticks | Policy takeoff minus floor reward | Expert takeoff minus floor reward | Balanced source accuracy |
| --- | ---: | ---: | ---: | ---: |
| Baseline, noise .002 | .1104 | +.2538 | +.2316 | 67.71% |
| Noise .006 | .1124 | +.2583 | +.2363 | 67.71% |
| Retrain without flags | .0391 | +.1144 | +.1040 | 66.32% |
| Random phases only | .1313 | +.2486 | +.2298 | 67.69% |
| Phases + consistency **1** | **.0703** | **+.2031** | **+.1821** | **67.59%** |
| Phases + consistency 4 | .0414 | +.1335 | +.1238 | 67.25% |
| Endpoint-only control | .0601 | +.2480 | +.2523 | 66.68% |
| Temporal physics, endpoint flags | .0602 | +.2802 | +.2497 | 67.18% |
| Temporal physics, endpoint flags + boost amount | **.0575** | **+.2738** | **+.2408** | **67.13%** |

Weight 1 reduces takeoff phase sensitivity by **36%** while retaining most of
the takeoff/floor contrast. Weight 4 and removing all flags produce greater
invariance but roughly halve that contrast. Source accuracy alone cannot select
a good imitation reward: a shortcut can also improve accuracy.

An additional targeted representation is particularly promising: retain every
physics frame but expose **only endpoint flags and boost amount**, rather than
their temporal changes. In the diagnostic harness those scalar slots are
repeated from the endpoint before relative encoding, at both training and
inference. This removes their history while preserving current-state semantics.
It reduces phase sensitivity **48%**, retaining the takeoff/floor contrast that
full flag removal weakens. This is a retrained feature restriction, not an
inference-only edit to a frozen production discriminator. The later online
continuation in [FS4_RECOVERY_AUDIT.md](FS4_RECOVERY_AUDIT.md) found no clear
aerial-contact benefit; its FS8 control also degraded. The production implementation is
available as `--discriminator-endpoint-scalars`; see [GAIFO.md](GAIFO.md#endpoint-only-flags-and-boost-temporal-history-erasure).

**Immediately available candidate: real-phase sampling plus soft consistency at
weight 1. Targeted representation candidate: endpoint-only scalar history.**
Noise .006 suppresses the isolated residual cue but does not address the measured
phase sensitivity. Random phase sampling alone is also insufficient in this
test. These classifiers again favor takeoff over floor; they do not reproduce
the production checkpoint's preference inversion. Long-run online causality
remains unresolved.

## Frozen-policy exploration interventions

Native one-tick evaluation, actions held for four ticks, seeds 111 and 112,
128 simulations × 60 seconds per seed, 75% bounded uniform replay resets. Aerial occupancy
and touches are measured at physics-tick time, including true terminal states.
“Touch ticks” count contact-containing physics ticks, not distinct strikes.
Age-conditioned occupancy excludes the first two seconds after each reset.
Complete floor departures and jump-hold bursts exclude episode/end-of-recording
censoring; floor departures can include low flips and incidental departures.

Only jump logits are modified. Temperature divides both jump logits. Hold bias
adds to the true-jump logit only while the controller reports an already-held,
active first jump younger than .2 seconds; action legality still applies.
These are evaluation interventions, not entropy-coefficient training runs.

| Intervention | Aerial after reset age ≥2 s | Mean jump hold | Holds ≥200 ms | All touch ticks/car/min | Aerial touch ticks/car/min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original policy | 5.19% | 98.2 ms | 15.2% | 20.19 | .357 |
| Jump temperature 2 | 6.84% | 86.0 ms | 11.3% | 15.91 | .500 |
| Hold logit bias +2 | 5.12% | 113.6 ms | 21.7% | 20.55 | .438 |
| Temperature 1.5 + hold bias 1 | 6.30% | 103.7 ms | 18.0% | 18.41 | .475 |

Table entries are means across the two seeds. Temperature 1.5 alone and hold
bias +1 alone were also screened on seed 111; full results are in its JSON.
Temperature 2 increases aerial exploration but reduces overall contact rate by
21% and shortens holds. Longer holds alone do not improve aerial occupancy.
The combined modest intervention retains holds/contact rate better, but still
reduces overall touches by 9%. These evaluation interventions do not establish
a good permanent action distribution for PPO training.
Completed floor departures reaching ≥.5 seconds above aerial height rise from
7.45% to 8.98% for the combined intervention, versus only 7.75% at temperature 2.
High-aerial floor launches and
high-aerial contacts remain rare. Raising altitude occupancy is not equivalent
to learning air dribbles or successful attacks.

## Short joint-AIL continuations

### Results

Two continuation seeds, each evaluated with its own independent rollout seed.
Entries below average the two seed-level metrics; these are not confidence
intervals or independent from-scratch training replicates. The phase row uses
the corrected independently sampled expert phase pairs described below.

| Setting | Aerial at age ≥2 s | High aerial at age ≥2 s | All touch ticks/car/min | Aerial touch ticks/car/min | Mean jump hold | Floor departures with ≥.5 s aerial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline: jump entropy .01, trace 32, lambda .98 | 4.64% | 1.09% | 17.04 | .383 | 99.4 ms | 7.41% |
| Jump entropy .03 | 5.07% | 1.35% | 16.78 | .453 | 89.0 ms | 6.45% |
| Jump entropy **.10** | **5.78%** | **1.56%** | **16.84** | **.477** | 88.7 ms | 7.48% |
| Jump .03, trace 64 | 4.71% | 1.06% | 17.28 | .367 | 92.3 ms | 8.00% |
| Jump .03, trace 64, lambda .99 | 4.77% | 1.07% | 18.39 | .365 | 94.0 ms | 7.14% |
| Jump .03, trace 32, phase consistency 1 | 4.67% | 1.37% | 16.27 | .381 | 87.7 ms | 6.07% |

Entropy .10 raises age-conditioned aerial occupancy **24%** and aerial contact
rate **24%** relative to the matched baseline, with mean overall touch rate
within 2%. However overall contact effects vary considerably by seed:
19.66 vs 16.54 in seed 121, but 14.02 vs 17.53 in seed 122. Holds shorten, and the
fraction of floor departures sustaining aerial height does not clearly improve.
This supports more exploration, not better complete aerial attacks or win rate.

Legal-floor-state mean jump entropy rises from .0210 to .0269 nats at coefficient
.10; mean jump probability rises from 2.16% to 2.67%. The policy remains strongly
concentrated. Simply increasing the coefficient does not quickly undo the
existing saturated jump policy or teach competent follow-through.

The 64-step/.03 arm looked promising in the first seed, but its occupancy fell
below the 32-step/.03 arm in the second (4.41% vs 5.16%). Lambda .99 also changes
direction across seeds. Neither warrants a claim of a reliable gain here.
Phase consistency's offline invariance benefit does not translate into improved
online aerial contacts in these small continuations.

### Design

The parameter comparison starts each arm from the same saved actor, critic,
online discriminator and target discriminator. Adam state restarts identically.
Each arm uses 128 self-play simulations, 32 iterations × 16,384 decisions =
**524,288 agent decisions / 2,097,152 agent-physics ticks**. Each iteration collects 64
decisions, trains the discriminator on all generated windows in two balanced
8,192-per-source updates, and uses two PPO epochs with minibatches of 2,048. Source
expert windows come from the fixed training-match cache above. Learning rates
are 3e-4, target tau .2, gamma .99, goal magnitude 50; other action-head entropy
coefficients remain .01. There is no aerial shaping reward.

To isolate credit length from batch size/update count, the 32-step GAE arms
compute two separate traces within the same 64-step collection, bootstrapping
the first from the pre-update value at step 32. The 64-step arms compute one
trace. This is a controlled trace-length comparison, not a full-scale benchmark
of changing the production `--rollout` argument. Lambda .98/.99 and jump entropy
.01/.03/.10 are compared separately. The phase arm uses the production
`JumpPhaseEnv`/`JumpPhaseRollout` and weight-1 consistency path.
Expert phase views are independently drawn with replacement from the two cached
canonical endpoints, matching the production sampler's possibility of identical
pairs. An initial pilot always paired distinct expert endpoints, overapplying
that part of the consistency penalty. Both phase arms were rerun with independent
draws; only those corrected results appear in the table.

Every final actor is evaluated for 60 seconds in 128 fresh simulations at the
ordinary FS4 action cadence, with physics-tick touch measurements. Both players
use that actor; this measures exploratory behavior, not win rate against a
fixed opponent. These are small continuations of one trained policy, not
independent from-scratch training replications or demonstrated skill mastery.

## Artifacts and reproduction

Frozen initial checkpoint: `gaifo-20261115-184227-1`, step 1,900,019,712.
Data and result JSONs: `/tmp/opencode/aerial-reward-audit/`.

- `phase-probe-{97,98}.json`, `phase-windows-{97,98}.pt`:
  `/tmp/opencode/aerial-phase-probe.py --seed SEED`
- `expert-match-heldout-pairs.pt`, `mitigation-{111,112}.json`:
  `/tmp/opencode/aerial-mitigation-sweep.py --seed SEED`
- `scalar-history-{111,112}.json`: same script with
  `--tag scalar-history --modes endpoint_flags endpoint_scalars`
- `residual-matched.json`: `/tmp/opencode/aerial-residual-check.py`
- `exploration-{111,112}.json`: `/tmp/opencode/aerial-exploration-sweep.py --seed SEED`
- `online-{121,122}.json`: `/tmp/opencode/aerial-online-sweep.py --seed SEED`.
  Its initial phase rows are superseded by the corrected runs below; the five
  ordinary arms remain valid.
- `online-phase-{121,122}.json`: same script with
  `--tag online-phase --modes ent03_phase` (independent expert phase draws).

Run with the repository `.venv/bin/python`, repository on `PYTHONPATH`, and
`OMP_NUM_THREADS=2`. Native experiments additionally require
`$HOME/.local/share/goddard/carl/stage` first on `PYTHONPATH`. The replay corpus
is read through the production dataset; the reset reservoir retains the
production four-second edge buffers. All experimental checkpoints are isolated
under the artifact directory.
