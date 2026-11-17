# Current FS8 run: expert state-occupancy audit

2026-11-16 · run `gaifo-20261115-225140-1` · source process 470383.

## Available checkpoints

The current trainer saves atomically to `latest.pt` every 100 million agent
steps, replacing the preceding checkpoint. This run has no historical opponent
pool and no archived checkpoint files were found. Consequently **a retrospective
full-state-distribution convergence curve cannot be reconstructed**.

For this audit, the available checkpoint was frozen at **25,400,180,736 agent
steps** (203,201,445,888 agent-physics ticks). The run uses FS8, factorized
ego/global temporal discriminators, forward-KL main reward, ratio cap 10,
goal reward 50, and .75 replay-reset probability. No trainer settings were
changed by the audit. This snapshot used the original uncentered positive
forward-KL reward; the later fixed-baseline-minus-one change is not represented
in these measurements.

Charts: [`/tmp/opencode/current-run-occupancy/report.html`](file:///tmp/opencode/current-run-occupancy/report.html).
The report includes six state histograms, horizontal position heatmaps,
age-filtered state-region occupancy and historical gameplay/reward plots.

## Current distribution versus expert

Two frozen-policy evaluation seeds, each 256 simulations × 120 seconds at
native FS8, produce **1,840,658 non-goal-terminal agent states**. Evaluation uses
stochastic actions and current-policy self-play, matching the run's opponent
configuration. The fixed replay-reset bank contains 65,536 bounded uniform
reset states; the live trainer uses its larger three-million-state bank.

Expert reference: 30,000 endpoints from 160 matches. The fixed 256-cluster
partition was learned from 100,000 endpoints from 1,418 other expert matches.
These are held-out matches **for the clustering/reference construction**, not
held out from the production policy's discriminator training.

| Statistic | Expert | Policy: all ages | Policy: ≥2s since reset | Policy: ≥10s since reset |
| --- | ---: | ---: | ---: | ---: |
| Wheel-contact states | 54.46% | 63.14% | 63.54% | 63.91% |
| Airborne above 182.5 uu | 15.08% | 12.42% | 12.39% | 12.44% |
| Airborne above 500 uu | 7.79% | 5.42% | 5.34% | 5.28% |
| Wheel contact above 100 uu | 6.71% | 7.22% | 7.27% | 7.46% |
| Near arena walls | 14.84% | 16.65% | 16.85% | 17.28% |
| Boost ≤1 point | 15.70% | 19.58% | 19.94% | 19.71% |
| Mean car height, uu | 169.2 | 139.3 | 138.2 | 138.3 |
| Mean ball height, uu | 522.2 | 461.9 | 462.3 | 457.2 |
| Mean car speed, uu/s | 1496.2 | 1447.3 | 1443.6 | 1452.8 |
| Mean ball speed, uu/s | 1356.7 | 1362.6 | 1378.8 | 1383.1 |
| Mean boost, points | 40.47 | 40.15 | 40.19 | 40.65 |
| Mean ego–ball distance, uu | 1323.6 | 1317.7 | 1299.0 | 1299.2 |

Wheel contact includes all arena surfaces; elevated wheel contact does not by
itself establish a walldash. Near-wall means `abs(x)>3800` or `abs(y)>4800`.

**Largest clear marginal gaps:** high aerial occupancy is approximately 30%
below expert, general aerial occupancy is approximately 18% below expert, and
wheel-contact occupancy is 8.7 percentage points higher. The policy spends more
time nearly out of boost even though its average boost is close to expert.
Mean distance and speed agreement does not imply matching the full distribution.

### Global coarse coverage

| Evaluation subset | Smoothed forward KL ↓ | Seed range | Jensen–Shannon ↓ | Total variation ↓ | Expert mass in ≥2× undercovered clusters ↓ |
| --- | ---: | --- | ---: | ---: | ---: |
| All states | .0806 | .0738–.0874 | .0184 | .1414 | 9.90% |
| ≥2s since reset | .1407 | .1352–.1461 | .0260 | .1518 | 10.69% |
| ≥10s since reset | .1699 | .1623–.1775 | .0313 | .1656 | 12.60% |

Metrics are means across the two seeds; ranges are not confidence intervals.
For context, held-out expert versus training-expert cluster occupancy gives
forward KL **.0075**, so the measured policy difference is larger than that
reference split's mismatch.

KL uses `q_smoothed=.99*q_policy+.01*p_expert`. These are **coarse histogram
distances**, not full continuous-state KL. Cluster standardization includes all
51 global scene features, including flags/boost. Differences may reflect both
behavior and replay/native observation-domain differences.

The reset-inclusive distribution is closer to expert. Removing early states
increases the measured gap, although aerial fractions remain almost unchanged.
Thus current aerial occupancy is not simply an immediate-reset artifact.
Age conditioning also selects surviving episodes, while the expert reference
is unconditioned; its increased KL cannot be attributed solely to policy drift.

## What the training history actually establishes

The run logs gameplay and discriminator summaries, not state-occupancy
histograms. The report streams these logs into one-billion-step bins through
step **25,570,050,048**. The last bin is partial.

| Agent steps | Aerial-touch intervals / 1,000 physics ticks | All-touch intervals / 1,000 ticks | Ratio cap fraction |
| --- | ---: | ---: | ---: |
| 0–1B | .0751 | 1.807 | 1.034% |
| 1–2B | .2218 | 2.766 | .649% |
| 5–6B | .4138 | 3.267 | .554% |
| 10–11B | .4647 | 3.378 | .508% |
| 15–16B | .4981 | 3.414 | .485% |
| 20–21B | .5159 | 3.412 | .465% |
| 25–25.57B | .5222 | 3.483 | .441% |

These count contact-containing decision intervals, not individual contact
onsets; they are not directly interchangeable with older audit contact-onset
rates. Aerial-touch frequency clearly improves, with slower gains later in
training. Mean discriminator reward declines while the discriminator itself
changes, so it is not a calibrated convergence curve.

**Conclusion:** strong behavioral progress is visible in the logs, and current
coarse occupancy is reasonably close on several broad statistics. Remaining
gaps concentrate partly in high aerial play, wheel-contact prevalence and boost
usage. Monotonic convergence toward expert occupancy is **not established**;
that requires archived policy checkpoints or fixed-reference occupancy metrics
recorded throughout training.

## Reproduction and artifacts

Under `/tmp/opencode/current-run-occupancy/`:

- `gaifo-20261115-225140-1-25400180736.pt`: frozen actor/config/step.
- `evaluate.py`: native frozen-policy histogram evaluation; no learned online
  discriminator is used to score similarity.
- `eval-25400180736-{301,302}.json`: per-seed raw counts and metrics.
- `log_history.py`, `history.json`: streamed scalar history and 1B-step summaries.
- `report.py`, `report.html`, `summary.json`: charts and pooled distributions.

Example evaluation command:

```bash
env PYTHONPATH=/home/jack/.local/share/goddard/carl/stage:/home/jack/Goddard \
  OMP_NUM_THREADS=2 .venv/bin/python /tmp/opencode/current-run-occupancy/evaluate.py \
  --checkpoint /tmp/opencode/current-run-occupancy/gaifo-20261115-225140-1-25400180736.pt \
  --seed 301
```

The evaluator records the loaded CARL binary path and SHA256, excludes
goal-terminated endpoints, retains pre-reset timeout successors, checks count
normalization, and uses independent seeds with the same frozen reference.
