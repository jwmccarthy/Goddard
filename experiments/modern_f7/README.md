# Modern BASIC trial: tested `f7cab81` reward and 30-second episodes

Started September 25, 2026 with local Goddard `b4ced8a3` plus the
uncommitted, parity-tested BASIC/Seer reward changes; CARL `6362f59f`,
JARL `8759825f`. The modern reward default matches the **whole** committed
`f7cab81` reward on 16 diagnostic events (including normalization). Later
experimental incentives and kickoff windows remain opt-in. BASIC defaults
to 30-second no-touch timeout and 36,000 maximum ticks; the separate DIFO
configuration is unchanged. No scripted controller is used in self-play.

On RunPod the source checkout is
`/workspace/goddard-modern-f7-20260925/{Goddard,CARL,JARL}`. The CARL
native module was compiled for the RTX 4090 (CUDA architecture 89) into
that checkout. Current JARL and Goddard sources are ahead of the installed
July 31 stack. The pinned `/Goddard` Python environment supplies Torch
2.11.0+cu128 and common dependencies; `PYTHONPATH` starts with the **modern**
Goddard, CARL, and JARL source directories, followed by the separately
staged small build/runtime dependencies. Imports were checked to resolve to
the current source and newly built CARL extension. The remote focused suite
passed 48 tests, and a 32-simulation, two-update PPO smoke run wrote a real
actor and optimizer checkpoint.

The untouched `/workspace/parsed_replays/` backup has 23,411 `.npy` files
and 23,402 safety sidecars (~17 GB). Duplicating it on `/workspace` exceeded
that volume's quota; scanning individual files over the volume took far too
long. A verified RAM-backed copy lives at
`/dev/shm/goddard-modern-f7-20260925/parsed_replays/`. Relative paths and
sampled file hashes match the persistent backup; all 23,411 replay files
load as the modern 1v1 reset dataset in roughly 10 seconds. That copy is
**ephemeral**: after a Pod restart, restore it from the persistent backup.
The source checkout's `parsed_replays` link resolves to the original backup.

The single modern trainer (PID 61128 at launch) uses seed 0, 8,192
simulations, 32-step rollouts, frameskip 8, 70% replay resets, and the BASIC
defaults. Its full 10B-step target keeps the learning-rate, entropy, and
discount schedules comparable while 500M is assessed. Live outputs are
under `/dev/shm/goddard-modern-f7-20260925/live/` and are mirrored every
two minutes to `/workspace/goddard-modern-f7-20260925/live/` by
`mirror_outputs.py` (PID 61225 at launch). The run is called
`modern-f7-default-8192x32-20260925`. It is also linked into the existing
TensorBoard server on port 6006, with `Gameplay/*`, `Seer/*`, and
`Heldout/*` charts. `Gameplay/timeout_fraction` only counts episodes that
reached **inactivity** timeout; use held-out `non_goal_fraction_completed`
for all non-goal endings, including match-limit truncations.

`watch.py` (PID 62040 at launch) sequentially evaluates the latest actor
at roughly 100M-step intervals on the full-length fixed-seed kickoff and
512-game replay-mixed first-episode tests. It writes JSONL records directly
to `/workspace/goddard-modern-f7-20260925/evaluations/` and publishes the
same results to TensorBoard. `evaluate.py` also supports independent
one-off evaluations. The replay-mixed evaluation uses the modern RAM-backed
reset corpus, not the July 31 corpus. Training never waits for evaluators.

An independent, pinned-stack entropy measurement of the much older August 1
archived actor and both `f7cab81` snapshots lives at
`/workspace/july31-recreation-20260924/archived_action_entropy_9212.json`.
The historic train-time entropy cannot be recovered from its actor export;
the JSON records **masked policy-action entropy on the same held-out states**.
See `experiments/july31_stack/HISTORICAL_TRIALS_20260925.md` for the full
measurement and the archive's 20B+ training-age caveat.

### Full-length same-seed first-episode results

| Modern steps | Frozen kickoff wins / 256 | Replay-mixed goals / 512 | Completed non-goal fraction |
| ---: | ---: | ---: | ---: |
| 107.2M | 53 | 308 | 197/505 (39.0%) |
| 199.0M | 131 | 397 | 106/503 (21.1%) |
| 309.6M | 225 | 417 | 87/504 (17.3%) |
| 421.6M | 246 | 443 | 63/506 (12.5%) |
| **500.3M** | **241** | **446** | **63/509 (12.4%)** |
| 524.3M | 243 | 455 | 57/512 (11.1%) |

The 500.3M policy touched the ball in all 256 kickoff games on seed 9210;
it also won 244/256 on independent seed 9211 (255 games with an observed
own touch). Only three mixed games were censored at the 5,000-action horizon
at 500.3M, compared with seven, nine, eight, and six at the earlier steps.
All 512 mixed games completed at the next routine 524.3M checkpoint, with
455 goals and 57 non-goal endings. The pinned July 31 `f7cab81` run at a
nearby 522.8M checkpoint had 453 goals and 58 non-goal endings among 511
completed games. Those runs use different replay-reset corpora and versions
of CARL/JARL, so this is a useful early-signal comparison, not a controlled
one-factor ablation.
This **passes the 500M sustained-learning gate**; the younger modern policy
is not expected to equal the archive's 20B+ competitive skill yet. Both the
500.3M actor snapshot and optimizer checkpoint are being mirrored to the
persistent volume. The actor at 500,279,721 steps has matching SHA-256
`66e1225de06780c1ff8ad28116885bde174efaf348bae75f65f0093a8cf3eab1`
in the live and mirrored checkpoint directories. Training continues toward
longer-horizon mechanics and competitive evaluation.

`/workspace/goddard-modern-f7-20260925/evaluations/archived_vs_modern_action_entropy_9212.json`
compares the archived, historical 522.8M `f7cab81`, and modern 500.3M
actors on **identical fixed archived-policy trajectories** (512 actor states
initially, 256 games, seed 9212). In nats across seven legal-action factors,
their mean entropy at steps 0/15/45/90/180 is respectively:

| Actor | Step 0 | Step 15 | Step 45 | Step 90 | Step 180 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Archived August 1 (20B+ steps) | 1.722 | 2.960 | 2.977 | 2.480 | 2.580 |
| Historical `f7cab81` 522.8M | 3.013 | 3.332 | 3.515 | 3.239 | 3.174 |
| Modern default 500.3M | 3.378 | 3.302 | 3.525 | 3.241 | 3.246 |

The current 500M policy has **not** collapsed to the historical run's
3B-step ~0.2-nat entropy on these states. These fixed observations beyond
the opening are generated by the archived actor, so the younger actors'
figures are off-policy; they should not be equated with training-time PPO
entropy. The continued live run can show whether any later collapse occurs.
