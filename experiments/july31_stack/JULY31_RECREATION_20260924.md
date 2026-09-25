# July 31 pipeline recreation (September 24, 2026)

The live RunPod installation at `/Goddard` uses the unmodified Goddard commit
`a817186d3c1441706d287faea4eea42d7e06b720`, CARL commit
`965cba1edc3cde0717ceb632b9baf8e994e7a1d9`, and JARL commit
`0c6ed3a689e06e97e64678d7834007ece1769ab8`. The previous installation
and its artifacts are preserved at `/Goddard-pre-july31-20260924`.

The original Ballchasing manifest lists 1,024 replays. All raw replay hashes
match the manifest (two replay files unavailable through the live API were
restored from the original archive). The unmodified July 31 `replay_dataset.py
parse --fps 10` produced 2,419,196 frames from 1,023 nonempty replays.
The July 31 dataset builder sampled 487,499 reset states. Its `frames.npy`
SHA-256 is `00ceb3d7e6ff25e5814d135512e220d9caaec82a3384efdce9900a0004531639`,
**identical to the archived reset corpus**. The active dataset is under
`/Goddard/data/ballchasing-ssl-1v1/reset_dataset`; raw replays and parsed
shards are on the persistent `/workspace/july31-recreation-20260924/` volume.

## Faithful default training (stopped September 24, 22:27 UTC)

Started from `/Goddard` with:

```sh
/Goddard/.venv/bin/python -u -B ppo.py --run-name july31-full-verified-20260924
```

Apart from the run name, all July 31 PPO defaults apply, including the
10-billion-transition training target, 1,024 simulations, 70% replay resets,
and seed zero. The initial actor snapshot SHA-256 is
`493f566440939bffc50e6d24fed641d923dc50a1216b9f204a471cfded2da891`.
The active `/Goddard/{runs,checkpoints}` directories point to
`/dev/shm/july31-recreation-20260924/`; `mirror_outputs.py` copies them to
`/workspace/july31-recreation-20260924/live/` every two minutes. The stopped
earlier local run remains in `/Goddard/{runs,checkpoints}-early-local-20260924`.

`watch_frozen_kickoffs.py` evaluated a checkpoint about every 100M steps against
the same 256 legal-random-opponent kickoff games (seed 9210, 900-action limit).
Its persistent output is `/workspace/july31-recreation-20260924/live/frozen_kickoff_evals.jsonl`.
The mirror stops training before the RAM-backed or overlay filesystem fills.
This run was stopped at approximately 421M steps to switch to the 8,192×32
configuration below. Its last `training_latest.pt` matches the persistent
backup byte-for-byte. The existing TensorBoard server has a symlink to this
run's events, so its training history remains visible.

At 17M, 80M, 111M, and 218M learner steps the faithful run touched the ball in
17/256, 17/256, 18/256, and 23/256 frozen kickoff games, respectively. The
original archived actor touched in 256/256 matched games. Matching the replay
corpus and dated code therefore did not recover that actor's early training
skill. The user later clarified that the polished archived export had
**upwards of 20B training steps**; comparing it with much earlier
checkpoints diagnoses the current skill gap, not whether the recreation
would eventually learn comparable play with more training.

## Reconfigured 8,192×32 training (stopped at 3.005B steps)

The historical trainer was started fresh from `/Goddard` with only the
following flags changed from its defaults:

```sh
/Goddard/.venv/bin/python -u -B ppo.py \
    --run-name july31-full-8192x32-20260924 \
    --num-simulations 8192 --rollout-steps 32
```

It retains the 10-billion-step target, seed zero, 70% replay resets, frameskip
8, sequence length 16, 32 PPO epochs, 65,536 minibatch size, BF16, and the
original July 31 reward. Its checkpoints and
events were mirrored to `/workspace/july31-recreation-20260924/live/` every
two minutes (`mirror-8192x32.log`), and its frozen kickoff evaluations are in
`live/frozen_kickoff_evals_8192x32.jsonl` about every 30M steps. Both this
run and the stopped default run appear in the existing TensorBoard server's
run list through symlinks in `/Goddard-pre-july31-20260924/runs/`. This
configuration did not match the reference trajectory: at 3.005B its fixed
kickoff evaluator saw 37/256 games with a touch, four goals, and 249/256
non-goal terminations. The 1.864B snapshot reached 82/256 touch games but
later lost most of that skill. See `HISTORICAL_TRIALS_20260925.md` for the
subsequent sequential training comparisons.

The original trainer does **not** log a timeout-fraction scalar. A separate
first-episode evaluation of 512 stochastic self-play games (both sides from
the checkpoint, 70% replay/30% kickoff starts, seed 9210, 1,500-step limit)
found the following non-goal termination rates; all games completed:

| Checkpoint | Combined | Replay starts | Kickoff starts |
| --- | ---: | ---: | ---: |
| Default run, ~421M | 437/512 (85.4%) | 281/352 (79.8%) | 156/160 (97.5%) |
| 8,192×32 run, ~39M | 446/512 (87.1%) | 287/352 (81.5%) | 159/160 (99.4%) |

These are held-out, same-policy self-play samples, not a live training-rollout
metric; matches against historical opponents can differ.

## Single-weight diagnostic

`july31_kickoff_ablation.py` runs the same July 31 PPO defaults, seed, and
replay corpus, changing **only** `SeerRewardWeights.kickoff` from `0.0` to
`0.1` at reward construction. Its initial actor snapshot is byte-for-byte
identical to the faithful run's; the ablation was stopped after the 33.9M
snapshot to return the GPU to the faithful run.

Across 256 matched kickoff games at each of seeds 9210 and 9211 (900-action
limit), the faithful 32.1M policy touched in 32/512 games and scored 0 goals;
the 33.9M kickoff-weight ablation touched in 58/512 and scored 8 goals. This
supports insufficient early kickoff incentives as **one contributor**, not
proof of the archived actor's training provenance or a complete fix.
`july31_preoccupancy_ablation.py` instead restores exactly the 14 weight
defaults removed by commit `e53f73e`, holding the pinned July 31 PPO code,
later reward mechanics, seed, and replays fixed. Its seed-zero actor snapshot
is also identical. At 33.9M steps it touched in 76/512 matched games and
scored 10 goals. Both diagnostic runs were stopped after this comparison;
their snapshots and training checkpoints are preserved by the output mirror.
The old July 31 reward also sets direct `ball_touch` weight to zero. Its
logged angular-velocity component was about 3 per episode at 65M, versus
about 1.3 from goals and 0.03 from touch acceleration at that reading;
these diagnostics are not a causal ablation.

The archived actor export is dated August 1, 08:52 EDT. There is no surviving
training manifest, critic, optimizer state, or step history for that actor,
so its source checkpoint, initialization, prior training, and exact flags
cannot presently be verified. Commit `e53f73e` disabled the old `0.1`
kickoff incentive on July 31 at 15:00 EDT; the later export date alone does
not tell us which reward revision trained the archived actor. The replay
archive also retains multiple older reset-dataset generations, so matching
its latest active generation does not prove which generation trained it.
