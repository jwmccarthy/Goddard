# Latent Behavior Imitation from Observation (LBIfO)

`lbifo.py` implements the state-only 1v1 method in `/home/bento/rl.pdf`.
Demonstrations supply scenes, **not actions**. The low-level policy learns by
maximum likelihood on its own CARL actions relabeled with the EMA scene encoder;
task rewards train only the separate plan-value model. There is no PPO, imitation
reward, or discriminator in this trainer.

## Replay data and training phases

Use parsed **1v1** replay folders containing 161-column `.npy` files. The three
data paths have separate roles:

| Flag | Used for | Typical source |
| --- | --- | --- |
| `--pretrain-replay-dir` | Random windows for representation pretraining | Broad/lower-ranked 1v1 games |
| `--target-replay-dir` (`--replay-dir`) | Pro-level segment inference, requested skills, target sequence prior and online expert updates | Pro 1v1 games |
| `--replay-reset-dir` | Optional **reset-only** physical states; never treated as expert demonstrations | A separately chosen 1v1 reset pool |

The low-level phase draws expert targets **only** from `--target-replay-dir`.
By default the simulator also resets from safe pro target segment starts.
If `--replay-reset-dir` is specified, `--external-reset-fraction` (default
`0.25`) additionally uses its safe states. Those starts are labeled as
reset-only, and their requests come from the prior or achieved policy skills;
they do not acquire an expert label just because they were used for a reset.
Policy-generated rollouts join representation learning on their own (weighted)
domain. Only pro demonstrations furnish expert targets. Replay reset filtering
excludes physics-unsafe impulses, near-goal frames, held-out frames and parser
corrections; simulator boost-pad cooldowns cannot be recovered from replay data.

Pretrain a representation on broader play before starting pro-only skills:

```bash
.venv/bin/python lbifo.py \
  --pretrain-replay-dir parsed_replays/ranked_1v1_fs4 \
  --pretrain-only \
  --pretrain-updates 10000 \
  --checkpoint-dir checkpoints/lbifo
```

Start the online stage from its `pretrain_latest.pt` (the broad pretraining
replays need not still be available once pretraining is complete):

```bash
.venv/bin/python lbifo.py \
  --resume-checkpoint checkpoints/lbifo/<pretrain-run>/pretrain_latest.pt \
  --target-replay-dir parsed_replays/pro_1v1_fs4 \
  --timesteps 10000000
```

To train both stages in one command, supply both replay directories without
`--pretrain-only`. Add `--replay-reset-dir PATH` if you want a separate pool
for simulator resets. `--external-reset-fraction 0` keeps all low-level starts
on pro target states; `--no-replay-reset-dir` removes an inherited external
reset pool on resume. Continue an online checkpoint with
`--resume-checkpoint <run>/lbifo_000000000000.pt --timesteps NEW_TOTAL`;
`--timesteps` is the minimum total number of **joint CARL actor-steps**, not an
additional-step count. The last CARL step runs all `2 * --n-sim` actors, so
the saved step count can exceed the requested total by fewer than that many
actor-steps; the requested total need not be divisible by the actor count.

## Method and controls

- An entity-anchored relational GRU infers per-car vMF latents. A masked scene
  decoder sees only visible context and a domain label; anchor-only, sparse,
  entity and causal masks plus prefix prediction and same-window contrastive
  pairs train the representation. Labels use its EMA encoder. Expert posterior
  concentration and the anchor-only error gap against random codes are logged;
  a weak diagnostic raises `--mask-ratio` before other representation settings.
- Random (never segmentation-selected) expert windows calibrate anchor and
  causal next-frame prediction errors by offset. A semi-Markov Viterbi search
  selects segments using those excess errors and the sequence prior's latent
  and duration code length. Initial segmentation uses a flat latent prior;
  refits alternate on the pro dataset. `--min-duration`, `--max-duration`,
  `--segment-steps` and `--prior-rounds` control this stage.
- The conditional prior denoises **joint** sequences on a product of spheres.
  Its shared duration/density heads also score boundaries and provide the
  time-aware termination baseline; realized expert/policy durations supervise
  its hazard. `--plan-horizon` can be `1` for single-skill planning.
- Joint rollouts, single-controlled-car reenactment against a demonstrated
  opponent (`--single-reenactments`), and full-car kinematic replay furnish
  experiences. Full-car kinematic and altered-timestep action replays provide
  cross-domain positive pairs at matching physical times. Disable the latter
  with `--no-dynamics-pairs` when measuring their contribution.
- Stored trajectories have **no frozen hindsight labels**. During online
  updates they are re-segmented under the current EMA and policy actions are
  fit to achieved behavior latents, weighted by relative posterior
  concentration. Anchored opponent actions are never policy training targets.
- At planning time the same joint latent prior samples candidates and inpaints
  plausible opponent futures. A separately trained, task-return-only value
  model ranks focal plans. During training the entire sampled plan is executed
  to obtain a full-horizon return, even across rollout batches; episode ends
  terminate the return early. `--replan-after` selects how many behaviors the
  saved viewer executes before requesting a new plan.

Run `watch_checkpoints.py --checkpoint-dir checkpoints/lbifo` to watch saved
`lbifo_*.pt` hierarchies against each other. The viewer also supports BASIC
and GAIFO. Representation-only checkpoints are not playable.

For a quick test on synthetic replay files, run
`.venv/bin/python -m unittest discover -s tests -p 'test_lbifo*.py' -v`.
The CARL integration smoke is opt-in:
`GODDARD_GPU_SMOKE=1 .venv/bin/python -m unittest discover -s tests -p test_lbifo_training.py -v`.
