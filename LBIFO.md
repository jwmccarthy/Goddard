# Latent Behavior Imitation from Observation (LBIfO)

`lbifo.py` implements the state-only 1v1 method in `/home/bento/rl.pdf`.
Demonstrations supply scenes, **not actions**. Online skill acquisition uses PPO
on a dense tracking reward: the EMA encoder compares each realized behavior
prefix with the requested expert-trajectory embedding, and rewards both its
alignment and the change in alignment. Environment task rewards train only the
separate high-level plan-value model. `LBIFO_PAPER_REVISION.md` specifies the
corresponding revision to the paper's reward-free skill-acquisition sections.

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
  --pretrain-updates 30000 \
  --checkpoint-dir checkpoints/lbifo
```

Start the online stage from its `pretrain_latest.pt` (the broad pretraining
replays need not still be available once pretraining is complete):

```bash
.venv/bin/python lbifo.py \
  --resume-checkpoint checkpoints/lbifo/<pretrain-run>/pretrain_latest.pt \
  --target-replay-dir parsed_replays/pro_1v1_fs4 \
  --timesteps 2000000000 \
  --checkpoint-dir /path/to/persistent/checkpoints/lbifo
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
For sustained online skill training, set `--timesteps` to the intended long-run
actor-step budget and put `--checkpoint-dir` on a persistent volume with enough
free space for complete checkpoints. An existing checkpoint from the previous
hindsight-supervised trainer can warm-start the policy, representation, prior,
and plan value; the new tracking critic and policy optimizer start fresh.

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
  updates they are re-segmented under the current EMA to train the termination
  hazard and realizability prior. PPO trains the policy and a separate skill
  critic on every active actor transition from the current joint CARL rollout.
  JARL handles the policy and critic's PPO loss, minibatches and optimizer
  steps. The augmented policy state comprises the CARL observation, requested
  expert latent and skill age. For each requested behavior, the reward is the
  cosine similarity of its embedding to the EMA embedding of the realized
  prefix, plus the difference from the preceding prefix. The EMA is frozen
  during each rollout; neither it nor the decoder receives a gradient from
  PPO. Anchored opponent actions are
  never policy training targets. `--ppo-epochs`, `--ppo-batch`,
  `--ppo-target-kl`, `--skill-critic-lr`, `--tracking-reward-weight` and
  `--tracking-progress-weight` control these updates; `tracking_cosine` and
  `tracking_progress` measure whether requested behaviors are actually being
  realized. Skill-boundary GAE prevents one requested behavior from receiving
  another's returns. PPO updates scale with active actor-steps, rather than
  sampling only a few trajectories from each large parallel rollout. Expert
  requests for new resets are re-encoded when the EMA and expert segments are
  refit, so long online runs do not compare current rollouts with stale target
  codes.
- At planning time the same joint latent prior samples candidates and inpaints
  plausible opponent futures. A separately trained, task-return-only value
  model ranks focal plans. During training the entire sampled plan is executed
  to obtain a full-horizon return, even across rollout batches; episode ends
  terminate the return early. `--replan-after` selects how many behaviors the
  saved viewer executes before requesting a new plan.

At `--frameskip 4`, the default `--min-duration 30` and `--max-duration 60`
allow one- to two-second expert behaviors. The 180-transition segmentation
excerpt spans six seconds, and the 180-step CARL rollout can hold a complete
three-skill plan even when each behavior lasts the maximum two seconds.
Segmentation samples nonoverlapping excerpts long enough for a full plan,
visiting distinct pro replay files before taking another excerpt from one
replay. If the corpus contains fewer usable excerpts than `--expert-sequences`,
the trainer reports the available count rather than duplicating the same
action. Online rollouts shorter than `--plan-horizon * --min-duration`
cannot supply complete multi-skill policy plans to the prior. Longer windows
increase training and segmentation cost substantially.

To reuse an older `pretrain_latest.pt` **or** `expert_segments.pt` with
short-duration settings and continue its representation training on longer
windows, run:

```bash
.venv/bin/python lbifo.py \
  --resume-checkpoint checkpoints/lbifo/<old-run>/expert_segments.pt \
  --pretrain-replay-dir parsed_replays/ranked_1v1_fs4 \
  --pretrain-only --pretrain-updates 30000 \
  --min-duration 30 --max-duration 60 --segment-steps 180 --rollout 180
.venv/bin/python lbifo.py \
  --resume-checkpoint checkpoints/lbifo/<new-pretrain>/pretrain_latest.pt \
  --target-replay-dir parsed_replays/pro_1v1_fs4 --segment-only
```

`--pretrain-updates` is the **total** number of updates, so it must exceed the
old checkpoint's `pretrain_step` to train on longer windows. This reuses the
encoder and optimizer but initializes a new duration prior and calibration.
When starting from `expert_segments.pt`, old inferred boundaries are discarded
and new, longer excerpts are segmented. Online checkpoints have already started
policy training and cannot change the duration-prior architecture. More PPO
timesteps cannot change an existing `expert_segments.pt`. Longer pretraining
may improve a weak representation, but same-window contrastive learning treats
other sampled windows as negatives even if they depict the same behavior; more
updates alone cannot guarantee cross-replay skill clusters.

Run `watch_checkpoints.py --checkpoint-dir checkpoints/lbifo` to watch saved
`lbifo_*.pt` hierarchies against each other. The viewer also supports BASIC
and GAIFO. Representation-only checkpoints are not playable.

After pro segmentation, `lbifo.py` saves `expert_segments.pt` immediately,
before the first CARL rollout. To stop there and inspect the demonstrations
before starting online training, run:

```bash
.venv/bin/python lbifo.py \
  --resume-checkpoint checkpoints/lbifo/<pretrain-run>/pretrain_latest.pt \
  --target-replay-dir parsed_replays/pro_1v1_fs4 --segment-only
.venv/bin/python watch_expert_skills.py \
  --checkpoint checkpoints/lbifo/<segmentation-run>/expert_segments.pt \
  --replay-dir parsed_replays/pro_1v1_fs4 --open
```

Resume online learning later from `expert_segments.pt` in the same way as from
`pretrain_latest.pt`. Existing online `lbifo_*.pt` checkpoints can also be
inspected; pretraining-only checkpoints contain no inferred expert skills.

The watcher groups the **actual pro replay clips sampled at segmentation**
by per-car latent: each displayed clip has cosine similarity at least the
selected cutoff to that group's representative latent. Select a skill latent,
then choose or play through all its matching saved clips. An expert clip may
appear under more than one latent when it matches both. The catalog reports
the saved clip-duration range and, when replay files are available, how many
distinct replays support each group. It shows one match per replay first;
several matches from a single replay do not establish a repeatable skill. A
source replay is counted only after its scenes match the saved excerpt.
The similarity slider (initial `--similarity`, default `0.9`) controls grouping
because the latent space is continuous, not a set of categorical labels. The
browser shows each trajectory's focal car, match score, segment boundaries,
3D playback, both cars' latent vectors and concentrations, and each evolving
prefix embedding. The inspector lists all 51 saved scene features for every
frame. If the original parsed replays are present (automatically from the
checkpoint's target replay path, or via `--replay-dir`), it also shows all 110
other parser fields: boost pads, relative ball/car and goal features, internal
state, touches, bumps and correction flags. Scene alignment is checked before
displaying raw rows. `--expert-sequences` caps how many nonoverlapping pro
excerpts are sampled for segmentation; the watcher does not scan every frame
of every replay. Viewing needs no GPU. The watcher defaults to port `8789`
(`--host` and `--port` are configurable).

For a quick test on synthetic replay files, run
`.venv/bin/python -m unittest discover -s tests -p 'test_lbifo*.py' -v`.
The CARL integration smoke is opt-in:
`GODDARD_GPU_SMOKE=1 .venv/bin/python -m unittest discover -s tests -p test_lbifo_training.py -v`.
