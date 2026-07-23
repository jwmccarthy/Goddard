# Goddard

Goddard trains recurrent PPO Rocket League policies in CARL with JARL. It supports 1v1 self play with either the Seer reward or sequence-level GAIfO.

## Requirements

Goddard requires Python 3.11 or newer, CUDA, and the CARL and JARL repositories at `../CARL` and `../JARL`.

Install the environment with:

```bash
uv sync
```

## Replay Dataset

Training requires a parsed replay dataset. Download SSL game average ranked duel replays with:

```bash
uv run python replay_dataset.py acquire --count 1024
```

Parse the files and build the dataset with:

```bash
uv run python replay_dataset.py parse --fps 10
```

Parsing keeps live gameplay states and removes states that occur less than five seconds before a goal.

Split the replay IDs and reparse the expert half directly from the raw replay files into frameskip-matched GAIfO observation sequences with:

```bash
uv run python imitation_dataset.py --expert-count 512 --frameskip 8 --history-length 64
```

Downloads and parser state are stored under `data/ballchasing-ssl-1v1`. `reset_dataset/CURRENT` names the active sampled starting-position generation. `expert_dataset/CURRENT` names the active contiguous observation-sequence generation. GAIfO only reads `expert_dataset`; it never forms sequences from `reset_dataset`.

The collector uses the `babytowniv-rl-dataset/1.0` user agent and limits requests to five per second. The download manifest stores file hashes and supports resumed runs.

## Training

Start training with:

```bash
uv run python train.py --total-timesteps 100000000
```

Training loads the reset dataset onto `cuda:0`. By default, 70 percent of completed simulations receive a conservative grounded replay state and the rest keep their normal kickoff state. Configure this with `--replay-reset-probability`. PPO uses the normalized zero sum Seer reward. TrueSkill evaluation uses normal kickoff states.

CARL observations are normalized by physical limits by default; use `--no-normalize` to retain raw observations. Actor and critic recurrent networks are independent. Learning rate, entropy coefficient, reward goal weight, and credit half-life follow linear schedules over actual learner transitions. The Seer-compatible defaults use `1e-5 -> 5e-6`, `0.01 -> 0.005`, `1.25 -> 1.45`, and `10s -> 20s`, respectively.

The default recurrent setup uses 512-step rollouts, 16-step sequences, 32 PPO epochs, and episodes up to 120 seconds. Every schedule is recorded under `Schedule/*` in TensorBoard. Passing `--gamma` replaces the credit half-life schedule with a constant discount.

The default setup runs 1,024 parallel 1v1 simulations. Self play uses the current policy in 80 percent of matches and a saved policy in 20 percent. TrueSkill evaluation runs every 32,000,000 learner transitions.

TrueSkill treats every checkpoint as an immutable player. Each evaluation matches the newest checkpoint against the initial anchor, its immediate predecessor, and stale older checkpoints as capacity permits. Draws and all match outcomes are persisted, then ratings are recomputed from a common prior.

TensorBoard data is written to `runs/<run_id>`. Policy snapshots, ratings, and the final model are written to `checkpoints/<run_id>`.

Run a small training check with:

```bash
uv run python train.py \
    --total-timesteps 256 \
    --num-simulations 16 \
    --rollout-steps 8 \
    --sequence-length 4 \
    --hidden-size 32 \
    --minibatch-size 256 \
    --self-play-current 1.0
```

Use `uv run python train.py --help` for all options.

## GAIfO Training

Train with the sequence-level imitation reward using:

```bash
uv run python gaifo.py --total-timesteps 100000000
```

`gaifo.py` requires an expert dataset whose history length is at least the configured training sequence length. Each expert history comes directly from contiguous raw replay samples at the policy control rate (`120 / frameskip` Hz), using replay IDs disjoint from the sampled starting-position dataset. Training randomly crops each longer expert history to match `--sequence-length`, so one 64-observation artifact supports 16-, 32-, and 64-observation discriminators. The recurrent discriminator classifies complete observation sequences rather than explicit transition pairs. The discriminator reward `softplus(-logit)` is emitted once at the final step of each valid sequence. Sequences that cross episode boundaries are excluded from discriminator updates and receive zero imitation reward.

GAIfO uses the full imitation reward throughout training. It begins with the default Seer reward components and linearly anneals all non-goal shaping to zero over total training timesteps; goal scoring remains active. Configure the initial shaping strength with `--shaping-coef`.

Run a small GAIfO training check with:

```bash
uv run python gaifo.py \
    --total-timesteps 512 \
    --num-simulations 16 \
    --rollout-steps 16 \
    --sequence-length 16 \
    --hidden-size 32 \
    --minibatch-size 512 \
    --discriminator-batch-size 16 \
    --self-play-current 1.0
```

Use `uv run python gaifo.py --help` for all GAIfO options.

## Checkpoint Viewer

Start the browser viewer with:

```bash
uv run python watch_checkpoints.py --open
```

The viewer runs at `http://127.0.0.1:8788`. It scans `checkpoints/` every five seconds. The menu selects a run directory and a checkpoint for each team. It also supports sampled actions and random replay starting states. Pass `--normalize` when viewing checkpoints trained with normalized observations; the default remains raw for older checkpoints.

Drag to orbit. Right drag to pan. Scroll to zoom. Use WASD, Space, and Ctrl to move the camera. Press `R` to reset the match.

The page loads Three.js from `unpkg.com`, so the browser needs internet access.

## Reward

`SeerReward` combines 16 Rocket League reward terms. It converts the result to zero sum team rewards and normalizes it with running statistics unless `--no-normalize-rewards` is passed. TensorBoard records per-component episode means and raw, zero-sum, and normalized aggregate means and RMS scales.
