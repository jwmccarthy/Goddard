# Goddard

Goddard trains a recurrent PPO Rocket League policy in CARL with JARL. Training uses 1v1 self play, sequence-level GAIfO, and expert replay starting states.

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

Split the replays into reset states and frameskip-matched GAIfO observation sequences with:

```bash
uv run python imitation_dataset.py --expert-count 512 --frameskip 8 --sequence-length 16
```

Downloads and parser state are stored under `data/ballchasing-ssl-1v1`. `reset_dataset/CURRENT` names the active reset generation. `expert_dataset/CURRENT` names the active observation sequence generation.

The collector uses the `babytowniv-rl-dataset/1.0` user agent and limits requests to five per second. The download manifest stores file hashes and supports resumed runs.

## Training

Start training with:

```bash
uv run python train.py --total-timesteps 100000000
```

Training loads the reset dataset onto `cuda:0`. By default, 70 percent of completed simulations receive a conservative grounded replay state and the rest keep their normal kickoff state. Configure this with `--replay-reset-probability`. PPO uses rewards from a recurrent GAIfO discriminator trained to classify complete 16-transition policy and expert sequences. TrueSkill evaluation uses normal kickoff states.

CARL observations are normalized by physical limits by default; use `--no-normalize` to retain raw observations. Actor and critic recurrent networks are independent. Learning rate, entropy coefficient, and credit half-life follow linear schedules over actual learner transitions.

The default recurrent setup uses 512-step rollouts, 16-step sequences, 32 PPO epochs, and episodes up to 120 seconds. Every schedule is recorded under `Schedule/*` in TensorBoard. Passing `--gamma` replaces the credit half-life schedule with a constant discount.

The default setup runs 1,024 parallel 1v1 simulations. Self play uses the current policy in 80 percent of matches and a saved policy in 20 percent. TrueSkill evaluation runs every 32,000,000 learner transitions.

TrueSkill treats every checkpoint as an immutable player. Each evaluation matches the newest checkpoint against the initial anchor, its immediate predecessor, and stale older checkpoints as capacity permits. Draws and all match outcomes are persisted, then ratings are recomputed from a common prior.

TensorBoard data is written to `runs/<run_id>`. Policy snapshots, ratings, and the final model are written to `checkpoints/<run_id>`.

Run a small training check with:

```bash
uv run python train.py \
    --total-timesteps 512 \
    --num-simulations 16 \
    --rollout-steps 16 \
    --sequence-length 16 \
    --hidden-size 32 \
    --minibatch-size 512 \
    --discriminator-batch-size 16 \
    --self-play-current 1.0
```

Use `uv run python train.py --help` for all options.

## Checkpoint Viewer

Start the browser viewer with:

```bash
uv run python watch_checkpoints.py --open
```

The viewer runs at `http://127.0.0.1:8788`. It scans `checkpoints/` every five seconds. The menu selects a run directory and a checkpoint for each team. It also supports sampled actions and random replay starting states. Pass `--normalize` when viewing checkpoints trained with normalized observations; the default remains raw for older checkpoints.

Drag to orbit. Right drag to pan. Scroll to zoom. Use WASD, Space, and Ctrl to move the camera. Press `R` to reset the match.

The page loads Three.js from `unpkg.com`, so the browser needs internet access.

## Reward

The discriminator projects each observation to ball, controlled-car, and opponent features, encodes every transition in a sequence with a GRU, and classifies the final sequence representation. Its negative logit is emitted once at the final step of each valid sequence. Sequences that cross episode boundaries are excluded from discriminator updates and receive zero imitation reward.
