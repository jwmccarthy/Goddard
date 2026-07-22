# Goddard

Goddard trains a recurrent PPO Rocket League policy in CARL with JARL. Training uses 1v1 self play, the Seer reward, and expert replay starting states.

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

Split the replays into reset states and frameskip matched GAIfO observation pairs with:

```bash
uv run python imitation_dataset.py --expert-count 512 --frameskip 8
```

Downloads and parser state are stored under `data/ballchasing-ssl-1v1`. `reset_dataset/CURRENT` names the active reset generation. `expert_dataset/CURRENT` names the active observation pair generation.

The collector uses the `babytowniv-rl-dataset/1.0` user agent and limits requests to five per second. The download manifest stores file hashes and supports resumed runs.

## Training

Start training with:

```bash
uv run python train.py --total-timesteps 100000000
```

Training loads the reset dataset onto `cuda:0` and samples a state at each episode reset. PPO uses the normalized zero sum Seer reward. TrueSkill evaluation uses normal kickoff states.

The default setup runs 1,024 parallel 1v1 simulations. Self play uses the current policy in 80 percent of matches and a saved policy in 20 percent. TrueSkill evaluation runs every 16,000,000 learner transitions.

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

## Checkpoint Viewer

Start the browser viewer with:

```bash
uv run python watch_checkpoints.py --open
```

The viewer runs at `http://127.0.0.1:8788`. It scans `checkpoints/` every five seconds. The menu selects a run directory and a checkpoint for each team. It also supports sampled actions and random replay starting states.

Drag to orbit. Right drag to pan. Scroll to zoom. Use WASD, Space, and Ctrl to move the camera. Press `R` to reset the match.

The page loads Three.js from `unpkg.com`, so the browser needs internet access.

## Reward

`SeerReward` combines 16 Rocket League reward terms. It converts the result to zero sum team rewards and normalizes it with running statistics.
