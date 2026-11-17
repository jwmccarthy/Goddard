## Run PPO

```bash
uv run --no-sync python ppo.py \
  --n-sim 8192 --rollout 32 --frameskip 4 \
  --timesteps 100_000_000_000 \
  --learning-rate 3e-4 --learning-rate-end 1e-5 --gamma 0.99 \
  --replay-dir replays/rlcs \
  --replay-reset-fraction 0.5 --replay-reset-size 500_000 \
  --opponent-pool-size 8 --opponent-snapshot-interval 10_000_000 \
  --historical-opponent-fraction 0.5
```

## Run GAIfO

```bash
PYTHONPATH=/tmp/opencode/reset-consistency/stage uv run --no-sync python gaifo.py \
  --frameskip 8 --replay-frameskip 4 --replay-dir replays/rlcs \
  --discriminator-motion-noise-scale 0.001 --target-discriminator
```

Optional `--experimental-jump-phase` adds real-tick discriminator phase sampling
and soft consistency regularization without changing policy action holds. It is
off by default, costs extra memory/runtime, and is not yet a verified gameplay
fix. See [GAIfO details](training/GAIFO.md#experimental-jump-phase-robustness).

Optional `--mlp-critic --critic-window-length 4 --critic-decision-stride 1`
gives only the critic a concatenated observation history. Add `--critic-relative`
for historical scene-physics deltas relative to the current observation. The
actor is unchanged; better calibration is not yet established. See
[critic history](training/OBSERVATIONS.md#optional-critic-only-observation-history).

## Benchmark native RocketSim GAIfO

`gaifo_cpu.py` runs RocketSim collection and GAIfO/PPO learning in C++, using
LibTorch with CPU or CUDA learning. See [build and comparison instructions](training/GAIFO_CPU.md).

## Run DIFO

Replace `gaifo.py` with `difo.py` to use a conditional diffusion discriminator
with the same PPO, scene-window, sampling and replay-reset settings. DIFO
defaults to log expert-probability reward and rejects `--amp` least squares;
`--bf16` is supported. Runs/checkpoints remain separate from GAIfO.
See [DIFO parameters and implementation details](training/DIFO.md).

## Run CALM

`calm.py` learns a reference-conditioned policy from overlapping **two-second
gameplay windows**, without skill labels. Its motion encoder learns through PPO;
a conditional discriminator rewards behavior resembling the selected reference.
Both teams use current-policy self-play. See [CALM setup and options](training/CALM.md).

```bash
PYTHONPATH="$HOME/.local/share/goddard/carl/stage" \
  uv run --no-sync python calm.py --replay-dir replays/rlcs --bf16 --compile
```

## Resume training

Resuming with `--resume runs/<run>/latest.pt` continues TensorBoard logging and
checkpoint saves in that checkpoint's directory. Points beyond the restored
step are hidden in TensorBoard (event files are not deleted). An explicit
`--run-name` instead selects a separate run under `--output-dir`.

## Run the viewer

```bash
uv run --no-sync python -m vis runs/<run>/latest.pt
```

Watch recorded human aerials instead of a bot with
`uv run --no-sync python -m vis --expert-demos --port 8001`, or pass the air-dribble
audit's `results.json` to `--expert-demos` for its exact candidate playlist.
Playback runs on CPU with pause, seek, slow motion and frame stepping; see
[expert demonstration controls](vis/README.md#expert-demonstrations-cpu-only).

## Preview a training pack

```bash
PYTHONPATH=../CARL/python uv run --no-sync python -m vis \
  --training-pack vis/example_pack.py
```
