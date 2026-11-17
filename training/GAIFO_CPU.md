# Native RocketSim GAIfO benchmark

`gaifo_cpu.py` prepares models/data and launches a **C++ collector and C++ LibTorch
learner**. RocketSim runs on CPU; `--device cuda` (default) runs inference,
discriminator updates, imitation rewards, GAE and PPO on the GPU. `--device cpu`
runs everything on CPU. Python is absent from the training loop.

The starting point is [RLGymPPO_CPP](https://github.com/ZealanL/RLGymPPO_CPP), pinned
to `ee4cc56fc8e43758cc898fdba92a9173a94e52f2`. We build its bundled RocketSim and
use its bundled nlohmann JSON. The collector/learner in `training/native_gaifo/`
is a Goddard-specific implementation inspired by its CPU simulation + LibTorch
architecture; this is **not a benchmark of the unmodified upstream PPO trainer**.
Collection uses an OpenMP arena batch with one batched policy inference per
decision, rather than upstream's independently collecting thread agents.

## Build and run

Requires Linux, a C++20 compiler, Make, OpenMP, Git, CMake and the project's
PyTorch environment. The installed PyTorch wheel supplies LibTorch, including
CUDA libraries when available; a separate CUDA toolkit/LibTorch download is
unnecessary. Collision meshes are discovered from `rlgym-rocket-league`, or
supplied with `--collision-meshes /path/to/collision_meshes` (containing `soccar/`).

```bash
uv pip install cmake
.venv/bin/python gaifo_cpu.py --build-only
.venv/bin/python gaifo_cpu.py --n-sim 256 --threads 16 --timesteps 1000000
```

The first build downloads the pinned upstream source under
`runs/gaifo-cpu-build/upstream`. Override with `--upstream-dir` and `--build-dir`.
An existing checkout must have the pinned HEAD. `--build-jobs` defaults to 4.
Models are traced on the requested learner device; keep that device when loading
the exported graphs. The native launcher doesn't require the installed CARL
extension to support the newer controller observations.

## Workload and supported options

Shared CLI flags use the definitions in `training/arguments.py`. Native defaults
override only simulation count (256), transition budget (one million), and
generated-history capacity (zero). In particular, frameskip is 8, rollout is 32,
the discriminator has two frames at stride 3, and each update uses four
discriminator epochs and four PPO epochs. All cars are learners.

Supported: 1v1/2v2/3v3, seven CARL action heads and masks, `ego-controller-v1`
observations, existing MLP actor/critic, transformer/GRU/MLP discriminators,
factorization, ego-only/relative inputs, scalar flags and boost controls,
motion noise, mirroring, target discriminator, generated history, softplus/log/
forward-KL/AMP rewards, explicit signed goal reward, value normalization/clipping,
entropy coefficients, gamma/PPO learning-rate annealing and KL early stopping.

Unsupported flags raise an error: resume, BF16, `torch.compile`, CUDA graphs,
`--compile-autotune`, joint actions, history critic, hierarchical/jump-phase discrimination, maneuver
sampling, gameplay shaping, historical opponents and automatic
goal-weight calibration. `--amp` means the least-squares imitation objective,
as in `gaifo.py`, not mixed precision.

The existing read-only expert and bounded-uniform replay-reset loaders are used,
including the four-second period-boundary exclusions. Source arrays are never
rewritten. Native reset sampling uses independent per-arena RNGs. Physics,
kickoff distributions, collision/contact state and demo respawns differ between
engines; matching layouts does not make trajectories identical. Replay rows
contain one grounded flag, so native reset initializes all four wheel contacts
from that flag; CARL reconstructs contacts geometrically. Subsequent RocketSim
ticks update contacts normally. The last-jump latch is restored separately from
the zeroed applied controls. Episodes end on goals or no-touch timeouts; tied
regulation expiry allows the same fixed 36,000-tick overtime limit as CARL.

## Comparing throughput

Use identical shared arguments for both trainers. For example:

```bash
COMMON=(--n-sim 256 --rollout 32 --timesteps 196608
        --expert-frame-limit 20000 --replay-reset-size 4096
        --discriminator-history-capacity 0 --discriminator-forward-kl-reward)
.venv/bin/python gaifo_cpu.py "${COMMON[@]}" --threads 16 --run-name native-256
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python gaifo.py "${COMMON[@]}" \
    --no-bf16 --run-name carl-256
```

CARL requires its current controller-enabled extension. Use `PYTHONPATH` to point
to a staged build if your installed wheel predates that schema.

For the faster factorized-MLP training profile, use the same model/objective
settings on both backends:

```bash
PROFILE=(--rollout 32 --frameskip 8 --discriminator-window-length 2
         --ppo-lr 3e-4 --mlp-discriminator --discriminator-relative
         --replay-dir replays/rlcs/parsed/ --discriminator-motion-noise-scale 0.001
         --gamma 0.99 --gamma-end 0.995 --gamma-anneal-steps 100000000000
         --discriminator-forward-kl-reward --target-discriminator
         --target-discriminator-tau 0.2 --matmul-precision medium
         --discriminator-epochs 1 --goal-reward-weight 50
         --replay-reset-size 3000000 --discriminator-decision-stride 3
         --discriminator-endpoint-scalars --factorize
         --discriminator-history-capacity 262144)
.venv/bin/python gaifo_cpu.py "${PROFILE[@]}" --n-sim 8192 --threads 16 \
    --timesteps 6291456 --warmup-iterations 3 --run-name native-mlp-8192
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python gaifo.py "${PROFILE[@]}" \
    --n-sim 8192 --timesteps 6291456 --compile --no-bf16 --run-name carl-mlp-8192
```

This compares CARL's compiled Python/Torch learner with the native
C++/TorchScript learner. Both use FP32 parameters and medium matmul precision.
`--no-bf16` explicitly matches native precision now that CARL training defaults
to BF16 autocast. Native keeps compilation/BF16 flags off by default. The current
CARL defaults also enable autotuning and broader PPO update compilation; historic
measurements below predate those defaults.
The explicit 100-billion-step gamma schedule preserves the production schedule
when shortening the benchmark's transition budget. Startup and compilation are
excluded using warmup iterations; do not compare this one-discriminator-epoch
MLP profile to the default four-epoch transformer profile.

Sweep simulation counts and collector threads for sustained comparisons. A
256-arena batch may underutilize CARL; separately report each engine's best tuned
throughput, with its batch sizes, rather than calling differently sized workloads
an equal-batch comparison. Match architecture, precision, discriminator cadence,
update epochs, source batch/microbatch sizes, PPO minibatches, replay caps,
history/target settings, reset fraction and frameskip. Run sequentially on an
otherwise idle machine and use longer runs for stable conclusions.

Each native run writes:

- `config.json`: resolved arguments and build/model provenance.
- `metrics.jsonl`: per-iteration timings, throughput and training losses.
- `summary.json`: aggregate post-warmup throughput (default: exclude 2 iterations).
- Initial and `latest-*.jit` models, optimizer archives and `latest.json` step
  metadata. These are native artifacts, not browser-compatible Python training
  checkpoints; native resume is not implemented.

Timing includes collection (simulation, observations, transfer, inference and
history construction), discriminator updates, reward/GAE and PPO. CUDA is
synchronized at stage boundaries. Startup/build/export/data loading, metrics
writes and checkpoint writes are excluded. CARL's existing total iteration timer
also includes its gameplay diagnostics; the native implementation reports a
leaner diagnostic set, so end-to-end ratios include that difference.

`agent_steps_per_second` counts car decisions. Multiply by frameskip for
`agent_physics_ticks_per_second`; divide that by `2 * team_size` for
`arena_physics_ticks_per_second`. These are nominal physics ticks represented by
the decisions, not wall-clock FPS. Compare to CARL's
`physics_ticks_per_second` tag, which also counts **agent** ticks.

`--compare-carl-log runs/carl-256` optionally prints a descriptive ratio after a
native run. It skips the same number of warmup samples and uses the harmonic mean
of CARL rates (appropriate for equal rollout sizes). It does not verify workload
equivalence. A run shorter than its warmup has zero measured steps.

### Host/device transfer accounting

Every CUDA run reports logical rollout `h2d_bytes` and `d2h_bytes`. For each
agent decision, this implementation transfers two FP32 observation vectors
(final and next), a float score and two byte flags to CUDA, and seven int64
actions back to CPU. For 1v1's 166-value observation, that is **1,334 bytes H2D
plus 56 bytes D2H per agent-step**. At 8,192 arenas and rollout 32, this is
667 MiB H2D plus 28 MiB D2H per rollout. This accounting excludes small training
metric/gradient-check scalars and checkpoints, and counts payload rather than
PCIe protocol traffic.

Use a separate `--profile-transfers` run to measure `h2d_seconds` and
`d2h_seconds`. Additional CUDA synchronization before the transfer blocks
separates pending inference from transfer time. These are **host-visible copy
path latencies**, including allocation/copy dispatch and completion, rather than
pure DMA event durations. The extra barriers perturb total throughput; compare
against an otherwise identical unprofiled run. Stage timing already includes
these copies, so do not add transfer seconds to total training time.

Expert frames and sampling indices stay on the learner device throughout
training. `expert_bank_bytes` and `expert_startup_transfer_seconds` measure that
bank's one-time transfer after reading the exported archive. `reset_bank_host_bytes`
reports native replay-reset storage, which stays on CPU beside RocketSim.
Increasing the expert/reset bank size therefore increases startup/memory cost,
but **does not add expert minibatch H2D copies in the training loop**. Increasing
the environment count or rollout length does increase rollout traffic linearly.
CARL keeps simulation observations/actions and replay resets on CUDA, avoiding
the native collector's bulk action/observation transfers; scalar diagnostics and
episode logging still synchronize/copy small amounts to the host.

## Verification

```bash
.venv/bin/python -m training.test_native_gaifo \
    --binary runs/gaifo-cpu-build/gaifo_native
```

This generates Python reference fixtures and checks native causal strided
windows across startup/chunks/resets, true final successors, timeout GAE, capped
forward-KL rewards, PPO loss and parameter gradients. Native environment probes
also cover scene ordering/team inversion, internal reset latch restoration,
action decoding, goal signs/terminal flags and final/reset separation for all
three team sizes.
Export additionally checks dynamic batch sizes and finite gradients through each
TorchScript model. CPU/CUDA end-to-end smoke runs should show finite updates to
actor, critic and discriminator parameters.

## Initial default-profile measurements (2026-11-16)

Hardware: Ryzen 9 7950X3D (16 cores/32 logical CPUs), RTX 4090, PyTorch
2.11.0+cu128. An existing GPU workload was active (about 12.4 GiB allocated and
75% utilization before testing). These short sequential measurements are
**contended preliminary results**, not an uncontended simulator speedup claim.
These used the default transformer/four discriminator epochs, **not the faster
factorized-MLP production profile above**.

At **2,048 arenas**, matched 1v1 runs measured **60,583 native RocketSim versus
72,658 CARL agent-steps/s**, or **1.20x CARL/native**. Both used the comparison
arguments above, changing `--n-sim 2048 --timesteps 1310720`: ten iterations,
with the first two excluded, yielding 1,048,576 measured agent transitions.
Native used 16 collector threads and one Torch thread; CARL used one OMP/MKL
thread and the staged controller-enabled reset-consistency build at
`~/.local/share/goddard/carl/reset-consistency-20261115-112936/stage`.

Native post-warmup stage totals were 3.920 s collection, 10.286 s discriminator,
0.404 s reward/GAE, and 2.698 s PPO, totaling 17.308 s. CARL's measured total was
14.432 s. The discriminator dominates this workload; the ratio is end-to-end
training throughput, not isolated simulation throughput. Local artifacts are
`runs/native-benchmark-2048/{config.json,metrics.jsonl,summary.json}` and
`runs/carl-benchmark-2048/` (TensorBoard and checkpoint configuration).
