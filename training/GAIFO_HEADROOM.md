# GAIfO throughput/headroom investigation — 2026-11-16

Hardware: Ryzen 9 7950X3D, RTX 4090, Torch 2.11.0+cu128. The user's other
training process was stopped for the baseline, compiled-diagnostics and
CUDA-graph measurements. It restarted at 17:35:23, before the autotuning run
finished; that later result is not a controlled idle-GPU comparison. CARL is the staged
controller-enabled reset-consistency build at
`~/.local/share/goddard/carl/reset-consistency-20261115-112936/stage`.

## Workload and units

Use the **factorized relative MLP profile** in [GAIFO_CPU.md](GAIFO_CPU.md),
including endpoint scalars, one discriminator epoch, four PPO epochs, medium
FP32 matmul precision, forward-KL reward, target tau .2, goal weight 50 and the
full replay dataset/three-million-state reset bank. The gamma anneal duration
stays 100 billion transitions when shortening a benchmark run.

At 8,192 arenas in 1v1, rollout 32 contains **524,288 agent-steps**. With
frameskip 8, those represent 4,194,304 agent physics ticks. A rate of 550k
agent-steps/s is the *same end-to-end rate* as 4.4M agent physics ticks/s. The
latter is not a separate collection-only benchmark.

## Measured stage breakdown

Three warmup iterations were excluded; four measured iterations per row.
The profiler synchronizes at stage boundaries. These are short sequential
measurements, not confidence intervals or demonstrated globally optimal tuning.

| Runtime path | Collection ms/rollout | PPO ms | Discriminator ms | Reward ms | Agent-steps/s |
|---|---:|---:|---:|---:|---:|
| Existing compiled model/loss path | 468.4 | 313.4 | 123.9 | 43.1 | 551,956 |
| Also compile gameplay diagnostics | 347.0 | 317.4 | 124.1 | 44.1 | 628,787 |
| Compiled diagnostics + existing `--cuda-graphs` | 345.3 | 298.0 | 124.1 | 30.1 | 656,512 |

GAE is about 0.3 ms/rollout. The original total is 949.9 ms/rollout, so
collection accounts for **49.3%**. Its collection-only rate is **1.12M
agent-steps/s**, versus 552k end-to-end. Compilation of the existing diagnostics
reduces collection time 26% and improves total throughput about 14%; adding
CUDA graphs gives about 19% over the starting path.
The autotuning experiment recorded 679,079 agent-steps/s, but overlapped the
restarted training process. **The previously claimed 23% controlled gain is
withdrawn.** The experimental `--compile-autotune` flag is wired alongside
`--compile --cuda-graphs`, but production-flag CUDA execution and an uncontended
A/B remain pending. Expect longer initial compilation. Discriminator loss keeps
CUDA-graph replay disabled while accumulating microbatch gradients, as before.

Artifact completion times were 17:27:42 (baseline), 17:33:20 (diagnostics),
17:35:08 (graphs), and 17:40:59 (autotuning). The 65,536-discriminator-microbatch
experiment was stopped after detecting concurrent training; it produced no
completed summary. No larger-microbatch speedup is established.

The completed native sweep's best result was **251,222 agent-steps/s at 2,048
arenas/32 threads**. CARL's initial uninstrumented 8,192-arena result was 556,345,
or 2.21x native. The user stopped the sweep before larger native configurations
completed, so this is best-completed, not a fully tuned upper envelope or a
benchmark of the unmodified upstream RLGymPPO_CPP program.

## Bottlenecks found

1. **Eager gameplay diagnostics on the collection path.** GAIfO computes the
   complete Nexto component dictionary and counters each step, even when the
   gameplay reward weight is zero. Those outputs are still useful logs, but
   their many small operations were eagerly dispatched. `gaifo.py` now compiles
   this existing callable with `--compile` for the Nexto diagnostic path.
   Component values, counters and reset behavior are retained. Reward-plus stays
   on its existing path. A CUDA numerical test covers repeated calls, counter
   resets and non-mutation of simulator inputs.
2. **PPO is substantial GPU work.** Four PPO epochs remain intact. The warmed
   trace contains thousands of GEMMs and separate bias/ReLU/backward kernels.
   CUDA graphs help launch overhead, but do not eliminate this compute.
3. **CARL physics itself is also material.** In the traced rollout,
   `carCarBallSolveKernel` took 160.5 ms, suspension raycasts 29.6 ms, arena
   solving 10.6 ms, and observation packing 14.1 ms. This is evidence against
   attributing all collection cost to Python. The trace has profiler overhead;
   these kernel times are not a standalone simulator throughput claim.
4. **Many small launches and synchronizations.** The warmed trace contains
   40,844 GPU kernels and 667 `cudaStreamSynchronize` calls. Profiling itself
   adds device synchronizations and CPU overhead. Host wait duration includes
   waiting for compute and must not be called PCIe transfer time or added to
   overlapping GPU durations.

### Where the stream synchronizations occur

The wrapper's explicit `_sync()` calls are conditional on `synchronize=True`;
the default is false and Goddard does not enable it. The relevant waits are
implicit synchronization from copies, scalar extraction and dynamic-size output
operations. CUDA stream/event ordering still has to be preserved.

The original trace attributes 403 stream waits to `aten::copy_` inside
`env.step` (198 ms of host waiting), 94 to `nonzero` inside `env.step`, and 129
to PPO scalar extraction (2.3 ms). The copy waits include eagerly created
Nexto goal-coordinate constants; they wait for preceding GPU work, so the
198 ms is **not 198 ms of copy engine work** and cannot simply be subtracted
from runtime. Compiling diagnostics removes much of this eager constant path.

Replay resets compact `reset_mask.nonzero()` and then boolean-select sampled
indices; episode reporting separately compacts `done.nonzero()`. This creates
host-dependent lengths within each decision. A GPU-mask-based fused reset path
and deferred episode reporting would remove those control dependencies while
retaining reset sampling/terminal successors. That requires a CARL interface
change and validation, rather than simply deleting `synchronize()` calls.
The 94 `nonzero` waits summed to only 1.52 ms in this trace. Nested profiling
also inserts per-step/policy barriers, so it cannot establish the throughput
gain from removing these waits in normal asynchronous execution. A targeted A/B
is needed before promising a large reset/reporting speedup.

### Relationship to the published benchmark

The [CARL post](https://jwmccarthy.github.io/posts/cuda-accelerated-rocket-league-intro/)
and [benchmark configuration](https://github.com/jwmccarthy/carl-benchmarks/blob/main/benchmarks/unified_configs.py)
use **one PPO epoch, 3x256 networks, kickoff resets, and goal/touch rewards**.
This GAIfO workload has four PPO epochs, 3x384 networks, replay resets and two
discriminator heads. The hidden-layer PPO matrix work alone scales roughly as
`4 * (384/256)^2 = 9` relative to the one-epoch setup, before discriminator work.
That explains part of the end-to-end difference, **not the collection-only gap**.

Published table sizes count actors: our 8,192 arenas correspond to 16,384 actors.
At that actor count, the post reports 6.52M collection and 2.86M total
agent-steps/s on the 5090. Collection must additionally be compared with the same
physics revision, reset-state distribution, rewards/diagnostics and wrapper path;
the current profiling shows both interface overhead and substantial physics cost.
The post's C++ peak is 95.8k on its i5-13600K/3060 setup, whereas our new native
GAIfO port reaches 251k on the 7950X3D/4090. Speedup ratios using those different
CPU baselines are not interchangeable.

One untested physics-state confound is the sleeping-ball fast path in
`Physics/BallArena/Solve.cuh`: exactly sleeping balls skip arena collision
solving, whereas replay initialization usually starts with moving balls. Short
kickoff benchmarks may exercise a different mix of this work. The prepared
`/tmp/opencode/physics_sleep_probe.py` has **not been run**, and this is not yet
an explanation of the published collection-rate gap.

## Transfer/data-size findings

The full dataset has 26,386,483 eligible expert windows. Native telemetry reports
**6,651,122,992 bytes** for GPU expert frames plus sampling metadata and
**1,068,000,000 bytes** for its CPU reset bank. The initial expert upload took
about **0.62 s**, excluded from steady-state timing. Expert sampling thereafter
is device-resident; a larger expert bank does not imply an H2D copy of each
training minibatch in either implementation.

The CARL trace measured only **3,788 H2D bytes and 2,947 D2H bytes** in one
524,288-agent-step rollout/update. Transfer device durations summed to about
**0.63 ms**. These small copies are constants, scalar checks and episode/logging
data; synchronization can cost much more than moving their payload. The trace
also includes about 3.06 GB of **device-to-device** copies, which are not PCIe
traffic.

By contrast, the native RocketSim collector necessarily exchanges actions and
observations. Its current interface accounts for 1,334 H2D bytes plus 56 D2H
bytes per agent-step: **695 MiB per 8,192-arena rollout**. Payload size scales
with environments, rollout length and observation width, not with the size of
the resident expert bank. `--profile-transfers` measures the native copy-path
latency separately; its explicit barriers perturb throughput, so it is a
separate profiling mode.

## Interpreting further headroom

### Implemented Goddard defaults after the measurements

PPO/GAIfO/DIFO now enable `--compile`, `--compile-ppo-update`,
`--compile-autotune`, `--cuda-graphs` and `--bf16` by default. The new PPO path
compiles minibatch gathering with loss/backward, gradient clipping, fused Adam
dispatch and metric accumulation. It retains a pre-Adam host finite check;
mutating optimizer/gradient/metric regions do not use CUDA graph trees. Static
rollout addresses avoid full-rollout input copies on each loss graph replay.
Tensor learning rates support annealing without specializing on each scalar LR.
See [GAIFO.md](GAIFO.md) for opt-outs and scope.

These defaults were implemented **after** the timing table above. Those
measurements used the old FP32 training configuration and do not establish the
speed of this new path. CPU Inductor tests cover eager/compiled parameter and
Adam-state parity, partial minibatches, BF16, changing rollout contents and
learning rates, optimizer-state reloads, and rejection of nonfinite losses and
gradients before Adam. The targeted suite ran 57 tests: 56 passed and the CUDA
graph/BF16 parity test was skipped with CUDA hidden to leave training undisturbed.
GPU execution validation and clean throughput measurements remain pending.

### CARL native-code audit (no concurrent GPU benchmarking)

Inspection of the staged source and its existing sm_89 binary found concrete
optimization candidates. `cuobjdump --dump-resource-usage` on
`reset-consistency-20261115-112936/build/_carl.cpython-311-x86_64-linux-gnu.so`
reports the following static resource requirements:

| Kernel | Registers/thread | Stack bytes/thread |
|---|---:|---:|
| `carCarBallSolveKernel` | 205 | 16,344 |
| `carArenaSolveKernel` | 172 | 1,288 |
| `carSuspensionRaycastKernel` | 109 | 32 |
| `packObsKernel` | 96 | 0 |

These are binary resource metadata, not measured memory traffic or occupancy.
The combined car-car/ball kernel was also the largest individual physics kernel
in the earlier trace (160.5 ms per rollout). Its high register requirement and
large stack make it the first native physics optimization target to investigate.

- **Smaller common-contact solver path, with full-capacity fallback.**
  `include/Physics/BallArena/Solve.cuh` allocates 68 contacts;
  `Solver.cuh` allocates 68 split-impulse rows. `Contact.cuh` also maintains two
  27-entry arrays and repeatedly merges spatial-cell triangle lists in native
  traversal order. Measure contact-count distribution, stack accesses and warp
  stalls; then test small-contact specialization and/or a more compact scratch
  layout. Preserve contact ordering, all contacts and solver iterations.
  Splitting the combined kernel may reduce resource coupling but adds launches
  and state traffic, so it needs an A/B rather than an assumed win.
- **Combined replay-reset application and final packing.** In `python/carl.cu`,
  both `setBall` and `setCar` independently repack observations, state and
  transition state. The Python replay-reset path invokes these setters in
  sequence. A single reset operation can avoid intermediate output packing while
  retaining contact reconstruction, final successors and reset-state semantics.
  Pair this with GPU-mask reset selection and deferred episode reporting.
- **CUDA graph replay for the native frameskip sequence.** The native `step`
  binding currently loops over physics ticks and dispatches each substage;
  it has no CUDA graph capture/replay path. Goddard's `--cuda-graphs` applies to
  compiled Torch callables, not this native physics loop. Capturing the fixed
  eight-tick sequence is a candidate for reducing CPU launch overhead, with
  correct input lifetimes, stream ordering and event-buffer updates.

None of these native changes has been implemented or benchmarked. At the
current 43% collection share, a hypothetical **2x speedup of all collection**
would improve end-to-end throughput about **28%** (657k to roughly 837k
agent-steps/s). Doubling physics alone would give a smaller gain; the collection
stage also includes policy inference, diagnostics and bookkeeping.

Keep model sizes, PPO/discriminator update counts, optimizer minibatches and
reward definitions fixed when attributing a speedup to implementation. Larger
discriminator *microbatches* can reduce dispatch without changing the effective
optimizer batch; larger PPO minibatches change optimization and are a different
experiment. GEMM/epilogue autotuning and more fusion are candidates to measure,
not assumed gains.

At the original measured split, making **all learner/reward work free** would
leave 468 ms collection time, or 1.12M agent-steps/s. With compiled diagnostics,
that collection-only limit becomes about 1.51M. Thus reaching 10x the completed
251k native result would require faster collection/physics as well as faster
learning; it cannot follow from removing Python learner overhead alone.

## Local artifacts

- `/tmp/opencode/headroom-carl-stages/`: coarse baseline stage timings.
- `/tmp/opencode/headroom-carl-nested/`: nested timing, CPU/CUDA operator tables,
  and `trace.json` (one warmed, instrumented iteration).
- `/tmp/opencode/headroom-carl-compiled-diagnostics/`: first targeted A/B.
- `/tmp/opencode/headroom-carl-graphs/`: diagnostics plus CUDA-graph A/B.
- `/tmp/opencode/profile_carl_stages.py`: profiling wrapper; source trainer calls
  are timed with explicit synchronization.
- `runs/headroom-carl-*/`: TensorBoard metrics and fresh benchmark checkpoints.
- `/tmp/opencode/gaifo-profile-sweep/results.json`: completed native results.

Numerical check: `.venv/bin/python -m unittest training.test_compiled_diagnostics`.
The CUDA diagnostics parity test passed before training resumed. After the
autotuning flag changes, 36 CPU tests passed across `training.test_diffusion`,
`training.test_forward_kl`, `training.test_calm` and
`training.test_discriminator_flags`. Further GPU profiling (including native
transfer latency and the sleeping-ball probe) is paused while training is active.
