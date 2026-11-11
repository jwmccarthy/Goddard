# 10 Hz discriminator observation spacing

`--discriminator-decision-stride 3` selects observations three policy decisions
apart for both dense replay experts and native CARL trajectories. It requires
`--frameskip 4`. The stride itself introduces no replay parsing, recorder,
quantization, noise, smoothing, or interpolation. Discriminator-training motion
noise is a separate setting: `--discriminator-motion-noise-scale` defaults to
0.001 for new invocations; set it to 0 for noise-free discriminator updates.
Policy observations, physics and reward inference stay unchanged.

`--discriminator-window-length` counts selected observations:

| Length | Raw decision history | Physics-tick span | Duration |
| --- | ---: | ---: | ---: |
| 2 | 4 observations | 12 | 100 ms |
| 8 | 22 observations | 84 | 0.7 s |

For length 8, a full window ending at decision `d` selects:

```text
d-21, d-18, d-15, d-12, d-9, d-6, d-3, d
```

The next window ends at `d+1`. Every 30 Hz decision receives a discriminator
reward, computed in batches after each rollout; rewards are not held for three
decisions. Policy inputs/actions, physics, PPO, and discounting remain unchanged.
`--discriminator-history-interval` separately controls how often earlier policy
windows are added to the discriminator history buffer.

Windowed actor and critic heads instead share `--policy-window-length` consecutive
decision-step observations (default 16). They have **no discriminator stride**:
at frameskip 4, 16 actor/critic frames span 0.5 s at 30 Hz, even when the
discriminator uses 2 frames spanning 0.1 s at 10 Hz. Use `--mlp-policy --mlp-critic`
for feed-forward MLPs over that history, or `--gru-policy --gru-critic` for GRUs.
Unflagged heads retain their original single-state MLP. Hidden widths default to
384, with an optional critic-only
`--critic-hidden-size` override. See [history semantics](README.md#dense-actorcritic-history).

Short prefixes are left-padded. Every intervening decision is checked for resets
or invalid replay frames, even if it is not selected. Histories carry across PPO
rollouts but never across episode resets or replay files. DAC still replaces a
goal successor with an absorbing frame and supplies absorbing self-loops;
timeouts use the actual final observation. Potential shaping, if enabled, uses
matching strided windows ending at the current and successor decisions.

The stride defaults to 1. Changing it requires a new run rather than silently
resuming a differently spaced discriminator. Checkpoints predating the option
mean stride 1. Configuration and TensorBoard record the observation frequency
and window span as `discriminator_observation_hz` / `discriminator/observation_hz`
and `discriminator_window_span_ticks` / `discriminator/window_span_ticks`.

Downsampling does not undo interpolation already present in dense expert files.
It may reduce fine-timescale shortcuts while retaining longer-term behavior;
neither shortcut elimination nor improved policy quality has been established.
Existing MLP/GRU/transformer, factorized/ego/global, and relative-input options
remain available. See [the scale command](../README.md#10-hz-discriminator-observations).
