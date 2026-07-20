# Goddard

Goddard is a Rocket League agent trained in CARL with JARL. The first baseline
is shared-policy self-play using PPO and the reward design from the Seer thesis.

Each car is one PPO rollout actor. CARL inverts orange observations and computes
the registered reward from canonical state transitions. All simulation,
reward, rollout, and learning tensors remain on `cuda:0`.

## Training

CARL and JARL must be installed in the active Python environment. With the
local `rlenv` environment used by this workspace:

```bash
python train.py --total-timesteps 100000000
```

Training metrics are written to timestamped directories such as
`runs/goddard-20260720-143000`, with matching snapshots and the final model in
`checkpoints/goddard-20260720-143000`. Use `--run-name NAME` to provide a
stable shared label.
Episode reward and length include only actors controlled by the actively
trained policy; frozen historical opponents are excluded. In current-policy
mirror games, both sides are active learners and contribute metrics. The
`historical_reward` and `historical_length` series isolate active learner
episodes played against frozen historical opponents.

The defaults run 1,024 parallel 1v1 simulations, collect 512 steps per update,
unroll 32-step GRU sequences with a 256-unit hidden state, and use eight PPO
epochs with 65,536-sample minibatches. Updates use BF16 autocast, TF32 matrix
multiplication, zero-copy rollout acquisition, reset-aware fused GRU batches,
and field-pruned recurrent sampling. Use `--precision float32` to disable BF16.
Episodes truncate after 4,096 physics
ticks. `--total-timesteps 256 --num-simulations 16
--rollout-steps 8 --sequence-length 4 --hidden-size 32 --minibatch-size 256` is
a useful smoke test.

The actor and critic are standard JARL modules with a shared linear-ReLU head
and GRU body, plus independent output feet. `ActorCritic` executes the shared
path once for policy and value estimation.

Self-play defaults to 80% current-policy mirrors and 20% games against
historical snapshots:

```bash
python train.py \
    --self-play-current 0.8 \
    --snapshot-interval 16 \
    --opponent-pool-size 8 \
    --historical-policies 4 \
    --team-spirit 1.0
```

The snapshot interval is measured in PPO updates. Other aligned defaults are a
constant `2.5e-4` learning rate, entropy coefficient `1e-3`, discount `0.999`,
and GAE lambda `0.99`.

The historical-game proportion is `1 - self-play-current`.
The learner's team is randomized in historical games. Both teams contribute
training samples in current-policy mirrors, while historical opponent samples
are excluded from PPO. Team spirit only changes behavior for multi-car teams
and has no effect in 1v1.

The progress counter reports learner-controlled per-car transitions, not CARL
simulation steps or historical-opponent actions.

CARL performs same-step autoreset but does not expose terminal observations.
JARL therefore disables value bootstrapping for CARL time-limit truncations
rather than using the next episode's kickoff state.

## Reward

`SeerReward` is registered directly with `CARLTorchVectorEnv`. It combines the
16 terms documented in *Seer: Reinforcement Learning in Rocket League*: goal
scored, boost difference, ball touch, demo, player-ball distance, ball-goal
distance, facing ball, ball-goal alignment, closest to ball, last touch,
behind ball, velocity toward ball, kickoff, velocity, boost amount, and forward
velocity.

The weighted reward is converted to zero-sum form by subtracting the opposing
team's mean reward, then normalized with running mean and variance. Touch decay
and last-touch possession are maintained on CUDA across steps. CARL does not
store the demolishing player, so demo credit is exact in 1v1 and distributed
across the opposing team in larger matches.
