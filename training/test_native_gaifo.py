"""Cross-language checks: python -m training.test_native_gaifo --binary BUILD/gaifo_native."""
import argparse
import json
import subprocess
import tempfile
from importlib.metadata import distribution
from pathlib import Path

import torch
from torch import nn

from gaifo_cpu import NativeActionCodec, PolicyExport, parser
from training.arguments import resolve_gaifo_defaults
from training.discriminator import discriminator_rewards
from training.imitation import TransitionRollout
from training.models import Agent
from training.ppo import ppo_loss
from training.rollout import Rollout, compute_gae


def check(binary, directory, team_size):
    torch.manual_seed(72)
    args = parser().parse_args([])
    resolve_gaifo_defaults(args)
    size = 9 + 21 * 2 * team_size + 68 + 12 * team_size + 8 + 27
    config = dict(vars(args), n_sim=2, team_size=team_size, threads=2,
                  observation_size=size, replay_reset_fraction=1.0, max_ticks=8,
                  no_touch_timeout=8 / 120,
                  clip_value_loss=True, value_clip_coef=0.15,
                  collision_meshes=str(distribution("rlgym-rocket-league").locate_file(
                      "rlgym/rocket_league/sim/collision_meshes")))
    (directory / "config.json").write_text(json.dumps(config, default=str))
    data = {}
    observations = torch.randn(13, 4, 5)
    done = torch.zeros(12, 4, dtype=torch.bool)
    done[2, 0] = done[3, 0] = done[4, 1] = done[6, 2] = done[9, 3] = True
    finals = torch.where(done[..., None], torch.randn(12, 4, 5), observations[1:])
    scenes = TransitionRollout(observations[0], 3, 5, 3, 2)
    windows = []
    # Reuse the Python rollout over four chunks, including histories longer than a chunk.
    for start in range(0, 12, 3):
        for t in range(start, start + 3):
            scenes.record(t - start, observations[t], observations[t + 1], finals[t],
                          done[t], done[t])
        windows.append(scenes.transitions(torch.arange(12)).reshape(3, 4, 3, 5))
    data.update(observations=observations, done=done, finals=finals, windows=torch.cat(windows))
    rollout = Rollout.allocate(12, 4, 5, 7, "cpu")
    rollout.rewards.normal_()
    rollout.values.normal_()
    rollout.dones.copy_(done)
    rollout.timeout_values.copy_(torch.randn(12, 4) * done)
    next_value = torch.randn(4)
    compute_gae(rollout, next_value, .97, .91)
    for name in ("rewards", "values", "advantages"):
        data[name] = getattr(rollout, name)
    data.update(timeouts=rollout.timeout_values, next_value=next_value)
    logits = torch.tensor([[-100., 100.], [0., 0.], [-1., 2.], [-3., -2.]])
    data.update(logits=logits, ratio_reward=discriminator_rewards(
        logits, discriminator_forward_kl_reward=True))
    agent = Agent(size, (3, 3, 3, 2, 2, 3, 2), NativeActionCodec(), 32, 1)
    obs = torch.randn(11, size)
    obs[:, 25] = torch.arange(11) % 2
    obs[:, -29] = torch.arange(11) % 3 == 0
    obs[:, -28] = torch.arange(11) / 300
    torch.jit.trace(PolicyExport(agent), obs).save(str(directory / "actor.jit"))
    torch.jit.trace(agent.critic, obs).save(str(directory / "critic.jit"))
    with torch.no_grad():
        actions, old_logp, _, values = agent.get_action_and_value(obs)
    old_logp += torch.linspace(-.5, .5, 11)
    returns, advantages = torch.randn(11), torch.randn(11)
    entropy = torch.linspace(.001, .02, 7)
    scale = torch.tensor(1.7)
    loss, _ = ppo_loss(agent, obs, actions, old_logp, advantages, returns, values,
                       args.clip_coef, entropy, args.vf_coef, True, scale, .15)
    loss.backward()
    data.update(ppo_obs=obs, ppo_actions=actions, ppo_logp=old_logp,
                ppo_values=values, ppo_returns=returns, ppo_advantages=advantages,
                ppo_entropy=entropy, ppo_scale=scale, ppo_loss=loss.detach(),
                ppo_gradients=torch.cat([p.grad.flatten() for p in agent.parameters()]))
    # A synthetic airborne reset makes wheel contacts unambiguous in either engine.
    reset = torch.zeros(1, 9 + 42 * team_size)
    reset[0, :9] = torch.tensor([.1, .2, .3, .01, .02, .03, .04, .05, .06])
    internal = torch.zeros(1, 2 * team_size, 19)
    for car in range(2 * team_size):
        start = 9 + 21 * car
        reset[0, start:start + 3] = torch.tensor([.1 * (car + 1), -.1, .4])
        reset[0, start + 9] = reset[0, start + 14] = 1
        reset[0, start + 15] = .5
        internal[0, car, 3] = internal[0, car, 5] = 1
        internal[0, car, 1] = .2
    controls = torch.tensor([[1, 2, 1, 1, 1, 2, 0], [2, 1, 2, 0, 0, 1, 1]])
    controls = controls.repeat(2 * team_size, 1)
    data.update(reset_frames=reset, reset_internal=internal, controls=controls)
    fixture = nn.Module()
    for name, tensor in data.items():
        fixture.register_buffer(name, tensor)
    torch.jit.script(fixture).save(str(directory / "checks.jit"))
    subprocess.run([str(binary), str(directory), "--check"], check=True)
    probe = torch.jit.load(str(directory / "probe.jit"))
    observed = probe.observations.reshape(2, 2 * team_size, size)
    cars = reset[0, 9:].reshape(2 * team_size, 21)
    signs = torch.tensor([-1, -1, 1] * 5 + [1] * 6)
    for ego in range(2 * team_size):
        allies = list(range(0, team_size) if ego < team_size else range(team_size, 2 * team_size))
        opponents = list(range(team_size, 2 * team_size) if ego < team_size else range(team_size))
        order = [ego] + [c for c in allies if c != ego] + opponents
        expected = torch.cat((reset[0, :9], cars[order].flatten()))
        if ego >= team_size:
            expected[:9] *= torch.tensor([-1, -1, 1] * 3)
            expected[9:] *= signs.repeat(2 * team_size)
        torch.testing.assert_close(observed[0, ego, :len(expected)], expected)
    torch.testing.assert_close(observed[0], observed[1])
    torch.testing.assert_close(observed[..., -27:-19], torch.zeros_like(observed[..., -27:-19]))
    assert observed[..., -19].eq(1).all()  # latch restored independently of applied controls
    assert probe.done.bool().all()  # timeout after one action; preserve pre-reset final_obs
    torch.testing.assert_close(probe.next, probe.observations)
    decoded = torch.tensor([[-1, -1, -1, 1, 1, 0, 1, 1], [1, 1, 1, -1, -1, 1, 0, 0.]])
    torch.testing.assert_close(probe.final[:, -27:-19], decoded.repeat(2 * team_size, 1))
    assert not torch.equal(probe.final, probe.next)
    print(f"{team_size}v{team_size}: scene ordering/inversion, reset latch, action decoding, final/reset separation passed")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--binary", required=True, type=Path)
    args = cli.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="native-gaifo-") as temporary:
        for team_size in (1, 2, 3):
            check(args.binary.resolve(), Path(temporary), team_size)


if __name__ == "__main__":
    main()
