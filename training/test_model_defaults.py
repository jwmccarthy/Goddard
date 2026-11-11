import argparse

import pytest

from training.arguments import add_ppo_arguments


@pytest.mark.parametrize("gaifo", [False, True])
def test_default_actor_and_critic_mlp_dimensions(gaifo):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=gaifo)
    args = parser.parse_args([])

    assert args.hidden_size == 384
    assert args.layers == 3


@pytest.mark.parametrize("gaifo", [False, True])
@pytest.mark.parametrize("flags,gamma,gae_lambda", [
    ([], 0.995, 0.98),
    (["--gamma", "0.99", "--gae-lambda", "0.95"], 0.99, 0.95),
])
def test_discount_defaults_and_explicit_overrides(gaifo, flags, gamma, gae_lambda):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=gaifo)
    args = parser.parse_args(flags)

    assert args.gamma == gamma
    assert args.gae_lambda == gae_lambda
    assert args.rollout == 32
