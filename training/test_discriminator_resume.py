from types import SimpleNamespace

import pytest
import torch

from imitation.test_amp import arguments
from training.arguments import resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, restore_discriminator_optimizer, save_checkpoint


def test_fresh_lr_default_and_explicit_override_survive_repeated_default_resolution():
    default = arguments()
    explicit = arguments("--discriminator-learning-rate", ".0001")
    assert default.discriminator_learning_rate == 3e-4
    assert default.discriminator_learning_rate_override is None
    assert explicit.discriminator_learning_rate_override == 1e-4
    resolve_gaifo_defaults(default)
    resolve_gaifo_defaults(explicit)
    assert default.discriminator_learning_rate_override is None
    assert explicit.discriminator_learning_rate_override == 1e-4


@pytest.mark.parametrize("override", [None, 1e-4, 3e-4])
def test_resume_preserves_moments_but_applies_only_explicit_lr_override(override):
    parameter = torch.nn.Parameter(torch.ones(()))
    saved = torch.optim.Adam([parameter], lr=3e-4)
    parameter.square().backward()
    saved.step()
    state = saved.state_dict()
    restored = torch.optim.Adam([parameter], lr=1e-5)
    args = SimpleNamespace(discriminator_learning_rate_override=override, discriminator_learning_rate=1e-5)
    if override == 1e-4:
        with pytest.warns(UserWarning, match="learning rate overridden"):
            restore_discriminator_optimizer(restored, state, args)
    else:
        restore_discriminator_optimizer(restored, state, args)
    assert restored.param_groups[0]["lr"] == args.discriminator_learning_rate == (override or 3e-4)
    torch.testing.assert_close(restored.state_dict()["state"], state["state"], rtol=0, atol=0)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_discriminator_lr_is_rejected(value):
    with pytest.raises(ValueError, match="discriminator learning rate"):
        arguments("--discriminator-learning-rate", value)


def test_legacy_resume_warns_that_padding_training_changed_but_keeps_weights(tmp_path):
    old, new = arguments(), arguments()
    del old.discriminator_padding_objective
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, "gaifo", old, model, optimizer, 12)
    with pytest.warns(UserWarning, match="padding matching changed"):
        payload = load_checkpoint(path, "gaifo", new, model, optimizer)
    assert payload["step"] == 12
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
