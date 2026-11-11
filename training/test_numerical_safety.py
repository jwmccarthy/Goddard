from types import SimpleNamespace

import pytest
import torch

from imitation.evaluation import evaluate_discriminator
from imitation.models import WindowHead
from training.checkpoints import save_checkpoint, validate_checkpoint_state
from training.ppo import clip_actor_critic_gradients


def test_nonfinite_gradients_stop_before_optimizer_step():
    agent = SimpleNamespace(actor=torch.nn.Linear(1, 1), critic=torch.nn.Linear(1, 1))
    agent.actor.weight.grad = torch.full_like(agent.actor.weight, float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        clip_actor_critic_gradients(agent, 0.5)


def test_nonfinite_checkpoint_does_not_replace_last_good(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "latest.pt"
    path.write_bytes(b"previous checkpoint")
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="checkpoint state"):
        save_checkpoint(path, "ppo", SimpleNamespace(), model, optimizer, 1)
    assert path.read_bytes() == b"previous checkpoint"


def test_nonfinite_optimizer_state_is_rejected():
    with pytest.raises(FloatingPointError, match="optimizer.state"):
        validate_checkpoint_state({"state": {0: {"exp_avg": torch.tensor([float("inf")])}}}, "optimizer")


def test_nonfinite_validation_logits_do_not_report_half_accuracy():
    class InvalidDiscriminator(torch.nn.Module):
        factorized = False

        def forward(self, windows):
            return torch.full((len(windows),), float("nan"))

    args = SimpleNamespace(discriminator_inference_microbatch_size=8, near_distance=1000)
    validation = torch.zeros(4, 2, 51)
    with pytest.raises(FloatingPointError, match="validation logits"):
        evaluate_discriminator(InvalidDiscriminator(), None, None, None, args, validation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA attention backend test")
def test_masked_transformer_training_uses_math_attention(monkeypatch):
    from imitation import models

    original = models.sdpa_kernel
    selected = []

    def track(backend):
        selected.append(backend)
        return original(backend)

    monkeypatch.setattr(models, "sdpa_kernel", track)
    model = WindowHead(12, 16, 8, 4).cuda()
    observations = torch.randn(8, 4, 12, device="cuda")
    padding = torch.tensor([[True, True, False, False]] * 8, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(observations, padding)
        loss = output.float().square().mean()
    loss.backward()
    assert selected == [models.SDPBackend.MATH]
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    selected.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(observations, padding)
    assert not selected
