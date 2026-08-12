import torch as th

from jarl.collect.capture import CaptureContext, CaptureBase, CriticCapture


class LatentCapture(CaptureBase):

    def __init__(
        self,
        latent_dim: int = 64,
        min_steps:  int = 1,
        max_steps:  int = 150,
        device:     str | th.device = "cuda:0",
        seed:       int = 0
    ) -> None:
        self.latent_dim = latent_dim
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.device = th.device(device)
        self.generator = th.Generator(device=self.device).manual_seed(seed)
        self.latent: th.Tensor | None = None
        self.current: th.Tensor | None = None
        self.steps: th.Tensor | None = None

    def reset(self, batch_size: int) -> None:
        self.latent = self._sample(batch_size)
        self.steps = self._sample_duration(batch_size)

    def _sample(self, count: int) -> th.Tensor:
        latent = th.randn(
            (count, self.latent_dim),
            device=self.device,
            generator=self.generator
        )
        return th.nn.functional.normalize(latent, dim=-1)

    def _sample_duration(self, count: int) -> th.Tensor:
        return th.randint(
            self.min_steps,
            self.max_steps + 1,
            (count,),
            device=self.device,
            generator=self.generator
        )

    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        if self.latent is None or self.steps is None:
            raise RuntimeError("latent capture must be reset before use")

        return {"latent": self.advance(context.env_step.done)}

    def advance(self, done: th.Tensor) -> th.Tensor:
        if self.latent is None or self.steps is None:
            raise RuntimeError("latent capture must be reset before use")

        self.current = self.latent.clone()
        done = th.as_tensor(
            done,
            dtype=th.bool,
            device=self.device
        )

        self.steps -= 1
        resample = done | self.steps.eq(0)
        count = int(resample.sum().item())

        if count:
            self.latent[resample] = self._sample(count)
            self.steps[resample] = self._sample_duration(count)

        return self.current


class LatentCriticCapture(CriticCapture):

    def __init__(self, critic, latents: LatentCapture) -> None:
        super().__init__(critic)
        self.latents = latents

    @th.no_grad()
    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        if self.latents.current is None or self.latents.latent is None:
            raise RuntimeError("latent capture must run before critic capture")

        next_obs = th.as_tensor(
            context.env_step.next_obs,
            device=context.observation.device
        )
        baseline_value = self.critic.value(
            (context.observation, self.latents.current),
            context.state
        )
        baseline_next_value = self.critic.value(
            (next_obs, self.latents.latent),
            context.policy_output.next_state
        )

        return {
            "baseline_value": baseline_value,
            "baseline_next_value": baseline_next_value
        }
