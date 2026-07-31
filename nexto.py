from pathlib import Path

import torch
import torch.nn as nn

from carl.gymnasium.state import BOOST_PAD_POSITIONS
from jarl.data.records import PolicyOutput


NEXTO_BOOST_POSITIONS = (
    (0.0, -4240.0, 70.0), (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0), (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0), (-940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0), (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0), (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0), (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0), (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0), (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0), (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0), (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0), (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0), (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0), (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0), (-940.0, 3310.0, 70.0),
    (940.0, 3308.0, 70.0), (-3072.0, 4096.0, 70.0),
    (3072.0, 4096.0, 70.0), (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0), (0.0, 4240.0, 70.0),
)

def _lookup_table() -> torch.Tensor:
    actions = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost and throttle != 1:
                        continue
                    actions.append(
                        [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
                    )
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump and yaw:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    return torch.tensor(actions, dtype=torch.float32)


class NextoPolicy(nn.Module):
    """Official Nexto TorchScript policy adapted to CARL's raw 1v1 state."""

    def __init__(self, checkpoint: Path, device: torch.device | str) -> None:
        super().__init__()
        # This small fixed batch is slower with the host's default thread count.
        torch.set_num_threads(min(8, torch.get_num_threads()))
        # The official TorchScript graph embeds its action table on CPU.
        self.model = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        carl_positions = torch.tensor(BOOST_PAD_POSITIONS, dtype=torch.float32)
        nexto_positions = torch.tensor(NEXTO_BOOST_POSITIONS, dtype=torch.float32)
        distances = torch.cdist(nexto_positions, carl_positions)
        nearest, indices = distances.min(dim=1)
        if nearest.gt(3.0).any():
            raise ValueError("CARL boost-pad layout does not match Nexto")
        self.register_buffer("boost_indices", indices.to(dtype=torch.int64))
        self.register_buffer("lookup", _lookup_table())
        self.register_buffer(
            "boost_positions",
            torch.tensor(NEXTO_BOOST_POSITIONS, dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return self.lookup.device

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, 8, device=self.device)

    @torch.no_grad()
    def act_from_raw(
        self,
        raw: torch.Tensor,
        state: torch.Tensor,
        car_index: int,
    ) -> PolicyOutput:
        action_device = raw.device
        query, values, mask = self._observation(
            raw.cpu(), state.cpu(), car_index
        )
        logits, _ = self.model((query, values, mask))
        controls = self.lookup[logits.argmax(dim=-1)]
        return PolicyOutput(
            action=self._encode_controls(controls).to(action_device),
            next_state=controls,
        )

    def _observation(
        self,
        raw: torch.Tensor,
        previous_action: torch.Tensor,
        car_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(raw)
        cars = raw[:, 9:53].view(batch_size, 2, 22)
        values = torch.zeros(batch_size, 37, 24, device=raw.device)
        values[:, car_index, 0] = 1
        values[:, car_index, 1] = 1
        values[:, 1 - car_index, 2] = 1
        values[:, 2, 3] = 1
        values[:, 3:, 4] = 1

        values[:, 2, 5:8] = raw[:, 0:3]
        values[:, 2, 8:11] = raw[:, 3:6]
        values[:, 2, 17:20] = raw[:, 6:9]
        values[:, 3:, 5:8] = self.boost_positions
        values[:, 3:, 20] = 0.12 + 0.88 * (self.boost_positions[:, 2] > 72)
        values[:, 3:, 21] = raw[:, 53:].index_select(1, self.boost_indices)

        values[:, :2, 5:8] = cars[:, :, 0:3]
        values[:, :2, 8:11] = cars[:, :, 3:6]
        values[:, :2, 11:14] = cars[:, :, 9:12]
        values[:, :2, 14:17] = cars[:, :, 12:15]
        values[:, :2, 17:20] = cars[:, :, 6:9]
        values[:, :2, 20] = cars[:, :, 15] / 100.0
        values[:, :2, 21] = cars[:, :, 17]
        values[:, :2, 22] = cars[:, :, 16]
        values[:, :2, 23] = ~(cars[:, :, 18].bool() | cars[:, :, 19].bool())

        if car_index:
            values[:, :, 5:7].neg_()
            values[:, :, 8:10].neg_()
            values[:, :, 11:13].neg_()
            values[:, :, 14:16].neg_()
            values[:, :, 17:19].neg_()

        normalizer = values.new_tensor(
            [1] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4
        )
        values /= normalizer
        query = torch.cat(
            (values[:, car_index : car_index + 1], previous_action[:, None]),
            dim=-1,
        )
        values[:, :, 5:8] -= query[:, :, 5:8]
        theta = torch.atan2(query[:, :, 11], query[:, :, 12])
        cosine = theta.cos()
        sine = theta.sin()
        for index in (5, 8, 11, 14, 17):
            x = values[:, :, index].clone()
            y = values[:, :, index + 1].clone()
            values[:, :, index] = cosine * x - sine * y
            values[:, :, index + 1] = sine * x + cosine * y
        return query, values, torch.zeros(batch_size, 37, dtype=torch.bool, device=raw.device)

    @staticmethod
    def _encode_controls(controls: torch.Tensor) -> torch.Tensor:
        horizontal = torch.where(
            controls[:, 1].ne(0), controls[:, 1], controls[:, 3]
        )
        analog = torch.stack(
            (horizontal, controls[:, 2], controls[:, 0], controls[:, 4]), dim=1
        )
        axis = torch.where(
            analog.lt(0),
            torch.ones_like(analog),
            torch.where(analog.gt(0), 2, 0),
        ).to(torch.int64)
        return torch.stack(
            (
                axis[:, 0], axis[:, 1], axis[:, 2], controls[:, 7].to(torch.int64),
                controls[:, 6].to(torch.int64), axis[:, 3], controls[:, 5].to(torch.int64),
            ),
            dim=-1,
        )
