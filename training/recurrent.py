import torch
from torch.autograd.function import FunctionCtx
from torch.nn import functional as F

from training.recurrent_cuda import gru_backward, gru_forward


class GRUSequence(torch.autograd.Function):
    """Reset-aware GRU with a linear-time, batched-weight-gradient backward."""

    @staticmethod
    def forward(
        ctx:       FunctionCtx,
        projected: torch.Tensor,
        initial:   torch.Tensor,
        resets:    torch.Tensor,
        weight:    torch.Tensor,
        bias:      torch.Tensor
    ) -> torch.Tensor:
        if projected.is_cuda:
            states, gates = gru_forward(
                projected, initial, resets, weight, bias
            )
            ctx.save_for_backward(initial, states, gates, resets, weight)

            return states

        state = initial
        states, gates = [], []

        for step in range(projected.shape[0]):
            state = state.masked_fill(resets[step, :, None], 0)
            inputs = projected[step].float().chunk(3, -1)
            hidden = F.linear(state.to(projected.dtype), weight, bias)
            hidden = hidden.float().chunk(3, -1)
            reset = (inputs[0] + hidden[0]).sigmoid()
            update = (inputs[1] + hidden[1]).sigmoid()
            candidate = (inputs[2] + reset * hidden[2]).tanh()
            gates.append(torch.cat((reset, update, candidate, hidden[2]), -1))
            state = candidate + update * (state - candidate)
            states.append(state)

        states = torch.stack(states)
        ctx.save_for_backward(
            initial, states, torch.stack(gates), resets, weight
        )

        return states

    @staticmethod
    def backward(
        ctx:      FunctionCtx,
        gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, torch.Tensor]:
        initial, states, gates, resets, weight = ctx.saved_tensors

        if states.is_cuda:
            inputs, carry, weight_grad, bias_grad = gru_backward(
                gradient, initial, states, gates, resets, weight
            )

            return inputs, carry, None, weight_grad, bias_grad

        previous = torch.cat((initial[None], states[:-1]), 0)
        previous = previous.masked_fill(resets[..., None], 0)
        carry = torch.zeros_like(initial)
        inputs, hidden = [], []

        for step in reversed(range(states.shape[0])):
            reset, update, candidate, recurrent = gates[step].chunk(4, -1)
            total = gradient[step] + carry
            dc = total * (1 - update) * (1 - candidate.square())
            dz = total * (previous[step] - candidate) * update * (1 - update)
            dr = dc * recurrent * reset * (1 - reset)
            inputs.append(torch.cat((dr, dz, dc), -1).to(weight.dtype))
            dh = torch.cat((dr, dz, dc * reset), -1).to(weight.dtype)
            hidden.append(dh)
            carry = total * update + F.linear(dh, weight.T).float()
            carry = carry.masked_fill(resets[step, :, None], 0)

        # Assemble each gradient once instead of scattering every frame into
        # a full sequence tensor. Accumulate recurrent weights in one GEMM.
        inputs = torch.stack(inputs[::-1])
        hidden = torch.stack(hidden[::-1]).flatten(0, 1)
        previous = previous.flatten(0, 1).to(weight.dtype)

        return inputs, carry, None, hidden.T @ previous, hidden.sum(0)
