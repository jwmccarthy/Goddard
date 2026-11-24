import torch
import triton
import triton.language as tl
from torch.nn import functional as F
from triton.language.extra.cuda import libdevice


@triton.jit
def _forward_gate(
    inputs, recurrent, previous, resets, next_resets, states, gates, carry,
    width: tl.constexpr, size: tl.constexpr, block: tl.constexpr
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    live = index < size
    row, column = index // width, index % width
    offset = row * (3 * width) + column
    x0 = tl.load(inputs + offset, live, 0).to(tl.float32)
    x1 = tl.load(inputs + offset + width, live, 0).to(tl.float32)
    x2 = tl.load(inputs + offset + 2 * width, live, 0).to(tl.float32)
    r0 = tl.load(recurrent + offset, live, 0).to(tl.float32)
    r1 = tl.load(recurrent + offset + width, live, 0).to(tl.float32)
    r2 = tl.load(recurrent + offset + 2 * width, live, 0).to(tl.float32)
    reset = tl.sigmoid(x0 + r0)
    update = tl.sigmoid(x1 + r1)
    candidate = libdevice.tanh(x2 + reset * r2)
    prior = tl.load(previous + index, live, 0)
    prior = tl.where(tl.load(resets + row, live, 0), 0., prior)
    state = candidate + update * (prior - candidate)
    tl.store(states + index, state, live)

    offset = row * (4 * width) + column
    tl.store(gates + offset, reset, live)
    tl.store(gates + offset + width, update, live)
    tl.store(gates + offset + 2 * width, candidate, live)
    tl.store(gates + offset + 3 * width, r2, live)
    state = tl.where(tl.load(next_resets + row, live, 0), 0., state)
    tl.store(carry + index, state, live)


@triton.jit
def _backward_gate(
    gradient, recurrent_gradient, direct, gates, previous, resets,
    next_resets, input_gradient, hidden_gradient, weight_inputs,
    width: tl.constexpr, size: tl.constexpr, block: tl.constexpr
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    live = index < size
    row, column = index // width, index % width
    offset = row * (4 * width) + column
    reset = tl.load(gates + offset, live, 0)
    update = tl.load(gates + offset + width, live, 0)
    candidate = tl.load(gates + offset + 2 * width, live, 0)
    recurrent = tl.load(gates + offset + 3 * width, live, 0)
    prior = tl.load(previous + index, live, 0)
    prior = tl.where(tl.load(resets + row, live, 0), 0., prior)
    future = tl.load(recurrent_gradient + index, live, 0).to(tl.float32)
    future += tl.load(direct + index, live, 0)
    future = tl.where(tl.load(next_resets + row, live, 0), 0., future)
    total = tl.load(gradient + index, live, 0).to(tl.float32) + future
    dc = total * (1 - update) * (1 - candidate * candidate)
    dz = total * (prior - candidate) * update * (1 - update)
    dr = dc * recurrent * reset * (1 - reset)

    offset = row * (3 * width) + column
    tl.store(input_gradient + offset, dr, live)
    tl.store(input_gradient + offset + width, dz, live)
    tl.store(input_gradient + offset + 2 * width, dc, live)
    tl.store(hidden_gradient + offset, dr, live)
    tl.store(hidden_gradient + offset + width, dz, live)
    tl.store(hidden_gradient + offset + 2 * width, dc * reset, live)
    tl.store(direct + index, total * update, live)
    tl.store(weight_inputs + index, prior, live)


@torch.library.custom_op("goddard::gru_forward", mutates_args=())
def gru_forward(
    projected: torch.Tensor,
    initial:   torch.Tensor,
    resets:    torch.Tensor,
    weight:    torch.Tensor,
    bias:      torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque to Inductor; CUDA graphs capture the fixed-size recurrent loop."""
    projected = projected.contiguous()
    initial = initial.contiguous()
    resets = resets.contiguous()
    length, batch, _ = projected.shape
    width = initial.shape[-1]
    states = initial.new_empty((length, batch, width))
    gates = initial.new_empty((length, batch, 4 * width))
    carry = initial.masked_fill(resets[0, :, None], 0).to(projected.dtype)
    grid = (triton.cdiv(batch * width, 256),)

    for step in range(length):
        recurrent = F.linear(carry, weight, bias)
        previous = initial if step == 0 else states[step - 1]
        _forward_gate[grid](
            projected[step], recurrent, previous, resets[step],
            resets[min(step + 1, length - 1)], states[step], gates[step], carry,
            width, batch * width, 256, enable_fp_fusion=False
        )

    return states, gates


@gru_forward.register_fake
def _fake_forward(
    projected: torch.Tensor,
    initial:   torch.Tensor,
    resets:    torch.Tensor,
    weight:    torch.Tensor,
    bias:      torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    length, batch, _ = projected.shape
    width = initial.shape[-1]

    return (
        initial.new_empty((length, batch, width)),
        initial.new_empty((length, batch, 4 * width))
    )


@torch.library.custom_op("goddard::gru_backward", mutates_args=())
def gru_backward(
    gradient: torch.Tensor,
    initial:  torch.Tensor,
    states:   torch.Tensor,
    gates:    torch.Tensor,
    resets:   torch.Tensor,
    weight:   torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length, batch, width = states.shape
    initial = initial.contiguous()
    resets = resets.contiguous()
    gradient = gradient.contiguous()
    inputs = weight.new_empty((length, batch, 3 * width))
    hidden = torch.empty_like(inputs)
    previous = weight.new_empty((length, batch, width))
    direct = torch.zeros_like(initial)
    carry = weight.new_zeros((batch, width))
    grid = (triton.cdiv(batch * width, 256),)

    for step in reversed(range(length)):
        prior = initial if step == 0 else states[step - 1]
        _backward_gate[grid](
            gradient[step], carry, direct, gates[step], prior, resets[step],
            resets[min(step + 1, length - 1)], inputs[step], hidden[step],
            previous[step], width, batch * width, 256, enable_fp_fusion=False
        )
        carry = F.linear(hidden[step], weight.T)

    initial_gradient = (direct + carry.float()).masked_fill(
        resets[0, :, None], 0
    )
    hidden = hidden.flatten(0, 1)

    return (
        inputs,
        initial_gradient,
        hidden.T @ previous.flatten(0, 1),
        hidden.sum(0)
    )


@gru_backward.register_fake
def _fake_backward(
    gradient: torch.Tensor,
    initial:  torch.Tensor,
    states:   torch.Tensor,
    gates:    torch.Tensor,
    resets:   torch.Tensor,
    weight:   torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        weight.new_empty((*states.shape[:2], weight.shape[0])),
        initial.new_empty(initial.shape),
        weight.new_empty(weight.shape),
        weight.new_empty(weight.shape[0])
    )
