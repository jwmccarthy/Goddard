import operator

import torch


def balanced_window_batch(generated, expert, *, absorbing=False):
    """Equalize paired source masks by hiding history, never inventing frames.

    Union masks handle either source having shorter causal history. Draw/pair
    windows before this operation, including any historical generated samples;
    keep it outside microbatching. Stored windows and reward inference stay raw.
    """
    if generated.ndim != 3 or generated.shape != expert.shape:
        raise ValueError("padding matching requires equal source window shapes")
    padding = torch.isnan(generated[..., 0]) | torch.isnan(expert[..., 0])
    original = torch.cat((generated, expert))
    matched = original.masked_fill(torch.cat((padding, padding))[..., None], float("nan"))
    if absorbing:
        # Preserve terminal/loop bookkeeping for diagnostics. Padded tokens are
        # ignored by every head, including their absorbing embeddings.
        matched[..., -1] = original[..., -1]
    return matched


def decision_history_length(window_length, decision_stride=1):
    """Raw decision observations spanned by a strided discriminator window."""
    try:
        window_length = operator.index(window_length)
        decision_stride = operator.index(decision_stride)
    except TypeError:
        raise ValueError("discriminator window length and decision stride must be integers") from None
    if window_length < 2:
        raise ValueError("discriminator window length must be at least 2")
    if decision_stride < 1:
        raise ValueError("discriminator decision stride must be positive")
    return (window_length - 1) * decision_stride + 1
