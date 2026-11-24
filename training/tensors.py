import logging
import warnings
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch._inductor import config


def plain_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if type(tensor) is torch.Tensor else tensor.as_subclass(torch.Tensor)


def autocast(device: torch.device, enabled: bool) -> AbstractContextManager[None]:
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled)


def compilation_mode(*, cuda_graphs: bool = True) -> str:
    return "max-autotune" if cuda_graphs else "max-autotune-no-cudagraphs"


def require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(
            f"non-finite {name}; training stopped before corrupting a checkpoint"
        )


def configure_compilation_warnings() -> None:
    config.autotune_num_choices_displayed = 0
    config.max_autotune_report_choices_stats = False
    logger = logging.getLogger("torch._inductor.select_algorithm")

    if _autotune_log_filter not in logger.filters:
        logger.addFilter(_autotune_log_filter)

    warnings.filterwarnings(
        "ignore",
        message=r"TypedStorage is deprecated\..*",
        category=UserWarning,
        module=r"torch\._inductor\.select_algorithm"
    )

    # This is Inductor choosing a correct split reduction, not a graph break,
    # recompile or numerical error. Keep all other compiler warnings visible.
    warnings.filterwarnings(
        "ignore",
        message=(
            r"(?s).*Online softmax is disabled on the fly since Inductor "
            r"decides to.*split the reduction.*"
        ),
        category=UserWarning,
        module=r"torch\._inductor\.lowering"
    )


def _autotune_log_filter(record: Any) -> bool:
    # Inductor logs rejected candidates as errors while continuing the search.
    return not (
        "Ignoring this choice." in record.getMessage()
        or record.msg == "Exception %s for benchmark choice %s"
    )
