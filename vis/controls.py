import torch


def axis(value: int) -> int:
    return -1 if value == 1 else 1 if value == 2 else 0


def control_values(
    actions:  torch.Tensor,
    discrete: bool = True
) -> list[dict[str, int | bool | float]]:
    """Decode CARL's seven action fields for display, in car order."""
    values = actions.detach().cpu().tolist()
    result: list[dict[str, int | bool | float]] = []

    for horizontal, vertical, throttle, slide, boost, roll, jump in values:
        if discrete:
            horizontal, vertical, throttle, roll = map(
                axis,
                (horizontal, vertical, throttle, roll)
            )

        result.append(
            {
                "horizontal": horizontal,
                "vertical":   vertical,
                "throttle":   throttle,
                "roll":       roll,
                "slide":      slide >= 0.5,
                "boost":      boost >= 0.5,
                "jump":       jump >= 0.5
            }
        )

    return result
