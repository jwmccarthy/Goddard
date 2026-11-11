import torch

from vis.controls import control_values


def test_discrete_controls_decode_carl_action_order():
    actions = torch.tensor([[1, 2, 0, 1, 0, 2, 1], [2, 0, 1, 0, 1, 0, 0]])
    assert control_values(actions) == [
        {"horizontal": -1, "vertical": 1, "throttle": 0, "roll": 1,
         "slide": True, "boost": False, "jump": True},
        {"horizontal": 1, "vertical": 0, "throttle": -1, "roll": 0,
         "slide": False, "boost": True, "jump": False},
    ]


def test_continuous_controls_keep_axes_and_button_thresholds():
    values = control_values(torch.tensor([[0.25, 0, -1, 0.49, 0.5, 1, 0]]), discrete=False)[0]
    assert values == {"horizontal": 0.25, "vertical": 0, "throttle": -1, "roll": 1,
                      "slide": False, "boost": True, "jump": False}
