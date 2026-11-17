"""Checkpoint layout selection and CARL schema compatibility."""

DEFAULT_SCHEMA = "ego-controller-v3"


def observation_options(config, *, legacy=False):
    schema = config.get("observation_schema", "base-v1" if legacy else DEFAULT_SCHEMA) or DEFAULT_SCHEMA
    controls = bool(config.get("previous_controls", False))
    if schema not in ("base-v1", "ego-controller-v1", DEFAULT_SCHEMA):
        raise ValueError(f"unsupported observation schema: {schema}")
    if controls and schema != DEFAULT_SCHEMA:
        raise ValueError("--previous-controls applies only to ego-controller-v3")
    return schema, controls


def observation_size(schema, n_cars, previous_controls=False):
    observation_options(dict(observation_schema=schema, previous_controls=previous_controls))
    extra = {"base-v1": 0, "ego-controller-v1": 27, DEFAULT_SCHEMA: 5}[schema]
    return 85 + 27 * n_cars + extra + 8 * previous_controls


def resolve_observation_arguments(args):
    previous = None
    if getattr(args, "resume", None):
        import torch

        previous = observation_options(
            torch.load(args.resume, map_location="cpu", weights_only=False)["config"], legacy=True,
        )
    if getattr(args, "observation_schema", None) is None:
        args.observation_schema = previous[0] if previous else DEFAULT_SCHEMA
    if getattr(args, "previous_controls", None) is None:
        args.previous_controls = previous[1] if previous else False
    current = observation_options(vars(args))
    if previous is not None and current != previous:
        raise ValueError("checkpoint observation schema/previous-controls setting differs; start a new run")
    return current
