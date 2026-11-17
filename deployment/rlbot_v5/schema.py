"""Observation layouts supported by the standalone RLBot adapter."""

DEFAULT_SCHEMA = "ego-controller-v3"
LEGACY_SCHEMA = "ego-controller-v1"


def observation_size(schema, previous_controls=False):
    if schema == DEFAULT_SCHEMA:
        return 144 + 8 * bool(previous_controls)
    if schema == LEGACY_SCHEMA and not previous_controls:
        return 166
    raise ValueError(f"unsupported RLBot observation layout: {schema}, controls={previous_controls}")
