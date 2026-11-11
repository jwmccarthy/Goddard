"""Offline aerial-phase annotations and bounded, causal parse-time candidates.

Annotations can see the future; stored discriminator scenes cannot. Labels are
heuristics, not judgments of expert intent or proof that an aerial was useful.
"""
import numpy as np

from replays.corpus import stable_seed
from replays.layout import INTERNAL_SIZE, team_observation_size

PHASES = ("representative", "setup", "takeoff", "execution", "recovery")
MIXTURE = (.40, .20, .15, .15, .10)
RESET_MIXTURE = (.50, .40, .05, .00, .05)
RESET_PROFILES = {"setup": RESET_MIXTURE, "midair": (.20, .20, .10, .40, .10),
                  "balanced": (4 / 15, 6 / 15, 1 / 15, 3 / 15, 1 / 15)}
HISTORY = 22
FORMAT = "aerial-selected-v1"


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def quality_mask(rows, fps=30):
    bad = ~np.isfinite(rows).all(axis=-1) | (rows[:, -2:] > .5).any(axis=-1)
    afk = np.zeros(len(rows), bool)
    for car in (9, 30):
        position = rows[:, car:car + 3] * (4108, 6000, 2076)
        ball = rows[:, :3] * (4108, 6000, 2076)
        stationary = (np.linalg.norm(rows[:, car + 3:car + 6] * 2300, axis=-1) < 50)
        stationary &= rows[:, car + 16] > .5
        stationary &= np.linalg.norm(ball - position, axis=-1) > 1000
        for lo, hi in runs(stationary & ~bad):
            if hi - lo >= 8 * fps and np.ptp(position[lo:hi], axis=0).max() < 100:
                afk[lo:hi] = True
    return bad | afk, int(afk.sum())


def annotate(rows, bad, fps=30):
    """Disjoint targeted phases plus an overlapping unconditional component."""
    n = len(rows)
    phase = np.full(n, -1, np.int8)
    event = np.full(n, -1, np.int32)
    origin = np.zeros(n, np.int8)  # 0 unknown/opportunity, 1 floor, 2 wall
    decision = np.zeros(n, np.int8)  # 0 ordinary, 1 attempted, 2 no detected takeoff
    touch = np.zeros(n, np.int8)  # detected replay-inferred touch, not ground truth
    pos = rows[:, 9:12] * (4108, 6000, 2076)
    ball = rows[:, :3] * (4108, 6000, 2076)
    ball_vel = rows[:, 3:6] * 6000
    distance = np.linalg.norm(ball - pos, axis=-1)
    other_distance = np.linalg.norm(ball - rows[:, 30:33] * (4108, 6000, 2076), axis=-1)
    role = np.zeros(n, np.int8)  # neutral/offensive
    role[(ball[:, 1] < -1000) | ((ball_vel[:, 1] < -400) & (ball[:, 1] < 1000))] = 1  # defense
    role[(np.abs(distance - other_distance) < 500) & (other_distance < 1600)] = 2  # contested
    grounded = rows[:, 25] > .5
    demo = rows[:, 26] > .5
    serial = 0
    events = []
    for valid_lo, valid_hi in runs(~bad & ~demo):
        airborne = ~grounded[valid_lo:valid_hi]
        for a, b in runs(airborne):
            lo, hi = valid_lo + a, valid_lo + b
            if hi - lo < int(.3 * fps) or pos[lo:hi, 2].max() < 300 or distance[lo:hi].min() > 1500:
                continue
            if lo == valid_lo:  # Already flying at a true segment boundary: no invented takeoff.
                start = lo
                takeoff_end = lo
            else:
                start = max(valid_lo, lo - int(1.5 * fps))
                takeoff_end = min(hi, lo + int(.35 * fps))
            contacts = np.flatnonzero(rows[lo:hi, -5] > .5) + lo
            closest = lo + int(distance[lo:hi].argmin())
            last_execution = int(contacts[-1]) if len(contacts) else closest
            execution_end = min(hi, max(takeoff_end + 1, last_execution + int(.2 * fps) + 1))
            end = min(valid_hi, hi + int(.8 * fps))
            serial += 1
            # Later attempts own their lead-in when neighborhoods overlap.
            phase[start:lo] = 1
            phase[lo:takeoff_end] = 2
            phase[takeoff_end:execution_end] = 3
            phase[execution_end:end] = 4
            event[start:end] = serial
            wall = grounded[max(valid_lo, lo - 1)] and pos[max(valid_lo, lo - 1), 2] > 180
            origin[start:end] = 2 if wall else (1 if lo > valid_lo else 0)
            decision[start:end] = 1
            touch[start:end] = int(bool(len(contacts)))
            events.append({"event": serial, "start": int(start), "takeoff": int(lo), "end": int(end),
                           "origin": "wall" if wall else "floor" if lo > valid_lo else "segment_airborne",
                           "touch_observed": bool(len(contacts)), "touch_provenance": "replay-inferred",
                           "goal_outcome": "see_segment_metadata"})
        # Comparable grounded/wall opportunities, including stays/aborts. No
        # success filter; boost-poor defensive opportunities are also retained.
        predicted_height = ball[:, 2] + ball_vel[:, 2] * .6 - .5 * 650 * .6 ** 2
        opportunity = grounded & ~demo & (distance < 2200) & (np.maximum(ball[:, 2], predicted_height) > 300)
        opportunity &= phase < 0
        for a, b in runs(opportunity[valid_lo:valid_hi]):
            a, b = a + valid_lo, b + valid_lo
            if b - a < 3:
                continue
            serial += 1
            phase[a:b], event[a:b], decision[a:b] = 1, serial, 2
    return {"phase": phase, "event": event, "origin": origin, "decision": decision, "touch": touch, "role": role}, events


def valid_endings(rows, bad, history=HISTORY, goal=False):
    ends = np.arange(1, len(rows))
    beginnings = np.maximum(0, ends - history + 1)
    cumulative = np.r_[0, np.cumsum(bad)]
    good = cumulative[ends + 1] == cumulative[beginnings]
    if goal:
        good[-1] = False
    return ends[good]


def diverse_choices(items, limit, rng, spacing=HISTORY):
    """Round-robin player/role/event groups; no long-flight duration weighting."""
    groups = {}
    for item in items:
        groups.setdefault((item["player"], item["period"], item["role"], item["event"]), []).append(item)
    keys = list(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    chosen, last = [], {}
    while keys and len(chosen) < limit:
        remaining = []
        for key in keys:
            group = groups[key]
            while group:
                item = group.pop()
                identity = (item["player"], item["period"])
                if all(abs(item["end"] - old) >= spacing for old in last.get(identity, [])):
                    chosen.append(item)
                    last.setdefault(identity, []).append(item["end"])
                    break
            if group:
                remaining.append(key)
            if len(chosen) == limit:
                break
        keys = remaining
    return chosen


def select_match(segments, source, seed, per_phase=32, history=HISTORY, reset_profile="setup", reset_selection="events",
                 reset_only=False):
    """Only this match's dense rows exist in memory. Persist bounded candidates."""
    reset_mixture = RESET_PROFILES[reset_profile]
    if reset_selection not in ("events", "diverse"):
        raise ValueError("unknown reset selection strategy")
    rng = np.random.default_rng(stable_seed(seed, source["match_key"], "candidates"))
    pools = [[] for _ in PHASES]
    reset_pools = [[] for _ in PHASES]
    terminals, event_records = [], []
    valid_total = goals_total = afk_total = 0
    by_period = {}
    for rows, metadata in segments:
        by_period.setdefault(metadata["period"], []).append((rows, metadata))
    for period, pair in sorted(by_period.items()):
        quality = [quality_mask(rows) for rows, _ in pair]
        common_bad = None
        if len(pair) == 2 and pair[0][0].shape == pair[1][0].shape:
            common_bad = quality[0][0] | quality[1][0]
        for actor, (rows, metadata) in enumerate(pair):
            if source.get("expert_ego_ids") is not None and metadata["ego_id"] not in source["expert_ego_ids"]:
                continue  # Counterpart is still available for exact reset reconstruction.
            bad = quality[actor][0].copy() if common_bad is None else common_bad.copy()
            afk_total += quality[actor][1]
            goal = metadata["goal_terminal"]
            if len(rows) < 2:
                continue
            labels, events = annotate(rows, bad)
            player = next((p for p in source["players"] if str(p["id"]["id"]).lower() == metadata["online_id"].lower()), None)
            if player is None:
                raise ValueError(f"unmapped replay account {metadata['online_id']}")
            identity = f"{player['id']['platform']}:{player['id']['id']}"
            event_records.extend({**e, "player": identity, "period": period, "goal_outcome": metadata["goal_outcome"]} for e in events)
            ends = valid_endings(rows, bad, history, goal)
            valid_total += len(ends)
            for component in (() if reset_only else range(5)):
                indices = ends if component == 0 else ends[labels["phase"][ends] == component]
                for end in indices:
                    pools[component].append({"rows": rows, "end": int(end), "period": period, "player": identity,
                        "ego_id": metadata["ego_id"], "component": component,
                        "event": int(end // 60) if component == 0 else int(labels["event"][end]),
                        **{name: int(labels[name][end]) for name in ("role", "origin", "decision", "touch")}})
            terminal_bad = bad.copy()
            terminal_bad[-1] = False  # Goal frame is replaced, as in ExpertDataset.
            if goal and not terminal_bad[max(0, len(rows) - history):].any():
                goals_total += 1
                if not reset_only:
                    terminals.append({"rows": rows, "end": len(rows) - 1, "period": period, "player": identity,
                                  "ego_id": metadata["ego_id"], "component": 5, "event": -1,
                                  "role": 0, "origin": 0, "decision": 0, "touch": 0})
            # Exact full internal state from the aligned counterpart POV.
            if common_bad is None:
                continue
            other = pair[1 - actor][0]
            expected = rows[:, 30:51].copy()
            for start in (0, 3, 6, 9, 12):
                expected[:, start:start + 2] *= -1
            aligned = np.isclose(expected, other[:, 9:30], rtol=1e-4, atol=1e-4).all(axis=-1)
            internal_start = team_observation_size(1)
            internal = np.stack((rows[:, internal_start:internal_start + INTERNAL_SIZE],
                                 other[:, internal_start:internal_start + INTERNAL_SIZE]), axis=1)
            eligible = ~bad & aligned & np.isfinite(internal).all(axis=(1, 2))
            eligible &= (np.abs(rows[:, 1] * 6000) < 5100) & (rows[:, 26] < .5) & (rows[:, 47] < .5)
            cars = rows[:, 9:51].reshape(-1, 2, 21)
            eligible &= (np.linalg.norm(np.cross(cars[..., 12:15], cars[..., 9:12]), axis=-1) > .9).all(axis=-1)
            if goal:
                eligible[-1] = False
            if reset_profile in ("midair", "balanced"):
                # Avoid spawning into an already active contact/penetration.
                # Preserve jump/flip internals: mechanical phases are not errors.
                positions = cars[..., :3] * (4108, 6000, 2076)
                ball = rows[:, None, :3] * (4108, 6000, 2076)
                eligible &= (np.linalg.norm(ball - positions, axis=-1) > 180).all(axis=-1)
                eligible &= np.linalg.norm(positions[:, 0] - positions[:, 1], axis=-1) > 250
                eligible &= (rows[:, -5:-2] <= .5).all(axis=-1)
            if reset_selection == "diverse":
                from replays.reset_diversity import reset_bins

                spatial, context = reset_bins(rows[:, :51])
            for component, fraction in enumerate(reset_mixture):
                if not fraction:
                    continue
                candidate = eligible & ((labels["phase"] == component) if component else True)
                if reset_profile in ("midair", "balanced"):
                    if component in (2, 3, 4):
                        candidate &= (rows[:, 25] < .5) & (rows[:, 11] * 2076 > 182.5)
                        if component == 3:
                            candidate &= np.linalg.norm((rows[:, :3] - rows[:, 9:12]) * (4108, 6000, 2076), axis=-1) < 1500
                    elif component == 1:
                        candidate &= rows[:, 25] > .5
                    elif reset_profile == "balanced":
                        candidate &= (rows[:, 25] > .5) & (rows[:, 11] * 2076 < 120) & (rows[:, 23] > .65)
                indices = np.flatnonzero(candidate)
                for end in indices:
                    item = {"rows": rows, "internal": internal, "end": int(end), "period": period,
                        "player": identity, "component": component, "role": int(labels["role"][end]),
                        "origin": int(labels["origin"][end]),
                        "event": int(end // 60) if component == 0 else int(labels["event"][end])}
                    if reset_selection == "diverse":
                        item.update(spatial=tuple(spatial[end]), context=tuple(context[end]))
                    reset_pools[component].append(item)
    limits = (per_phase * 4, per_phase * 2, per_phase * 2, per_phase * 2, per_phase * 2)
    selected = [item for pool, limit in zip(pools, limits) for item in diverse_choices(pool, limit, rng, history)]
    # Preserve both scored/conceded examples, bounded to one original goal per POV.
    terminal_players = sorted({item["player"] for item in terminals})
    for player in terminal_players:
        choices = [item for item in terminals if item["player"] == player]
        selected.append(choices[int(rng.integers(len(choices)))])
    if reset_selection == "diverse":
        from replays.reset_diversity import diverse_reset_choices

        reset_rng = np.random.default_rng(stable_seed(seed, source["match_key"], reset_profile, "diverse-resets"))
        resets = [item for pool, limit in zip(reset_pools, (768, 768, 384, 1536, 768))
                  for item in diverse_reset_choices(pool, limit, reset_rng)]
    else:
        reset_limits = (96, 64, 64, 128, 64) if reset_profile in ("midair", "balanced") else (96, 64, 32, 0, 32)
        resets = [item for pool, limit in zip(reset_pools, reset_limits)
                  for item in diverse_choices(pool, limit, rng, history)]
    windows = np.full((len(selected), history, 51), np.nan, np.float32)
    lengths = np.empty(len(selected), np.int16)
    for index, item in enumerate(selected):
        length = min(history, item["end"] + 1)
        windows[index, -length:] = item["rows"][item["end"] - length + 1:item["end"] + 1, :51]
        lengths[index] = length
    payload = {"windows": windows, "lengths": lengths}
    for name in ("component", "end", "period", "event", "role", "origin", "decision", "touch"):
        payload[name] = np.asarray([item[name] for item in selected], np.int32)
    payload["players"] = np.asarray([item["player"] for item in selected])
    payload["ego_ids"] = np.asarray([item["ego_id"] for item in selected])
    payload["reset_scenes"] = np.asarray([item["rows"][item["end"], :51] for item in resets], np.float32).reshape(-1, 51)
    payload["reset_internal"] = np.asarray([item["internal"][item["end"]] for item in resets], np.float32).reshape(-1, 2, 19)
    for name in ("component", "end", "period"):
        payload["reset_" + name] = np.asarray([item[name] for item in resets], np.int32)
    payload["reset_players"] = np.asarray([item["player"] for item in resets])
    if reset_selection == "diverse":
        from replays.reset_diversity import reset_signature

        for name in ("event", "role", "origin", "spatial", "context"):
            payload["reset_" + name] = np.asarray([item[name] for item in resets], np.int16 if name in ("spatial", "context") else np.int32)
        payload["reset_signature"] = np.asarray([reset_signature(scene, state) for scene, state in
            zip(payload["reset_scenes"], payload["reset_internal"])], dtype='S32')
    return payload, {"valid_live_endings": valid_total, "valid_goals": goals_total, "afk_rows_excluded": afk_total,
                     "events": event_records, "candidates": len(selected), "reset_candidates": len(resets)}
