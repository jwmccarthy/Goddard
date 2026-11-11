"""Conservative, observation-only outcome annotations for complete duel periods.

Future rows qualify events, never become privileged model inputs. ``success``
means that a declared rule passed, not that replay-inferred contacts or intent
are ground truth. Unsupported/uncertain outcomes fail closed.
"""
from dataclasses import asdict, dataclass

import numpy as np

from replays.aerial import runs
from replays.layout import team_replay_row_size

SUCCESS_FORMAT = "successful-selected-v1"
OUTCOME_VERSION = "successful-events-v1"
SKILLS = ("goal_touch", "save", "air_dribble")
PHASES = ("setup", "takeoff", "execution", "recovery")
POSITION_SCALE = np.asarray((4108., 6000., 2076.))
BALL_RADIUS = 91.25
GOAL_Y = 5120.
GOAL_HALF_WIDTH = 892.755
GOAL_HEIGHT = 642.775


@dataclass(frozen=True)
class OutcomeConfig:
    fps: int = 30
    goal_horizon: float = 3.
    save_horizon: float = 2.
    setup_seconds: float = 1.5
    recovery_seconds: float = .8
    max_event_seconds: float = 12.
    touch_gap_seconds: float = .1
    contact_distance: float = 250.
    prediction_margin: float = 25.
    clearance_distance: float = 500.
    carry_seconds: float = .75
    carry_distance: float = 350.
    carry_relative_speed: float = 800.
    carry_progress: float = 300.
    possession_seconds: float = .3
    possession_distance: float = 450.

    def __post_init__(self):
        if self.fps != 30:
            raise ValueError("outcome selection currently requires dense 30 Hz duel observations")
        for name, value in asdict(self).items():
            minimum = 0 if name in ("setup_seconds", "recovery_seconds") else np.nextafter(0., 1.)
            if not np.isfinite(value) or value < minimum:
                raise ValueError(f"outcome {name} must be finite and {'nonnegative' if minimum == 0 else 'positive'}")
        if self.prediction_margin >= min(GOAL_HALF_WIDTH - BALL_RADIUS, (GOAL_HEIGHT - 2 * BALL_RADIUS) / 2):
            raise ValueError("prediction margin leaves no goal aperture")

    def frames(self, seconds):
        return int(np.ceil(seconds * self.fps))


def contact_clusters(mask, gap=3):
    """Merge nearby flagged rows into one contact episode, not many touches."""
    result = []
    for lo, hi in runs(np.asarray(mask, bool)):
        if result and lo - result[-1][1] < gap:
            result[-1] = (result[-1][0], hi)
        else:
            result.append((lo, hi))
    return result


def goal_threat(position, velocity, direction=-1, horizon=2., margin=25.):
    """Predict a direct goal without car contacts, bounces or arena collisions.

    This deliberately declines bank/floor/wall shots rather than approximating
    unmodelled bounces. It is a conservative labeler, not a full arena simulator.
    Inputs are world units in the ego-normalized field orientation.
    """
    position, velocity = np.asarray(position, float), np.asarray(velocity, float)
    if (position.shape != (3,) or velocity.shape != (3,) or direction not in (-1, 1)
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()
            or not np.isfinite(horizon) or horizon <= 0 or not np.isfinite(margin) or margin < 0):
        raise ValueError("invalid direct-goal prediction inputs")
    if (direction * velocity[1] <= 0 or abs(position[1]) >= GOAL_Y
            or abs(position[0]) >= 4096 - BALL_RADIUS or not BALL_RADIUS < position[2] < 2044 - BALL_RADIUS):
        return None
    dt = 1 / 120
    decay = .97 ** dt
    previous = position.copy()
    for step in range(int(np.ceil(horizon / dt))):
        next_velocity = velocity * decay
        next_velocity[2] -= 650 * dt
        current = previous + .5 * (velocity + next_velocity) * dt
        # Check entry through the aperture AND the whole ball crossing the line.
        # Merely reaching y=5120 could still hit the floor/post before scoring.
        for depth in (GOAL_Y, GOAL_Y + BALL_RADIUS):
            if direction * previous[1] < depth <= direction * current[1]:
                fraction = (direction * depth - previous[1]) / (current[1] - previous[1])
                crossing = previous + fraction * (current - previous)
                if not (abs(crossing[0]) < GOAL_HALF_WIDTH - BALL_RADIUS - margin
                        and BALL_RADIUS + margin < crossing[2] < GOAL_HEIGHT - BALL_RADIUS - margin):
                    return None
                if depth > GOAL_Y:
                    crossing_time = (step + fraction) * dt
                    return crossing_time if crossing_time <= horizon else None
        if (current[2] <= BALL_RADIUS or current[2] >= 2044 - BALL_RADIUS
                or abs(current[0]) >= 4096 - BALL_RADIUS):
            return None
        previous, velocity = current, next_velocity
    return None


def annotate_outcomes(rows, metadata, config=None):
    """Return auditable success/failed/unknown candidates; only successes qualify.

    Row indices are relative to one original player POV/gameplay period. A goal
    marker is evidence only for a nearby uncontested last ego touch. Conceded
    goals and period-level outcomes never qualify arbitrary earlier attempts.
    """
    config = config or OutcomeConfig()
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != team_replay_row_size(1):
        raise ValueError("outcome annotation requires complete dense duel replay rows, not compact scenes")
    terminal = metadata.get("goal_terminal", False)
    outcome = metadata.get("goal_outcome")
    if not isinstance(terminal, (bool, np.bool_)) or (terminal and outcome not in ("scored", "conceded")):
        raise ValueError("goal terminal requires a known ego-relative scored/conceded outcome")
    if not terminal and outcome is not None:
        raise ValueError("nonterminal period cannot claim a goal outcome")
    n = len(rows)
    live_end = n - int(terminal)
    if live_end < 2:
        return []
    # Do not inherit the old AFK/phase-selection heuristics. Hard invalidity and
    # explicit reconstruction flags are checked over context AND verification.
    bad = ~np.isfinite(rows).all(axis=1) | (rows[:, -2:] > .5).any(axis=1)
    bad |= (rows[:, (26, 47)] > .5).any(axis=1)
    if terminal:
        bad[-1] = False  # Only the recorded goal marker is used, never its physics.
    ball = rows[:, :3] * POSITION_SCALE
    ball_velocity = rows[:, 3:6] * 6000
    car = rows[:, 9:12] * POSITION_SCALE
    car_velocity = rows[:, 12:15] * 2300
    opponent = rows[:, 30:33] * POSITION_SCALE
    distance = np.linalg.norm(ball - car, axis=1)
    relative_speed = np.linalg.norm(ball_velocity - car_velocity, axis=1)
    other_touch = rows[:, -4] > .5
    touches = contact_clusters(rows[:live_end, -5] > .5, config.frames(config.touch_gap_seconds))
    flights = runs((rows[:live_end, 25] < .5) & ~bad[:live_end])
    events = []

    def record(kind, contact, status, reason, **evidence):
        event = {"kind": kind, "contact_start": int(contact[0]), "contact_end": int(contact[1]),
                 "status": status, "reason": reason, "evidence": evidence,
                 "touch_provenance": "replay-inferred", "outcome_version": OUTCOME_VERSION}
        events.append(event)
        return event

    def flight_for(contact):
        return next(((lo, hi) for lo, hi in flights if lo < contact[1] and contact[0] < hi), None)

    def observed_launch(flight):
        return flight is None or (flight[0] > 0 and not bad[flight[0] - 1] and rows[flight[0] - 1, 25] > .5)

    def clean(lo, hi):
        return 0 <= lo < hi <= n and not bad[lo:hi].any()

    def observed_through(frame):
        return frame <= live_end or (terminal and outcome == "scored")

    def finish(event, contact, flight, action_end, verification_end, outcome_frame, goal=False):
        action_start = flight[0] if flight is not None else contact[0]
        if not observed_launch(flight):
            event.update(status="unknown", reason="unobserved_takeoff")
            return
        start = max(0, action_start - config.frames(config.setup_seconds))
        # No preceding unrelated flight or opponent possession in the lead-in.
        for lo, hi in flights:
            if hi <= action_start:
                start = max(start, hi)
        previous_opponent = np.flatnonzero(other_touch[:action_start])
        if len(previous_opponent):
            start = max(start, int(previous_opponent[-1]) + 1)
        previous_bad = np.flatnonzero(bad[:action_start])
        if len(previous_bad):
            start = max(start, int(previous_bad[-1]) + 1)
        end = live_end if goal else action_end + config.frames(config.recovery_seconds)
        # A later attempt is not recovery from this successful event.
        for lo, hi in flights:
            if lo >= action_end and (flight is None or lo != flight[0]):
                end = min(end, lo)
                break
        required = max(end, verification_end)
        if not observed_through(required):
            event.update(status="unknown", reason="censored_recovery_or_outcome")
            return
        end, required = min(end, live_end), min(required, live_end)
        if not clean(start, required) or end < action_end or end <= start:
            event.update(status="unknown", reason="quality_gap_or_incomplete_event")
            return
        if end - start > config.frames(config.max_event_seconds):
            event.update(status="unknown", reason="event_exceeds_clip_budget")
            return
        if other_touch[contact[0]:required].any():
            event.update(status="unknown", reason="contested_verification")
            return
        execution_start = action_start
        if flight is not None:
            # A clustered contact can begin on the ground and continue through
            # departure. It must not create a negative-duration takeoff phase.
            execution_start = max(action_start, min(contact[0], action_start + config.frames(.35)))
        execution_end = min(end, max(execution_start + 1, action_end))
        event.update(start=int(start), end=int(end), outcome_frame=int(outcome_frame),
                     verification_start=int(contact[0]), verification_end=int(required),
                     terminal=bool(goal and end == live_end),
                     phases={"setup": [int(start), int(action_start)],
                             "takeoff": [int(action_start), int(execution_start)],
                             "execution": [int(execution_start), int(execution_end)],
                             "recovery": [int(execution_end), int(end)]},
                     confidence="rule-verified; heuristic contacts, not ground truth")

    goal_event = None
    if touches and terminal and outcome == "scored":
        contact = touches[-1]
        gap = (live_end - contact[1] + 1) / config.fps
        event = record("goal_touch", contact, "success", "uncontested_last_touch_to_scored_goal", goal_delay=gap)
        if gap > config.goal_horizon:
            event.update(status="failed", reason="goal_outside_credit_horizon")
        elif distance[contact[0]:contact[1]].min() > config.contact_distance:
            event.update(status="unknown", reason="implausible_contact_geometry")
        elif other_touch[contact[0]:n].any():
            event.update(status="unknown", reason="opponent_touch_or_deflection_before_goal")
        else:
            flight = flight_for(contact)
            action_end = min(live_end, max(contact[1] + config.frames(.2), flight[1] if flight else 0))
            finish(event, contact, flight, action_end, live_end, live_end, goal=True)
            if event["status"] == "success":
                goal_event = event

    for contact in touches:
        lo, hi = contact
        # Use the pre-contact row, never the outgoing impulse as the threat.
        if lo < 2 or not clean(lo - 2, hi + 1) or distance[lo:hi].min() > config.contact_distance:
            continue
        incoming = goal_threat(ball[lo - 2], ball_velocity[lo - 2], horizon=config.save_horizon,
                               margin=config.prediction_margin)
        if incoming is None:
            continue
        event = record("save", contact, "success", "direct_goal_threat_cleared", incoming_goal_seconds=incoming,
                       prediction="no-bounce-drag-gravity-v1")
        verification_end = hi + config.frames(config.save_horizon)
        if terminal and outcome == "conceded" and live_end <= verification_end:
            event.update(status="failed", reason="conceded_during_save_verification")
            continue
        if not observed_through(verification_end):
            event.update(status="unknown", reason="censored_save_outcome")
            continue
        verify_stop = min(verification_end, live_end)
        outgoing = min(hi + 1, live_end - 1)
        if (not clean(lo - 2, verify_stop) or other_touch[lo:verify_stop].any()):
            event.update(status="unknown", reason="contested_or_invalid_save_outcome")
            continue
        if goal_threat(ball[outgoing], ball_velocity[outgoing], horizon=config.save_horizon,
                       margin=0) is not None or ball_velocity[outgoing, 1] < 250:
            event.update(status="failed", reason="no_verified_clearance")
            continue
        progress = ball[outgoing:verify_stop, 1].max() - ball[lo - 2, 1]
        if progress < config.clearance_distance:
            event.update(status="failed", reason="insufficient_clearance")
            continue
        flight = flight_for(contact)
        action_end = max(hi + config.frames(.2), flight[1] if flight else 0)
        linked_goal = (goal_event is not None and goal_event["status"] == "success"
                       and goal_event["contact_start"] == lo)
        finish(event, contact, flight, min(action_end, live_end) if linked_goal else action_end,
               verification_end, live_end if linked_goal else verify_stop - 1, goal=linked_goal)
        if event["status"] == "success" and linked_goal:
            event["evidence"]["also_goal_touch"] = True
            goal_event.update(status="superseded", reason="covered_by_successful_save")

    for flight in flights:
        lo, hi = flight
        contacts = [contact for contact in touches if lo <= contact[0] < hi]
        if not contacts or car[lo:hi, 2].max() < 200:
            continue
        controlled = ((car[lo:hi, 2] > 200) & (ball[lo:hi, 2] > 300)
                      & (distance[lo:hi] <= config.carry_distance)
                      & (relative_speed[lo:hi] <= config.carry_relative_speed)
                      & (ball[lo:hi, 2] - car[lo:hi, 2] > 20)
                      & (ball[lo:hi, 2] - car[lo:hi, 2] < 300))
        carries = [(lo + a, lo + b) for a, b in runs(controlled)
                   if b - a >= config.frames(config.carry_seconds)]
        if not carries:
            continue
        contact = (contacts[0][0], contacts[-1][1])
        event = record("air_dribble", contact, "success", "sustained_control_with_verified_finish")
        carry = next(((a, b) for a, b in carries if
                      (sum(a <= t[0] < b for t in contacts) >= 2
                       or any(min(b, t[1]) - max(a, t[0]) >= config.frames(.25) for t in contacts))
                      and ball[b - 1, 1] - ball[a, 1] >= config.carry_progress), None)
        if carry is None:
            event.update(status="unknown", reason="unverified_repeated_control_or_progress")
            continue
        event["evidence"].update(control_start=int(carry[0]), control_end=int(carry[1]),
                                 control_seconds=(carry[1] - carry[0]) / config.fps,
                                 contact_episodes=len(contacts))
        linked_goal = (goal_event is not None and goal_event["status"] == "success"
                       and lo <= goal_event["contact_start"] < hi)
        if linked_goal:
            finish(event, contact, flight, hi, live_end, live_end, goal=True)
            if event["status"] == "success":
                event["evidence"]["finish"] = "goal_touch"
                goal_event.update(status="superseded", reason="covered_by_complete_air_dribble")
            continue
        settled = hi + config.frames(config.possession_seconds)
        verification_end = hi + max(config.frames(config.recovery_seconds), config.frames(config.possession_seconds))
        if terminal and outcome == "conceded" and live_end <= verification_end:
            event.update(status="failed", reason="conceded_during_air_dribble_finish")
            continue
        if hi == live_end or not observed_through(verification_end):
            event.update(status="unknown", reason="censored_air_dribble_finish")
            continue
        if (settled > live_end or not clean(lo, verification_end)
                or other_touch[lo:verification_end].any()):
            event.update(status="unknown", reason="contested_or_invalid_air_dribble_finish")
            continue
        retained = ((rows[hi:settled, 25] > .5) & (distance[hi:settled] < config.possession_distance)
                    & (relative_speed[hi:settled] < config.carry_relative_speed)
                    & (distance[hi:settled] + 100 < np.linalg.norm(ball[hi:settled] - opponent[hi:settled], axis=1)))
        if not retained.all():
            event.update(status="failed", reason="lost_control_at_finish")
            continue
        finish(event, contact, flight, hi, verification_end, settled - 1)
        if event["status"] == "success":
            event["evidence"]["finish"] = "landed_with_retained_possession"

    for index, event in enumerate(sorted(events, key=lambda e: (e["contact_start"], e["kind"]))):
        event["event"] = index
    return sorted(events, key=lambda e: e["event"])
