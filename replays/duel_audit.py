"""Fail-closed header and end-clock checks for ranked duel candidates.

Uses rrrocket's JSON output, not a new training-data parser. Passing these
checks establishes clock completion, not an absence of AFK play or bad effort.
"""

import json
import math
import subprocess
from datetime import datetime, timezone

from replays.duel_ranks import timestamp

FORMAT = "duel-clock-audit-v2"
CLOCK = "TAGame.GameEvent_Soccar_TA:SecondsRemaining"
OVERTIME = "TAGame.GameEvent_Soccar_TA:bOverTime"
FORFEIT = "TAGame.GameEvent_Team_TA:bForfeit"
ENDED = "TAGame.GameEvent_Soccar_TA:bMatchEnded"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def header_identity(player):
    fields = player.get("PlayerID", {}).get("fields", {})
    platform = player.get("Platform", fields.get("Platform", {})).get("value")
    if platform == "OnlinePlatform_Steam":
        account = fields.get("Uid") or player.get("OnlineID")
        return f"steam:{account}" if account and str(account) != "0" else None
    if platform == "OnlinePlatform_Epic":
        account = fields.get("EpicAccountId")
        return f"epic:{account}" if account else None
    # Console header IDs do not always use the same representation as the API.
    # Do not substitute a display name for an account ID.
    return None


def header_audit(decoded, listing, config):
    props = decoded.get("properties", {})
    pair = props.get("PlayerStats", [])
    report = {"format": FORMAT, "completion_verified": False,
              "forfeit": props.get("bForfeit"), "total_seconds_played": props.get("TotalSecondsPlayed"),
              "match_guid": props.get("MatchGUID"), "match_start_epoch": props.get("MatchStartEpoch"),
              "goals": [props.get("Team0Score"), props.get("Team1Score")],
              "players": [{"identity": header_identity(p), "team": p.get("Team")} for p in pair]}
    reason = None
    if props.get("bForfeit") is True:
        reason = "forfeit"
    elif props.get("TeamSize") != 1 or props.get("MatchType") != "Online":
        reason = "format"
    elif not finite(props.get("TotalSecondsPlayed")) or props["TotalSecondsPlayed"] < config["min_play_seconds"]:
        reason = "short_play"
    elif report["goals"] != [listing[color]["goals"] for color in ("blue", "orange")]:
        reason = "score_conflict"
    elif len(pair) != 2 or sorted(p.get("Team", -1) for p in pair) != [0, 1]:
        reason = "players"
    else:
        for p in pair:
            expected = listing[("blue", "orange")[p["Team"]]]["players"][0]["id"]
            if header_identity(p) != f"{expected['platform']}:{expected['id']}" or p.get("bBot") is not False:
                reason = "identity_unverified"
                break
            if (not finite(p.get("Score")) or p["Score"] < config["min_player_score"]
                    or not finite(p.get("Shots")) or p["Shots"] < config["min_player_shots"]):
                reason = "activity"
                break
    if reason is None and (config.get("replay_date_after") or config.get("replay_date_before")):
        try:
            epoch = float(props.get("MatchStartEpoch"))
            if not math.isfinite(epoch):
                raise ValueError("non-finite match epoch")
            date = datetime.fromtimestamp(epoch, timezone.utc)
            report["match_start_utc"] = date.isoformat()
        except (TypeError, ValueError, OverflowError, OSError):
            reason = "date_unverified"
        else:
            if config.get("replay_date_after") and date <= timestamp(config["replay_date_after"]):
                reason = "date_outside_range"
            if config.get("replay_date_before") and date >= timestamp(config["replay_date_before"]):
                reason = "date_outside_range"
    report["rejection"] = reason
    return report


def clock_audit(decoded, report, *, overtime=False):
    report = dict(report)
    objects = decoded.get("objects", [])
    frames = (decoded.get("network_frames") or {}).get("frames", [])
    clock, clock_max, overtime_at, ended = None, None, None, False
    regulation_max = None
    initial_clock_at, zero_clock_at = None, None
    for frame in frames:
        for update in frame.get("updated_actors", []):
            index = update.get("object_id")
            if not isinstance(index, int) or not 0 <= index < len(objects):
                continue
            name, attribute = objects[index], update.get("attribute", {})
            if name == FORFEIT and attribute.get("Boolean") is True:
                report.update(forfeit=True, rejection="forfeit")
            elif name == CLOCK:
                value = attribute.get("Int")
                if finite(value) and value >= 0:
                    clock = value
                    clock_max = value if clock_max is None else max(clock_max, value)
                    if overtime_at is None:
                        regulation_max = value if regulation_max is None else max(regulation_max, value)
                    if value == 300 and initial_clock_at is None:
                        initial_clock_at = frame.get("time")
                    if value == 0 and zero_clock_at is None:
                        zero_clock_at = frame.get("time")
            elif name == OVERTIME and attribute.get("Boolean") is True and overtime_at is None:
                overtime_at = frame.get("time")
            elif name == ENDED and attribute.get("Boolean") is True:
                ended = True
    report.update(final_clock=clock, maximum_clock=clock_max, overtime_start=overtime_at,
                  explicit_match_end=ended, frames=len(frames),
                  initial_clock_at=initial_clock_at, zero_clock_at=zero_clock_at,
                  maximum_regulation_clock=regulation_max)
    if report["rejection"]:
        return report
    is_overtime = overtime or overtime_at is not None
    if (regulation_max != 300 or (not is_overtime and clock != 0)
            or not finite(initial_clock_at) or not finite(zero_clock_at)
            or zero_clock_at - initial_clock_at < 299):
        report["rejection"] = "end_clock_unverified"
    elif is_overtime:
        # Sudden death needs a goal AFTER overtime began, not just a zero clock.
        goals = decoded.get("properties", {}).get("Goals", [])
        goal_times = [frames[g["frame"]]["time"] for g in goals
                      if isinstance(g.get("frame"), int) and 0 <= g["frame"] < len(frames)]
        if (not finite(overtime_at) or zero_clock_at > overtime_at
                or not any(t > overtime_at for t in goal_times)
                or abs(report["goals"][0] - report["goals"][1]) != 1):
            report["rejection"] = "overtime_finish_unverified"
    report["completion_verified"] = report["rejection"] is None
    return report


def decode_replay(path, executable, *, network=False):
    command = [str(executable), *(["-n"] if network else []), str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
    return json.loads(result.stdout)


def audit_replay(path, executable, listing, config):
    try:
        report = header_audit(decode_replay(path, executable), listing, config)
        if not report["rejection"]:
            report = clock_audit(decode_replay(path, executable, network=True), report,
                                 overtime=bool(listing.get("overtime")))
        return report
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {"format": FORMAT, "completion_verified": False, "rejection": "decode_failed"}
