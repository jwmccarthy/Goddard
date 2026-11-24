"""Select the actual playing roster for each decoded gameplay period."""

from copy import copy


def source_teams(metadata: dict | None) -> dict[tuple, bool]:
    """Map (platform, id) to is_orange from source replay metadata."""
    teams = {}

    for orange, team in enumerate(("blue", "orange")):
        for player in (metadata or {}).get(team, {}).get("players", []):
            identity = player.get("id", {})
            key = (identity.get("platform", "").lower(), identity.get("id"))

            if key in teams and teams[key] != bool(orange):
                raise ValueError("player belongs to both teams in source metadata")

            teams[key] = bool(orange)

    return teams


def period_end(replay, period: dict, index: int, diagnostics: list | None) -> int:
    end = period.get("goal_frame")

    if end is not None:
        return end

    end = period["end_frame"]

    # The upstream converter trims at a sleeping ball but still indexes
    # the original end for non-goal periods, overrunning truncated files.
    ball = replay.ball_df.loc[period["start_frame"]:end]

    sleeping = ball.index[
        (ball["is_sleeping"] == 1) & (ball["pos_x"] != 0) & (ball["pos_y"] != 0)
    ]

    if not len(sleeping) or sleeping[0] >= end:
        return end

    end = period["end_frame"] = int(sleeping[0])

    if diagnostics is not None:
        diagnostics.append({
            "period":    index,
            "action":    "trimmed_sleeping_tail",
            "end_frame": end
        })

    return end


def period_roster(
    spans:       dict,
    players:     dict,
    teams:       dict,
    start:       int,
    end:         int,
    index:       int,
    diagnostics: list | None
) -> list[dict]:
    roster = []

    for uid, (first, last) in spans.items():
        if first > end or last < start:
            continue

        player = dict(players[uid])
        key = (player.get("online_id_kind", "").lower(), player.get("online_id"))

        if player.get("is_orange") is None and key in teams:
            player["is_orange"] = teams[key]

            if diagnostics is not None:
                diagnostics.append({
                    "period":    index,
                    "action":    "team_from_source",
                    "player":    uid,
                    "is_orange": teams[key]
                })

        roster.append(player)

    return roster


def balanced(roster: list[dict], size: int) -> bool:
    return (
        len(roster) == 2 * size
        and sum(p.get("is_orange") is True for p in roster) == size
        and sum(p.get("is_orange") is False for p in roster) == size
    )


def period_replays(replay, team_size=None, metadata=None, diagnostics=None):
    """Keep reconnect actors separate and omit periods spanning roster changes."""
    teams = source_teams(metadata)
    players = {p["unique_id"]: p for p in replay.metadata["players"]}

    spans = {}

    for uid, data in replay.player_dfs.items():
        valid = data["pos_x"].dropna()

        if not len(valid):
            continue

        spans[uid] = (valid.index[0], valid.index[-1])

    expected_size = team_size

    for index, original in enumerate(replay.analyzer["gameplay_periods"]):
        period = dict(original)
        start = period["start_frame"]
        end = period_end(replay, period, index, diagnostics)

        roster = period_roster(
            spans, players, teams, start, end, index, diagnostics
        )

        size = expected_size if expected_size is not None else len(roster) // 2

        if not roster or not balanced(roster, size):
            if diagnostics is not None:
                diagnostics.append({
                    "period":      index,
                    "action":      "omitted_roster_change",
                    "start_frame": start,
                    "end_frame":   end,
                    "players":     [p["unique_id"] for p in roster]
                })

            yield index, None

            continue

        expected_size = size

        view = copy(replay)
        view.metadata = {**replay.metadata, "players": roster}

        view.player_dfs = {
            p["unique_id"]: replay.player_dfs[p["unique_id"]] for p in roster
        }

        view.analyzer = {**replay.analyzer, "gameplay_periods": [period]}

        yield index, view
