"""Read-only source discovery and SHA-bound, fail-closed admission for aerial data."""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from replays.duel_audit import FORMAT, clock_audit, decode_replay, header_audit, header_identity

ADMISSION_VERSION = "aerial-admission-v2"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def source_records(duels, pros):
    records = []
    duels, pros = Path(duels), Path(pros)
    for line in (duels / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        records.append({**row, "source": "ssl_duels", "raw": str((duels / "raw" / f"{row['id']}.replay").resolve()),
                        "metadata": str((duels / "metadata" / f"{row['id']}.json").resolve()),
                        "admission_config": json.loads((duels / "config.json").read_text())})
    provenance = json.loads((pros / "provenance.json").read_text())["replays"]
    # Use exactly the existing admitted pro matches, not the other downloaded files.
    allowed = json.loads((pros / "dac_parsed/pro_files.json").read_text())["files"]
    ids = {Path(name).stem[-36:] for name in allowed}
    for replay_id in sorted(ids):
        metadata = provenance[replay_id]["metadata"]
        players = [p for color in ("blue", "orange") for p in metadata[color]["players"]]
        records.append({"id": replay_id, "source": "pro_playoffs", "raw": str((pros / "raw" / f"{replay_id}.replay").resolve()),
                        "listing": metadata, "players": players, "date": metadata.get("date"),
                        "pro_players": [f"{p['id']['platform']}:{p['id']['id']}" for p in players if p.get("pro")],
                        "rocket_league_id": metadata.get("rocket_league_id"), "event_path": provenance[replay_id]["event_path"],
                        "expert_ego_ids": sorted({Path(name).stem.split("-", 1)[0] for name in allowed if replay_id in name})})
    return records


def audit_source(job):
    row, output, executable = job
    path = Path(output) / "audits" / f"{row['id']}.json"
    try:
        sha = hashlib.sha256(Path(row["raw"]).read_bytes()).hexdigest()
        listing = json.loads(Path(row["metadata"]).read_text()) if row["source"] == "ssl_duels" else row["listing"]
        source_digest = hashlib.sha256(json.dumps({"listing": listing, "players": row["players"],
            "goals": row.get("goals"), "match_key": row.get("match_key"), "rocket_league_id": row.get("rocket_league_id"),
            "config": row.get("admission_config")}, sort_keys=True).encode()).hexdigest()
        if path.is_file():
            cached = json.loads(path.read_text())
            if (cached.get("sha256") == sha and cached.get("format") == FORMAT
                    and cached.get("admission_version") == ADMISSION_VERSION and cached.get("source_digest") == source_digest):
                return cached
        report = {"format": FORMAT, "completion_verified": False, "rejection": None}
        if row.get("sha256") and sha != row["sha256"]:
            report["rejection"] = "sha256_conflict"
        else:
            decoded = decode_replay(row["raw"], executable)
            props = decoded["properties"]
            if row["source"] == "ssl_duels":
                for color, goals in zip(("blue", "orange"), row["goals"]):
                    listing[color]["goals"] = goals
                ranks = [p.get("rank", {}).get("id") for p in row["players"]]
                report = header_audit(decoded, listing, row["admission_config"])
                if ranks != ["supersonic-legend"] * 2:
                    report["rejection"] = "explicit_ssl_missing"
                elif abs(row["goals"][0] - row["goals"][1]) not in (1, 2):
                    report["rejection"] = "score_margin"
                elif (props.get("MatchGUID") and str(props["MatchGUID"]).lower() != row["match_key"].lower()):
                    report["rejection"] = "match_guid_conflict"
                elif str(props.get("Id", "")).lower() != row.get("rocket_league_id", "").lower():
                    report["rejection"] = "replay_id_conflict"
                report["identity_evidence"] = "raw-guid-and-id" if props.get("MatchGUID") else "raw-replay-id; listing-guid"
            else:
                pair = props.get("PlayerStats", [])
                expected = {f"{p['id']['platform']}:{p['id']['id']}" for p in row["players"]}
                report.update(goals=[props.get("Team0Score"), props.get("Team1Score")], match_guid=props.get("MatchGUID"),
                              policy="existing-pro-tournament-plus-clock", match_type=props.get("MatchType"))
                # Private lobbies often advertise capacity four with only two
                # active players. The raw pair and dense parser establish 1v1.
                if (len(pair) != 2 or sorted(p.get("Team", -1) for p in pair) != [0, 1]
                        or {header_identity(p) for p in pair} != expected or any(p.get("bBot") is not False for p in pair)):
                    report["rejection"] = "tournament_identity_unverified"
                elif any(listing[c].get("goals") is not None and listing[c]["goals"] != report["goals"][i]
                         for i, c in enumerate(("blue", "orange"))):
                    report["rejection"] = "score_conflict"
                elif props.get("bForfeit") is True:
                    report["rejection"] = "forfeit"
            if not report["rejection"]:
                report = clock_audit(decode_replay(row["raw"], executable, network=True), report,
                                     overtime=bool(listing.get("overtime")))
            report["raw_replay_id"] = str(props.get("Id") or "").lower()
            report["raw_match_key"] = str(props.get("MatchGUID") or row.get("match_key") or props.get("Id") or "").lower()
            if not report["raw_match_key"]:
                report.update(completion_verified=False, rejection="match_identity_missing")
        report.update(id=row["id"], source=row["source"], sha256=sha, admission_version=ADMISSION_VERSION, source_digest=source_digest)
    except Exception as error:
        report = {"format": FORMAT, "id": row["id"], "source": row["source"], "completion_verified": False,
                  "rejection": "audit_failed", "error": f"{type(error).__name__}: {error}"}
    write_json(path, report)
    return report


def audit_corpus(duels, pros, output, executable, workers=4, seed=20261109):
    output = Path(output)
    (output / "audits").mkdir(parents=True, exist_ok=True)
    rows = source_records(duels, pros)
    accepted, rejected, seen, seen_replays = [], [], set(), set()
    # Priority is source policy, not a behavior statistic. All POVs of a raw GUID share a split.
    rows.sort(key=lambda r: (r["source"] != "pro_playoffs", r["id"]))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, (row, report) in enumerate(zip(rows, executor.map(audit_source,
                ((r, str(output), executable) for r in rows), chunksize=1))):
            if report["completion_verified"]:
                key = report["raw_match_key"]
                replay_key = report.get("raw_replay_id")
                if key in seen or (replay_key and replay_key in seen_replays):
                    rejected.append({"id": row["id"], "rejection": "duplicate_raw_match", "match_key": key})
                else:
                    seen.add(key)
                    if replay_key:
                        seen_replays.add(replay_key)
                    row.update(match_key=key, sha256=report["sha256"], completion_verified=True,
                               split="heldout" if stable_seed(seed, key, "split") % 10 == 0 else "train")
                    accepted.append(row)
            else:
                rejected.append(report)
            if (index + 1) % 100 == 0:
                print(json.dumps({"audited": index + 1, "total": len(rows), "accepted": len(accepted), "rejected": len(rejected)}), flush=True)
    write_json(output / "sources.json", {"format": "aerial-sources-v1", "seed": seed, "records": accepted})
    write_json(output / "quarantine.json", rejected)
    report = {"sources": len(rows), "accepted": len(accepted), "rejections": dict(Counter(r["rejection"] for r in rejected)),
              "accepted_by_source": dict(Counter(r["source"] for r in accepted)),
              "split": dict(Counter(r["split"] for r in accepted))}
    write_json(output / "audit_report.json", report)
    print(json.dumps(report), flush=True)
    return accepted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duels", type=Path, default=Path("replays/ssl_duels"))
    parser.add_argument("--pros", type=Path, default=Path("replays/rlcs_2026_1v1_playoffs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rrrocket", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    audit_corpus(args.duels, args.pros, args.output, args.rrrocket, args.workers)
