"""Download official RLCS event trees, optionally selecting Swiss/group stages onward."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from replays.api import CachedClient
from replays.download import Root, list_group_children
from replays.utils import file_sha256, write_json

ROOTS = {2025: Root('1v1-events-spqzm3tx17', '1v1', 'RLCS 2025 1v1 Events'),
         2026: Root('1v1-events-q0z69hlqmt', '1v1', 'RLCS 2026 1v1 Events')}
DOUBLES_ROOTS = {2026: Root('2v2-events-gnz7nvaem1', '2v2', 'RLCS 2026 2v2 Events')}
MAIN_EVENT_STAGES = {'swiss', 'group_stage', 'playoffs', 'championship'}
OUT = Path(__file__).resolve().parent / 'rlcs'


def discover(client, root, stages='all'):
    entries, visited = {}, set()

    def walk(group, path, stage=None):
        if group['id'] in visited:
            return
        visited.add(group['id'])
        name = group.get('name', group['id'])
        if 'swiss' in name.lower():
            stage = 'swiss'
        elif 'group stage' in name.lower() or 'gsl stage' in name.lower():
            stage = 'group_stage'
        elif 'playoff' in name.lower():
            stage = 'playoffs'
        elif 'world championship' in name.lower():
            stage = 'championship'
        elif 'bracket' in name.lower():
            # A bracket inside a selected main-event stage remains eligible;
            # an unclassified qualifier bracket does not establish eligibility.
            stage = stage if stage in MAIN_EVENT_STAGES else 'bracket'
        path = path + [name]
        direct = group.get('direct_replays', 0)
        if direct and (stages == 'all' or stage in MAIN_EVENT_STAGES):
            items = client.find_replay_entries(group=group['id'], count=200, sort_by='replay-date', sort_dir='asc')
            for item in items:
                uid = item['id'].lower()
                record = entries.setdefault(uid, {'id': uid, 'metadata': item, 'occurrences': []})
                record['occurrences'].append({'group_id': group['id'], 'path': path, 'stage': stage or 'unclassified'})
        if group.get('total_replays', group.get('indirect_replays')) == direct and direct:
            return
        for child in list_group_children(client, group['id']):
            walk(child, path, stage)

    for child in list_group_children(client, root.group_id):
        walk(child, [root.event])
    return entries


def validate_resume(manifest, mode, years, stages):
    if not manifest:
        return
    # Legacy manifests came from the unfiltered, two-year 1v1 acquisition.
    previous = (manifest.get('mode', '1v1'),
                sorted(manifest.get('years', [2025, 2026])),
                manifest.get('stage_selection', 'all'))
    if previous != (mode, sorted(set(years)), stages):
        raise ValueError('Existing manifest has a different mode/year/stage selection; use a separate --output')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('1v1', '2v2'), default='1v1')
    parser.add_argument('--years', type=int, nargs='+', choices=(2025, 2026), default=[2025, 2026])
    parser.add_argument('--stages', choices=('all', 'swiss-onward'),
                        help='default: all for 1v1; Swiss/group stages and playoffs onward for 2v2')
    args = parser.parse_args()
    roots = ROOTS if args.mode == '1v1' else DOUBLES_ROOTS
    for year in args.years:
        if year not in roots:
            print(f'No official RLCS {year} {args.mode} event tree: RLCS 2v2 began in 2026.', flush=True)
    if not any(year in roots for year in args.years):
        return
    token = os.environ.get('BALLCHASING_TOKEN')
    if not token:
        parser.error('BALLCHASING_TOKEN is required in the environment or project .env')
    stages = args.stages or ('all' if args.mode == '1v1' else 'swiss-onward')
    output = (args.output or (OUT if args.mode == '1v1' else OUT / '2v2')).resolve()
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    try:
        validate_resume(manifest, args.mode, args.years, stages)
    except ValueError as error:
        parser.error(str(error))
    records = {record['id']: record for record in manifest.get('records', [])}
    raw = output / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    client = CachedClient(token, output / 'provenance/api_cache')
    errors = {}
    for year in sorted(set(args.years)):
        if year not in roots:
            continue
        root = roots[year]
        entries = discover(client, root, stages)
        write_json(output / 'provenance' / f'inventory-{year}.json', entries)
        pending = [uid for uid in entries if not (raw / (uid + '.replay')).exists()]
        print(f'{year} {args.mode}: {len(entries):,} selected, {len(pending):,} downloads ({stages}).', flush=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(client.download_replay, uid, raw): uid for uid in pending}
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    if not future.result():
                        errors[uid] = 'not_found'
                except Exception as error:
                    errors[uid] = type(error).__name__
        for uid, item in entries.items():
            path = raw / (uid + '.replay')
            if not path.is_file():
                continue
            sha = file_sha256(path)
            previous = records.get(uid, {})
            if previous.get('raw_sha256', sha) != sha:
                raise ValueError(f'raw replay checksum changed: {uid}')
            records[uid] = {**previous, 'id': uid, 'year': year, 'raw': str(path), 'raw_sha256': sha,
                            'stage': item['occurrences'][0]['stage'], 'provenance': item,
                            'team_size': int(args.mode[0]), 'mode': args.mode}
        manifest.update(records=sorted(records.values(), key=lambda record: record['id']), behavior_filter=False)
        manifest.update(mode=args.mode, years=sorted(set(args.years)), stage_selection=stages)
        write_json(manifest_path, manifest)
    write_json(output / 'provenance/download-errors.json', errors)
    if errors:
        raise RuntimeError(f'{len(errors)} downloads failed; raw inputs and errors were preserved')


if __name__ == '__main__':
    main()
