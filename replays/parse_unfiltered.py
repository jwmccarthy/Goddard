"""Full replay baseline: no rank, pro-POV, outcome or quality selection.

Completed exports are verified, never silently reconstructed. New
replays retain the full decoder output plus ordinary goal-aware observation
periods. Unsupported/malformed data remains in raw/decoded form with errors.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from replays.layout import team_replay_row_size
from replays.utils import file_sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'replays/rlcs'


def implementation_identity():
    return hashlib.sha256((ROOT / 'replays/parse.py').read_bytes()).hexdigest()


def parse_one(job):
    record, output = job
    output = Path(output)
    uid = record['id']
    team_size = record.get('team_size', 1)
    if type(team_size) is not int or team_size not in (1, 2, 3):
        raise ValueError('unsupported team size')
    raw = Path(record['raw'])
    identity = {'raw_sha256': file_sha256(raw), 'frameskip': 4, 'all_povs': True,
                'include_goals': True, 'success_filter': False, 'pro_filter': False,
                 'quality_filter': False, 'observation_parser_sha256': implementation_identity()}
    if team_size != 1:
        identity['team_size'] = team_size
    report_path = output / 'reports' / (uid + '.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous['identity']['raw_sha256'] != identity['raw_sha256']:
            raise ValueError('raw identity differs; preserving prior baseline')
        if previous['identity'].get('team_size', 1) != team_size:
            raise ValueError('team size differs; preserving prior baseline')
        if previous['status'] == 'done':
            # A source-code update or relocation must not force a new export
            # of already completed, SHA-bound observations.
            for item in previous['files']:
                assert file_sha256(output / item['file']) == item['sha256']
            return {**previous, 'reused_completed_baseline': True}
        if previous['identity'] != identity:
            raise ValueError('parser identity differs; preserve failures and use a separate output to retry')
    result = {'id': uid, 'year': record['year'], 'stage': record.get('stage', 'unknown'),
              'raw': str(raw), 'identity': identity, 'status': 'running', 'files': [],
              'success_filtered': False, 'training_admitted': False}
    dense = output / 'parsed'
    dense.mkdir(parents=True, exist_ok=True)
    try:
        from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay, process_replay

        from replays.parse import parse_replay

        decoded_parent = output / 'decoded'
        decoded_parent.mkdir(parents=True, exist_ok=True)
        decoded = decoded_parent / uid
        marker = decoded / 'baseline-complete.json'
        if marker.exists():
            assert json.loads(marker.read_text())['raw_sha256'] == identity['raw_sha256']
        elif decoded.exists():
            raise ValueError('incomplete decoded output preserved; inspect before retry')
        else:
            with tempfile.TemporaryDirectory(prefix='decode-', dir=decoded_parent) as temporary:
                completed = process_replay(str(raw), temporary, skip_existing=False)
                if completed is None or completed.returncode:
                    failure = output / 'decode_failures' / uid
                    failure.parent.mkdir(parents=True, exist_ok=True)
                    if not failure.exists():
                        shutil.copytree(temporary, failure)
                    raise ValueError('replay decoder failed; raw and decoder diagnostics retained')
                produced = Path(temporary) / raw.stem
                write_json(produced / 'baseline-complete.json', {'raw_sha256': identity['raw_sha256']})
                produced.rename(decoded)
        replay = ParsedReplay.load(decoded)
        written = parse_replay(replay, uid, dense, 4, team_size=team_size, include_goals=True)
        result.update(parse_origin='new_full_decode_and_all_pov_observations',
                      decoder_files=str(decoded.relative_to(output)), raw_decode_repeated=False,
                      emitted_periods=written)
        rows, nonfinite, flagged = 0, 0, 0
        periods = sorted(dense.glob('*-*-' + uid + '.npy'))
        for path in periods:
            data = np.load(path, mmap_mode='r')
            assert data.ndim == 2 and data.shape[1] == team_replay_row_size(team_size)
            rows += len(data)
            nonfinite += int((~np.isfinite(data).all(axis=1)).sum())
            flagged += int((data[:, -2:] > .5).any(axis=1).sum())
        for path in sorted(dense.glob('*-*-' + uid + '.*')):
            result['files'].append({'file': str(path.relative_to(output)), 'sha256': file_sha256(path),
                                    'bytes': path.stat().st_size})
        result.update(status='done', periods=len(periods), observation_rows=rows,
                      nonfinite_rows_retained=nonfinite, quality_flagged_rows_retained=flagged,
                      note='Standard converter segments active gameplay and may decline structurally invalid periods; '
                            'new full decoder output is retained without success selection.')
        assert file_sha256(raw) == identity['raw_sha256']
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    write_json(report_path, result)
    return result


def run(records, output=OUT, workers=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'source_manifest.json', {'records': records, 'selection': 'all supplied replay records; no behavior filtering'})
    write_json(output / 'run_state.json', {'stage': 'parsing', 'sources': len(records), 'training_launched': False})
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for report in executor.map(parse_one, [(r, str(output)) for r in records], chunksize=1):
            reports.append(report)
            write_json(output / 'progress.json', {'processed': len(reports), 'sources': len(records),
                'done': sum(r['status'] == 'done' for r in reports), 'failed': sum(r['status'] == 'failed' for r in reports)})
    summary = {'sources': len(records), 'done': sum(r['status'] == 'done' for r in reports),
               'failed': sum(r['status'] == 'failed' for r in reports),
               'legacy_parses_reused': sum(r.get('parse_origin') == 'reused_full_legacy_two_pov_cache' for r in reports),
               'observation_rows': sum(r.get('observation_rows', 0) for r in reports),
               'quality_flagged_rows_retained': sum(r.get('quality_flagged_rows_retained', 0) for r in reports),
               'training_launched': False, 'success_filter': False}
    write_json(output / 'report.json', summary)
    write_json(output / 'run_state.json', {'stage': 'completed_with_failures' if summary['failed'] else 'completed', **summary})
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=OUT / 'manifest.json')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())['records']
    run(records, args.output, args.workers)
