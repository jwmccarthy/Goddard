import copy
import hashlib
import json

import pytest

from replays import corpus
from replays.test_collect_duels import CONFIG, listing
from replays.test_duel_audit import decoded_replay


def source(tmp_path):
    raw = tmp_path / 'sample.replay'
    raw.write_bytes(b'immutable replay bytes')
    metadata = tmp_path / 'listing.json'
    value = listing()
    metadata.write_text(json.dumps(value))
    (tmp_path / 'audits').mkdir()
    return {'id': 'sample', 'raw': str(raw), 'metadata': str(metadata), 'source': 'ssl_duels',
            'players': [p for color in ('blue', 'orange') for p in value[color]['players']],
            'goals': [5, 4], 'match_key': 'test-match', 'rocket_league_id': 'raw-id',
            'admission_config': CONFIG, 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}


def test_admission_is_sha_bound_and_does_not_edit_original_manifest(tmp_path, monkeypatch):
    row = source(tmp_path)
    decoded = decoded_replay()
    decoded['properties']['Id'] = 'raw-id'
    original = copy.deepcopy(row)
    monkeypatch.setattr(corpus, 'decode_replay', lambda *args, **kwargs: copy.deepcopy(decoded))
    report = corpus.audit_source((row, str(tmp_path), 'unused'))
    assert report['completion_verified'] and row == original
    assert report['source_digest']
    assert report['sha256'] == row['sha256']
    # Cached admission cannot survive different rank evidence or changed bytes.
    row['players'][0]['rank'] = {'id': 'grand-champion-3'}
    assert corpus.audit_source((row, str(tmp_path), 'unused'))['rejection'] == 'explicit_ssl_missing'
    row = original
    (tmp_path / 'sample.replay').write_bytes(b'different')
    assert corpus.audit_source((row, str(tmp_path), 'unused'))['rejection'] == 'sha256_conflict'


@pytest.mark.parametrize('case,reason', [('missing_clock', 'end_clock_unverified'),
                                        ('missing_header', 'short_play'), ('missing_rank', 'explicit_ssl_missing')])
def test_missing_admission_evidence_never_implies_verified(tmp_path, monkeypatch, case, reason):
    row = source(tmp_path)
    decoded = decoded_replay()
    decoded['properties']['Id'] = 'raw-id'
    if case == 'missing_clock':
        decoded['network_frames'] = None
    elif case == 'missing_header':
        decoded['properties'].pop('TotalSecondsPlayed')
    else:
        row['players'][0].pop('rank')
    monkeypatch.setattr(corpus, 'decode_replay', lambda *args, **kwargs: copy.deepcopy(decoded))
    report = corpus.audit_source((row, str(tmp_path), 'unused'))
    assert not report['completion_verified'] and report['rejection'] == reason


def test_absent_guid_can_bind_to_exact_raw_replay_id_but_not_an_unrelated_id(tmp_path, monkeypatch):
    row = source(tmp_path)
    decoded = decoded_replay()
    decoded['properties'].pop('MatchGUID')
    decoded['properties']['Id'] = 'raw-id'
    monkeypatch.setattr(corpus, 'decode_replay', lambda *args, **kwargs: copy.deepcopy(decoded))
    report = corpus.audit_source((row, str(tmp_path), 'unused'))
    assert report['completion_verified']
    assert report['identity_evidence'] == 'raw-replay-id; listing-guid'
    row['rocket_league_id'] = 'unrelated'
    assert corpus.audit_source((row, str(tmp_path), 'unused'))['rejection'] == 'replay_id_conflict'
