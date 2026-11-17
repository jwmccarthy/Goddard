import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from replays.download import Root
from replays.download_rlcs import discover, validate_resume
from replays.layout import team_replay_row_size
from replays.parse_unfiltered import parse_one


class AcquisitionTests(unittest.TestCase):
    def test_stage_inheritance_exclusion_and_duplicate_provenance(self):
        def group(uid, name, direct=0):
            return {'id': uid, 'name': name, 'direct_replays': direct}

        tree = {
            'root': [group('region', 'Europe'), group('worlds', 'World Championship', 1)],
            'region': [group('qualifier', 'Qualifier Bracket', 1),
                       group('swiss', 'Swiss', 1), group('gsl', 'GSL Stage', 1),
                       group('groups', 'Group Stage'), group('playoffs', 'Playoffs')],
            'groups': [group('series', 'A vs B', 1)],
            'playoffs': [group('upper', 'Upper Bracket', 1)],
        }
        client = Mock()
        client.find_replay_entries.side_effect = lambda **kw: [
            {'id': 'SHARED' if kw['group'] in ('swiss', 'upper') else kw['group']}]
        with patch('replays.download_rlcs.list_group_children',
                   side_effect=lambda client, uid: tree.get(uid, [])):
            selected = discover(client, Root('root', '2v2', 'RLCS 2026'), 'swiss-onward')
            self.assertEqual(set(selected), {'worlds', 'shared', 'gsl', 'series'})
            self.assertEqual({o['stage'] for o in selected['shared']['occurrences']},
                             {'swiss', 'playoffs'})
            self.assertEqual(selected['series']['occurrences'][0]['stage'], 'group_stage')
            self.assertNotIn('qualifier', [call.kwargs['group']
                                          for call in client.find_replay_entries.call_args_list])
            self.assertIn('qualifier', discover(client, Root('root', '2v2', 'RLCS 2026')))

    def test_resume_cannot_mix_acquisition_selections(self):
        legacy = {'records': [{'id': 'old'}]}
        validate_resume(legacy, '1v1', [2025, 2026], 'all')
        for mode, years, stages in [('2v2', [2025, 2026], 'all'),
                                    ('1v1', [2026], 'all'),
                                    ('1v1', [2025, 2026], 'swiss-onward')]:
            with self.assertRaisesRegex(ValueError, 'different'):
                validate_resume(legacy, mode, years, stages)
        doubles = {'mode': '2v2', 'years': [2026, 2025], 'stage_selection': 'swiss-onward'}
        validate_resume(doubles, '2v2', [2025, 2026], 'swiss-onward')

    def test_doubles_pipeline_checks_width_and_preserves_completed_exports(self):
        # Substitute only the expensive decoder/converter boundary. Exercise the
        # real baseline export validation, identity, report, and resume paths.
        for width in (team_replay_row_size(2), team_replay_row_size(1)):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                raw = output / 'game.replay'
                raw.write_bytes(b'replay fixture')
                record = {'id': 'game', 'raw': str(raw), 'year': 2026, 'team_size': 2}

                def decode(raw_path, temporary, **kwargs):
                    (Path(temporary) / Path(raw_path).stem).mkdir()
                    return SimpleNamespace(returncode=0)

                def export(replay, uid, dense, frameskip, **kwargs):
                    self.assertEqual(kwargs, {'team_size': 2, 'include_goals': True})
                    np.save(dense / f'0-player-{uid}.npy', np.zeros((3, width), dtype=np.float32))
                    return 1

                decoder = Mock(side_effect=decode)
                converter = Mock(side_effect=export)
                modules = {
                    'rlgym_tools.rocket_league.replays.parsed_replay':
                        SimpleNamespace(ParsedReplay=Mock(), process_replay=decoder),
                    'replays.parse': SimpleNamespace(parse_replay=converter),
                }
                with patch.dict('sys.modules', modules):
                    report = parse_one((record, output))
                    self.assertEqual(report['identity']['team_size'], 2)
                    if width != team_replay_row_size(2):
                        self.assertEqual(report['status'], 'failed')
                        continue
                    self.assertEqual(report['status'], 'done')
                    self.assertEqual(report['observation_rows'], 3)
                    self.assertTrue(parse_one((record, output))['reused_completed_baseline'])
                    self.assertEqual(decoder.call_count, 1)
                    self.assertEqual(converter.call_count, 1)
                    with self.assertRaisesRegex(ValueError, 'team size differs'):
                        parse_one(({**record, 'team_size': 1}, output))
                    saved = json.loads((output / 'reports/game.json').read_text())
                    self.assertEqual(saved['status'], 'done')


if __name__ == '__main__':
    unittest.main()
