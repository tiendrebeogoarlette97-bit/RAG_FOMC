"""Deployment wiring tests; downloads and remote LLM calls are not performed."""
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class CloudBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = runpy.run_path(str(ROOT / 'app/cloud_app.py'), run_name='bootstrap_test')

    def test_all_pinned_models_and_indexes(self):
        prepare = self.ns['prepare_models'].__wrapped__
        with patch.dict(prepare.__globals__, snapshot_download=lambda **kw: calls.append(kw)), \
                patch('build_indexes.load_indexes') as validate:
            calls = []
            prepare()
        self.assertEqual(len(calls), 3)
        for call, kind in zip(calls, ('dense', 'colbert', 'cross_encoder')):
            self.assertEqual(call['repo_id'], self.ns['CONFIG'][kind + '_model'])
            self.assertEqual(call['revision'], self.ns['CONFIG'][kind + '_revision'])
            self.assertEqual(call['local_dir'], str(self.ns['model_dir'](kind)))
            self.assertIn('onnx_config.json', call['allow_patterns'])
        validate.assert_called_once()

    def test_missing_index_fails_before_download(self):
        prepare = self.ns['prepare_models'].__wrapped__
        with patch('pathlib.Path.is_file', return_value=False), \
                patch.dict(prepare.__globals__, snapshot_download=lambda **kw: self.fail('Unexpected download')):
            with self.assertRaisesRegex(FileNotFoundError, 'chunks.jsonl'):
                prepare()


if __name__ == '__main__':
    unittest.main()
