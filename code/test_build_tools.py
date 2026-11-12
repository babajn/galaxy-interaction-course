#!/usr/bin/env python3
"""第0章の再生成で実行結果や実測環境が失われないことを確認します。"""
from pathlib import Path
import ast
import copy
import json
import platform
import re
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parent / 'chap00/build_chap00.py'
module = ast.parse(SOURCE.read_text())
functions = ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in ['preserve_outputs', 'generate_values']], type_ignores=[])
ns = {'json': json, 're': re}
exec(compile(functions, str(SOURCE), 'exec'), ns)


class BuildToolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'generated').mkdir()
        ns['ROOT'] = self.root
        self.path = self.root / 'saved.ipynb'
        self.fresh = {'cells': [{'cell_type': 'code', 'source': 'print(1)', 'execution_count': None, 'outputs': []}], 'metadata': {}}
        self.saved = copy.deepcopy(self.fresh)
        self.saved['cells'][0].update(execution_count=1, outputs=[{'output_type': 'stream', 'name': 'stdout', 'text': ['Python: 3.12.7\n', 'NumPy: 2.1.3\n', 'Matplotlib: 3.9.2\n']}])
        self.saved['metadata']['language_info'] = {'name': 'python', 'version': '3.12.7'}
        self.path.write_text(json.dumps(self.saved))

    def test_matching_sources_preserve_output_and_environment(self):
        result = ns['preserve_outputs'](self.fresh, self.path)
        self.assertEqual(result, self.saved)

    def test_changed_source_does_not_keep_stale_output(self):
        self.fresh['cells'][0]['source'] = 'print(2)'
        result = ns['preserve_outputs'](self.fresh, self.path)
        self.assertIsNone(result['cells'][0]['execution_count'])
        self.assertEqual(result['cells'][0]['outputs'], [])

    def test_missing_saved_notebook_is_unchanged(self):
        self.assertEqual(ns['preserve_outputs'](self.fresh, self.root / 'missing.ipynb'), self.fresh)

    def test_environment_values_use_executed_output(self):
        ns['generate_values'](self.saved)
        result = json.loads((self.root / 'data/validation.json').read_text())
        self.assertEqual(result, {'python_version': '3.12.7', 'numpy_version': '2.1.3', 'matplotlib_version': '3.9.2'})
        self.assertIn('3.12.7', (self.root / 'generated/values.tex').read_text())

    def test_repeated_runtime_table_has_one_cpu_macro(self):
        source = SOURCE.parent.parent / 'rebuild.py'
        tree = ast.parse(source.read_text())
        functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'write_runtime_table'], type_ignores=[])
        context = {'re': re, 'platform': platform, 'TEXT': self.root}
        exec(compile(functions, str(source), 'exec'), context)
        folder = self.root / 'generated/chap00'
        folder.mkdir()
        values = folder / 'values.tex'
        values.write_text('')
        report = {'environment': {'cpu': 'test CPU'}, 'chapters': []}
        context['write_runtime_table'](report)
        first = values.read_text()
        context['write_runtime_table'](report)
        self.assertEqual(first, values.read_text())
        self.assertEqual(first.count('chzerovalCPU'), 1)

    def test_unexecuted_notebook_cannot_claim_environment(self):
        with self.assertRaises(RuntimeError):
            ns['generate_values'](self.fresh)


if __name__ == '__main__':
    unittest.main()
