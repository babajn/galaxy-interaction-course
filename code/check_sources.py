#!/usr/bin/env python3
"""標準ライブラリだけで正本・Notebook・本文抜粋の同期を確認します。数値の再実行ではありません。"""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def source_cells(source, student=False):
    blocks = re.split(r'(?m)^# %%(.*)\n', source)
    cells = []
    for i in range(1, len(blocks), 2):
        header, body = blocks[i], blocks[i + 1].strip('\n')
        kind = 'markdown' if '[markdown]' in header else 'code'
        if kind == 'markdown':
            body = '\n'.join(re.sub(r'^# ?', '', line) for line in body.splitlines())
        else:
            lines, hidden = [], False
            for line in body.splitlines():
                if student and line.strip() == '### BEGIN SOLUTION':
                    if hidden:
                        raise ValueError('Nested solution block')
                    indent = line[:len(line) - len(line.lstrip())]
                    lines.extend([indent + '# TODO: 本文の式を参照して、この部分を実装してください。', indent + 'raise NotImplementedError("Complete the TODO block before running this cell.")'])
                    hidden = True
                elif student and line.strip() == '### END SOLUTION':
                    if not hidden:
                        raise ValueError('Unmatched solution end')
                    hidden = False
                elif not hidden and '# --8<--' not in line:
                    lines.append(line)
            if hidden:
                raise ValueError('Unclosed solution block')
            body = '\n'.join(lines)
        cells.append((kind, body))
    return cells


def main():
    checks, errors = [], []
    def check(name, condition):
        (checks if condition else errors).append(name)
    report = json.loads((ROOT / 'validation/execution_report.json').read_text())
    recorded = {r['chapter']: r for r in report['chapters']}
    for number in range(8):
        tag = f'chap{number:02d}'
        folder = ROOT / 'code' / tag
        source = (folder / f'{tag}_master.py').read_text()
        sha = hashlib.sha256(source.encode()).hexdigest()
        check(f'{tag}: recorded executed master hash', recorded[number]['master_sha256'] == sha)
        for base in [folder / 'generated', ROOT / 'textbook/generated' / tag]:
            manifest = json.loads((base / 'source_manifest.json').read_text())
            check(f'{base.relative_to(ROOT)}: master hash', manifest['master_sha256'] == sha)
        for kind in ['complete', 'student']:
            path = ROOT / 'notebook' / kind / f'{tag}_{kind}.ipynb'
            notebook = json.loads(path.read_text())
            local = json.loads((folder / path.name).read_text())
            check(f'{tag} {kind}: distributed/build notebook equality', notebook == local)
            expected = source_cells(source, student=kind == 'student')
            check(f'{tag} {kind}: cell count', len(expected) == len(notebook['cells']))
            for index, ((cell_type, text), actual) in enumerate(zip(expected, notebook['cells'])):
                check(f'{tag} {kind} cell {index}: type', actual['cell_type'] == cell_type)
                if cell_type == 'code' or kind == 'complete':
                    check(f'{tag} {kind} cell {index}: source', ''.join(actual['source']) == text)
                if cell_type == 'code':
                    ast.parse(''.join(actual['source']), filename=str(path))
                    if kind == 'complete':
                        check(f'{tag} cell {index}: stored execution', actual.get('execution_count') is not None)
                        check(f'{tag} cell {index}: stored error-free output', not any(o.get('output_type') == 'error' for o in actual.get('outputs', [])))
        for match in re.finditer(r'(?ms)^# --8<-- \[start:([^\]]+)\]\n(.*?)^# --8<-- \[end:\1\]', source):
            name, text = match.groups()
            text = '\n'.join(line for line in text.splitlines() if line.strip() not in ['### BEGIN SOLUTION', '### END SOLUTION']) + '\n'
            for base in [folder / 'generated', ROOT / 'textbook/generated' / tag]:
                path = base / f'{name}.py'
                check(f'{path.relative_to(ROOT)}: extracted source', path.is_file() and path.read_text() == text)
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    checks.append('All Python sources parse')
    metadata = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for key in ['outline', 'style_guide', 'validation']:
        check(f'Release metadata: {key} exists', (ROOT / metadata[key]).is_file())
    print(f'Source checks: {len(checks)} passed, {len(errors)} failed. Stored execution is not a fresh numerical test.')
    for error in errors:
        print('FAILED:', error)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
