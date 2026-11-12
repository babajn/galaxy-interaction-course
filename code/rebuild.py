#!/usr/bin/env python3
"""正本 -> 完成版/学生版Notebook -> 実行結果 -> 本文の抜粋/図表 -> PDF。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
TEXT = ROOT / 'textbook'
PREFIX = ['chzero', 'chone', 'chtwo', 'chthree', 'chfour', 'chfive', 'chsix', 'chseven']


def run(command, cwd, log):
    start = perf_counter()
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run(command, cwd=cwd, env={**os.environ, 'MPLBACKEND': 'Agg'}, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Failed: {command}\n{log.read_text(encoding="utf-8")[-10000:]}')
    return perf_counter() - start


def synchronize(number):
    tag = f'chap{number:02d}'
    folder = ROOT / 'code' / tag
    generated = folder / 'generated'
    target = TEXT / 'generated' / tag
    target.mkdir(parents=True, exist_ok=True)
    value_text = (generated / 'values.tex').read_text(encoding='utf-8')
    names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', value_text)
    mapping = {name: PREFIX[number] + name for name in names}
    for source in generated.iterdir():
        if source.suffix == '.tex':
            text = source.read_text(encoding='utf-8')
            text = re.sub(r'\\([A-Za-z]+)', lambda m: '\\' + mapping.get(m[1], m[1]), text)
            (target / source.name).write_text(text, encoding='utf-8')
        elif source.is_file():
            shutil.copy2(source, target / source.name)
    for kind in ['complete', 'student']:
        shutil.copy2(folder / f'{tag}_{kind}.ipynb', ROOT / 'notebook' / kind / f'{tag}_{kind}.ipynb')
    for source in (folder / 'figures').glob('*'):
        if source.is_file() and source.suffix.lower() in ['.pdf', '.png', '.svg']:
            shutil.copy2(source, TEXT / 'figures' / source.name)


def write_runtime_table(report):
    values = TEXT / 'generated/chap00/values.tex'
    text = values.read_text(encoding='utf-8')
    cpu = report['environment'].get('cpu', platform.machine())
    cpu = cpu.replace('_', r'\_').replace('&', r'\&')
    text = re.sub(r'(?m)^\\newcommand\{\\chzerovalCPU\}\{[^\n]*\}\n?', '', text)
    text += rf'\newcommand{{\chzerovalCPU}}{{{cpu}}}' + '\n'
    values.write_text(text, encoding='utf-8')
    rows = []
    for item in report['chapters']:
        if 'execution_seconds' in item:
            rows.append(f"{item['chapter']} & {item['code_cells']} & {item['execution_seconds']:.1f}" + r' \\')
    result = r'\begin{tabular}{crr}\toprule 章 & コードセル数 & 全実行 [s] \\\midrule' + '\n'
    result += '\n'.join(rows) + '\n' + r'\bottomrule\end{tabular}' + '\n'
    (TEXT / 'generated/chap00/runtime_table.tex').write_text(result, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='完成版全セルを新しいカーネルで実行')
    parser.add_argument('--test', action='store_true', help='章別の独立テストを実行')
    parser.add_argument('--pdf', action='store_true', help='本文をXeLaTeXで3回組版')
    parser.add_argument('--chapters', nargs='+', type=int, default=list(range(8)))
    args = parser.parse_args()
    if any(n not in range(8) for n in args.chapters):
        parser.error('chapter must be 0..7')
    import nbformat
    logs = ROOT / 'validation'
    logs.mkdir(exist_ok=True)
    report_path = logs / 'execution_report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {'chapters': []}
    cpu = platform.processor() or platform.machine()
    if Path('/proc/cpuinfo').exists():
        found = re.search(r'model name\s*:\s*(.+)', Path('/proc/cpuinfo').read_text())
        if found: cpu = found[1]
    import numpy as np
    import matplotlib
    current_environment = dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__, system=platform.platform(), cpu=cpu)
    report['last_build_environment'] = current_environment
    if args.execute or 'environment' not in report:
        report['environment'] = current_environment
    existing = {r['chapter']: r for r in report['chapters']}
    for number in args.chapters:
        tag = f'chap{number:02d}'
        folder = ROOT / 'code' / tag
        item = existing.get(number, {'chapter': number})
        command = [sys.executable, f'build_{tag}.py'] + (['--execute'] if args.execute else [])
        seconds = run(command, folder, logs / f'{tag}_build.log')
        nb = nbformat.read(folder / f'{tag}_complete.ipynb', as_version=4)
        code = [cell for cell in nb.cells if cell.cell_type == 'code']
        item['code_cells'] = len(code)
        if args.execute:
            if not all(cell.execution_count is not None for cell in code):
                raise RuntimeError(f'{tag}: unexecuted cell')
            if any(out.output_type == 'error' for c in code for out in c.outputs):
                raise RuntimeError(f'{tag}: notebook error')
            item['execution_seconds'] = seconds
            item['execution_environment'] = current_environment
            item['master_sha256'] = hashlib.sha256((folder / f'{tag}_master.py').read_bytes()).hexdigest()
        synchronize(number)
        if args.test:
            run([sys.executable, f'test_{tag}.py'], folder, logs / f'{tag}_tests.log')
            output = (logs / f'{tag}_tests.log').read_text()
            match = re.search(r'(\d+) passed, 0 failed', output) or re.search(r'All (\d+) independent checks passed', output) or re.search(r'(\d+) checks passed', output)
            if match:
                item['independent_tests'] = int(match[1])
            elif output.lstrip().startswith('{'):
                check_data = json.loads(output)
                if 'checks' in check_data and all(check_data['checks'].values()):
                    item['independent_tests'] = len(check_data['checks'])
        existing[number] = item
        report['chapters'] = [existing[n] for n in sorted(existing)]
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (ROOT / 'code/benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{tag}: {len(code)} cells; build {seconds:.1f} s; tests {item.get("independent_tests", "not requested")}', flush=True)
    if args.pdf:
        for count in range(3):
            run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], TEXT, logs / f'xelatex_{count+1}.log')
        run([sys.executable, 'export_chapter_pdfs.py'], TEXT, logs / 'chapter_pdf_export.log')
        for name, folder in [('tikz_figures', TEXT), ('style_guide_v1_8', TEXT / 'guides'), ('outline_v9', TEXT / 'guides')]:
            for count in range(2):
                run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', name+'.tex'], folder, logs / f'{name}_{count+1}.log')
        print('Integrated PDF and Chapter0--7 PDFs rebuilt.', flush=True)


if __name__ == '__main__':
    main()
