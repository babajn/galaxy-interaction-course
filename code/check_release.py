#!/usr/bin/env python3
"""配布物の正本・生成物・本文構造を検証します。物理の独立テストは各章にあります。"""
from pathlib import Path
import hashlib
import json
import re
import nbformat
from check_sources import main as check_sources

check_sources()

ROOT = Path(__file__).resolve().parents[1]
TEXT = ROOT / 'textbook'
checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


for number in range(8):
    tag = f'chap{number:02d}'
    folder = ROOT / 'code' / tag
    source = (folder / f'{tag}_master.py').read_text(encoding='utf-8')
    chapter = (TEXT / 'chapters' / f'{tag}.tex').read_text(encoding='utf-8')
    goals = re.findall(r'\\begin\{goalbox\}.*?\\end\{goalbox\}', chapter, re.S)
    if number == 0:
        check(f'{tag}: preparation chapter has no goalbox', len(goals)==0)
        check(f'{tag}: preparation chapter has no exercises', r'\chapterexercises' not in chapter)
    else:
        check(f'{tag}: one goalbox, 3--5 items', len(goals)==1 and 3 <= goals[0].count(r'\item') <= 5)
        check(f'{tag}: chapter ends with exercises only', chapter.count(r'\chapterexercises')==1 and r'\chapteranswers' not in chapter and r'\begin{chapterbib}' not in chapter)
    check(f'{tag}: no handwritten Python verbatim', r'\begin{verbatim}' not in chapter)
    check(f'{tag}: no whole-TikZ resizing', r'\resizebox' not in chapter)
    check(f'{tag}: current title', '銀河形成シミュレーション' not in chapter)
    region_pattern = r'(?ms)^# --8<-- \[start:([^\]]+)\]\n(.*?)^# --8<-- \[end:\1\]'
    for match in re.finditer(region_pattern, source):
        name, text = match.groups()
        text = '\n'.join(line for line in text.splitlines() if line.strip() not in ['### BEGIN SOLUTION','### END SOLUTION']) + '\n'
        exported = TEXT / 'generated' / tag / f'{name}.py'
        check(f'{tag}: region {name}', exported.is_file() and exported.read_text(encoding='utf-8')==text)
    nb = nbformat.read(ROOT / 'notebook/complete' / f'{tag}_complete.ipynb', as_version=4)
    code = [c for c in nb.cells if c.cell_type=='code']
    check(f'{tag}: complete notebook executed', all(c.execution_count is not None for c in code))
    check(f'{tag}: no execution errors', not any(o.output_type=='error' for c in code for o in c.outputs))
    student = nbformat.read(ROOT / 'notebook/student' / f'{tag}_student.ipynb', as_version=4)
    check(f'{tag}: student title', any('学生版' in c.source for c in student.cells if c.cell_type=='markdown'))
    if number == 0:
        check(f'{tag}: no exercise TODO in preparation notebook', not any('NotImplementedError' in c.source for c in student.cells if c.cell_type=='code'))
    else:
        check(f'{tag}: student TODO guard', any('NotImplementedError' in c.source for c in student.cells if c.cell_type=='code'))
    manifest = json.loads((TEXT / 'generated' / tag / 'source_manifest.json').read_text())
    check(f'{tag}: extracted-source hash', manifest['master_sha256']==hashlib.sha256(source.encode()).hexdigest())

all_text = '\n'.join(p.read_text() for p in (TEXT/'chapters').glob('chap*.tex'))
for item in json.loads((TEXT/'figures/tikz/figure_index.json').read_text()):
    label = item['label']
    check('TikZ label: '+label, all_text.count(r'\label{'+label+'}')==1 and r'\ref{'+label+'}' in all_text)
    check('TikZ source: '+item['file'], (TEXT/'figures/tikz'/f"{item['file']}.tex").is_file())
log = (TEXT/'main.log').read_text(errors='replace')
for forbidden in ['Undefined control sequence','There were undefined references','multiply defined','Overfull \\hbox','Overfull \\vbox','Missing character:','Package pgfkeys Error']:
    check('TeX: '+forbidden, forbidden not in log)
aux = (TEXT/'main.aux').read_text()
check('Chapter0 table numbering', bool(re.search(r'\\newlabel\{tab:ch00-files\}\{\{0\.1\}', aux)))
check('No font files in package', not any(p.suffix.lower() in ['.ttf','.otf','.ttc','.woff','.woff2'] for p in ROOT.rglob('*')))
check('Required runtime records', (ROOT/'code/benchmark.json').is_file() and (ROOT/'code/constraints-tested.txt').is_file())
check('Backmatter sources', (TEXT/'backmatter/answers.tex').is_file() and (TEXT/'backmatter/references.tex').is_file())
check('No superseded basic textbook citations', all(key not in all_text for key in ['chap03:landau', 'chap04:landau', 'chap01:butcher', 'mitsuda2023']))

check('Current guide PDFs', (TEXT/'guides/style_guide_v1_8.pdf').is_file() and (TEXT/'guides/outline_v9.pdf').is_file())
report = json.loads((ROOT/'validation/execution_report.json').read_text())
for item in report['chapters']:
    tag = f"chap{item['chapter']:02d}"
    current = hashlib.sha256((ROOT/'code'/tag/f'{tag}_master.py').read_bytes()).hexdigest()
    check(tag+': executed master matches release', current==item['master_sha256'])

import fitz
pdf = fitz.open(TEXT/'main.pdf')
top_titles=[r[1] for r in pdf.get_toc() if r[0]==1]
check('Eight core chapter bookmarks', all(title in top_titles for title in ['計算環境を準備する','Pythonで天体の運動を計算する','2体問題を計算する','銀河の中で星を動かす','テスト粒子で円盤銀河を作る','2つの銀河を接近させる','銀河相互作用を調べる','動力学的摩擦で軌道減衰を調べる']))
check('Backmatter bookmarks', '演習の解答・着眼点' in top_titles and '参考図書' in top_titles)
check('Physical page size', abs(pdf[0].rect.width*25.4/72-157)<0.1 and abs(pdf[0].rect.height*25.4/72-221)<0.1)
result = dict(passed=len(checks), failed=0, pages=len(pdf), figures=len(re.findall(r'\\begin\{figure\}',all_text)),
              complete_code_cells=sum(r['code_cells'] for r in report['chapters']),
              independent_tests=sum(r['independent_tests'] for r in report['chapters']), checks=checks)
(ROOT/'validation/release_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(f"Release checks: {len(checks)} passed; PDF {len(pdf)} pages; {result['complete_code_cells']} executed cells; {result['independent_tests']} independent checks.")
