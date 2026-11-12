#!/usr/bin/env python3
"""Jupytext masterからNotebook・コード抜粋・数値表を再生成します。"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import shutil
import sys
import nbformat
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "chap06_master.py"


def hide_solutions(text):
    lines, result = text.splitlines(), []
    inside = False
    for line in lines:
        if line.strip() == "### BEGIN SOLUTION":
            if inside:
                raise ValueError("Nested solution block")
            indent = line[:len(line) - len(line.lstrip())]
            result.extend([indent + "# TODO: 本文の式を参照して、この部分を実装してください。",
                           indent + 'raise NotImplementedError("Complete the TODO block before running this cell.")'])
            inside = True
        elif line.strip() == "### END SOLUTION":
            if not inside:
                raise ValueError("Unmatched solution end")
            inside = False
        elif not inside:
            result.append(line)
    if inside:
        raise ValueError("Unclosed solution block")
    return "\n".join(result)

def make_notebook(source, student=False):
    cells = []
    blocks = re.split(r"(?m)^# %%(.*)\n", source)
    for i in range(1, len(blocks), 2):
        header, text = blocks[i], blocks[i + 1].strip("\n")
        if "[markdown]" in header:
            text = "\n".join(re.sub(r"^# ?", "", line) for line in text.splitlines())
            if student:
                text = text.replace("**完成版 / Chap.6--7分離版 v2.3**", "**学生版 / Chap.6--7分離版 v2.3**\n\n未解答セルは `NotImplementedError` で停止します。本文を参照してTODOを埋めてから先へ進んでください。")
            cell = nbformat.v4.new_markdown_cell(text)
        else:
            if student:
                text = hide_solutions(text)
            # Notebookでは編集・実行に不要な抜粋用マーカーだけ除きます。
            text = "\n".join(line for line in text.splitlines() if "# --8<--" not in line)
            cell = nbformat.v4.new_code_cell(text)
            if "NotImplementedError" in text:
                cell.metadata["tags"] = ["exercise"]
        cell["id"] = hashlib.sha256(f"{i}:{cell.source}".encode()).hexdigest()[:12]
        cells.append(cell)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "chapter_version": "2.3"}
    nbformat.validate(nb)
    return nb

def extract_regions(source):
    folder = ROOT / "generated"
    folder.mkdir(exist_ok=True)
    pattern = r"(?ms)^# --8<-- \[start:([^\]]+)\]\n(.*?)^# --8<-- \[end:\1\]"
    regions = {}
    for match in re.finditer(pattern, source):
        name, text = match.groups()
        text = "\n".join(line for line in text.splitlines() if line.strip() not in
                         ["### BEGIN SOLUTION", "### END SOLUTION"])
        (folder / f"{name}.py").write_text(text + "\n", encoding="utf-8")
        regions[name] = hashlib.sha256(text.encode()).hexdigest()
    (folder / "source_manifest.json").write_text(json.dumps({"master_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                                          "regions": regions}, indent=2), encoding="utf-8")
    return regions

def sci(value, digits=3):
    mantissa, exponent = f"{value:.{digits-1}e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"

def generate_values():
    s = json.loads((ROOT / 'data/validation.json').read_text(encoding='utf-8'))
    by_name = {row['name']: row for row in s['comparison']}
    values = {
        'valBaseOne': f"{by_name['PP']['fractions'][0]:.3f}",
        'valBaseTwo': f"{by_name['PP']['fractions'][1]:.3f}",
        'valRetroTwo': f"{by_name['PR']['fractions'][1]:.3f}",
        'valSpinDifference': f"{by_name['PP']['fractions'][1]-by_name['PR']['fractions'][1]:.3f}",
        'valBaseTime': f"{by_name['PP']['t_used']:.3f}", 'valBaseTau': f"{by_name['PP']['tau_used']:.6f}",
        'valQsmallTime': f"{by_name['Q0.1']['t_peri']:.6f}",
        'valFinalPPTwo': f"{by_name['PP']['fractions_final'][1]:.3f}", 'valFinalPRTwo': f"{by_name['PR']['fractions_final'][1]:.3f}",
        'valPeriSeries': ','.join(f"{by_name[k]['fractions'][1]:.3f}" for k in ['P5','PP','P12']),
        'valMassSeriesOne': ','.join(f"{by_name[k]['fractions'][0]:.3f}" for k in ['PP','Q0.5','Q0.1']),
        'valMassSeriesTwo': ','.join(f"{by_name[k]['fractions'][1]:.3f}" for k in ['PP','Q0.5','Q0.1']),
        'valSmallTauFixed': f"{by_name['PP']['t_used']-by_name['Q0.1']['t_peri']:.6f}",
        'valEpochDifferenceMyr': f"{1000*(by_name['Q0.1']['t_peri']-by_name['PP']['t_peri']):.0f}",
        'valThresholdDifferences': ','.join(f'{row[3]:.3f}' for row in s['threshold_rows']),
        'valSeedDifferences': ','.join(f'{row[1]-row[2]:.3f}' for row in s['seed_rows']),
        'valPfiveMax': sci(next(r['max_difference'] for r in s['resolution'] if r['name']=='P5')),
        'valPfiveRms': sci(next(r['rms_difference'] for r in s['resolution'] if r['name']=='P5')),
        'valSoftFive': f"{s['softening_rows'][0][2]:.3f}",
        'valSoftRatios': ','.join(f'{row[2]:.3f}' for row in s['softening_rows']),
    }
    (ROOT/'generated/values.tex').write_text('\n'.join(rf'\newcommand{{\{k}}}{{{v}}}' for k,v in values.items())+'\n',encoding='utf-8')
    def ms(v): return '$0$' if v == 0 else '$'+sci(v)+'$'
    def table(name, cols, head, rows):
        text = r'\begin{tabular}{'+cols+r'}\toprule'+'\n'+head+r' \\'+'\n'+r'\midrule'+'\n'
        text += '\n'.join(' & '.join(row)+r' \\' for row in rows)
        text += '\n'+r'\bottomrule\end{tabular}'+'\n'
        (ROOT/'generated'/f'{name}.tex').write_text(text,encoding='utf-8')
    table('timing_table', 'lrrrrr', r'ケース & $q$ & $D_{\rm p}$ [kpc] & $t_{\rm p}$ [Gyr] & $t_{\rm used}$ [Gyr] & $\tau_{\rm used}$ [Gyr]',
          [[r['name'], f"{r['q']:.1f}", f"{r['d_peri']:.0f}", f"{r['t_peri']:.5f}", f"{r['t_used']:.3f}", f"{r['tau_used']:.5f}"] for r in s['comparison']])
    table('response_table','lrrrr', r'ケース & $f_1(\tau\simeq0.20)$ & $f_2(\tau\simeq0.20)$ & $f_1(t=0.60)$ & $f_2(t=0.60)$',
          [[r['name']]+[f'{x:.3f}' for x in r['fractions']+r['fractions_final']] for r in s['comparison']])
    table('threshold_table','rrrr',r'$s_{\rm cut}$ [kpc] & $f_{2,\rm PP}$ & $f_{2,\rm PR}$ & PP $-$ PR',[[f'{r[0]:.0f}']+[f'{x:.3f}' for x in r[1:]] for r in s['threshold_rows']])
    table('seed_table','rrrr',r'seed & $f_{2,\rm PP}$ & $f_{2,\rm PR}$ & PP $-$ PR',[[str(r[0]),f'{r[1]:.3f}',f'{r[2]:.3f}',f'{r[1]-r[2]:.3f}'] for r in s['seed_rows']])
    table('resolution_table','lrrrrr',r'ケース & $\delta_{\max}$ & $\delta_{\rm rms}$ & $\Delta f_1$ & $\Delta f_2$ & 細かい側の $e_{E,\rm c}$',[[r['name'],ms(r['max_difference']),ms(r['rms_difference']),f"{r['fine_f'][0]-r['coarse_f'][0]:+.3f}",f"{r['fine_f'][1]-r['coarse_f'][1]:+.3f}",ms(r['fine_energy_error'])] for r in s['resolution']])
    table('softening_table','rrr',r'$D$ [kpc] & $\sqrt{D^2+b^2}$ [kpc] & $A_{\rm eff}/A_{\rm point}$',[[f'{r[0]:.0f}', f'{r[1]:.3f}', f'{r[2]:.3f}'] for r in s['softening_rows']])
    import csv
    with (ROOT/'data/comparison_summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream); writer.writerow(['case','q','Dperi_kpc','Dmin_kpc','tperi_Gyr','t_used_Gyr','tau_used_Gyr','f1_aligned','f2_aligned','f1_final','f2_final'])
        for r in s['comparison']: writer.writerow([r['name'],r['q'],r['d_peri'],r['d_min'],r['t_peri'],r['t_used'],r['tau_used'],*r['fractions'],*r['fractions_final']])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--pdf',action='store_true')
    args=parser.parse_args()
    source=SOURCE.read_text(encoding='utf-8')
    regions=extract_regions(source)
    complete=make_notebook(source)
    student=make_notebook(source,student=True)
    nbformat.write(student,ROOT/'chap06_student.ipynb')
    if args.execute:
        from nbclient import NotebookClient
        client=NotebookClient(complete,timeout=240,kernel_name='python3',allow_errors=False,resources={'metadata':{'path':str(ROOT)}})
        client.execute()
    elif (ROOT/'chap06_complete.ipynb').exists():
        existing=nbformat.read(ROOT/'chap06_complete.ipynb',as_version=4)
        if [c.source for c in existing.cells]==[c.source for c in complete.cells]: complete=existing
    nbformat.write(complete,ROOT/'chap06_complete.ipynb')
    if (ROOT/'data/validation.json').exists(): generate_values()
    if args.pdf:
        subprocess.run([sys.executable, str(ROOT.parent / "rebuild.py"), "--pdf"], check=True)
    print(f'Generated {len(regions)} code regions and two notebooks.')

if __name__=='__main__': main()
