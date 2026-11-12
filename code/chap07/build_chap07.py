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
SOURCE = ROOT / "chap07_master.py"


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
    values = {
        'valRdf': f"{s['df_final_radius']:.6f}", 'valRno': f"{s['no_df_final_radius']:.6f}",
        'valEno': sci(s['no_df_energy_error']), 'valLno': sci(s['no_df_angular_error']),
        'valEdf': f"{s['df_final_energy_ratio']:.6f}", 'valLdf': f"{s['df_final_angular_ratio']:.6f}",
        'valWorkResidual': sci(s['work_residual']), 'valDfRatio': f"{s['df_ratios'][-1]:.5f}",
        'valSatMax': sci(s['sat_maxdiff']), 'valSatRms': sci(s['sat_rmsdiff']),
        'valLambdaSeries': ','.join(f"{r[1]:.3f}" for r in s['lambda_rows']),
    }
    (ROOT/'generated/values.tex').write_text('\n'.join(rf'\newcommand{{\{k}}}{{{v}}}' for k,v in values.items())+'\n',encoding='utf-8')
    def ms(v): return '$0$' if v == 0 else '$'+sci(v)+'$'
    def table(name, cols, head, rows):
        text = r'\begin{tabular}{'+cols+r'}\toprule'+'\n'+head+r' \\'+'\n'+r'\midrule'+'\n'
        text += '\n'.join(' & '.join(row)+r' \\' for row in rows)
        text += '\n'+r'\bottomrule\end{tabular}'+'\n'
        (ROOT/'generated'/f'{name}.tex').write_text(text,encoding='utf-8')
    rows=[]
    for j,r in enumerate(s['df_resolution']):
        diff='---' if j==len(s['df_resolution'])-1 else ms(s['df_self'][j][2])
        ratio='---' if j==0 or j==len(s['df_resolution'])-1 else f"{s['df_ratios'][j-1]:.5f}"
        rows.append([str(int(r[0])),f'{1000*r[1]:.1f}',f'{r[2]:.8f}',diff,ratio,ms(r[3])])
    table('df_resolution_table','rrrrrr',r'$N_{\rm step}$ & $\Delta t$ [Myr] & 最終半径 [kpc] & 次行との $\delta_{\rm s}$ & 差の比 & $e_{\rm work}$', rows)
    table('lambda_table','rr',r'$\ln\Lambda$ & 1.50 Gyr後の半径 [kpc]',[[f'{r[0]:.6f}',f'{r[1]:.6f}'] for r in s['lambda_rows']])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--pdf',action='store_true')
    args=parser.parse_args()
    source=SOURCE.read_text(encoding='utf-8')
    regions=extract_regions(source)
    complete=make_notebook(source)
    student=make_notebook(source,student=True)
    nbformat.write(student,ROOT/'chap07_student.ipynb')
    if args.execute:
        from nbclient import NotebookClient
        client=NotebookClient(complete,timeout=240,kernel_name='python3',allow_errors=False,resources={'metadata':{'path':str(ROOT)}})
        client.execute()
    elif (ROOT/'chap07_complete.ipynb').exists():
        existing=nbformat.read(ROOT/'chap07_complete.ipynb',as_version=4)
        if [c.source for c in existing.cells]==[c.source for c in complete.cells]: complete=existing
    nbformat.write(complete,ROOT/'chap07_complete.ipynb')
    if (ROOT/'data/validation.json').exists(): generate_values()
    if args.pdf:
        subprocess.run([sys.executable, str(ROOT.parent / "rebuild.py"), "--pdf"], check=True)
    print(f'Generated {len(regions)} code regions and two notebooks.')

if __name__=='__main__': main()
