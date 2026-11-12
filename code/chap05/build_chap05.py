#!/usr/bin/env python3
"""Jupytext percentの正本からNotebook・コード抜粋・数値表を生成します。"""
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
SOURCE = ROOT / "chap05_master.py"


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
                text = text.replace("**完成版 / 全章改訂 v2.0**", "**学生版 / 全章改訂 v2.0**\n\n未解答セルは `NotImplementedError` で停止します。本文を参照してTODOを埋めてから先へ進んでください。")
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
                   "language_info": {"name": "python"}, "chapter_version": "2.0"}
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
    s = json.loads((ROOT / "data" / "validation.json").read_text(encoding="utf-8"))
    v = s["initial_relative_velocity_kms"]
    values = {
        "valVr": f"{v[0]:.6f}", "valVt": f"{v[1]:.6f}", "valSpeed": f'{s["initial_relative_speed_kms"]:.6f}',
        "valPeri": f'{s["sampled_dmin_kpc"]:.9f}', "valTperi": f'{s["sampled_tperi_Gyr"]:.6f}',
        "valDt": f'{s["dt_Gyr"]*1000:.3f}', "valNstep": str(s["n_steps"]), "valForce": str(s["n_steps"]+1),
        "valSaveEvery": str(s["save_every"]), "valHistoryMB": f'{s["history_MB"]:.6f}',
        "valCenterMB": f'{s["center_history_MB"]:.6f}',
        "valEnergyError": sci(s["center_errors"][0]), "valAngularError": sci(s["center_errors"][1]),
        "valOuterOne": f'{s["outer_fraction_final"][0]:.3f}', "valOuterTwo": f'{s["outer_fraction_final"][1]:.3f}',
        "valLastRatio": f'{s["self_ratios"][-1]:.5f}',
        "valLastDifference": sci(s["self_convergence"][-1][4]),
        "valLastDifferenceKpc": f'{s["self_convergence"][-1][4]*s["disk_scale_kpc"]:.3f}',
        "valLastRMS": sci(s["self_convergence"][-1][5]),
        "valWatchRadius": f'{s["watch_initial_radius_kpc"]:.6f}',
        "valWatchChange": f'{s["watch_final_energy_change"]:.6f}',
        "valBoostDiff": sci(s["boost_difference_Rd"]),
        "valRetroOne": f'{s["retrograde_outer_fraction_final"][0]:.3f}',
        "valRetroTwo": f'{s["retrograde_outer_fraction_final"][1]:.3f}'
    }
    total_steps, number = s['n_steps'], 2 + sum(s['counts'])
    saved = [len(set(list(range(0,total_steps+1,stride))+[total_steps])) for stride in [1,s['save_every'],200]]
    values['valMemorySeries'] = ','.join(f'{8*4*number*count/1e6:.3f}' for count in saved)
    values['valSavedSeries'] = ','.join(str(count) for count in saved)
    (ROOT / "generated" / "values.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()) + "\n", encoding="utf-8")

    def ms(v):
        return "$0$" if v == 0 else "$" + sci(v) + "$"

    def table(name, columns, header, rows):
        text = r"\begin{tabular}{" + columns + r"}\toprule" + "\n" + header + r" \\" + "\n" + r"\midrule" + "\n"
        text += "\n".join(" & ".join(row) + r" \\" for row in rows)
        text += "\n" + r"\bottomrule\end{tabular}" + "\n"
        (ROOT / "generated" / f"{name}.tex").write_text(text, encoding="utf-8")

    table("center_errors_table", "lr", r"指標 & 実測値", [[label, ms(val)] for label,val in zip(
        [r"エネルギー $e_{E,{\rm c}}$", r"角運動量 $e_{L,{\rm c}}$", r"運動量 $e_{P,{\rm c}}$", r"重心位置 $e_{\rm cm}$"], s["center_errors"])])
    rows=[]
    for row in s["resolution_summary"]:
        rows.append([str(int(row[0])), f"{row[1]*1000:.3f}", f"{row[2]:.9f}", ms(row[3]), f"{row[7]:.3f}", f"{row[8]:.3f}"])
    table("resolution_table", "rrrrrr", r"$N_{\rm step}$ & $\Delta t\ [\mathrm{Myr}]$ & $D_{\min}\ [\mathrm{kpc}]$ & $e_{E,{\rm c}}$ & $f_1$ & $f_2$", rows)
    rows=[]
    for j,row in enumerate(s["self_convergence"]):
        ratio="---" if j==0 else f'{s["self_ratios"][j-1]:.5f}'
        rows.append([str(int(row[0])), str(int(row[1])), ms(row[3]), ms(row[4]), ms(row[5]), ratio])
    table("self_convergence_table", "rrrrrr", r"粗い計算 & 細かい計算 & $\delta_{\rm c}$ & $\delta_{\rm t}$ & $\delta_{\rm rms}$ & 前行との比", rows)
    import numpy as np
    values=np.loadtxt(ROOT / "data" / "outer_fraction.csv", delimiter=",", skiprows=1)
    rows=[]
    for t in s["snapshot_times_Gyr"]:
        j=int(np.argmin(np.abs(values[:,0]-t)))
        rows.append([f"{t:.2f}", f"{values[j,1]:.3f}", f"{values[j,2]:.3f}"])
    table("outer_fraction_table", "rrr", r"$t\ [\mathrm{Gyr}]$ & $f_1$ & $f_2$", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="完成版Notebookを実行して図と数値表を更新します")
    parser.add_argument("--pdf", action="store_true", help="XeLaTeXを2回実行してPDFを作ります")
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    regions = extract_regions(source)
    complete = make_notebook(source)
    student = make_notebook(source, student=True)
    nbformat.write(student, ROOT / "chap05_student.ipynb")
    if args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(complete, timeout=120, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
    if not args.execute and (ROOT / "chap05_complete.ipynb").exists():
        existing = nbformat.read(ROOT / "chap05_complete.ipynb", as_version=4)
        if [c.source for c in existing.cells] == [c.source for c in complete.cells]:
            complete = existing
    nbformat.write(complete, ROOT / "chap05_complete.ipynb")
    if (ROOT / "data" / "validation.json").exists():
        generate_values()
    if args.pdf:
        subprocess.run([sys.executable, str(ROOT.parent / "rebuild.py"), "--pdf"], check=True)
    print(f"Generated {len(regions)} code regions and two notebooks.")

if __name__ == "__main__":
    main()
