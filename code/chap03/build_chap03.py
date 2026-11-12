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
SOURCE = ROOT / "chap03_master.py"


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
    values = {
        "valRhoZero": sci(s["density_central"], 5), "valVc": f'{s["vc_kms"]:.6f}',
        "valVcInt": f'{s["vc_internal"]:.6f}', "valTc": f'{s["tc_Gyr"]:.6f}',
        "valTend": f'{s["t_end_Gyr"]:.6f}', "valDt": f'{s["dt_Myr"]:.6f}',
        "valMassError": sci(s["mass_error"]), "valMassOrder": f'{s["mass_orders"][-1]:.4f}',
        "valRh": f'{s["r_half"]:.9f}', "valRhExact": f'{s["r_half_exact"]:.9f}',
        "valRhError": sci(s["root_error"]), "valRootResidual": sci(abs(s["root_residual"])),
        "valTrueResidual": sci(abs(s["true_mass_residual"])), "valRootWidth": sci(s["root_trace"][-1][5]),
        "valRootHalvings": str(int(s["root_trace"][-1][0])), "valRootCalls": str(len(s["root_trace"])),
        "valOrbitOrder": f'{s["orbit_orders"][-1]:.5f}', "valPeri": f'{s["r_peri"]:.9f}',
        "valPeriSaved": f'{s["orbit_errors"]["Noncircular"][2]:.9f}',
        "valCircEnergy": sci(s["orbit_errors"]["Circular"][0]),
        "valNoncEnergy": sci(s["orbit_errors"]["Noncircular"][0]),
        "valRootFloor": sci(next(row[2] for row in s["root_convergence"] if row[0] == 64)),
        "valNoncRatio": f'{s["nonc_ratios"][-1]:.5f}'
    }
    values.update({
        'valAccelX': sci(-s['vc_internal']**2/s['r0_kpc'], 5),
        'valOutsideFractions': ','.join(f'{1-row[1]:.6f}' for row in s['cutoffs']),
        'valScaleVcSeries': ','.join(f'{row[1]:.3f}' for row in s['scale_rows']),
        'valScaleRhSeries': ','.join(f'{row[2]:.6f}' for row in s['scale_rows']),
    })
    (ROOT / "generated" / "values.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()) + "\n", encoding="utf-8")

    def ms(value):
        return "$0$" if value == 0.0 else f"${sci(value)}$"

    def table(name, columns, header, rows):
        text = r"\begin{tabular}{" + columns + r"}\toprule" + "\n" + header + r" \\" + "\n" + r"\midrule" + "\n"
        text += "\n".join(" & ".join(row) + r" \\" for row in rows)
        text += "\n" + r"\bottomrule\end{tabular}" + "\n"
        (ROOT / "generated" / f"{name}.tex").write_text(text, encoding="utf-8")

    rows=[]
    for j,row in enumerate(s["mass_convergence"]):
        order = "---" if j==0 else f'{s["mass_orders"][j-1]:.4f}'
        rows.append([str(int(row[0])), f"{row[1]:.6f}", ms(row[2]), order])
    table("mass_convergence_table", "rrrr", r"$N_r$ & $\Delta r\ [\mathrm{kpc}]$ & $e_M$ & $p_M$", rows)
    rows=[[f"{row[0]:.0f}", f"{row[1]:.9f}", f"{1-row[1]:.9f}"] for row in s["cutoffs"]]
    table("cutoff_table", "rrr", r"$r_{\rm tab}/a$ & $M(<r_{\rm tab})/M$ & 範囲外の質量$/M$", rows)
    rows=[]
    for j,row in enumerate(s["root_convergence"]):
        rows.append([str(int(row[0])), f"{row[1]:.9f}", ms(row[2]), ms(row[3])])
    table("root_convergence_table", "rrrr", r"$N_r$ & $\widehat r_h\ [\mathrm{kpc}]$ & $|\widehat r_h-r_h|\ [\mathrm{kpc}]$ & $|F(\widehat r_h)|$", rows)
    rows=[[ms(row[0]), ms(row[1]), ms(row[2]), ms(row[3])] for row in s["tolerances"]]
    table("root_tolerance_table", "rrrr", r"$\tau_r\ [\mathrm{kpc}]$ & 最終半区間幅$\ [\mathrm{kpc}]$ & 真の半径との差$\ [\mathrm{kpc}]$ & $|F|$", rows)
    rows=[[label, ms(s["orbit_errors"][name][0]), ms(s["orbit_errors"][name][1]), f'{s["orbit_errors"][name][2]:.6f}'] for name,label in [("Circular","円"),("Noncircular","非円")]]
    table("conservation_table", "lrrr", r"軌道 & $e_\varepsilon$ & $e_\ell$ & 保存された最小半径$\ [\mathrm{kpc}]$", rows)
    rows=[]
    for j,row in enumerate(s["orbit_convergence"]):
        order = "---" if j==0 else f'{s["orbit_orders"][j-1]:.5f}'
        rows.append([str(int(row[0])), f"{1000*row[1]:.6f}", ms(row[2]), order])
    table("orbit_convergence_table", "rrrr", r"$N_{\rm step}$ & $\Delta t\ [\mathrm{Myr}]$ & $e_r$ & $p_r$", rows)
    rows=[[str(s["nonc_counts"][j]), str(s["nonc_counts"][j+1]), ms(v), "---" if j==0 else f'{s["nonc_ratios"][j-1]:.5f}'] for j,v in enumerate(s["nonc_differences"])]
    table("noncircular_table", "rrrr", r"粗い計算のステップ数 & 細かい計算のステップ数 & 終点の差$/r_0$ & 前の行との比", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="完成版Notebookを実行して図と数値表を更新します")
    parser.add_argument("--pdf", action="store_true", help="XeLaTeXを2回実行してPDFを作ります")
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    regions = extract_regions(source)
    complete = make_notebook(source)
    student = make_notebook(source, student=True)
    nbformat.write(student, ROOT / "chap03_student.ipynb")
    if args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(complete, timeout=120, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
    if not args.execute and (ROOT / "chap03_complete.ipynb").exists():
        existing = nbformat.read(ROOT / "chap03_complete.ipynb", as_version=4)
        if [c.source for c in existing.cells] == [c.source for c in complete.cells]:
            complete = existing
    nbformat.write(complete, ROOT / "chap03_complete.ipynb")
    if (ROOT / "data" / "validation.json").exists():
        generate_values()
    if args.pdf:
        subprocess.run([sys.executable, str(ROOT.parent / "rebuild.py"), "--pdf"], check=True)
    print(f"Generated {len(regions)} code regions and two notebooks.")

if __name__ == "__main__":
    main()
