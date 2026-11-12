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
SOURCE = ROOT / "chap02_master.py"


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
    summary = json.loads((ROOT / "data" / "validation.json").read_text(encoding="utf-8"))
    values = {
        "valVc": f'{summary["v_c_kms"]:.6f}', "valVcInt": f'{summary["v_c_internal"]:.6f}',
        "valTc": f'{summary["tc_Gyr"]:.6f}', "valDt": f'{summary["dt_Myr"]:.6f}',
        "valMu": sci(summary["mu"], 6), "valMred": sci(summary["m_red"], 6),
        "valEllA": f'{summary["ellipse"]["a_kpc"]:.6f}', "valEllP": f'{summary["ellipse"]["period_Gyr"]:.6f}',
        "valEllRp": f'{summary["ellipse"]["rp_kpc"]:.6f}', "valEllMin": f'{summary["ellipse"]["sampled_rmin_kpc"]:.6f}',
        "valEllErrE": sci(summary["base_errors"]["Elliptic"][0]),
        "valOrderC": f'{summary["orders_circle"][-1]:.4f}', "valOrderEll": f'{summary["orders_ellipse"][-1]:.4f}',
        "valEncMin": f'{summary["encounter"]["rmin_kpc"]:.6f}', "valEncTime": f'{summary["encounter"]["sampled_time_Gyr"]:.4f}',
        "valEncVr": f'{summary["encounter"]["initial_relative_velocity_kms"][0]:.3f}',
        "valEncVt": f'{summary["encounter"]["initial_relative_velocity_kms"][1]:.3f}',
        "valBoostRel": sci(summary["boost_relative_error"]), "valBoostCm": sci(summary["boost_errors"][3]),
        "valReverseR": sci(summary["time_reversal"][0]), "valReverseV": sci(summary["time_reversal"][1]),
        "valHyperD": f'{summary["hyperbolic_final_D"]:.3f}'
    }
    m1, m2 = summary['masses']
    values.update({
        'valEllPeriods': f"{summary['tc_Gyr']/summary['ellipse']['period_Gyr']:.3f}",
        'valInitialXone': f"{-m2/(m1+m2)*summary['d0_kpc']:.6f}",
        'valInitialXtwo': f"{m1/(m1+m2)*summary['d0_kpc']:.6f}",
        'valInitialVone': f"{-m2/(m1+m2)*summary['v_c_kms']:.6f}",
        'valInitialVtwo': f"{m1/(m1+m2)*summary['v_c_kms']:.6f}",
        'valRadiusSeriesOne': ','.join(f'{row[1]:.6f}' for row in summary['mass_ratio']),
        'valRadiusSeriesTwo': ','.join(f'{row[2]:.6f}' for row in summary['mass_ratio']),
    })
    (ROOT / "generated" / "values.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()) + "\n", encoding="utf-8")

    def math_sci(value):
        return "$0$" if value == 0.0 else f"${sci(value)}$"

    def table(name, columns, header, rows):
        text = r"\begin{tabular}{" + columns + r"}\toprule" + "\n" + header + r" \\" + "\n" + r"\midrule" + "\n"
        text += "\n".join(" & ".join(row) + r" \\" for row in rows)
        text += "\n" + r"\bottomrule\end{tabular}" + "\n"
        (ROOT / "generated" / f"{name}.tex").write_text(text, encoding="utf-8")

    rows = [[label, *[math_sci(x) for x in summary["base_errors"][key]]] for key, label in [("Circular", "円"), ("Elliptic", "楕円"), ("Parabolic", "放物線")]]
    table("conservation_table", "lrrrr", r"軌道 & $e_E$ & $e_L$ & $e_P$ & $e_{\rm cm}$", rows)
    rows = []
    for i, row in enumerate(summary["convergence"]):
        oc = "---" if i == 0 else f'{summary["orders_circle"][i-1]:.4f}'
        oe = "---" if i == 0 else f'{summary["orders_ellipse"][i-1]:.4f}'
        rows.append([str(int(row[0])), math_sci(row[3]), oc, math_sci(row[4]), oe])
    table("convergence_table", "rrrrr", r"$N_{\rm step}$ & $e_{\rm circ}$ & $p$ & $e_{\rm ret}$ & $p$", rows)
    rows = [[f"{row[0]:.1f}", f"{row[1]:.6f}", f"{row[2]:.6f}", math_sci(row[3])] for row in summary["mass_ratio"]]
    table("mass_ratio_table", "rrrr", r"$q=M_2/M_1$ & $|\Rvec_1(0)|\ [\mathrm{kpc}]$ & $|\Rvec_2(0)|\ [\mathrm{kpc}]$ & 相対軌道の差$/D_0$", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="完成版Notebookを実行して図と数値表を更新します")
    parser.add_argument("--pdf", action="store_true", help="XeLaTeXを2回実行してPDFを作ります")
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    regions = extract_regions(source)
    complete = make_notebook(source)
    student = make_notebook(source, student=True)
    nbformat.write(student, ROOT / "chap02_student.ipynb")
    if args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(complete, timeout=120, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
    if not args.execute and (ROOT / "chap02_complete.ipynb").exists():
        existing = nbformat.read(ROOT / "chap02_complete.ipynb", as_version=4)
        if [c.source for c in existing.cells] == [c.source for c in complete.cells]:
            complete = existing
    nbformat.write(complete, ROOT / "chap02_complete.ipynb")
    if (ROOT / "data" / "validation.json").exists():
        generate_values()
    if args.pdf:
        subprocess.run([sys.executable, str(ROOT.parent / "rebuild.py"), "--pdf"], check=True)
    print(f"Generated {len(regions)} code regions and two notebooks.")

if __name__ == "__main__":
    main()
