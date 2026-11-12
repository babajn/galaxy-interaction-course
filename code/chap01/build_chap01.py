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
SOURCE = ROOT / "chap01_master.py"


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
    numeric = {
        "valVc": f'{summary["vc_km_s"]:.3f}', "valPeriod": f'{summary["period_Gyr"]:.6f}',
        "valDtMyr": f'{summary["dt_Gyr"]*1000:.4f}', "valMu": sci(summary["mu"], 7),
        "valEnergy": sci(summary["initial_specific_energy"], 5),
        "valAngular": sci(summary["initial_specific_angular_momentum"], 5),
        "valAcceleration": sci(summary["initial_acceleration_x"], 5),
        "errLfPos": sci(summary["leapfrog_errors"][0]), "errLfEnergy": sci(summary["leapfrog_errors"][1]),
        "errLfAngular": sci(summary["leapfrog_errors"][2]),
        "errRkPos": sci(summary["rk4_errors"][0]), "errRkEnergy": sci(summary["rk4_errors"][1]),
        "errRkAngular": sci(summary["rk4_errors"][2]), "errEulerPos": sci(summary["euler_errors"][0]),
        "errEulerEnergy": sci(summary["euler_errors"][1]), "errEulerAngular": sci(summary["euler_errors"][2]),
        "valLfOrder": f'{summary["orders_leapfrog"][-1]:.3f}',
        "valRkOrder": f'{summary["orders_rk4"][-1]:.3f}',
        "exDragEnergy": f'{summary["linear_drag_final_energy_ratio"]:.5f}',
        "longLfMax": sci(summary["long_time"]["leapfrog_max_signed_energy"]),
        "longRkFinal": sci(summary["long_time"]["rk4_final_signed_energy"]),
        "costLfPos": sci(summary["equal_cost"]["leapfrog_errors"][0]),
        "costRkPos": sci(summary["equal_cost"]["rk4_errors"][0])
    }
    for word, idx in [("Fifty", 0), ("Hundred", 1), ("TwoHundred", 2)]:
        numeric["exLf" + word] = sci(summary["exercise_leapfrog_errors"][idx])
        numeric["exRk" + word] = sci(summary["exercise_rk4_errors"][idx])
    numeric['valVcInternal'] = f"{(summary['mu']/summary['r0_kpc'])**0.5:.3f}"
    (ROOT / "generated" / "values.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in numeric.items()) + "\n", encoding="utf-8")

    def write_table(name, columns, heading, rows):
        text = r"\begin{tabular}{" + columns + r"}\toprule" + "\n"
        text += heading + r" \\" + "\n" + r"\midrule" + "\n"
        text += "\n".join(" & ".join(row) + r" \\" for row in rows)
        text += "\n" + r"\bottomrule\end{tabular}" + "\n"
        (ROOT / "generated" / f"{name}.tex").write_text(text, encoding="utf-8")

    base_rows = []
    for name, key in [("Euler", "euler_errors"), ("Leapfrog (KDK)", "leapfrog_errors"), ("RK4", "rk4_errors")]:
        calls = summary["force_calls"]["Leapfrog" if name.startswith("Leapfrog") else name]
        base_rows.append([name, str(calls), *[f"${sci(v)}$" for v in summary[key]]])
    write_table("base_errors_table", "lrrrr", r"方法 & 力評価回数 & $e_r$ & $e_\varepsilon$ & $e_\ell$", base_rows)

    rows = []
    for i, (lf, rk) in enumerate(zip(summary["convergence_leapfrog"], summary["convergence_rk4"])):
        plf = "---" if i == 0 else f'{summary["orders_leapfrog"][i-1]:.3f}'
        prk = "---" if i == 0 else f'{summary["orders_rk4"][i-1]:.3f}'
        rows.append([str(int(lf[0])), f"{lf[1]*1000:.4f}", f"${sci(lf[2])}$", plf, f"${sci(rk[2])}$", prk])
    write_table("convergence_table", "rrrrrr", r"$N_{\rm step}$ & $\Delta t\ [\mathrm{Myr}]$ & $e_r$ (KDK) & $p_r$ & $e_r$ (RK4) & $p_r$", rows)

    cost = summary["equal_cost"]
    rows = []
    for name, count, key in [("Leapfrog (KDK)", cost["n_leapfrog"], "leapfrog_errors"), ("RK4", cost["n_rk4"], "rk4_errors")]:
        rows.append([name, str(count), str(cost["force_budget"]), f"${sci(cost[key][0])}$", f"${sci(cost[key][1])}$"])
    write_table("cost_table", "lrrrr", r"方法 & $N_{\rm step}$ & 力評価回数 & $e_r$ & $e_\varepsilon$", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="完成版Notebookを実行して図と数値表を更新します")
    parser.add_argument("--pdf", action="store_true", help="XeLaTeXを2回実行してPDFを作ります")
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    regions = extract_regions(source)
    complete = make_notebook(source)
    student = make_notebook(source, student=True)
    nbformat.write(student, ROOT / "chap01_student.ipynb")
    if args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(complete, timeout=120, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
    if not args.execute and (ROOT / "chap01_complete.ipynb").exists():
        existing = nbformat.read(ROOT / "chap01_complete.ipynb", as_version=4)
        if [c.source for c in existing.cells] == [c.source for c in complete.cells]:
            complete = existing
    nbformat.write(complete, ROOT / "chap01_complete.ipynb")
    if (ROOT / "data" / "validation.json").exists():
        generate_values()
    if args.pdf:
        subprocess.run([sys.executable, str(ROOT.parent / "rebuild.py"), "--pdf"], check=True)
    print(f"Generated {len(regions)} code regions and two notebooks.")

if __name__ == "__main__":
    main()
