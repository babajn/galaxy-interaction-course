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
SOURCE = ROOT / "chap04_master.py"


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
    metrics = s["standard_errors"]
    values = {
        "valCq": f'{s["c_q"]:.9f}', "valNodeError": sci(s["cdf_node_error"]),
        "valInverseError": sci(s["inverse_probability_error"]), "valHalfInterp": f'{s["half_interpolated_kpc"]:.9f}',
        "valHalfExact": f'{s["half_exact_kpc"]:.9f}', "valHalfSample": f'{s["half_sample_kpc"]:.6f}',
        "valFractionRd": f'{s["fraction_inside_rd"]:.9f}', "valMinRadius": f'{s["min_radius_kpc"]:.6f}',
        "valMaxRadius": f'{s["max_radius_kpc"]:.6f}', "valSampleError": f'{s["cdf_sample_error"]:.6f}',
        "valFinalCDF": f'{s["cdf_final_error"]:.6f}', "valVectorDiff": sci(s["vector_difference"]),
        "valCorePeriod": f'{s["central_period_Gyr"]*1000:.3f}', "valPosError": sci(metrics[0]),
        "valRadiusError": sci(metrics[1]), "valEnergyError": sci(metrics[2]), "valAngularError": sci(metrics[3]),
        "valOrbitOrder": f'{s["orbit_orders"][-1]:.5f}', "valReverseError": sci(s["reverse_errors"][0]),
        "valSubMedian": f'{s["subcircular_final_median_kpc"]:.6f}', "valSubEnergy": sci(s["subcircular_energy_error"])
    }
    total_steps, number = s['n_steps'], s['n_particles']
    saved = [len(set(list(range(0,total_steps+1,stride))+[total_steps])) for stride in [1,s['save_every'],50]]
    memory = [8*4*number*count/1e6 for count in saved]
    values.update({
        'valAllMemory': f'{memory[0]:.3f}', 'valSavedMemory': f'{memory[1]:.3f}',
        'valMemorySeries': ','.join(f'{value:.3f}' for value in memory),
        'valSavedSeries': ','.join(str(count) for count in saved),
        'valPlummerHalf': f"{s['a_kpc']/(2**(2/3)-1)**0.5:.4f}",
    })
    (ROOT / "generated" / "values.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()) + "\n", encoding="utf-8")

    def ms(value):
        return "$0$" if value == 0.0 else f"${sci(value)}$"

    def table(name, columns, header, rows):
        text = r"\begin{tabular}{" + columns + r"}\toprule" + "\n" + header + r" \\" + "\n" + r"\midrule" + "\n"
        text += "\n".join(" & ".join(row) + r" \\" for row in rows)
        text += "\n" + r"\bottomrule\end{tabular}" + "\n"
        (ROOT / "generated" / f"{name}.tex").write_text(text, encoding="utf-8")

    rows = []
    for j, row in enumerate(s["cdf_convergence"]):
        order = "---" if j == 0 else f'{s["cdf_orders"][j - 1]:.4f}'
        rows.append([str(int(row[0])), f"{row[1]:.6f}", ms(row[2]), ms(row[3]), order])
    table("cdf_convergence_table", "rrrrr", r"$N_R$ & $\Delta R\ [\mathrm{kpc}]$ & $e_{F,\rm node}$ & $e_u$ & $p_u$", rows)
    rows = [[str(int(row[0])), *[f"{x:.6f}" for x in row[1:]]] for row in s["sampling_summary"]]
    table("sampling_table", "rrrrr", r"$N$ & $D_N$ (seed 42) & 平均 & 最小 & 最大", rows)
    rows = [[label, ms(value)] for label, value in zip([r"位置 $e_{\rm pos}$", r"半径 $e_R$", r"比エネルギー $e_\varepsilon$", r"比角運動量 $e_\ell$"], metrics)]
    table("errors_table", "lr", r"指標 & 値", rows)
    rows = []
    for j, row in enumerate(s["orbit_convergence"]):
        order = "---" if j == 0 else f'{s["orbit_orders"][j - 1]:.5f}'
        rows.append([str(int(row[0])), f"{row[1]*1000:.4f}", ms(row[2]), order, str(int(row[0])+1)])
    table("orbit_convergence_table", "rrrrr", r"$N_{\rm step}$ & $\Delta t\ [\mathrm{Myr}]$ & $e_{\rm pos}$ & $p$ & 力評価回数", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="完成版Notebookを実行して図と数値表を更新します")
    parser.add_argument("--pdf", action="store_true", help="XeLaTeXを2回実行してPDFを作ります")
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    regions = extract_regions(source)
    complete = make_notebook(source)
    student = make_notebook(source, student=True)
    nbformat.write(student, ROOT / "chap04_student.ipynb")
    if args.execute:
        from nbclient import NotebookClient
        client = NotebookClient(complete, timeout=120, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
    if not args.execute and (ROOT / "chap04_complete.ipynb").exists():
        existing = nbformat.read(ROOT / "chap04_complete.ipynb", as_version=4)
        if [c.source for c in existing.cells] == [c.source for c in complete.cells]:
            complete = existing
    nbformat.write(complete, ROOT / "chap04_complete.ipynb")
    if (ROOT / "data" / "validation.json").exists():
        generate_values()
    if args.pdf:
        subprocess.run([sys.executable, str(ROOT.parent / "rebuild.py"), "--pdf"], check=True)
    print(f"Generated {len(regions)} code regions and two notebooks.")

if __name__ == "__main__":
    main()
