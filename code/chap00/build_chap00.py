#!/usr/bin/env python3
"""第0章のJupytext masterからNotebookと本文用コード抜粋を再生成します。"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys
import nbformat

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "chap00_master.py"


def make_notebook(source, student=False):
    cells = []
    blocks = re.split(r"(?m)^# %%(.*)\n", source)
    for i in range(1, len(blocks), 2):
        header, body = blocks[i], blocks[i + 1].strip("\n")
        if "[markdown]" in header:
            body = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
            if student:
                body = body.replace("**完成版 / v2.2**", "**学生版 / v2.2**")
            cell = nbformat.v4.new_markdown_cell(body)
        else:
            body = "\n".join(line for line in body.splitlines() if "# --8<--" not in line)
            cell = nbformat.v4.new_code_cell(body)
        cell["id"] = hashlib.sha256(f"{i}:{cell.source}".encode()).hexdigest()[:12]
        cells.append(cell)
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "chapter_version": "2.2"}
    nbformat.validate(nb)
    return nb


def extract_regions(source):
    folder = ROOT / "generated"
    if folder.exists():
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
    else:
        folder.mkdir()
    pattern = r"(?ms)^# --8<-- \[start:([^\]]+)\]\n(.*?)^# --8<-- \[end:\1\]"
    regions = {}
    for match in re.finditer(pattern, source):
        name, text = match.groups()
        (folder / f"{name}.py").write_text(text.rstrip() + "\n", encoding="utf-8")
        regions[name] = hashlib.sha256(text.encode()).hexdigest()
    (folder / "source_manifest.json").write_text(json.dumps({"master_sha256": hashlib.sha256(source.encode()).hexdigest(), "regions": regions}, indent=2), encoding="utf-8")
    return regions


def preserve_outputs(notebook, path):
    """全セルの種類・内容が同じ場合に限り、保存済みの実行出力を引き継ぎます。"""
    if not path.is_file():
        return notebook
    previous = json.loads(path.read_text(encoding="utf-8"))
    signature = lambda nb: [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]
    if signature(previous) != signature(notebook):
        return notebook
    for new, old in zip(notebook["cells"], previous["cells"]):
        if new["cell_type"] == "code":
            new["execution_count"] = old.get("execution_count")
            new["outputs"] = old.get("outputs", [])
    notebook["metadata"]["language_info"] = previous.get("metadata", {}).get("language_info", {"name": "python"})
    return notebook


def generate_values(notebook):
    # PDFのみの再生成時も、ビルド用PythonでなくNotebookの実測環境を記録します。
    output = "".join("".join(o.get("text", "")) for c in notebook["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []) if o.get("output_type") == "stream")
    versions = dict(re.findall(r"^(Python|NumPy|Matplotlib):\s*(\S+)", output, re.M))
    if set(versions) != {"Python", "NumPy", "Matplotlib"}:
        raise RuntimeError("Chapter 0 has no complete environment record; run with --execute.")
    data = {"python_version": versions["Python"], "numpy_version": versions["NumPy"], "matplotlib_version": versions["Matplotlib"]}
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data/validation.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    values = {"valPython": data["python_version"], "valNumpy": data["numpy_version"], "valMatplotlib": data["matplotlib_version"]}
    (ROOT / "generated/values.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in values.items()) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    source = SOURCE.read_text(encoding="utf-8")
    regions = extract_regions(source)
    complete = make_notebook(source, student=False)
    student = make_notebook(source, student=True)
    if args.execute:
        from nbclient import NotebookClient
        NotebookClient(complete, timeout=120, kernel_name="python3", allow_errors=False, resources={"metadata": {"path": str(ROOT)}}).execute()
        NotebookClient(student, timeout=120, kernel_name="python3", allow_errors=False, resources={"metadata": {"path": str(ROOT)}}).execute()
    else:
        complete = preserve_outputs(complete, ROOT / "chap00_complete.ipynb")
        student = preserve_outputs(student, ROOT / "chap00_student.ipynb")
    nbformat.write(nbformat.from_dict(complete), ROOT / "chap00_complete.ipynb")
    nbformat.write(nbformat.from_dict(student), ROOT / "chap00_student.ipynb")
    generate_values(complete)
    if args.pdf:
        subprocess.run([sys.executable, str(ROOT.parent / "rebuild.py"), "--pdf"], check=True)
    print(f"Generated {len(regions)} code region and two environment-check notebooks.")

if __name__ == "__main__":
    main()
