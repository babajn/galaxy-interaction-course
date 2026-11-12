#!/usr/bin/env python3
"""Extract chapter PDFs from the integrated PDF (requires PyMuPDF, not needed for simulations)."""
from pathlib import Path
import json
import re
import shutil
try:
    import fitz
except ImportError as exc:
    raise SystemExit("PyMuPDF is required only for this PDF extraction utility.") from exc

base = Path(__file__).resolve().parent
source = fitz.open(base / "main.pdf")
top = [item for item in source.get_toc() if item[0] == 1]
chapter_titles = ["計算環境を準備する", "Pythonで天体の運動を計算する", "2体問題を計算する", "銀河の中で星を動かす", "テスト粒子で円盤銀河を作る", "2つの銀河を接近させる", "銀河相互作用を調べる", "動力学的摩擦で軌道減衰を調べる"]
chapters = [item for item in top if item[1] in chapter_titles]
if [item[1] for item in chapters] != chapter_titles:
    raise ValueError("Expected the eight core chapter bookmarks (Chapter 0--7).")
back_starts = [item[2] for item in top if item[1] == "演習の解答・着眼点"]
if len(back_starts) != 1:
    raise ValueError("Expected one backmatter answer bookmark.")
output = base / "pdf" / "chapters"
output.mkdir(parents=True, exist_ok=True)
for i, (_, title, start) in enumerate(chapters):
    end = chapters[i + 1][2] - 1 if i + 1 < len(chapters) else back_starts[0] - 1
    chapter_no = i
    target = fitz.open()
    target.insert_pdf(source, from_page=start - 1, to_page=end - 1)
    target.set_metadata({"title": f"第{chapter_no}章 {title}", "author": "馬場淳一"})
    target.save(output / f"chap{chapter_no:02d}.pdf", garbage=4, deflate=True)
    target.close()
    print(f"chap{chapter_no:02d}.pdf: PDF pages {start}--{end}")
# Figure numbers/pages are metadata derived from the current TeX labels, never hand maintained.
index_path = base / "figures" / "tikz" / "figure_index.json"
if index_path.exists():
    index = json.loads(index_path.read_text(encoding="utf-8"))
    aux = (base / "main.aux").read_text(encoding="utf-8")
    labels = dict((label, (number, page)) for label, number, page in re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}", aux))
    frontmatter_pages = chapters[0][2] - 1
    for item in index:
        label = item["label"]
        if label not in labels:
            raise ValueError(f"Missing compiled figure label: {label}")
        number, page = labels[label]
        item["figure_number"] = number
        item["book_page"] = int(page)
        item["pdf_page"] = int(page) + frontmatter_pages
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
shutil.copy2(output / "chap00.pdf", base / "chap00_standalone.pdf")
source.close()
