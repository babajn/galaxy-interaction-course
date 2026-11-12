# 銀河相互作用のシミュレーション入門

**Pythonで学ぶ制限三体近似**

Pythonの基礎と、Newtonの運動方程式・中心力・保存則・2体問題を学んだ学生向けの、PDFテキストとJupyter Notebookです。

- [PDFテキスト](textbook.pdf)
- [完成版Notebook（第0–7章）](notebook/complete)：説明・コード・実行結果付き
- [実習版Notebook（第0–7章）](notebook/student)：第1–7章は演習用TODO付き

**教材作成：馬場淳一（鹿児島大学/国立天文台）**

## 本書でいう「制限三体近似」

古典的な制限三体問題では、互いの重力で運動する2つの点質量天体と、それらの重力を受ける、質量を無視できる第3の天体を考えます。第3の天体から他の2天体への重力の反作用を省くのが「制限」です。

本書の第5・6章では、この考え方を「2つの銀河と1個の星」に対応させ、銀河を形の固定されたPlummerポテンシャル、星をテスト粒子で表します。多数の星を置いても、星同士の重力を省くため、各星の軌道を独立に計算できます。広がった銀河の重力場へ拡張した近似として扱います。第7章では主銀河を原点に固定し、衛星銀河の中心に動力学的摩擦を加えます。

## 使い方

GitHubの **Code → Download ZIP** でダウンロードして展開します。Python 3.10以上を使い、展開したフォルダで以下を実行してください。これは端末用の命令です。

```tcsh
python3 -m venv .venv
source .venv/bin/activate.csh
python -m pip install -r requirements.txt
jupyter lab
```

bash/zshの場合は、有効化の行を `source .venv/bin/activate` に置き換えます。Windows PowerShellでは、`python -m venv .venv` で作成し、`.venv\Scripts\Activate.ps1` で有効にします。以降のパッケージ導入・起動手順は共通です。

> 旧PDF（v2.5）の §0.1–0.2 にある `code/requirements.txt` は、公開版では直下の `requirements.txt` に読み替えてください。

まず [第0章の環境確認](notebook/complete/chap00_complete.ipynb) を開き、その後は第1章から順に進めてください。

- **完成版**：説明・コード・実行結果を確認するNotebookです。カーネルを再起動して、上から順に実行できます。
- **実習版**：本文を参照しながらTODOを埋めるNotebookです。第0章にはTODOがなく、第1〜7章で実装に取り組みます。

任意演習は、標準計算と結果の保存の後にあります。取り組まない場合は、該当するセル全体を飛ばしてください。本体のTODOを解答済みなら、「すべて実行」が未解答の任意演習で停止しても、標準結果は保存されています。

パラメータを変更して計算し直すときは、カーネルを再起動し、設定セルから順に実行してください。

各章のNotebookは単独で動作し、別のPythonファイルや入力データは不要です。図・CSVなどは、作業フォルダ内の `output/chapXX/` に保存します。完成版の図や動画は保存済みですが、GitHubのプレビューでは動画が表示されないことがあります。その場合はJupyterで開いてください。

## 内容

| 章 | 内容 | 完成版 | 実習版 |
|---|---|---|---|
| 0 | 計算環境を準備する | [完成版](notebook/complete/chap00_complete.ipynb) | [実習版](notebook/student/chap00_student.ipynb) |
| 1 | Pythonで天体の運動を計算する | [完成版](notebook/complete/chap01_complete.ipynb) | [実習版](notebook/student/chap01_student.ipynb) |
| 2 | 2体問題を計算する | [完成版](notebook/complete/chap02_complete.ipynb) | [実習版](notebook/student/chap02_student.ipynb) |
| 3 | 銀河の中で星を動かす | [完成版](notebook/complete/chap03_complete.ipynb) | [実習版](notebook/student/chap03_student.ipynb) |
| 4 | テスト粒子で円盤銀河を作る | [完成版](notebook/complete/chap04_complete.ipynb) | [実習版](notebook/student/chap04_student.ipynb) |
| 5 | 2つの銀河を接近させる | [完成版](notebook/complete/chap05_complete.ipynb) | [実習版](notebook/student/chap05_student.ipynb) |
| 6 | 銀河相互作用を調べる | [完成版](notebook/complete/chap06_complete.ipynb) | [実習版](notebook/student/chap06_student.ipynb) |
| 7 | 動力学的摩擦で軌道減衰を調べる | [完成版](notebook/complete/chap07_complete.ipynb) | [実習版](notebook/student/chap07_student.ipynb) |

## 参考文献

- Binney, J. & Tremaine, S. (2008). [*Galactic Dynamics*, 2nd ed.](https://press.princeton.edu/books/paperback/9780691130279/galactic-dynamics) Princeton University Press.
  銀河ポテンシャル、恒星の軌道、動力学的摩擦などを詳しく学ぶための参考書です。

## 関連する公開教材

- 馬場淳一『[シミュレーション天文学：銀河動力学を中心に](https://github.com/babajn/simulation-stellar-dynamics)』（2024年度 総研大講義資料）
- 馬場淳一『[銀河力学で学ぶ数値計算法](https://github.com/babajn/galactic-dynamics-numerical-methods)』（2026年度 鹿児島大学物理宇宙セミナー資料）

## 利用条件

利用ライセンスは未指定です。
