# Pythonによる はじめての銀河シミュレーション
## テスト粒子で学ぶ重力・軌道・銀河相互作用 — 見出し統一版 v2.5

対象はPythonの基礎を学んだ学生です。Python 3.10以上を使用してください。

第0章を学生向けの準備章とし、章・節・小節の見出しを動詞句へ統一した改訂版です。第0章はPython環境の構築とJupyter Notebookの利用法だけを扱います。構成案v9、スタイルガイドv1.8に対応します。

### まず実行する

配布物の最上位で、必要なパッケージを入れます。端末用の命令であり、Pythonセルには貼りません。

```tcsh
python3 -m venv .venv
source .venv/bin/activate.csh
python -m pip install -r code/requirements.txt
jupyter lab
```

bash/zshなら有効化は `source .venv/bin/activate` です。授業で指定する環境があればそちらを使います。
`notebook/complete/chap00_complete.ipynb` で環境確認を行ってから第1章へ進みます。第0章には演習用TODOはありません。第1--7章では完成版は出力付き、学生版はTODO付きです。公開版はこのリポジトリから取得できます。

### フォルダ

```text
code/chap00/ ... chap07/  正本・章別生成スクリプト・数値データ・独立テスト
code/rebuild.py            全章の再生成・同期・組版の入口
code/reference_models/    執筆前の探索的最小モデル（本文の標準コードとは別）
notebook/complete/         完成版Notebook
notebook/student/          学生版Notebook
notebook/reference/        探索用の参考Notebook
textbook/main.tex          統合本文の入口（XeLaTeX）
textbook/chapters/          第0–7章の文章の正本
textbook/generated/        正本から抽出したコード・数値表・数値マクロ
textbook/figures/           数値図とTikZ概念図
textbook/guides/            構成案v9、スタイルガイドv1.8（TeX/PDF）
validation/                全セル実行・独立テスト・組版の記録
```

### 正本からすべて再生成

```tcsh
python code/rebuild.py --execute --test --pdf
```

この命令は完成版Notebookを新しいカーネルで実行し、図とデータを作り、本文用のコード抜粋・数値・表を更新してから組版します。Pythonの全セル実行とテストは数分程度かかる場合があります。実測環境と全章の所要時間は `code/benchmark.json` にあり、学生PCでの保証値ではありません。

標準結果が既にある配布版では `python code/rebuild.py --pdf` で保存済みデータから本文を更新できます。古い結果を修正した計算の結果として使わないよう、masterを変更した場合は必ず `--execute` を付けてください。章を限定する例は `python code/rebuild.py --chapters 6 7 --execute --test` です。

`code/chapXX/` 内のNotebookは章別ビルド用の生成物です。授業で開くファイルは `notebook/complete/` と `notebook/student/` にまとめています。

Notebookだけを手で変更しても、再生成時はmasterが優先されます。教材として残す修正は `code/chapXX/chapXX_master.py` へ戻します。コード・表・数値マクロを `textbook/generated/` だけで修正しません。


### 章末と巻末資料

第1--7章は「本章のまとめ → 演習」で終わります。第0章は環境準備だけを扱い、まとめ・演習は置きません。演習の解答・着眼点は第1--7章分を巻末にまとめ、参考図書も巻末へ集約しています。銀河力学は Ryden と Binney & Tremaine、数値計算法は和書を優先して参照します。

### PDFとOverleaf

[配布版PDF（171ページ）](textbook/ms.pdf) を開けます。このPDFは既存の配布版を維持しており、今回の修正後の再組版はしていません。

本文は157×221 mmです。ローカルでは `cd textbook` の後、XeLaTeXで `main.tex` を3回実行します。tcsh用 `build.tcsh` もあります。TeXにはNoto CJKフォントとTikZ・xeCJKなどが必要です。フォントファイルは同梱していません。

Overleafには別配布のOverleaf用ZIPを「新規プロジェクト」として読み込み、**XeLaTeX / main.tex** を選ぶ方法を推奨します。このZIPは本文に必要なものをルートへ配置しています。全コード一式のフォルダと混ぜないでください。本文だけを古いプロジェクトへ追加すると新しいマクロが欠けるので、preamble・chapters・generated・figures・commandsを一組で更新します。実際のOverleafサーバーでの組版ではなく、ローカルXeLaTeXで検証した配布物です。

### 改訂内容と範囲

詳細は `REVISION_NOTES_v2.md`、`REVISION_NOTES_v2_1.md`、`REVISION_NOTES_v2_2.md`、`REVISION_NOTES_v2_3.md` を参照してください。基本の重力則、初期条件、粒子数、seed、時間刻みは変更していません。第6章の非単調な近点系列も残しています。動力学的摩擦は第7章へ独立させ、標準の摩擦パラメータと数値結果は分割前から変更していません。


### 公開準備時の検証（2026年11月12日）

本作業で確認した範囲と未完了の検証は `REVIEW_2026-11-12.txt` を参照してください。
同梱の過去の実行記録と、今回の新しい実行結果を区別してください。

```tcsh
python3 code/check_sources.py
python3 code/test_build_tools.py
```

前者は正本・完成版・学生版・本文用抜粋の同期を検査します。保存済み出力の存在を確認しても、新たに軌道計算を実行したことにはなりません。
数値の再検証には `python code/rebuild.py --execute --test` を実行してください。
GitHub Actionsにも同じ検査を設定しています。PDFの再組版・レイアウト確認は別途必要です。

公開先：https://github.com/babajn/galaxy-interaction-course

2026年11月12日のGitHub Actionsで、全107コードセルの実行と343項目の独立数値検査が合格しました。最新の全検査の状態と実行ログは [Actions](https://github.com/babajn/galaxy-interaction-course/actions) を参照してください。
利用ライセンスは未指定です。再配布・改変の条件は、著者がライセンスを指定するまで確定していません。
