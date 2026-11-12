# Pythonによる はじめての銀河シミュレーション

**テスト粒子で学ぶ重力・軌道・銀河相互作用** — 馬場淳一

Pythonの基礎を学んだ学生向けの、PDFテキストとJupyter Notebookです。

- [PDFテキスト（171ページ）](textbook.pdf)
- [完成版Notebook（第0–7章）](notebook/complete)：説明・コード・実行結果付き
- [学生版Notebook（第0–7章）](notebook/student)：第1–7章は演習用TODO付き

## 使い方

GitHubの **Code → Download ZIP** でダウンロードして展開します。Python 3.10以上を使い、展開したフォルダで以下を実行してください。これは端末用の命令です。

```tcsh
python3 -m venv .venv
source .venv/bin/activate.csh
python -m pip install -r requirements.txt
jupyter lab
```

bash/zshの場合は、有効化の行を `source .venv/bin/activate` に置き換えます。

まず [第0章の環境確認](notebook/complete/chap00_complete.ipynb) を開きます。その後、第1章から順に進めてください。完成版は、カーネルを再起動して上から順に実行できます。学生版の第1–7章はTODOを実装してから実行します。

各章のNotebookは単独で動作し、別のPythonファイルや入力データは不要です。図・CSVなどは、作業フォルダ内の `output/chapXX/` に保存します。完成版の図や動画は保存済みですが、GitHubのプレビューでは動画が表示されないことがあります。その場合はJupyterで開いてください。

## 内容

| 章 | 内容 |
|---|---|
| 0 | 計算環境を準備する |
| 1 | Pythonで天体の運動を計算する |
| 2 | 2体問題を計算する |
| 3 | 銀河の中で星を動かす |
| 4 | テスト粒子で円盤銀河を作る |
| 5 | 2つの銀河を接近させる |
| 6 | 銀河相互作用を調べる |
| 7 | 動力学的摩擦で軌道減衰を調べる |

## 配布版について

このリポジトリには公開用のPDF・Notebookと、実行案内・依存パッケージ一覧・自動検査設定を置いています。TeX原稿、生成スクリプト、編集メモ、検証用の内部資料は著者の教材一式で別管理しています。

PDFは既存のv2.5配布版です。本文にある正本コード・再生成ツール等のフォルダ案内は著者管理用です。公開版ではこのREADMEの手順でNotebookを実行してください。公開版のNotebookでは計算式と数値条件を維持し、出力先と配布に関する説明を調整しています。

完成版全8冊を、教材一式のない環境で実行する自動検査を設定しています。[最新の検査結果](https://github.com/babajn/galaxy-interaction-course/actions)を参照してください。

利用ライセンスは未指定です。
