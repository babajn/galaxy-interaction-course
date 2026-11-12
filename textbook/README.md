# 統合本文 — 参考図書整理版 v2.4

正式書名：Pythonによる はじめての銀河シミュレーション — テスト粒子で学ぶ重力・軌道・銀河相互作用。

`main.tex` が第0–7章の統合入口です。157×221 mm、XeLaTeXを使います。数値データは `../code/rebuild.py` で再生成・同期します。単に組版する場合は本フォルダで `tcsh build.tcsh` または同等のXeLaTeXコマンドを実行します。

TikZのスタイル、goalbox、章番号付きの図表・式番号は `preamble.tex` に共通定義します。第0章は準備章なのでgoalbox・まとめ・演習を置きません。`tikz_styles.tex` は後方互換の空の入口です。古いpreambleだけを残して章本文を交換しないでください。

`guides/` の構成案v8・スタイルガイドv1.7が現行です。全章のPDFは `pdf/chapters/` にあり、統合PDFから抽出した現行本文です。旧版の独立章PDFではありません。

図一覧は `tikz_figures.tex` です。公開URLの未確定注記はpreambleの `distributionnote` に集約しています。
