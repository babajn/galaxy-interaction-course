# コードと再生成

各 `chapXX_master.py` が実行コードの正本です。全章横断の処理は最上位から `python code/rebuild.py --execute --test --pdf` で行います。Pythonセル、図、数値表、本文中の値をここから再生成します。章別 `build_chapXX.py` のみでは統合本文への同期を行わないため、通常は共通入口を使ってください。

`requirements.txt` は許容する主な依存範囲、`constraints-tested.txt` は今回実際に試験したPythonパッケージ版です。Python本体の版とOSは `benchmark.json` を参照してください。計時範囲は全Notebookのビルド・実行（カーネル、比較計算、図、動画HTMLを含む）です。

`reference_models/` は執筆前の探索的な最小モデルです。本文の標準計算は各章のmasterです。
