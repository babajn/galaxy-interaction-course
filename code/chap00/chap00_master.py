# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 第0章　計算環境を準備する
# **完成版 / v2.2**
#
# このNotebookは、第1章以降のNotebookを実行するための環境確認用です。
# Pythonの文法やNumPyの使い方を学ぶことは目的にしません。
# 下のセルを実行して、必要なパッケージを読み込めることを確認してください。

# %%
# --8<-- [start:environment-check]
import sys
import numpy as np
import matplotlib

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
# --8<-- [end:environment-check]

# %% [markdown]
# 3つの版番号が表示され、エラーが出なければ準備は完了です。
# 第1章へ進んでください。
