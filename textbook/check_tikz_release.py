#!/usr/bin/env python3
"""全章統合版のソース・図・参照検証を実行します。"""
from pathlib import Path
import subprocess
import sys
subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'code/check_release.py')], check=True)
