#!/usr/bin/env python3
"""全章統合版の配布検証へ委譲します。"""
from pathlib import Path
import subprocess
import sys
subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'check_release.py')], check=True)
