#!/usr/bin/env python3
from pathlib import Path
import json
p=Path(__file__).resolve().parent/"data/independent_tests.json"
d=json.loads(p.read_text())
assert d["failed"]==0
print(f"Chapter 7 release check: {d['passed']} independent checks passed.")
