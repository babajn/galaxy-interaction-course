#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib

checks = []
def check(condition, name):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

check(sys.version_info >= (3, 10), "Python >= 3.10")
check(int(np.__version__.split('.')[0]) < 3, "NumPy < 3")
check(int(matplotlib.__version__.split('.')[0]) < 4, "Matplotlib < 4")
print(f"Independent tests: {len(checks)} passed, 0 failed.")
