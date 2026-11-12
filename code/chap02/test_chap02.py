#!/usr/bin/env python3
"""描画や実験セルを走らせず、正本の関数を独立に検証します。"""
from pathlib import Path
import ast
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
source = (ROOT / "chap02_master.py").read_text(encoding="utf-8")
functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
namespace = {"np": np, "G": 4.30091e-6 * 1.0227121650537077**2}
exec(compile(ast.Module(body=functions, type_ignores=[]), "chap02_master.py", "exec"), namespace)
acceleration = namespace["acceleration_two_body"]
initial_state = namespace["make_initial_state"]
integrate = namespace["integrate_leapfrog"]
step = namespace["leapfrog_step"]
diag = namespace["diagnostics"]
errors = namespace["conservation_errors"]
encounter = namespace["make_parabolic_encounter"]
G = namespace["G"]
masses = np.array([1.0e11, 3.7e10])
mu = G * masses.sum()
d0 = np.array([17.0, 6.0])
u0 = np.array([-23.0, 132.0])
rcm = np.array([2.0, -1.0])
vcm = np.array([4.0, 7.0])
report = []


def check(name, condition, value=None):
    if not condition:
        raise AssertionError(f"FAILED: {name}; value={value}")
    report.append({"test": name, "passed": True, "value": value})


def close(a, b, scale=1.0, tolerance=1e-12):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) <= tolerance * scale


def raises_value_error(call):
    try:
        call()
    except ValueError:
        return True
    return False


s0 = initial_state(masses, d0, u0, rcm, vcm)
val = diag(s0[None], masses)
check("initial relative position / velocity round trip", close(val["dr"][0], d0, 20) and close(val["dv"][0], u0, 200))
check("initial center of mass / bulk velocity", close(val["Rcm"][0], rcm, 20) and close(val["P"][0] / masses.sum(), vcm, 200))
a0 = acceleration(s0[:, :2], masses)
fsum = masses[0] * a0[0] + masses[1] * a0[1]
check("Newton third law for unequal masses", np.linalg.norm(fsum) / np.linalg.norm(masses[0] * a0[0]) < 1e-14)
arel = -mu * d0 / np.linalg.norm(d0)**3
check("relative acceleration uses total mass", close(a0[1] - a0[0], arel, np.linalg.norm(arel)))
rot = np.array([[np.cos(0.63), -np.sin(0.63)], [np.sin(0.63), np.cos(0.63)]])
check("force rotates as a vector", close(acceleration(s0[:, :2] @ rot.T, masses), a0 @ rot.T, np.max(np.abs(a0))))
check("force is translation invariant", close(acceleration(s0[:, :2] + [11.0, -9.0], masses), a0, np.max(np.abs(a0))))
check("body-label exchange symmetry", close(acceleration(s0[::-1, :2], masses[::-1])[::-1], a0, np.max(np.abs(a0))))

before = s0.copy()
pos = s0[:, :2].copy()
vel = s0[:, 2:].copy()
acc = acceleration(pos, masses)
p1, v1, a1 = step(acceleration, pos, vel, acc, 0.0002, masses)
p2, v2, _ = step(acceleration, p1, v1, a1, -0.0002, masses)
check("one-step KDK inversion", close(p2, pos, 20) and close(v2, vel, 200))
check("step leaves its input arrays unchanged", np.array_equal(pos, before[:, :2]) and np.array_equal(vel, before[:, 2:]))

calls = [0]
def counted(pos, masses):
    calls[0] += 1
    return acceleration(pos, masses)
t, hist = integrate(counted, s0, 0.2, 200, masses)
check("cached KDK uses Nstep+1 force calls", calls[0] == 201, calls[0])
check("trajectory shape and synchronized saved velocity", hist.shape == (201, 2, 4) and close(hist[1, :, 2:], vel + 0.5 * 0.001 * (acceleration(pos, masses) + acceleration(hist[1, :, :2], masses)), 200))
check("integrator preserves input initial state", np.array_equal(s0, before))

val = diag(hist, masses)
vc = np.sqrt(mu / np.linalg.norm(d0))
q = masses[0] * masses[1] / masses.sum()
eps = 0.5 * np.sum(val["dv"]**2, axis=1) - mu / val["D"]
bulk = val["P"] / masses.sum()
energy_from_relative = 0.5 * masses.sum() * np.sum(bulk**2, axis=1) + q * eps
check("energy decomposition in a moving COM frame", close(val["E"], energy_from_relative, G * masses.prod() / np.linalg.norm(d0)))
ell = val["dr"][:, 0] * val["dv"][:, 1] - val["dr"][:, 1] * val["dv"][:, 0]
bulk_l = masses.sum() * (val["Rcm"][:, 0] * bulk[:, 1] - val["Rcm"][:, 1] * bulk[:, 0])
check("angular-momentum decomposition about a fixed origin", close(val["L"], bulk_l + q * ell, abs(val["L"][0])))
check("total momentum and angular momentum", errors(t, hist, masses)[1] < 1e-12 and errors(t, hist, masses)[2] < 1e-12)
check("center of mass follows its predicted straight line", errors(t, hist, masses)[3] < 1e-12)

# 二つの中心を動かす実装と、相対方程式だけを動かす実装を照合します。
d, u = d0.copy(), u0.copy()
ar = -mu * d / np.linalg.norm(d)**3
for _ in range(200):
    uh = u + 0.0005 * ar
    d = d + 0.001 * uh
    ar = -mu * d / np.linalg.norm(d)**3
    u = uh + 0.0005 * ar
check("two-center update matches reduced relative equation", close(val["dr"][-1], d, 20) and close(val["dv"][-1], u, 200))

r = 20.0
vc = np.sqrt(mu / r)
period = 2 * np.pi * r / vc
s_par = initial_state(masses, [r, 0], [0, np.sqrt(2) * vc])
tp, hp = integrate(acceleration, s_par, period, 400, masses)
check("parabolic energy normalization remains finite", np.all(np.isfinite(errors(tp, hp, masses))))
s_enc = encounter(masses, 60.0, 8.0)
d_enc, u_enc = s_enc[1, :2] - s_enc[0, :2], s_enc[1, 2:] - s_enc[0, 2:]
e_enc = 0.5 * np.dot(u_enc, u_enc) - mu / 60.0
l_enc = d_enc[0] * u_enc[1] - d_enc[1] * u_enc[0]
check("incoming encounter has zero specific energy", abs(e_enc) / (mu / 60.0) < 1e-14)
check("incoming encounter has specified pericenter angular momentum", abs(l_enc**2 / (2 * mu) - 8.0) < 1e-12 and u_enc[0] < 0)

error_list = []
for n in (100, 200, 400):
    initial = initial_state(masses, [r, 0], [0, vc])
    ts, hs = integrate(acceleration, initial, period, n, masses)
    exact = r * np.column_stack((np.cos(2 * np.pi * ts / period), np.sin(2 * np.pi * ts / period)))
    relative = hs[:, 1, :2] - hs[:, 0, :2]
    error_list.append(float(np.max(np.linalg.norm(relative - exact, axis=1)) / r))
order = float(np.log2(error_list[-2] / error_list[-1]))
check("second-order circular trajectory convergence", 1.98 < order < 2.02, order)
check("point-mass collision is rejected", raises_value_error(lambda: acceleration(np.zeros((2, 2)), masses)))
check("negative mass is rejected", raises_value_error(lambda: initial_state([1e11, -1e10], d0, u0)))
check("invalid encounter pericenter is rejected", raises_value_error(lambda: encounter(masses, 8.0, 60.0)))
check("invalid initial-state shape is rejected", raises_value_error(lambda: integrate(acceleration, np.zeros((1, 4)), 1.0, 100, masses)))
check("nonfinite initial state is rejected", raises_value_error(lambda: integrate(acceleration, np.full((2, 4), np.nan), 1.0, 100, masses)))
check("invalid step count is rejected", all(raises_value_error(lambda n=n: integrate(acceleration, s0, 1.0, n, masses)) for n in [0, -1, 5.5, True]))
check("nonfinite final time is rejected", raises_value_error(lambda: integrate(acceleration, s0, np.inf, 100, masses)))

# NotebookやPDFへのコードの抜粋を照合します。
import build_chap02
regions = build_chap02.extract_regions(source)
check("all code excerpts compile", all(compile((ROOT / "generated" / f"{name}.py").read_text(), name, "exec") is not None for name in regions))
student = build_chap02.make_notebook(source, student=True)
for cell in student.cells:
    if cell.cell_type == "code":
        compile(cell.source, "student_cell", "exec")
check("student code cells are syntactically valid", True)
(ROOT / "data" / "independent_tests.json").write_text(json.dumps({"count": len(report), "tests": report}, indent=2), encoding="utf-8")
print(f"All {len(report)} independent checks passed.")
for item in report:
    print("PASS", item["test"])
