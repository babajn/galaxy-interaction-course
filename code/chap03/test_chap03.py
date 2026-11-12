#!/usr/bin/env python3
"""本文コードの独立試験。描画セルは実行せず、正本の関数定義を検証します。"""
from pathlib import Path
import ast
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
source = (ROOT / "chap03_master.py").read_text(encoding="utf-8")
parsed = ast.parse(source)
functions = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
G = 4.30091e-6 * 1.0227121650537077**2
mass, a, n_mass = 1.0e11, 3.0, 256
ns = {"np": np, "G": G, "mass": mass, "a": a, "n_mass": n_mass}
exec(compile(functions, "chap03_master.py", "exec"), ns)
trap = ns["cumulative_trapezoid"]
quad = ns["enclosed_mass_trapezoid"]
exact = ns["mass_plummer_exact"]
rho = ns["density_plummer"]
phi = ns["potential_plummer"]
accel = ns["acceleration_plummer"]
vc = ns["circular_speed_plummer"]
bisect = ns["bisection"]
initial = ns["make_initial_state"]
step = ns["leapfrog_step"]
integrate = ns["integrate_leapfrog"]
diag = ns["diagnostics"]
checks = []


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    checks.append({"name": name, "passed": True})


def raises(name, exception, func):
    try:
        func()
    except exception:
        check(name, True)
    else:
        check(name, False)


# 解析式とは独立した小問題で、台形則の添字と重みを確認します。
x = np.array([0.0, 0.1, 0.4, 1.0, 2.0])
check("trapezoid: constant on nonuniform grid", np.allclose(trap(x, 3.0 * np.ones_like(x)), 3.0 * x))
check("trapezoid: linear function on nonuniform grid", np.allclose(trap(x, 2.0 * x + 1.0), x**2 + x))
x_shift = x + 2.0
check("trapezoid: cumulative integral begins at grid origin", np.allclose(trap(x_shift, np.ones_like(x)), x))
errs = []
for n in [10, 20, 40]:
    grid = np.linspace(0.0, 1.0, n + 1)
    errs.append(abs(trap(grid, grid**2)[-1] - 1.0 / 3.0))
check("trapezoid: quadratic error ratio four", np.allclose(np.array(errs[:-1]) / errs[1:], 4.0, rtol=1.0e-10))
raises("grid: duplicate nodes rejected", ValueError, lambda: trap([0, 0, 1], [0, 1, 2]))
raises("grid: decreasing nodes rejected", ValueError, lambda: trap([1, 0], [1, 0]))
raises("grid: shape mismatch rejected", ValueError, lambda: trap([0, 1], [0, 1, 2]))
raises("grid: nonfinite values rejected", ValueError, lambda: trap([0, 1], [0, np.nan]))
raises("quadrature: negative radius rejected", ValueError, lambda: quad(-1.0, mass, a, 8))
raises("quadrature: zero interval count rejected", ValueError, lambda: quad(a, mass, a, 0))
raises("quadrature: bool interval count rejected", ValueError, lambda: quad(a, mass, a, True))
check("quadrature: zero radius returns zero", quad(0.0, mass, a, 8) == 0.0)

# Plummerの物理的な恒等式を、別の形の式で照合します。
check("Plummer: finite density at origin", np.isclose(rho(0.0, mass, a), 3.0 * mass / (4.0 * np.pi * a**3)))
check("Plummer: finite central potential", np.isclose(phi(0.0, mass, a), -G * mass / a))
check("Plummer: zero central force", np.array_equal(accel(np.zeros((1, 2)), mass, a), np.zeros((1, 2))))
check("Plummer: mass at scale radius", np.isclose(exact(a, mass, a) / mass, 2.0**(-1.5)))
check("Plummer: enclosed mass tends to total", abs(exact(1.0e4 * a, mass, a) / mass - 1.0) < 2.0e-8)
pos = np.array([[4.0, -3.0]])
r = np.linalg.norm(pos)
force_mass = -G * exact(r, mass, a) * pos / r**3
check("Plummer: force from enclosed mass", np.allclose(accel(pos, mass, a), force_mass, rtol=2.0e-14))
h = 1.0e-4
numeric_grad = np.empty((1, 2))
for axis in range(2):
    offset = np.zeros((1, 2)); offset[0, axis] = h
    numeric_grad[0, axis] = -(phi(np.linalg.norm(pos + offset), mass, a) - phi(np.linalg.norm(pos - offset), mass, a)) / (2.0 * h)
check("Plummer: force equals minus potential gradient", np.allclose(accel(pos, mass, a), numeric_grad, rtol=1.0e-8))
dmdr = (exact(r + h, mass, a) - exact(r - h, mass, a)) / (2.0 * h)
check("Plummer: mass derivative matches shell density", np.isclose(dmdr, 4.0 * np.pi * r**2 * rho(r, mass, a), rtol=1.0e-8))
check("Plummer: circular acceleration", np.isclose(vc(r, mass, a)**2 / r, np.linalg.norm(accel(pos, mass, a))))
check("Plummer: odd force under reflection", np.allclose(accel(-pos, mass, a), -accel(pos, mass, a)))

# 二分法は解析根・単調性の異なる例・失敗条件で確認します。
root, trace = bisect(lambda z: z*z - 2.0, 0.0, 2.0, 1.0e-10)
check("bisection: known root lies in reported bracket", np.all((trace[:, 1] <= np.sqrt(2.0)) & (np.sqrt(2.0) <= trace[:, 2])))
check("bisection: actual error within half-width", abs(root - np.sqrt(2.0)) <= trace[-1, 5] + 1.0e-15)
check("bisection: bracket shrinks by two", np.allclose(trace[1:, 5], 0.5 * trace[:-1, 5]))
check("bisection: residual sign straddles zero", np.all((trace[:, 1]**2 - 2.0) * (trace[:, 2]**2 - 2.0) <= 0.0))
root_decreasing, _ = bisect(lambda z: 2.0 - z*z, 0.0, 2.0, 1.0e-10)
check("bisection: decreasing function works", abs(root_decreasing - np.sqrt(2.0)) <= 1.0e-10)
check("bisection: exact lower endpoint", bisect(lambda z: z, 0.0, 2.0, 1.0e-6)[0] == 0.0)
check("bisection: exact upper endpoint", bisect(lambda z: z-2.0, 0.0, 2.0, 1.0e-6)[0] == 2.0)
raises("bisection: missing sign bracket rejected", ValueError, lambda: bisect(lambda z: z*z + 1, -1, 1, 1.0e-6))
raises("bisection: reversed interval rejected", ValueError, lambda: bisect(lambda z: z, 1, -1, 1.0e-6))
raises("bisection: invalid tolerance rejected", ValueError, lambda: bisect(lambda z: z, -1, 1, 0.0))
raises("bisection: nonfinite endpoint rejected", ValueError, lambda: bisect(lambda z: np.nan, 0, 1, 1.0e-6))
raises("bisection: iteration limit raises", RuntimeError, lambda: bisect(lambda z: z*z - 2, 0, 2, 1.0e-12, max_iter=2))
analytic_half = a / np.sqrt(2.0**(2.0/3.0) - 1.0)
root_analytic, _ = bisect(lambda z: exact(z, mass, a) / mass - 0.5, 0.0, 5*a, 1.0e-10)
check("half-mass: analytic function root", abs(root_analytic - analytic_half) < 1.0e-10)
roots = []
for n in [64, 128, 256]:
    root, _ = bisect(lambda z: quad(z, mass, a, n) / mass - 0.5, 0.0, 5*a, 1.0e-9)
    roots.append(root)
root_errors = np.abs(np.array(roots) - analytic_half)
check("half-mass: quadrature refinement improves actual radius", np.all(np.diff(root_errors) < 0))
check("half-mass: quadrature-limited error ratio four", np.allclose(root_errors[:-1] / root_errors[1:], 4, rtol=0.003))
check("half-mass: tiny bracket does not bound quadrature error", root_errors[-1] > 1000 * 1.0e-9)

# KDKは関数呼出しを実際に数え、保存量とは別に解の精度も確認します。
r0 = 2*a
t_c = 2*np.pi*r0/vc(r0, mass, a)
s0 = initial(r0, 1.0, mass, a)
s0_copy = s0.copy()
calls = [0]
def counted_acceleration(pos, mass, a):
    calls[0] += 1
    return accel(pos, mass, a)
times, history = integrate(counted_acceleration, s0, t_c, 200, mass, a)
check("KDK: initial state unchanged", np.array_equal(s0, s0_copy))
check("KDK: force evaluations equal Nstep+1", calls[0] == 201)
check("KDK: history shape and final time", history.shape == (201, 1, 4) and times[-1] == t_c)
pos, vel = s0[:, :2].copy(), s0[:, 2:].copy()
ac = accel(pos, mass, a)
pos_new, vel_new, ac_new = step(accel, pos, vel, ac, t_c/200, mass, a)
pos_back, vel_back, _ = step(accel, pos_new, vel_new, ac_new, -t_c/200, mass, a)
check("KDK: single-step inverse", np.allclose(pos_back, pos, atol=1.0e-14) and np.allclose(vel_back, vel, atol=1.0e-12))
check("KDK: cached acceleration matches new position", np.array_equal(ac_new, accel(pos_new, mass, a)))
_,energy,angular = diag(history, mass, a)
check("KDK: circular energy deviation small", np.max(np.abs(energy-energy[0]))/abs(energy[0]) < 1.0e-5)
check("KDK: central angular momentum conserved", np.max(np.abs(angular-angular[0]))/abs(angular[0]) < 1.0e-12)
position_errors = []
for n in [100, 200, 400]:
    ts, hist = integrate(accel, s0, t_c, n, mass, a)
    reference = ns["exact_circle"](ts, r0, mass, a)
    position_errors.append(np.max(np.linalg.norm(hist[:, 0, :2]-reference, axis=1))/r0)
check("KDK: circular position second-order convergence", np.allclose(np.array(position_errors[:-1])/position_errors[1:], 4.0, rtol=0.002))
reverse = history[-1].copy(); reverse[:, 2:] *= -1
_, back = integrate(accel, reverse, t_c, 200, mass, a)
check("KDK: full time reversal", np.allclose(back[-1, :, :2], s0[:, :2], atol=1.0e-10) and np.allclose(back[-1, :, 2:], -s0[:, 2:], atol=1.0e-8))
constant = np.array([[1.0, -2.0]])
def const_acc(pos, mass, a):
    return constant.copy()
ts, hs = integrate(const_acc, s0, 0.2, 20, mass, a)
check("KDK: constant acceleration exact", np.allclose(hs[-1, :, :2], s0[:, :2]+0.2*s0[:, 2:]+0.5*0.2**2*constant) and np.allclose(hs[-1, :, 2:], s0[:, 2:]+0.2*constant))
_, origin_h = integrate(accel, np.zeros((1,4)), 0.1, 10, mass, a)
check("KDK: Plummer origin valid and stationary", np.array_equal(origin_h, np.zeros_like(origin_h)))
raises("model: zero scale rejected", ValueError, lambda: integrate(accel, s0, 1, 10, mass, 0.0))
raises("integration: invalid state shape rejected", ValueError, lambda: integrate(accel, np.zeros(4), 1, 10, mass, a))
raises("integration: zero duration rejected", ValueError, lambda: integrate(accel, s0, 0, 10, mass, a))
raises("integration: nonfinite initial state rejected", ValueError, lambda: integrate(accel, np.full((1,4), np.nan), 1, 10, mass, a))

report = {"chapter": 3, "version": "1.0", "passed": len(checks), "failed": 0, "checks": checks}
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data" / "independent_tests.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Independent tests: {len(checks)} passed, 0 failed.")
