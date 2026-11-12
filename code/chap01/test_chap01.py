#!/usr/bin/env python3
"""正本の関数を独立に検証します。図は生成しません。"""
from pathlib import Path
import ast
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
source = (ROOT / "chap01_master.py").read_text(encoding="utf-8")
keep = {"acceleration_kepler", "rhs_kepler", "euler_step", "leapfrog_step", "rk4_step", "prepare_history",
        "integrate_leapfrog", "integrate", "exact_circle", "diagnostics", "error_metrics"}
functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in keep]
ns = {"np": np}
exec(compile(ast.Module(body=functions, type_ignores=[]), "chap01_master.py", "exec"), ns)
checks = {}
state = np.array([[1.0, 0.0, 0.0, 1.0]])
mu = 1.0
np.testing.assert_allclose(ns["rhs_kepler"](0.0, state, mu), [[0.0, 1.0, -1.0, 0.0]])
checks["gravity_direction_and_state_order"] = True
original = state.copy()
ns["rk4_step"](ns["rhs_kepler"], 0.0, state, 0.01, mu)
ns["integrate_leapfrog"](ns["acceleration_kepler"], state, 1.0, 100, mu)
np.testing.assert_array_equal(state, original)
checks["no_input_mutation"] = True

# KDKの3段階が、等加速度の位置と整数時刻の速度を正しく与えるか確認します。
a0 = np.array([[0.3, -0.7]])
def constant_acc(pos, mu):
    return a0.copy()
h = 0.2
r, v, a = ns["leapfrog_step"](constant_acc, state[:, :2], state[:, 2:], a0, h, mu)
np.testing.assert_allclose(r, state[:, :2] + h * state[:, 2:] + 0.5 * h*h * a0, atol=2e-16)
np.testing.assert_allclose(v, state[:, 2:] + h * a0, atol=2e-16)
checks["kdk_constant_acceleration_and_synchronized_velocity"] = True

# 1ステップの時間反転（負の刻み）を確認します。
r0, v0 = state[:, :2].copy(), state[:, 2:].copy()
a0 = ns["acceleration_kepler"](r0, mu)
r1, v1, a1 = ns["leapfrog_step"](ns["acceleration_kepler"], r0, v0, a0, 0.03, mu)
r2, v2, _ = ns["leapfrog_step"](ns["acceleration_kepler"], r1, v1, a1, -0.03, mu)
np.testing.assert_allclose(r2, r0, atol=5e-16, rtol=0)
np.testing.assert_allclose(v2, v0, atol=5e-16, rtol=0)
checks["kdk_one_step_time_reversal"] = True

# 実際の関数呼び出し数で確認します。診断・描画はカウントしません。
counts = {"lf": 0, "rk": 0, "eu": 0}
def counted_acc(pos, mu):
    counts["lf"] += 1
    return ns["acceleration_kepler"](pos, mu)
def counted_rk_rhs(t, state, mu):
    counts["rk"] += 1
    return ns["rhs_kepler"](t, state, mu)
def counted_eu_rhs(t, state, mu):
    counts["eu"] += 1
    return ns["rhs_kepler"](t, state, mu)
ns["integrate_leapfrog"](counted_acc, state, 1.0, 200, mu)
ns["integrate"](counted_rk_rhs, ns["rk4_step"], state, 1.0, 200, mu)
ns["integrate"](counted_eu_rhs, ns["euler_step"], state, 1.0, 200, mu)
assert counts == {"lf": 201, "rk": 800, "eu": 200}, counts
checks["force_evaluation_counts"] = True

# RK4の4次多項式と段階時刻を確認します。
def linear_rhs(t, state, mu):
    return state
h = 0.1
factor = 1 + h + h**2/2 + h**3/6 + h**4/24
np.testing.assert_allclose(ns["rk4_step"](linear_rhs, 0.0, state, h, mu), state * factor, rtol=2e-15)
checks["rk4_stability_polynomial"] = True
def timed_rhs(t, state, mu):
    return np.ones_like(state) * t**3
expected = state + ((0.4 + 0.2)**4 - 0.4**4) / 4
np.testing.assert_allclose(ns["rk4_step"](timed_rhs, 0.4, state, 0.2, mu), expected, rtol=2e-15)
checks["rk4_stage_times"] = True

# 原点と不正入力は、根拠のないsofteningを入れずに停止します。
for name, thunk in [
    ("singularity_guard", lambda: ns["acceleration_kepler"](np.zeros((1, 2)), mu)),
    ("step_count_guard", lambda: ns["integrate_leapfrog"](counted_acc, state, 1.0, 0, mu)),
    ("nan_time_guard", lambda: ns["integrate"](counted_rk_rhs, ns["rk4_step"], state, np.nan, 20, mu)),
    ("shape_guard", lambda: ns["prepare_history"](np.zeros(4), 1.0, 20))]:
    try:
        thunk()
    except ValueError:
        checks[name] = True
    else:
        raise AssertionError(name)

errors_lf, errors_rk = [], []
for n in [160, 320, 640]:
    t, hist_lf = ns["integrate_leapfrog"](ns["acceleration_kepler"], state, 2*np.pi, n, mu)
    _, hist_rk = ns["integrate"](ns["rhs_kepler"], ns["rk4_step"], state, 2*np.pi, n, mu)
    errors_lf.append(ns["error_metrics"](t, hist_lf, mu, 1.0)[0])
    errors_rk.append(ns["error_metrics"](t, hist_rk, mu, 1.0)[0])
orders_lf = np.log2(np.array(errors_lf[:-1]) / errors_lf[1:])
orders_rk = np.log2(np.array(errors_rk[:-1]) / errors_rk[1:])
assert 1.95 < orders_lf[-1] < 2.05
assert 3.9 < orders_rk[-1] < 4.2
checks["independent_orders_2_and_4"] = True
_, _, angular = ns["diagnostics"](hist_lf, mu)
assert np.max(np.abs(angular - angular[0])) < 1e-12
checks["kdk_central_force_angular_momentum"] = True

# 多数ステップ後に速度を反転して同じステップ数戻します。
t, hforward = ns["integrate_leapfrog"](ns["acceleration_kepler"], state, 2*np.pi, 200, mu)
back_state = hforward[-1].copy()
back_state[:, 2:] *= -1
_, hback = ns["integrate_leapfrog"](ns["acceleration_kepler"], back_state, 2*np.pi, 200, mu)
target = state.copy()
target[:, 2:] *= -1
reversal_error = float(np.max(np.abs(hback[-1] - target)))
assert reversal_error < 1e-12
checks["kdk_full_period_velocity_reversal"] = True

out = {"checks": checks, "all_passed": all(checks.values()), "force_calls_200_steps": counts,
       "dimensionless_position_errors_leapfrog": errors_lf, "dimensionless_position_errors_rk4": errors_rk,
       "dimensionless_orders_leapfrog": orders_lf.tolist(), "dimensionless_orders_rk4": orders_rk.tolist(),
       "time_reversal_error": reversal_error}
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "data" / "independent_tests.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
