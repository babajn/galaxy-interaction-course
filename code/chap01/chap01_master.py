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
# # 第1章　Pythonで天体の運動を計算する
# **完成版 / 全章改訂 v2.0**
#
# 原点に固定した質量 $M$ の重力場で、1個の粒子を動かします。
# Euler法を入口とし、重力計算の標準法としてleapfrog（KDK）を実装します。
# RK4も本文で学ぶ主要手法として実装し、速度依存力への拡張に使います。
# 解析的な円軌道、保存量、時間刻み、力評価回数で、両者を比較します。
#
# 構成案v4・スタイルガイドv1.3に対応します。詳しい導出は本文を参照してください。
# 本章の基準例は点質量です。2体運動とPlummer銀河は後の章で扱います。
# 旧版と同じ数値条件を保ち、今回追加した長時間・同一コストの実験は本教材での設定です。
# 実行にはNumPyとMatplotlibを使います。上から順に実行してください。
# 図・CSV・検証JSONは、実行フォルダの `figures/` と `data/` に保存します。

# %%
from pathlib import Path
import json
import platform
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for folder in (ROOT, *ROOT.parents):
    chapter_folder = folder / "code" / "chap01"
    if (chapter_folder / "chap01_master.py").is_file():
        ROOT = chapter_folder
        break
FIGDIR, DATADIR = ROOT / "figures", ROOT / "data"
FIGDIR.mkdir(exist_ok=True)
DATADIR.mkdir(exist_ok=True)
BLUE, VERMILLION, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#666666"
plt.rcParams.update({"font.size": 11, "figure.dpi": 110, "savefig.dpi": 180})


def save_figure(fig, name):
    """同じ図を本文用PDFとプレビュー用PNGで保存します。"""
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## 1. 条件と単位を決める（本文1.2・1.3）
# 内部単位はkpc、Gyr、$M_\odot$です。内部速度はkpc/Gyrで扱います。
# $\mu=GM$、$v_c=\sqrt{\mu/r_0}$、$T=2\pi r_0/v_c$です。
# 標準比較は円軌道を1周期、200ステップで積分します。

# %%
# --8<-- [start:parameters]
KM_S_TO_KPC_GYR = 1.0227121650537077
G = 4.30091e-6 * KM_S_TO_KPC_GYR**2  # kpc^3 / (Msun Gyr^2)
mass = 1.0e11                      # 中心質量 [Msun]
r0 = 10.0                          # 初期半径 [kpc]
mu = G * mass
vc = np.sqrt(mu / r0)               # 円運動速度 [kpc/Gyr]
period = 2.0 * np.pi * r0 / vc       # 軌道周期 [Gyr]
pos0 = np.array([[r0, 0.0]])         # shape = (1, 2)
vel0 = np.array([[0.0, vc]])         # shape = (1, 2)
state0 = np.column_stack((pos0, vel0))  # RK4用：[x, y, vx, vy]
n_steps = 200
t_end = period
dt = t_end / n_steps
# --8<-- [end:parameters]
print(f"Circular speed = {vc / KM_S_TO_KPC_GYR:.6f} km/s")
print(f"Period = {period:.9f} Gyr; dt = {dt * 1000:.6f} Myr")
print("pos0.shape =", pos0.shape, "; state0.shape =", state0.shape)

# %% [markdown]
# ## 2. 重力を1粒子の成分で計算する（本文1.2・1.7）
# $\boldsymbol a=-\mu\boldsymbol r/r^3$をコードに移します。
# 原点では式が発散するため、本章では原点へ衝突する軌道を扱いません。

# %%
# --8<-- [start:acceleration-scalar]
x, y = pos0[0]
r2 = x**2 + y**2
ax = -mu * x / r2**1.5
ay = -mu * y / r2**1.5
print(f"ax = {ax:.6f}, ay = {ay:.6f} [kpc/Gyr^2]")
# --8<-- [end:acceleration-scalar]

# %% [markdown]
# 同じ式を加速度関数にまとめます。`pos`も返り値も形は `(1, 2)` です。
# 多数粒子への一般化は第4章で行います。

# %%
# --8<-- [start:acceleration-kepler]
def acceleration_kepler(pos, mu):
    """1粒子の重力加速度。本文 eq:ch1-acceleration-components。"""
    x, y = pos[0]
    r2 = x**2 + y**2
    if r2 <= 0.0:
        raise ValueError("Point-mass gravity is undefined at r = 0.")
    ax = -mu * x / r2**1.5
    ay = -mu * y / r2**1.5
    return np.array([[ax, ay]])
# --8<-- [end:acceleration-kepler]
# --8<-- [start:rhs-kepler]
def rhs_kepler(t, state, mu):
    """状態[x, y, vx, vy]の右辺。本文 eq:ch1-state-ode。"""
    pos, vel = state[:, :2], state[:, 2:]
    acc = acceleration_kepler(pos, mu)
    return np.column_stack((vel, acc))
# --8<-- [end:rhs-kepler]

# %% [markdown]
# ## 3. Euler法：時間積分の入口（本文1.4）
# 位置と速度の両方に、更新前の同じ状態を使います。
# この方法は導入・比較用であり、後の重力計算の標準法にはしません。

# %%
# --8<-- [start:euler-step]
def euler_step(rhs, t, state, dt, mu):
    return state + dt * rhs(t, state, mu)
# --8<-- [end:euler-step]

# %% [markdown]
# ## 4. leapfrog（KDK）：重力計算の標準法（本文1.5）
# 半ステップのkick → 1ステップのdrift → 半ステップのkick、の順で進めます。
#
# $$\boldsymbol v^{n+1/2}=\boldsymbol v^n+\frac{\Delta t}{2}\boldsymbol a^n,$$
# $$\boldsymbol r^{n+1}=\boldsymbol r^n+\Delta t\boldsymbol v^{n+1/2},$$
# $$\boldsymbol v^{n+1}=\boldsymbol v^{n+1/2}+\frac{\Delta t}{2}\boldsymbol a^{n+1}.$$
#
# `vel_half`は更新途中だけで使います。保存する速度は位置と同じ整数時刻の `vel_new` です。
# 新しい位置で求めた `acc_new` を次のステップへ渡し、同じ加速度を再計算しません。

# %%
# --8<-- [start:leapfrog-step]
def leapfrog_step(acceleration, pos, vel, acc, dt, mu):
    """KDKで1ステップ。accは現在位置での加速度です。"""
    ### BEGIN SOLUTION
    vel_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_half
    acc_new = acceleration(pos_new, mu)
    vel_new = vel_half + 0.5 * dt * acc_new
    return pos_new, vel_new, acc_new
    ### END SOLUTION
# --8<-- [end:leapfrog-step]

# %% [markdown]
# **確認：** `acc_new`を返さず、毎ステップの最初に加速度を計算し直すと、
# 1ステップあたりの力評価回数はどう変わりますか。
# このKDKは加速度が位置だけで決まる問題に使います。速度依存力を単純に足さないでください。

# %% [markdown]
# ## 5. RK4：一般の常微分方程式へ（本文1.6）
# $\boldsymbol y=(\boldsymbol r,\boldsymbol v)$を4段階で予測します。
# $\boldsymbol k_j$自体に$\Delta t$は含めません。位置と速度の両方を同じ段階の状態で評価します。
# 1ステップ4回の右辺評価を使います。最後の速度比例抵抗の演習でも、この関数を再利用します。

# %%
# --8<-- [start:rk4-step]
def rk4_step(rhs, t, state, dt, mu):
    """RK4の4段階。本文 eq:ch1-rk4-stages, eq:ch1-rk4-update。"""
    ### BEGIN SOLUTION
    k1 = rhs(t, state, mu)
    k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1, mu)
    k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2, mu)
    k4 = rhs(t + dt, state + dt * k3, mu)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    ### END SOLUTION
# --8<-- [end:rk4-step]

# %% [markdown]
# ## 6. 初期条件から繰り返し計算する（本文1.7）
# 最終時刻を固定して$\Delta t=t_{\rm end}/N_{\rm step}$とします。
# 入力をコピーし、比較計算で初期条件を変更しないようにします。
# 履歴は `(出力時刻, 粒子, 4成分)` です。KDKも診断用に位置と速度を同じ時刻で保存します。

# %%
def prepare_history(state0, t_end, n_steps):
    """入力を確認し、共通の時刻配列と保存領域を用意します。"""
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    if not np.isfinite(t_end) or t_end <= 0.0:
        raise ValueError("t_end must be finite and positive.")
    state = np.array(state0, dtype=float, copy=True)
    if state.shape != (1, 4) or not np.all(np.isfinite(state)):
        raise ValueError("state0 must be a finite array with shape (1, 4).")
    dt = t_end / n_steps
    times = np.arange(n_steps + 1) * dt
    history = np.empty((n_steps + 1, 1, 4))
    history[0] = state
    return state, times, history, dt

# --8<-- [start:integrate-leapfrog]
def integrate_leapfrog(acceleration, state0, t_end, n_steps, mu):
    """初回の力を計算し、以降は前ステップのacc_newを使います。"""
    state, times, history, dt = prepare_history(state0, t_end, n_steps)
    pos, vel = state[:, :2].copy(), state[:, 2:].copy()
    acc = acceleration(pos, mu)  # 初回だけ、ループの前に計算します。
    for n in range(n_steps):
        pos, vel, acc = leapfrog_step(acceleration, pos, vel, acc, dt, mu)
        history[n + 1] = np.column_stack((pos, vel))
        if not np.all(np.isfinite(history[n + 1])):
            raise FloatingPointError("Non-finite state; check the time step.")
    return times, history
# --8<-- [end:integrate-leapfrog]

# --8<-- [start:integrate]
def integrate(rhs, stepper, state0, t_end, n_steps, mu):
    """Euler法・RK4用。右辺関数を指定して等間隔で積分します。"""
    state, times, history, dt = prepare_history(state0, t_end, n_steps)
    for n in range(n_steps):
        state = stepper(rhs, times[n], state, dt, mu)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("Non-finite state; check the time step.")
        history[n + 1] = state
    return times, history
# --8<-- [end:integrate]

# %% [markdown]
# 同じ初期条件・最終時刻・ステップ数で3つの方法を実行します。
# `history`は本章の標準法leapfrogの結果です。これは同じコストでの比較ではありません。

# %%
run_start = time.perf_counter()
# --8<-- [start:run-orbit]
times, history = integrate_leapfrog(acceleration_kepler, state0, t_end, n_steps, mu)
_, history_euler = integrate(rhs_kepler, euler_step, state0, t_end, n_steps, mu)
_, history_rk4 = integrate(rhs_kepler, rk4_step, state0, t_end, n_steps, mu)
# --8<-- [end:run-orbit]
print(f"Three integrations: {time.perf_counter() - run_start:.3f} s (this environment)")
print(f"Force evaluations: Euler={n_steps}, leapfrog={n_steps + 1}, RK4={4 * n_steps}")

# %% [markdown]
# ## 7. 解析的な円軌道と比較する（本文1.8）
# $\Omega=\sqrt{\mu/r_0^3}$、$\boldsymbol r_{\rm exact}=r_0(\cos\Omega t,\sin\Omega t)$です。
# 解析解は数値解からではなく、初期条件から計算します。同じ時刻の位置を比べてください。

# %%
# --8<-- [start:exact-circle]
def exact_circle(times, r0, mu):
    omega = np.sqrt(mu / r0**3)
    phase = omega * np.asarray(times)
    pos_exact = np.column_stack((r0 * np.cos(phase), r0 * np.sin(phase)))
    vel_exact = np.column_stack((-r0 * omega * np.sin(phase), r0 * omega * np.cos(phase)))
    return pos_exact, vel_exact

pos_exact, vel_exact = exact_circle(times, r0, mu)
pos_history = history[:, 0, :2]  # 第1軸は出力時刻です。
vel_history = history[:, 0, 2:]
# --8<-- [end:exact-circle]

# %%
# --8<-- [start:plot-orbit]
fig, ax = plt.subplots(figsize=(5.6, 5.2), layout="constrained")
ax.plot(pos_exact[:, 0], pos_exact[:, 1], color=GRAY, ls="--", lw=2, label="Exact circle")
ax.plot(pos_history[:, 0], pos_history[:, 1], color=BLUE, lw=1.3, label="Leapfrog (KDK)")
ax.scatter([0.0], [0.0], marker="*", s=90, color="black", label="Central mass")
sample_ids = np.rint(np.arange(4) * n_steps / 4).astype(int)
for n in sample_ids:
    ax.plot(*pos_history[n], marker="s", color=BLUE, ms=5)
    label = f"{times[n] / period:.2f} T"
    ax.annotate(label, pos_history[n], xytext=(7, 7), textcoords="offset points")
ax.set(xlabel="x [kpc]", ylabel="y [kpc]", xlim=(-13, 13), ylim=(-13, 13))
ax.set_aspect("equal")
ax.legend(loc="lower right", fontsize=9)
save_figure(fig, "ch01_orbit")
# --8<-- [end:plot-orbit]

# %%
fig, ax = plt.subplots(figsize=(5.8, 5.3), layout="constrained")
ax.plot(pos_exact[:, 0], pos_exact[:, 1], color=GRAY, ls=":", lw=2, label="Exact circle")
for h, color, style, name in [(history_euler, VERMILLION, "-.", "Euler"),
                              (history, BLUE, "-", "Leapfrog (KDK)"),
                              (history_rk4, GREEN, "--", "RK4")]:
    ax.plot(h[:, 0, 0], h[:, 0, 1], color=color, ls=style, lw=1.3, label=name)
ax.scatter([0.0], [0.0], marker="*", s=80, color="black")
ax.scatter([r0], [0.0], marker="o", facecolors="none", edgecolors="black", label="Start")
ax.set(xlabel="x [kpc]", ylabel="y [kpc]", title=f"One period; {n_steps} steps per method")
ax.set_aspect("equal")
ax.legend(loc="best", fontsize=9)
save_figure(fig, "ch01_method_comparison")

# %% [markdown]
# ## 8. 保存量を確認する（本文1.9）
# $\varepsilon=v^2/2-\mu/r$と$\ell_z=xv_y-yv_x$は、この静的中心力場の保存量です。
# **重要：** KDKの診断には、半整数時刻の `vel_half` ではなく整数時刻の速度を使います。
# leapfrogもエネルギーを厳密には保存しません。中心力の角運動量は、各kickとdriftで保たれます。

# %%
# --8<-- [start:diagnostics]
def diagnostics(history, mu):
    pos = history[:, 0, :2]
    vel = history[:, 0, 2:]
    radius = np.sqrt(np.sum(pos**2, axis=1))
    ### BEGIN SOLUTION
    energy = 0.5 * np.sum(vel**2, axis=1) - mu / radius
    angular_momentum = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    ### END SOLUTION
    return radius, energy, angular_momentum
# --8<-- [end:diagnostics]

# --8<-- [start:error-metrics]
def error_metrics(times, history, mu, r0):
    pos_exact, _ = exact_circle(times, r0, mu)
    radius, energy, angular_momentum = diagnostics(history, mu)
    position_error = np.linalg.norm(history[:, 0, :2] - pos_exact, axis=1) / r0
    energy_error = np.abs(energy - energy[0]) / abs(energy[0])
    angular_error = np.abs(angular_momentum - angular_momentum[0]) / abs(angular_momentum[0])
    return np.max(position_error), np.max(energy_error), np.max(angular_error)
# --8<-- [end:error-metrics]
radius, energy, angular_momentum = diagnostics(history, mu)
_, energy_euler, angular_euler = diagnostics(history_euler, mu)
_, energy_rk4, angular_rk4 = diagnostics(history_rk4, mu)
base_errors = error_metrics(times, history, mu, r0)
rk4_errors = error_metrics(times, history_rk4, mu, r0)
euler_errors = error_metrics(times, history_euler, mu, r0)
print("method       force calls   max position/r0    max |dE|/|E0|    max |dL|/|L0|")
for name, calls, errors in [("Euler", n_steps, euler_errors), ("Leapfrog", n_steps + 1, base_errors),
                            ("RK4", 4 * n_steps, rk4_errors)]:
    print(f"{name:10s} {calls:7d}       {errors[0]:.6e}       {errors[1]:.6e}       {errors[2]:.6e}")

# %%
fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
for values, color, style, label in [(energy_euler, VERMILLION, "-.", "Euler"),
                                    (energy, BLUE, "-", "Leapfrog (KDK)"),
                                    (energy_rk4, GREEN, "--", "RK4")]:
    error = np.abs(values - values[0]) / abs(values[0])
    mask = error > 0.0  # 対数軸で0を描かないだけで、データは変更しません。
    ax.semilogy(times[mask] / period, error[mask], color=color, ls=style, label=label)
ax.set(xlabel="Time / T", ylabel="Absolute relative energy error")
ax.legend()
save_figure(fig, "ch01_energy_error")

# %% [markdown]
# ## 9. 時間刻みと次数（本文1.10）
# 初期条件と最終時刻$T$を固定し、位置ベクトルの誤差を測ります。
# 十分小さい刻みでは、leapfrogは2次、RK4は4次です。
# 円軌道のエネルギー誤差は特別な打ち消しを含むため、それだけで手法の次数を判定しません。

# %%
# --8<-- [start:convergence]
step_counts = [20, 40, 80, 160, 320]
rows_lf, rows_rk4 = [], []
for count in step_counts:
    t, h_lf = integrate_leapfrog(acceleration_kepler, state0, period, count, mu)
    _, h_rk4 = integrate(rhs_kepler, rk4_step, state0, period, count, mu)
    rows_lf.append([count, period / count, *error_metrics(t, h_lf, mu, r0)])
    rows_rk4.append([count, period / count, *error_metrics(t, h_rk4, mu, r0)])
convergence_lf, convergence_rk4 = np.array(rows_lf), np.array(rows_rk4)
orders_lf = np.log2(convergence_lf[:-1, 2] / convergence_lf[1:, 2])
orders_rk4 = np.log2(convergence_rk4[:-1, 2] / convergence_rk4[1:, 2])
# --8<-- [end:convergence]
print("steps   dt [Myr]    position error (LF)   order     position error (RK4)   order")
for i, (lf, rk) in enumerate(zip(convergence_lf, convergence_rk4)):
    plf = "--" if i == 0 else f"{orders_lf[i - 1]:.3f}"
    prk = "--" if i == 0 else f"{orders_rk4[i - 1]:.3f}"
    print(f"{int(lf[0]):5d}   {lf[1]*1000:9.4f}   {lf[2]:.6e}   {plf:>6s}    {rk[2]:.6e}   {prk:>6s}")
for name, values in [("leapfrog", convergence_lf), ("rk4", convergence_rk4)]:
    np.savetxt(DATADIR / f"convergence_{name}.csv", values, delimiter=",",
               header="n_steps,dt_Gyr,max_position_error,max_energy_error,max_angular_error", comments="")

# %%
fig, ax = plt.subplots(figsize=(6.3, 4.5), layout="constrained")
stepsize = convergence_lf[:, 1] / period
ax.loglog(stepsize, convergence_lf[:, 2], "o-", color=BLUE, label="Leapfrog (KDK)")
ax.loglog(stepsize, convergence_rk4[:, 2], "s--", color=GREEN, label="RK4")
ref_lf = convergence_lf[-1, 2] * (stepsize / stepsize[-1])**2
ref_rk = convergence_rk4[-1, 2] * (stepsize / stepsize[-1])**4
ax.loglog(stepsize, ref_lf, ":", color=GRAY, label=r"Reference: $(\Delta t/T)^2$")
ax.loglog(stepsize, ref_rk, "-.", color=GRAY, label=r"Reference: $(\Delta t/T)^4$")
ax.set(xlabel=r"$\Delta t / T$", ylabel="Maximum position error / initial radius")
ax.legend(fontsize=9)
save_figure(fig, "ch01_convergence")

# %% [markdown]
# ## 10. 計算コストも比べる（本文1.11）
# 積分の力評価は、Eulerが$N_{\rm step}$回、キャッシュ付きKDKが$N_{\rm step}+1$回、
# RK4が$4N_{\rm step}$回です。診断・描画の費用は含めません。
# 同じ力評価回数800回なら、KDKは799ステップ、RK4は200ステップになります。
# 1粒子の実時間はPythonの処理や配列生成にも影響されます。単純に「4倍高速」とは言えません。

# %%
# --8<-- [start:equal-cost]
force_budget = 800
n_lf_cost = force_budget - 1
n_rk_cost = force_budget // 4
t_lf_cost, h_lf_cost = integrate_leapfrog(acceleration_kepler, state0, period, n_lf_cost, mu)
t_rk_cost, h_rk_cost = integrate(rhs_kepler, rk4_step, state0, period, n_rk_cost, mu)
cost_lf_errors = error_metrics(t_lf_cost, h_lf_cost, mu, r0)
cost_rk4_errors = error_metrics(t_rk_cost, h_rk_cost, mu, r0)
# --8<-- [end:equal-cost]
for name, count, errors in [("Leapfrog", n_lf_cost, cost_lf_errors), ("RK4", n_rk_cost, cost_rk4_errors)]:
    print(f"{name:8s}: steps={count}, force calls=800, position error={errors[0]:.6e}")

# %% [markdown]
# ## 11. 長時間のエネルギー変化（本文1.11）
# 同じ円軌道を100周期まで計算します。ここだけ$\Delta t=T/100$を使います。
# 同じ刻みでの比較であり、力評価回数はほぼ4倍異なります。
# 符号付きのエネルギー差を描き、振動と一方向のずれを区別してください。
# この実験だけで、任意の初期条件や刻みでも同じ振る舞いになるとは結論できません。

# %%
# --8<-- [start:long-time]
n_periods = 100
steps_per_period = 100
n_long = n_periods * steps_per_period
t_long, h_long_lf = integrate_leapfrog(acceleration_kepler, state0, n_periods * period, n_long, mu)
_, h_long_rk4 = integrate(rhs_kepler, rk4_step, state0, n_periods * period, n_long, mu)
_, e_long_lf, l_long_lf = diagnostics(h_long_lf, mu)
_, e_long_rk4, l_long_rk4 = diagnostics(h_long_rk4, mu)
signed_lf = (e_long_lf - e_long_lf[0]) / abs(e_long_lf[0])
signed_rk4 = (e_long_rk4 - e_long_rk4[0]) / abs(e_long_rk4[0])
# --8<-- [end:long-time]
fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
ax.plot(t_long / period, signed_lf, color=BLUE, lw=0.65, label="Leapfrog (KDK)")
ax.plot(t_long / period, signed_rk4, color=GREEN, ls="--", lw=1.4, label="RK4")
ax.axhline(0.0, color=GRAY, ls=":", lw=0.8)
ax.set(xlabel="Time / T", ylabel=r"$(\varepsilon(t)-\varepsilon(0))/|\varepsilon(0)|$")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
ax.legend()
save_figure(fig, "ch01_long_time_energy")
print("Long-time signed energy range (LF):", float(signed_lf.min()), float(signed_lf.max()))
print("Long-time final signed energy error (RK4):", float(signed_rk4[-1]))
np.savetxt(DATADIR / "long_time.csv", np.column_stack((t_long, signed_lf, signed_rk4)), delimiter=",",
           header="time_Gyr,signed_relative_energy_leapfrog,signed_relative_energy_rk4", comments="")

# %% [markdown]
# ## 12. 演習（本文の演習1.1〜1.7）
# **確認1.1：** $v_c,T,\varepsilon_0,\ell_{z,0}$と単位を求めてください。
# **確認1.2：** KDKの3段階と加速度を再利用する理由を説明してください。
# **確認1.3：** RK4の `k2` と `k3` の状態の違いを説明してください。
# **応用1.4：** 50、100、200ステップでKDKとRK4の位置誤差を比べてください。
# **応用1.5：** 初期の$y$速度だけを逆向きにし、KDKで軌道と角運動量を調べてください。
# **応用1.6：** 800回の力評価での比較を再現し、同じ刻みでの比較との違いを説明してください。
# **発展1.7：** 速度比例抵抗$-\gamma\boldsymbol v$（$\gamma=0.05/T$）をRK4で扱ってください。
# この抵抗はChandrasekhar公式ではありません。

# %%
### BEGIN SOLUTION
exercise_lf_errors, exercise_rk4_errors = [], []
for count in [50, 100, 200]:
    t_ex, h_lf_ex = integrate_leapfrog(acceleration_kepler, state0, period, count, mu)
    _, h_rk_ex = integrate(rhs_kepler, rk4_step, state0, period, count, mu)
    exercise_lf_errors.append(error_metrics(t_ex, h_lf_ex, mu, r0)[0])
    exercise_rk4_errors.append(error_metrics(t_ex, h_rk_ex, mu, r0)[0])
print("応用1.4の位置誤差（KDK）：", exercise_lf_errors)
print("応用1.4の位置誤差（RK4）：", exercise_rk4_errors)
reverse_state = state0.copy()
reverse_state[0, 3] *= -1.0
_, reverse_history = integrate_leapfrog(acceleration_kepler, reverse_state, period, n_steps, mu)
mirror = history.copy()
mirror[:, 0, [1, 3]] *= -1.0
print("応用1.5の鏡映対称性の最大差：", np.max(np.abs(reverse_history - mirror)))
### END SOLUTION

# %%
# --8<-- [start:linear-drag-exercise]
def rhs_linear_drag(t, state, mu):
    """発展演習：速度比例抵抗。Chandrasekhar摩擦ではありません。"""
    ### BEGIN SOLUTION
    derivative = rhs_kepler(t, state, mu)
    derivative[:, 2:] -= gamma * state[:, 2:]
    return derivative
    ### END SOLUTION
# --8<-- [end:linear-drag-exercise]
gamma = 0.05 / period
times_drag, history_drag = integrate(rhs_linear_drag, rk4_step, state0, period, 400, mu)
radius_drag, energy_drag, angular_drag = diagnostics(history_drag, mu)
print(f"Linear-drag exercise: E(T)/|E(0)| = {energy_drag[-1] / abs(energy_drag[0]):.6f}")

# %% [markdown]
# ## 13. 検証結果を保存する
# 標準条件に対する確認です。質量や刻みを変更する実験では、閾値も見直してください。
# 力評価回数、KDKの時間反転、等加速度、RK4の段階時刻などの独立試験は `test_chap01.py` にあります。

# %%
assert np.array_equal(state0, np.column_stack((pos0, vel0)))
assert history.shape == (n_steps + 1, 1, 4)
assert np.all(np.isfinite(history))
assert base_errors[0] < 0.005 and base_errors[2] < 1.0e-12
assert rk4_errors[0] < 1.0e-5 and rk4_errors[1] < 1.0e-7
assert np.all(np.diff(convergence_lf[:, 2]) < 0.0)
assert np.all(np.diff(convergence_rk4[:, 2]) < 0.0)
assert 1.9 < orders_lf[-1] < 2.1
assert 3.8 < orders_rk4[-1] < 4.3
assert energy_drag[-1] < energy_drag[0]
summary = {
    "version": "2.0", "outline_version": "3", "style_version": "1.2",
    "model": "fixed point mass; one test particle; circular initial orbit",
    "units": {"length": "kpc", "time": "Gyr", "mass": "Msun", "internal_velocity": "kpc/Gyr"},
    "mass_Msun": mass, "r0_kpc": r0, "mu": mu, "vc_km_s": vc / KM_S_TO_KPC_GYR,
    "period_Gyr": period, "n_steps": n_steps, "dt_Gyr": dt,
    "initial_specific_energy": energy[0], "initial_specific_angular_momentum": angular_momentum[0],
    "initial_acceleration_x": -mu / r0**2,
    "leapfrog_errors": list(base_errors), "rk4_errors": list(rk4_errors), "euler_errors": list(euler_errors),
    "force_calls": {"Euler": n_steps, "Leapfrog": n_steps + 1, "RK4": 4 * n_steps},
    "convergence_leapfrog": convergence_lf.tolist(), "convergence_rk4": convergence_rk4.tolist(),
    "orders_leapfrog": orders_lf.tolist(), "orders_rk4": orders_rk4.tolist(),
    "equal_cost": {"force_budget": force_budget, "n_leapfrog": n_lf_cost, "n_rk4": n_rk_cost,
                   "leapfrog_errors": list(cost_lf_errors), "rk4_errors": list(cost_rk4_errors)},
    "long_time": {"n_periods": n_periods, "steps_per_period": steps_per_period,
                  "leapfrog_min_signed_energy": float(signed_lf.min()),
                  "leapfrog_max_signed_energy": float(signed_lf.max()),
                  "rk4_final_signed_energy": float(signed_rk4[-1])},
    "exercise_leapfrog_errors": exercise_lf_errors, "exercise_rk4_errors": exercise_rk4_errors,
    "linear_drag_final_energy_ratio": float(energy_drag[-1] / abs(energy_drag[0])),
    "checks_passed": True,
    "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__, "platform": platform.platform()}
}
(DATADIR / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
for name, h in [("leapfrog", history), ("rk4", history_rk4), ("euler", history_euler)]:
    np.savetxt(DATADIR / f"orbit_{name}.csv", np.column_stack((times, h[:, 0, :])), delimiter=",",
               header="time_Gyr,x_kpc,y_kpc,vx_kpc_Gyr,vy_kpc_Gyr", comments="")
print("標準モデルの検証を通過しました。保存先：", DATADIR)

# %% [markdown]
# ## 14. アニメーション（本文1.8）
# KDKの計算済み出力から作ります。`interval`は表示の間隔であって積分刻みではありません。
# 動画エンコーダは不要です。スクリプト単独実行時は、この表示を省きます。

# %%
try:
    from IPython import get_ipython
    in_notebook = get_ipython() is not None
except ImportError:
    in_notebook = False

if in_notebook:
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display
    fig, ax = plt.subplots(figsize=(5.0, 4.8), layout="constrained")
    ax.plot(pos_exact[:, 0], pos_exact[:, 1], color=GRAY, ls="--", label="Exact circle")
    ax.scatter([0], [0], color="black", marker="*", s=70)
    trail, = ax.plot([], [], color=BLUE, lw=1.2, label="Leapfrog (KDK)")
    particle, = ax.plot([], [], "o", color=VERMILLION, ms=6)
    clock = ax.text(0.03, 0.94, "", transform=ax.transAxes)
    ax.set(xlabel="x [kpc]", ylabel="y [kpc]", xlim=(-12, 12), ylim=(-12, 12))
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)

    def update_frame(n):
        trail.set_data(pos_history[:n+1, 0], pos_history[:n+1, 1])
        particle.set_data([pos_history[n, 0]], [pos_history[n, 1]])
        clock.set_text(f"t / T = {times[n] / period:.2f}")
        return trail, particle, clock

    frame_ids = np.unique(np.linspace(0, n_steps, min(n_steps + 1, 61), dtype=int))
    anim = FuncAnimation(fig, update_frame, frames=frame_ids, interval=70, repeat=False)
    html = anim.to_jshtml()
    plt.close(fig)
    (FIGDIR / "ch01_orbit_animation.html").write_text(html, encoding="utf-8")
    display(HTML(html))
else:
    print("アニメーションの再生は完成版Notebookで行います。")

# %% [markdown]
# ## 参考
# - 杉浦洋『数値計算の基礎と応用［新訂版］』第11章：Euler法・Runge--Kutta法。
# - 幸谷智紀『Python数値計算プログラミング』第13章：PythonによるEuler法・古典的Runge--Kutta法の実装例。
# - J. Binney & S. Tremaine (2008), *Galactic Dynamics*, 2nd ed., §§3.1, 3.4.1, 3.4.2.
#
# 長時間の保存特性の説明は、固定刻みで十分に分解された滑らかな静的Hamilton系についてのものです。
# 「leapfrogはエネルギーを厳密に保存する」「RK4だけが速度依存力を扱える」という意味ではありません。
# 構成案・スタイルガイドは教育方針の資料です。図・数値表は本章のコードを実行して得たものです。
