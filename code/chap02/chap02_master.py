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
# # 第2章　2体問題を計算する
# **完成版 / 全章改訂 v2.0**
#
# 第1章のKDK leapfrogを、互いに重力を及ぼす2天体へ拡張します。
# 本章は**点質量2体のNewton重力**を計算します。移動するPlummerポテンシャルは第3・5章で扱います。
#
# 重心・相対運動 → 初期条件 → 同時更新 → 円・楕円・放物線 → 保存量 → 刻みの確認、の順に進めます。
# 解析解の数式はRyden『Celestial and Stellar Dynamics』§1.2–1.3に対応します。
# 本Notebookの質量・距離・刻み・演習条件は、本教材用に設定した数値実験です。
#
# **実行方法：** カーネルを再起動して上から順に実行します。パラメータ変更後も上から実行してください。
# 依存はNumPyとMatplotlibです。Notebookの表示にはJupyter/IPythonを使います。

# %%
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for folder in (ROOT, *ROOT.parents):
    chapter_folder = folder / "code" / "chap02"
    if (chapter_folder / "chap02_master.py").is_file():
        ROOT = chapter_folder
        break
FIGDIR, DATADIR = ROOT / "figures", ROOT / "data"
FIGDIR.mkdir(exist_ok=True)
DATADIR.mkdir(exist_ok=True)
IN_NOTEBOOK = "get_ipython" in globals()
BLUE, VERMILLION, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#777777"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})


def save_figure(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", dpi=180, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

# %% [markdown]
# ## 1. 単位と計算条件
# 内部単位はkpc, Gyr, $M_\odot$です。速度の内部単位はkpc/Gyrで、km/sへの変換は表示時だけ行います。
#
# $\boldsymbol D=\boldsymbol R_2-\boldsymbol R_1$、$\boldsymbol U=\boldsymbol V_2-\boldsymbol V_1$とします。
# 相対運動の重力定数は$\mu=G(M_1+M_2)$です。換算質量$m_{\rm red}=M_1M_2/(M_1+M_2)$とは別の量です。
#
# 円・楕円・放物線の比較では、同じ初期距離$D_0$から接線速度だけを変えます。
# 全ケースを同じ基準時間$T_{\rm c}=2\pi\sqrt{D_0^3/\mu}$まで積分します。

# %%
# --8<-- [start:parameters]
KM_S_TO_KPC_GYR = 1.0227121650537077
G = 4.30091e-6 * KM_S_TO_KPC_GYR**2  # kpc^3 / (Msun Gyr^2)
masses = np.array([1.0e11, 5.0e10])  # M1, M2 [Msun]
d0 = 20.0                            # 2天体の初期距離 [kpc]
mtot = np.sum(masses)
mu = G * mtot
m_red = masses[0] * masses[1] / mtot
v_c = np.sqrt(mu / d0)               # 相対運動の円運動速度 [kpc/Gyr]
t_c = 2.0 * np.pi * np.sqrt(d0**3 / mu)
n_steps = 1000
t_end = t_c
dt = t_end / n_steps
eta_values = {"Circular": 1.0, "Elliptic": 0.8, "Parabolic": np.sqrt(2.0)}
# --8<-- [end:parameters]
print(f"Relative circular speed = {v_c / KM_S_TO_KPC_GYR:.6f} km/s")
print(f"Reference period Tc = {t_c:.9f} Gyr; dt = {dt * 1000:.6f} Myr")
print(f"Reduced mass = {m_red:.6e} Msun")

# %% [markdown]
# ## 2. 重心系の初期条件を作る
# 重心を$\boldsymbol R_{\rm cm}$、その速度を$\boldsymbol V_{\rm cm}$とすると、
# $$\boldsymbol R_1=\boldsymbol R_{\rm cm}-\frac{M_2}{M_1+M_2}\boldsymbol D,\qquad
# \boldsymbol R_2=\boldsymbol R_{\rm cm}+\frac{M_1}{M_1+M_2}\boldsymbol D.$$
# 速度にも同じ質量比を使います。重い天体ほど重心に近く、遅く動きます。
# `state`の行0が天体1、行1が天体2、各行は`[x, y, vx, vy]`です。
# 位置・速度はそれぞれ`(2, 2)`、まとめた状態は`(2, 4)`になります。

# %%
def check_masses(masses):
    masses = np.asarray(masses, dtype=float)
    if masses.shape != (2,) or not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("masses must contain two finite, positive masses.")
    return masses


# --8<-- [start:initial-state]
def make_initial_state(masses, separation, relative_velocity, r_cm=None, v_cm=None):
    """相対位置・速度から2天体を配置します。本文 eq:ch2-inverse-transform。"""
    masses = check_masses(masses)
    separation = np.asarray(separation, dtype=float)
    relative_velocity = np.asarray(relative_velocity, dtype=float)
    r_cm = np.zeros(2) if r_cm is None else np.asarray(r_cm, dtype=float)
    v_cm = np.zeros(2) if v_cm is None else np.asarray(v_cm, dtype=float)
    for vector in (separation, relative_velocity, r_cm, v_cm):
        if vector.shape != (2,) or not np.all(np.isfinite(vector)):
            raise ValueError("Positions and velocities must be finite 2D vectors.")
    if np.linalg.norm(separation) == 0.0:
        raise ValueError("The two point masses must not coincide.")
    ### BEGIN SOLUTION
    mtot = np.sum(masses)
    pos1 = r_cm - masses[1] / mtot * separation
    pos2 = r_cm + masses[0] / mtot * separation
    vel1 = v_cm - masses[1] / mtot * relative_velocity
    vel2 = v_cm + masses[0] / mtot * relative_velocity
    ### END SOLUTION
    return np.column_stack((np.array([pos1, pos2]), np.array([vel1, vel2])))
# --8<-- [end:initial-state]

state0 = make_initial_state(masses, [d0, 0.0], [0.0, 0.8 * v_c])
print("Initial state [kpc, kpc, kpc/Gyr, kpc/Gyr]:")
print(state0)

# %% [markdown]
# ## 3. 一つの粒子対から両方の加速度を計算する
# $\boldsymbol A_1=GM_2\boldsymbol D/D^3$、$\boldsymbol A_2=-GM_1\boldsymbol D/D^3$です。
# 作用・反作用は**力**の関係$M_1\boldsymbol A_1+M_2\boldsymbol A_2=0$であり、
# 質量が違えば加速度の大きさは同じではありません。

# %%
# --8<-- [start:acceleration-scalar]
x1, y1 = state0[0, :2]
x2, y2 = state0[1, :2]
dx, dy = x2 - x1, y2 - y1
sep2 = dx**2 + dy**2
ax1, ay1 = G * masses[1] * dx / sep2**1.5, G * masses[1] * dy / sep2**1.5
ax2, ay2 = -G * masses[0] * dx / sep2**1.5, -G * masses[0] * dy / sep2**1.5
# --8<-- [end:acceleration-scalar]
print(f"A1 = ({ax1:.6f}, {ay1:.6f}); A2 = ({ax2:.6f}, {ay2:.6f}) [kpc/Gyr^2]")

# --8<-- [start:acceleration-two-body]
def acceleration_two_body(pos, masses):
    """2天体の同じ時刻の位置から両方の加速度を返します。"""
    dr = pos[1] - pos[0]
    sep2 = np.dot(dr, dr)
    if not np.isfinite(sep2) or sep2 <= 0.0:
        raise ValueError("Point-mass separation must be finite and nonzero.")
    ### BEGIN SOLUTION
    common = G * dr / sep2**1.5
    acc1 = masses[1] * common
    acc2 = -masses[0] * common
    ### END SOLUTION
    return np.array([acc1, acc2])
# --8<-- [end:acceleration-two-body]
acc0 = acceleration_two_body(state0[:, :2], masses)
force_sum = masses[0] * acc0[0] + masses[1] * acc0[1]
print("Sum of internal forces:", force_sum)

# %% [markdown]
# ## 4. 第1章のKDKを2天体へ適用する
# 積分法は変えません。最後の引数を`mu`から`masses`へ替えて加速度関数に渡します。
# **両天体を半kick → 両天体をdrift → 新しい両天体の位置から力を評価 → 両天体を半kick**です。
# 最後に計算した加速度を次のステップで再利用します。

# %%
# --8<-- [start:leapfrog-step]
def leapfrog_step(acceleration, pos, vel, acc, dt, masses):
    """KDK。accは現在のposに対応する加速度です。"""
    vel_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_half
    acc_new = acceleration(pos_new, masses)
    vel_new = vel_half + 0.5 * dt * acc_new
    return pos_new, vel_new, acc_new
# --8<-- [end:leapfrog-step]


def prepare_history(state0, t_end, n_steps):
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer.")
    if not np.isfinite(t_end) or t_end <= 0.0:
        raise ValueError("t_end must be finite and positive.")
    state = np.array(state0, dtype=float, copy=True)
    if state.shape != (2, 4) or not np.all(np.isfinite(state)):
        raise ValueError("state0 must be a finite array with shape (2, 4).")
    dt = t_end / n_steps
    times = np.arange(n_steps + 1) * dt
    history = np.empty((n_steps + 1, 2, 4))
    history[0] = state
    return state, times, history, dt


# --8<-- [start:integrate-leapfrog]
def integrate_leapfrog(acceleration, state0, t_end, n_steps, masses):
    masses = check_masses(masses)
    state, times, history, dt = prepare_history(state0, t_end, n_steps)
    pos, vel = state[:, :2].copy(), state[:, 2:].copy()
    acc = acceleration(pos, masses)  # 初回だけ計算します。
    for n in range(n_steps):
        pos, vel, acc = leapfrog_step(acceleration, pos, vel, acc, dt, masses)
        history[n + 1] = np.column_stack((pos, vel))
        if not np.all(np.isfinite(history[n + 1])):
            raise FloatingPointError("Non-finite state; check the time step.")
    return times, history
# --8<-- [end:integrate-leapfrog]

# %% [markdown]
# ## 5. 接線速度だけを変えた3ケース
# $\boldsymbol D(0)=(D_0,0)$、$\boldsymbol U(0)=(0,\eta v_{\rm c})$とします。
# $\eta=1$は円、$\eta=0.8$は遠点から始まる楕円、$\eta=\sqrt{2}$は近点から始まる放物線です。
# 全ケースで$M_1,M_2,D_0,t_{\rm end}=T_{\rm c},N_{\rm step}$をそろえます。

# %%
# --8<-- [start:run-cases]
cases = {}
start = perf_counter()
for name, eta in eta_values.items():
    initial = make_initial_state(masses, [d0, 0.0], [0.0, eta * v_c])
    times, history = integrate_leapfrog(acceleration_two_body, initial, t_end, n_steps, masses)
    cases[name] = {"eta": eta, "initial": initial, "times": times, "history": history}
# --8<-- [end:run-cases]
print(f"Three integrations: {perf_counter() - start:.3f} s (this environment)")
print("Force evaluations per integration:", n_steps + 1)

# %% [markdown]
# ## 6. 重心系で2天体の軌道を描く
# 楕円の場合を例に、天体1・2を別々に描きます。原点は重心です。
# 同時刻の2天体と重心は一直線に並び、重心からの距離の比は$M_2/M_1$です。
# 色だけでなく線種とマーカーでも区別します。

# %%
# --8<-- [start:plot-barycentric]
history = cases["Elliptic"]["history"]
fig, ax = plt.subplots(figsize=(5.8, 5.2), layout="constrained")
ax.plot(history[:, 0, 0], history[:, 0, 1], color=BLUE, label="Body 1")
ax.plot(history[:, 1, 0], history[:, 1, 1], color=VERMILLION, ls="--", label="Body 2")
ax.plot(history[0, 0, 0], history[0, 0, 1], "o", color=BLUE, ms=6)
ax.plot(history[0, 1, 0], history[0, 1, 1], "s", color=VERMILLION, ms=6)
ax.scatter([0.0], [0.0], marker="*", color="black", s=100, label="Center of mass")
ax.set(xlabel="x [kpc]", ylabel="y [kpc]", title="Elliptic orbit; mass ratio 2:1")
ax.set_aspect("equal")
ax.legend(fontsize=9)
save_figure(fig, "ch02_barycentric")
# --8<-- [end:plot-barycentric]

# %% [markdown]
# ## 7. 相対軌道と解析的な軌道形状
# 本章の初期条件では、相対位置の極角を$\phi$として
# $$D(\phi)=\frac{\eta^2D_0}{1+(\eta^2-1)\cos\phi}$$
# と書けます。これは軌道の**形**であり、同時刻の位置を与える式ではありません。
# 円軌道の時間依存は第1章と同じ三角関数で別に確認します。

# %%
styles = {"Circular": (BLUE, "-"), "Elliptic": (VERMILLION, "--"), "Parabolic": (GREEN, "-.")}
fig, ax = plt.subplots(figsize=(6.1, 5.4), layout="constrained")
for name, case in cases.items():
    relative = case["history"][:, 1, :2] - case["history"][:, 0, :2]
    color, ls = styles[name]
    ax.plot(relative[:, 0], relative[:, 1], color=color, ls=ls, label=name)
    phase = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
    phi = np.linspace(0.0, min(phase[-1], 2.0 * np.pi), 19)
    eta = case["eta"]
    r_exact = eta**2 * d0 / (1.0 + (eta**2 - 1.0) * np.cos(phi))
    ax.plot(r_exact * np.cos(phi), r_exact * np.sin(phi), "o", ms=3, mfc="none", mec=color)
ax.plot(d0, 0.0, "k*", ms=10, label="Initial relative position")
ax.set(xlabel="Relative x [kpc]", ylabel="Relative y [kpc]", title="Same elapsed time: Tc; open markers = exact conics")
ax.set_aspect("equal")
ax.legend(fontsize=9)
save_figure(fig, "ch02_relative_orbits")

# %% [markdown]
# ## 8. 全エネルギー・全角運動量・全運動量・重心
# 全エネルギーは
# $$E=\frac{M_1V_1^2}{2}+\frac{M_2V_2^2}{2}-\frac{GM_1M_2}{D}.$$
# ポテンシャルエネルギーは**粒子対ごとに1回**です。重心が静止する系では$E=m_{\rm red}\varepsilon$です。
#
# 放物線の初期エネルギーは0なので、$|E(0)|$では割りません。
# $E_{\rm scale}=GM_1M_2/D_0$を3ケースに共通して用います。

# %%
# --8<-- [start:diagnostics]
def diagnostics(history, masses):
    pos, vel = history[:, :, :2], history[:, :, 2:]
    dr, dv = pos[:, 1] - pos[:, 0], vel[:, 1] - vel[:, 0]
    separation = np.linalg.norm(dr, axis=1)
    ### BEGIN SOLUTION
    kinetic = 0.5 * masses[0] * np.sum(vel[:, 0]**2, axis=1)
    kinetic += 0.5 * masses[1] * np.sum(vel[:, 1]**2, axis=1)
    potential = -G * masses[0] * masses[1] / separation
    energy = kinetic + potential
    momentum = masses[0] * vel[:, 0] + masses[1] * vel[:, 1]
    r_cm = (masses[0] * pos[:, 0] + masses[1] * pos[:, 1]) / np.sum(masses)
    angular = masses[0] * (pos[:, 0, 0] * vel[:, 0, 1] - pos[:, 0, 1] * vel[:, 0, 0])
    angular += masses[1] * (pos[:, 1, 0] * vel[:, 1, 1] - pos[:, 1, 1] * vel[:, 1, 0])
    ### END SOLUTION
    return {"D": separation, "dr": dr, "dv": dv, "K": kinetic, "W": potential,
            "E": energy, "L": angular, "P": momentum, "Rcm": r_cm}
# --8<-- [end:diagnostics]


# --8<-- [start:error-metrics]
def conservation_errors(times, history, masses):
    values = diagnostics(history, masses)
    d_initial = values["D"][0]
    vc_initial = np.sqrt(G * np.sum(masses) / d_initial)
    e_scale = G * masses[0] * masses[1] / d_initial
    p_scale = np.sum(masses) * vc_initial
    r_cm_exact = values["Rcm"][0] + times[:, None] * values["P"][0] / np.sum(masses)
    err_e = np.max(np.abs(values["E"] - values["E"][0])) / e_scale
    l_scale = abs(values["L"][0])
    if l_scale == 0.0:
        raise ValueError("This angular-momentum normalization requires nonzero L(0).")
    err_l = np.max(np.abs(values["L"] - values["L"][0])) / l_scale
    err_p = np.max(np.linalg.norm(values["P"] - values["P"][0], axis=1)) / p_scale
    err_cm = np.max(np.linalg.norm(values["Rcm"] - r_cm_exact, axis=1)) / d_initial
    return np.array([err_e, err_l, err_p, err_cm])
# --8<-- [end:error-metrics]

base_errors = {}
print("Case          energy/Escale     angular/|L0|    momentum/Pscale   CM error/D0")
for name, case in cases.items():
    base_errors[name] = conservation_errors(case["times"], case["history"], masses)
    print(f"{name:10s}  " + "  ".join(f"{x:.6e}" for x in base_errors[name]))

# %%
fig, ax = plt.subplots(figsize=(6.2, 4.1), layout="constrained")
for name, case in cases.items():
    values = diagnostics(case["history"], masses)
    e_scale = G * masses[0] * masses[1] / d0
    color, ls = styles[name]
    ax.plot(case["times"] / t_c, (values["E"] - values["E"][0]) / e_scale, color=color, ls=ls, label=name)
ax.set(xlabel="Time / Tc", ylabel="[E(t) - E(0)] / E_scale", title="Signed energy error; KDK; 1000 steps")
ax.legend(fontsize=9)
save_figure(fig, "ch02_energy")

# %% [markdown]
# ## 9. 距離の変化と楕円の近点・遠点
# $0<\eta<1$の場合、初期位置は遠点です。
# $$e=1-\eta^2,\quad a_{\rm orb}=\frac{D_0}{2-\eta^2},\quad
# D_{\rm p}=\frac{\eta^2D_0}{2-\eta^2},\quad D_{\rm a}=D_0.$$
# 楕円の周期は$T_{\rm ell}=2\pi\sqrt{a_{\rm orb}^3/\mu}$です。
# 保存した距離の最小値は、連続時間の近点そのものではありません。

# %%
eta_ell = eta_values["Elliptic"]
a_ell = d0 / (2.0 - eta_ell**2)
e_ell = 1.0 - eta_ell**2
r_peri_ell = a_ell * (1.0 - e_ell)
t_ell = 2.0 * np.pi * np.sqrt(a_ell**3 / mu)
values_ell = diagnostics(cases["Elliptic"]["history"], masses)
i_min_ell = np.argmin(values_ell["D"])
print(f"Ellipse: e = {e_ell:.6f}; a = {a_ell:.6f} kpc")
print(f"Analytic pericenter = {r_peri_ell:.6f} kpc; period = {t_ell:.9f} Gyr")
print(f"Smallest sampled separation = {values_ell['D'][i_min_ell]:.6f} kpc")
fig, ax = plt.subplots(figsize=(6.2, 4.1), layout="constrained")
for name, case in cases.items():
    color, ls = styles[name]
    separation = diagnostics(case["history"], masses)["D"]
    ax.plot(case["times"] / t_c, separation, color=color, ls=ls, label=name)
ax.axhline(r_peri_ell, color=GRAY, ls=":", lw=1, label="Exact elliptic pericenter")
ax.set(xlabel="Time / Tc", ylabel="Separation [kpc]")
ax.legend(fontsize=9)
save_figure(fig, "ch02_separation")

# %% [markdown]
# ## 10. 時間刻みの確認
# 円軌道は同時刻の解析解との差を使います。楕円では、**解析的な周期**$T_{\rm ell}$だけ積分し、
# 初期位置にどれだけ戻れたかを調べます。楕円の終点を見てから周期を調整してはいけません。
# 円と楕円では最終時刻が異なります。各系列の中では、初期条件と最終時刻を固定します。
# 保存量とは独立に位置を調べることで、位相のずれも確認できます。

# %%
# --8<-- [start:exact-circle]
def exact_relative_circle(times, d0, mu):
    omega = np.sqrt(mu / d0**3)
    phase = omega * np.asarray(times)
    return np.column_stack((d0 * np.cos(phase), d0 * np.sin(phase)))
# --8<-- [end:exact-circle]

# --8<-- [start:convergence]
step_counts = [100, 200, 400, 800]
convergence_rows = []
for count in step_counts:
    tc, hc = integrate_leapfrog(acceleration_two_body, cases["Circular"]["initial"], t_c, count, masses)
    dr_c = hc[:, 1, :2] - hc[:, 0, :2]
    exact_c = exact_relative_circle(tc, d0, mu)
    err_c = np.max(np.linalg.norm(dr_c - exact_c, axis=1)) / d0
    te, he = integrate_leapfrog(acceleration_two_body, cases["Elliptic"]["initial"], t_ell, count, masses)
    dr_end = he[-1, 1, :2] - he[-1, 0, :2]
    err_ell = np.linalg.norm(dr_end - np.array([d0, 0.0])) / d0
    err_energy_ell = conservation_errors(te, he, masses)[0]
    convergence_rows.append([count, t_c / count, t_ell / count, err_c, err_ell, err_energy_ell])
convergence = np.array(convergence_rows)
orders_c = np.log2(convergence[:-1, 3] / convergence[1:, 3])
orders_ell = np.log2(convergence[:-1, 4] / convergence[1:, 4])
# --8<-- [end:convergence]
print("steps   circular position error  order     elliptic return error  order")
for i, row in enumerate(convergence):
    oc = "--" if i == 0 else f"{orders_c[i-1]:.4f}"
    oe = "--" if i == 0 else f"{orders_ell[i-1]:.4f}"
    print(f"{int(row[0]):5d}   {row[3]:.6e}  {oc:>7s}    {row[4]:.6e}  {oe:>7s}")
fig, ax = plt.subplots(figsize=(6.1, 4.1), layout="constrained")
x = 1.0 / convergence[:, 0]
ax.loglog(x, convergence[:, 3], "o-", color=BLUE, label="Circular: max position error")
ax.loglog(x, convergence[:, 4], "s--", color=VERMILLION, label="Elliptic: return error")
ax.loglog(x, convergence[-1, 3] * (x / x[-1])**2, ":", color=GRAY, label="Reference slope 2")
ax.set(xlabel="Time step / integration interval", ylabel="Position error / initial separation")
ax.legend(fontsize=9)
save_figure(fig, "ch02_convergence")

# %% [markdown]
# ## 11. 近点距離を指定した接近軌道
# 第5章への準備として、離れた位置から**内向き**に始まる放物線軌道を作ります。
# これはまだ点質量2体です。Plummer型の中心間重力へ替えると、この初期速度の式も見直します。
# $$\ell_z^2=2\mu D_{\rm p},\quad U_t=\ell_z/D_0,\quad
# U_r=-\sqrt{2\mu/D_0-U_t^2}.$$
# 根号内の条件と負号を確認してください。

# %%
# --8<-- [start:parabolic-encounter]
def make_parabolic_encounter(masses, d_start, d_peri):
    masses = check_masses(masses)
    if not (np.isfinite(d_start) and np.isfinite(d_peri) and 0.0 < d_peri < d_start):
        raise ValueError("Require finite distances with 0 < d_peri < d_start.")
    ### BEGIN SOLUTION
    mu = G * np.sum(masses)
    ell = np.sqrt(2.0 * mu * d_peri)
    vt = ell / d_start
    vr = -np.sqrt(2.0 * mu / d_start - vt**2)
    ### END SOLUTION
    return make_initial_state(masses, [d_start, 0.0], [vr, vt])

encounter_initial = make_parabolic_encounter(masses, 60.0, 8.0)
t_enc, h_enc = integrate_leapfrog(acceleration_two_body, encounter_initial, 0.60, 1200, masses)
enc_values = diagnostics(h_enc, masses)
i_peri = np.argmin(enc_values["D"])
# --8<-- [end:parabolic-encounter]
print(f"Encounter: smallest sampled distance = {enc_values['D'][i_peri]:.6f} kpc")
print(f"Time of the smallest sampled distance = {t_enc[i_peri]:.6f} Gyr")
print("Initial relative velocity [km/s]:", (encounter_initial[1, 2:] - encounter_initial[0, 2:]) / KM_S_TO_KPC_GYR)
fig, ax = plt.subplots(figsize=(6.2, 4.0), layout="constrained")
ax.plot(t_enc, enc_values["D"], color=BLUE, label="Incoming parabolic encounter")
ax.axhline(8.0, color=GRAY, ls="--", label="Target pericenter: 8 kpc")
ax.plot(t_enc[i_peri], enc_values["D"][i_peri], "s", color=VERMILLION)
ax.set(xlabel="Time [Gyr]", ylabel="Separation [kpc]")
ax.legend(fontsize=9)
save_figure(fig, "ch02_incoming_encounter")

# %% [markdown]
# ## 12. 応用：系全体を等速で動かす
# 両天体に同じ位置・速度を加えても相対運動は同じです。
# 新しい系での重心は$\boldsymbol R_{\rm cm}(0)+\boldsymbol V_{\rm cm}(0)t$に従います。
# 毎ステップ重心を原点へ戻す補正はしません。補正すると誤差を隠してしまいます。

# %%
# --8<-- [start:boost-experiment]
r_boost = np.array([5.0, -3.0])
v_boost = np.array([30.0, -20.0]) * KM_S_TO_KPC_GYR
boost_initial = make_initial_state(masses, [d0, 0.0], [0.0, 0.8 * v_c], r_boost, v_boost)
t_boost, h_boost = integrate_leapfrog(acceleration_two_body, boost_initial, t_c, n_steps, masses)
bv = diagnostics(h_boost, masses)
base = diagnostics(cases["Elliptic"]["history"], masses)
boost_relative_error = np.max(np.linalg.norm(bv["dr"] - base["dr"], axis=1)) / d0
boost_errors = conservation_errors(t_boost, h_boost, masses)
# --8<-- [end:boost-experiment]
print(f"Relative orbit change under boost / D0 = {boost_relative_error:.6e}")
print(f"Center-of-mass straight-line error / D0 = {boost_errors[3]:.6e}")

# %% [markdown]
# ## 13. 応用：全質量を固定して質量比を変える
# $M_1+M_2$、$\boldsymbol D(0)$、$\boldsymbol U(0)$を固定し、$q=M_2/M_1$だけを変えます。
# 相対運動は同じですが、重心からの各天体の距離は変わります。
# $M_1$だけを固定して$M_2$を変える実験とは違います。

# %%
ratio_rows = []
for q in (1.0, 0.5, 0.1):
    changed_masses = np.array([mtot / (1.0 + q), mtot * q / (1.0 + q)])
    initial = make_initial_state(changed_masses, [d0, 0.0], [0.0, 0.8 * v_c])
    tq, hq = integrate_leapfrog(acceleration_two_body, initial, t_c, n_steps, changed_masses)
    drq = hq[:, 1, :2] - hq[:, 0, :2]
    difference = np.max(np.linalg.norm(drq - base["dr"], axis=1)) / d0
    ratio_rows.append([q, abs(initial[0, 0]), abs(initial[1, 0]), difference])
    print(f"q={q:.1f}; initial COM radii={abs(initial[0, 0]):.6f}, {abs(initial[1, 0]):.6f} kpc; relative change={difference:.3e}")

# %% [markdown]
# ## 14. 発展：双曲線と時間反転
# 本文の発展演習では$\eta=1.6>\sqrt{2}$も試します。新しい解法は不要です。
# 時間反転の確認は、数値誤差がないという証明ではなく、KDKの対称な更新が保たれているかの試験です。

# %%
### BEGIN SOLUTION
hyper_initial = make_initial_state(masses, [d0, 0.0], [0.0, 1.6 * v_c])
t_hyp, h_hyp = integrate_leapfrog(acceleration_two_body, hyper_initial, t_c, n_steps, masses)
hyper_values = diagnostics(h_hyp, masses)
reverse_initial = cases["Elliptic"]["history"][-1].copy()
reverse_initial[:, 2:] *= -1.0
_, h_reverse = integrate_leapfrog(acceleration_two_body, reverse_initial, t_c, n_steps, masses)
reverse_expected = cases["Elliptic"]["initial"].copy()
reverse_expected[:, 2:] *= -1.0
reverse_pos_error = np.max(np.linalg.norm(h_reverse[-1, :, :2] - reverse_expected[:, :2], axis=1)) / d0
reverse_vel_error = np.max(np.linalg.norm(h_reverse[-1, :, 2:] - reverse_expected[:, 2:], axis=1)) / v_c
### END SOLUTION
print(f"Hyperbolic final separation = {hyper_values['D'][-1]:.6f} kpc")
print(f"Time reversal errors: position/D0={reverse_pos_error:.3e}; velocity/vc={reverse_vel_error:.3e}")

# %% [markdown]
# ## 15. 数値データと確認結果を保存する
# 表と本文中の数値も、この実行結果から生成します。
# $P(0)=0$や$E(0)=0$を分母に使わず、物理的な基準量を指定してあります。
# 以下の判定値は本章の標準条件の回帰試験です。任意の軌道の必要精度を保証する閾値ではありません。

# %%
for name, case in cases.items():
    values = diagnostics(case["history"], masses)
    columns = np.column_stack((case["times"], case["history"].reshape(len(case["times"]), 8), values["D"], values["E"], values["L"], values["P"], values["Rcm"]))
    np.savetxt(DATADIR / f"orbit_{name.lower()}.csv", columns, delimiter=",", header="time_Gyr,x1,y1,vx1,vy1,x2,y2,vx2,vy2,D,E,Lz,Px,Py,Rcm_x,Rcm_y", comments="")
np.savetxt(DATADIR / "convergence.csv", convergence, delimiter=",", header="n_steps,dt_circle_Gyr,dt_ellipse_Gyr,circle_max_position_error,ellipse_return_error,ellipse_energy_error", comments="")
np.savetxt(DATADIR / "incoming_encounter.csv", np.column_stack((t_enc, h_enc.reshape(len(t_enc), 8), enc_values["D"])), delimiter=",", header="time_Gyr,x1,y1,vx1,vy1,x2,y2,vx2,vy2,D", comments="")
np.savetxt(DATADIR / "mass_ratio.csv", np.array(ratio_rows), delimiter=",", header="q,R1_initial_kpc,R2_initial_kpc,relative_orbit_change", comments="")
validation = {"version": "1.0", "masses": masses.tolist(), "d0_kpc": d0, "mu": mu, "m_red": m_red,
              "v_c_kms": v_c / KM_S_TO_KPC_GYR, "v_c_internal": v_c, "tc_Gyr": t_c, "n_steps": n_steps,
              "dt_Myr": dt * 1000, "force_calls": n_steps + 1, "base_errors": {key: val.tolist() for key, val in base_errors.items()},
              "ellipse": {"e": e_ell, "a_kpc": a_ell, "rp_kpc": r_peri_ell, "period_Gyr": t_ell, "sampled_rmin_kpc": float(values_ell["D"][i_min_ell])},
              "convergence": convergence.tolist(), "orders_circle": orders_c.tolist(), "orders_ellipse": orders_ell.tolist(),
              "encounter": {"rmin_kpc": float(enc_values["D"][i_peri]), "sampled_time_Gyr": float(t_enc[i_peri]),
                            "initial_relative_velocity_kms": ((encounter_initial[1, 2:] - encounter_initial[0, 2:]) / KM_S_TO_KPC_GYR).tolist(),
                            "errors": conservation_errors(t_enc, h_enc, masses).tolist()},
              "boost_relative_error": boost_relative_error, "boost_errors": boost_errors.tolist(), "mass_ratio": ratio_rows,
              "time_reversal": [reverse_pos_error, reverse_vel_error], "hyperbolic_final_D": float(hyper_values["D"][-1])}
(DATADIR / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
assert base_errors["Elliptic"][0] < 1.0e-3
assert max(values[1] for values in base_errors.values()) < 1.0e-11
assert max(values[3] for values in base_errors.values()) < 1.0e-11
assert 1.9 < orders_c[-1] < 2.1 and 1.9 < orders_ell[-1] < 2.1
assert abs(enc_values["D"][i_peri] - 8.0) < 0.03
assert boost_relative_error < 1.0e-9
assert reverse_pos_error < 1.0e-9
print("Standard-model checks passed. Data directory:", DATADIR)

# %% [markdown]
# ## 16. アニメーション
# 計算済みの楕円軌道を表示します。再生速度は物理的な時間刻みとは無関係です。
# 青丸は天体1、朱色四角は天体2、黒い星印は重心です。静止画は上の節で既に保存しています。

# %%
MAKE_ANIMATION = IN_NOTEBOOK
if MAKE_ANIMATION:
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display
    h = cases["Elliptic"]["history"]
    t = cases["Elliptic"]["times"]
    fig, ax = plt.subplots(figsize=(5.2, 4.8), layout="constrained")
    ax.plot(h[:, 0, 0], h[:, 0, 1], color=BLUE, alpha=0.25)
    ax.plot(h[:, 1, 0], h[:, 1, 1], color=VERMILLION, ls="--", alpha=0.25)
    point1, = ax.plot([], [], "o", color=BLUE, ms=7, label="Body 1")
    point2, = ax.plot([], [], "s", color=VERMILLION, ms=7, label="Body 2")
    ax.plot(0.0, 0.0, "k*", ms=11)
    stamp = ax.text(0.04, 0.94, "", transform=ax.transAxes)
    ax.set(xlabel="x [kpc]", ylabel="y [kpc]", xlim=(-9, 15), ylim=(-10, 10))
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    def update_frame(index):
        point1.set_data([h[index, 0, 0]], [h[index, 0, 1]])
        point2.set_data([h[index, 1, 0]], [h[index, 1, 1]])
        stamp.set_text(f"Time = {t[index]:.3f} Gyr")
        return point1, point2, stamp
    frame_ids = np.unique(np.linspace(0, n_steps, 80).astype(int))
    animation = FuncAnimation(fig, update_frame, frames=frame_ids, interval=70, blit=False)
    html = animation.to_jshtml(fps=15)
    plt.close(fig)
    (FIGDIR / "ch02_orbit_animation.html").write_text(html, encoding="utf-8")
    display(HTML(html))
else:
    print("Animation is available in the completed Notebook.")

# %% [markdown]
# ## まとめと演習
# 本文の演習2.1–2.7と対応させて、初期値、軌道形状、保存量、刻み依存性を確認してください。
# 本章の2天体は有限の質量を持ちますが、有限サイズを持つPlummer球ではありません。
# 第3章で球対称の広がった重力場を学び、第5章で2つの移動するポテンシャルへ進みます。
#
# **参考：** Barbara Ryden, *Celestial and Stellar Dynamics* (2025), §1.2–1.3。
# J. Binney and S. Tremaine, *Galactic Dynamics*, 2nd ed. (2008), §3.4.1。
# KDKの更新とデータ配置は、本教材の第1章（改訂v2.0）から引き継いでいます。
