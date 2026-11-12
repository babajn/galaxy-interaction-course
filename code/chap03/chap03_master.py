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
# # 第3章　銀河の中で星を動かす
# **完成版 / 全章改訂 v2.0**
#
# 固定した球対称のPlummerポテンシャルを使います。密度から質量を**台形則**で積分し、
# **二分法**で半質量半径を求め、その後に1個のテスト粒子を**KDK leapfrog**で動かします。
# 球の密度は3次元ですが、粒子の軌道は$x$–$y$平面内で計算します。
#
# 本文の長い導出はPDFを参照してください。密度・ポテンシャルはBinney & Tremaine (2008), §2.2に基づき、
# 本Notebookのパラメータ・比較実験・Python実装は本教材用に作成しています。
# 球のスケール長$a$は外縁ではありません。円盤の生成と補間は第4章です。
#
# **実行：** カーネルを再起動して上から順に実行します。基本依存はNumPyとMatplotlibです。
# 完成版は末尾でアニメーションも表示します。表示しない場合は`MAKE_ANIMATION = False`にします。

# %%
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for folder in (ROOT, *ROOT.parents):
    chapter_folder = folder / "code" / "chap03"
    if (chapter_folder / "chap03_master.py").is_file():
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
# ## 1. 単位とPlummerモデル
# 距離はkpc、時間はGyr、質量は$M_\odot$です。内部速度はkpc/Gyr、表示する速度だけkm/sにします。
# $M$は球全体の質量、$a$はPlummerスケール長です。第2章の軌道長半径$a_{\rm orb}$とは別の量です。
#
# $$\rho(r)=\frac{3M}{4\pi a^3}\left(1+\frac{r^2}{a^2}\right)^{-5/2},\qquad
# \Phi(r)=-\frac{GM}{\sqrt{r^2+a^2}}.$$
#
# 質量表の上限`r_tab`は**積分・描画の範囲**であり、密度をそこで切断する設定ではありません。

# %%
# --8<-- [start:parameters]
KM_S_TO_KPC_GYR = 1.0227121650537077
G = 4.30091e-6 * KM_S_TO_KPC_GYR**2  # kpc^3 / (Msun Gyr^2)
mass = 1.0e11                      # 球全体の質量 [Msun]
a = 3.0                            # Plummerスケール長 [kpc]
mu = G * mass
r_tab = 5.0 * a                     # 質量表の範囲 [kpc]。球の外縁ではありません。
n_mass = 256                       # 台形則の区間数。分点数はn_mass + 1です。
root_xtol = 1.0e-8                  # 二分法の半区間幅の許容値 [kpc]
r0 = 2.0 * a                       # 軌道の初期半径 [kpc]
# --8<-- [end:parameters]


def check_model(mass, a):
    if not np.isfinite(mass) or not np.isfinite(a) or mass <= 0.0 or a <= 0.0:
        raise ValueError("mass and Plummer scale must be finite and positive.")


def check_panels(n_panels):
    if isinstance(n_panels, (bool, np.bool_)) or not isinstance(n_panels, (int, np.integer)) or n_panels < 1:
        raise ValueError("n_panels must be a positive integer.")


# --8<-- [start:plummer-model]
def density_plummer(radius, mass, a):
    """3次元の質量密度 [Msun/kpc^3]。本文 eq:ch3-density。"""
    radius = np.asarray(radius, dtype=float)
    ### BEGIN SOLUTION
    return 3.0 * mass / (4.0 * np.pi * a**3) * (1.0 + (radius / a)**2)**(-2.5)
    ### END SOLUTION


def potential_plummer(radius, mass, a):
    """単位質量当たりのポテンシャル [kpc^2/Gyr^2]。"""
    return -G * mass / np.sqrt(np.asarray(radius)**2 + a**2)


def mass_plummer_exact(radius, mass, a):
    """半径radius以内の解析的な質量。本文 eq:ch3-mass-exact。"""
    radius = np.asarray(radius, dtype=float)
    return mass * radius**3 / (radius**2 + a**2)**1.5
# --8<-- [end:plummer-model]
check_model(mass, a)
print(f"Central density = {density_plummer(0.0, mass, a):.6e} Msun/kpc^3")
print(f"M(<a)/M = {mass_plummer_exact(a, mass, a) / mass:.9f}")

# %% [markdown]
# ## 2. 球殻の質量を積分する
# 球対称の密度では、薄い球殻の質量は$4\pi r^2\rho(r)\,dr$です。
# $$M(<r)=\int_0^r g(s)\,ds,\qquad g(s)=4\pi s^2\rho(s).$$
# 軌道を平面で計算しても、ここを$2\pi r$に替えてはいけません。
# 台形の面積$[g_j+g_{j+1}](r_{j+1}-r_j)/2$を順に加えます。
# 下の関数は、0から始まる累積積分を返します。原点以外から積分するときも使える一般の形です。

# %%
def check_grid(x, values):
    x, values = np.asarray(x, dtype=float), np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2 or values.shape != x.shape:
        raise ValueError("x and values must be equal-length 1D arrays with at least two nodes.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(values)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("Use finite values and strictly increasing nodes.")
    return x, values


# --8<-- [start:trapezoid]
def cumulative_trapezoid(x, values):
    """各分点までの累積積分。本文 eq:ch3-cumulative-trapezoid。"""
    x, values = check_grid(x, values)
    integral = np.zeros_like(x)
    ### BEGIN SOLUTION
    for j in range(len(x) - 1):
        area = 0.5 * (values[j] + values[j + 1]) * (x[j + 1] - x[j])
        integral[j + 1] = integral[j] + area
    ### END SOLUTION
    return integral
# --8<-- [end:trapezoid]

# --8<-- [start:mass-table]
r_grid = np.linspace(0.0, r_tab, n_mass + 1)
rho_grid = density_plummer(r_grid, mass, a)
grid_shell = 4.0 * np.pi * r_grid**2 * rho_grid
mass_numeric = cumulative_trapezoid(r_grid, grid_shell)
mass_exact = mass_plummer_exact(r_grid, mass, a)
# --8<-- [end:mass-table]
# 同じ和を配列のスライスで書いて、ループ版との一致も確かめます。
mass_sliced = np.zeros_like(r_grid)
mass_sliced[1:] = np.cumsum(0.5 * (grid_shell[:-1] + grid_shell[1:]) * np.diff(r_grid))
print(f"M(<r_tab)/M: trapezoid={mass_numeric[-1] / mass:.9f}, exact={mass_exact[-1] / mass:.9f}")
print(f"Maximum mass error / M = {np.max(np.abs(mass_numeric - mass_exact)) / mass:.6e}")
assert np.allclose(mass_numeric, mass_sliced, rtol=1.0e-14, atol=1.0e-14 * mass)

# %%
# 台形則で近似している関数はrhoそのものではなくgです。
x_coarse = np.linspace(0.0, 2.0, 9)
x_fine = np.linspace(0.0, 2.0, 401)
f_coarse = 3.0 * x_coarse**2 / (1.0 + x_coarse**2)**2.5
f_fine = 3.0 * x_fine**2 / (1.0 + x_fine**2)**2.5
fig, ax = plt.subplots(figsize=(5.8, 3.7), layout="constrained")
ax.plot(x_fine, f_fine, color=BLUE, label="Shell mass integrand")
ax.plot(x_coarse, f_coarse, "o--", color=VERMILLION, label="Trapezoidal approximation")
ax.fill_between(x_coarse, f_coarse, color=VERMILLION, alpha=0.12)
ax.vlines(x_coarse, 0.0, f_coarse, color=GRAY, lw=0.6)
ax.set(xlabel="r / a", ylabel="Shell mass integrand / (M/a)")
ax.legend(fontsize=9)
save_figure(fig, "ch03_trapezoid")

fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
ax.plot(r_grid / a, mass_exact / mass, color=GRAY, ls="--", label="Analytic mass")
ax.plot(r_grid / a, mass_numeric / mass, color=BLUE, label=f"Trapezoid; {n_mass} intervals")
ax.axhline(1.0, color=GRAY, ls=":", label="Total mass at infinity")
ax.set(xlabel="r / a", ylabel="Enclosed mass / M", ylim=(0.0, 1.08))
ax.legend(fontsize=9)
save_figure(fig, "ch03_enclosed_mass")

# %% [markdown]
# ## 3. 分割数を変えて精度を調べる
# 比較範囲$0\leq r\leq5a$を固定します。各分点での解析解と比較し、
# $$e_M=\max_j\frac{|\widehat M(<r_j)-M(<r_j)|}{M}$$
# を測ります。$M(<0)=0$で割らないよう、分母は球全体の$M$です。
# 一方、$r_{\rm tab}$より外の質量は、分割数を増やしても表には入りません。
# 数値積分の誤差と、積分範囲の外にある質量を分けます。

# %%
# --8<-- [start:mass-convergence]
panel_counts = [20, 40, 80, 160, 320]
mass_rows = []
for count in panel_counts:
    r = np.linspace(0.0, r_tab, count + 1)
    shell = 4.0 * np.pi * r**2 * density_plummer(r, mass, a)
    calculated = cumulative_trapezoid(r, shell)
    expected = mass_plummer_exact(r, mass, a)
    error = np.max(np.abs(calculated - expected)) / mass
    mass_rows.append([count, r_tab / count, error])
mass_convergence = np.asarray(mass_rows)
mass_orders = np.log2(mass_convergence[:-1, 2] / mass_convergence[1:, 2])
# --8<-- [end:mass-convergence]
print("intervals    dr [kpc]     maximum mass error/M    order")
for j, row in enumerate(mass_convergence):
    order = "--" if j == 0 else f"{mass_orders[j - 1]:.4f}"
    print(f"{int(row[0]):9d}    {row[1]:.6f}     {row[2]:.7e}    {order}")
cutoff_rows = [[factor, mass_plummer_exact(factor * a, mass, a) / mass] for factor in [5.0, 10.0, 20.0]]
print("Finite-radius mass fractions [r/a, M(<r)/M]:", cutoff_rows)

fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
x = mass_convergence[:, 1] / a
y = mass_convergence[:, 2]
ax.loglog(x, y, "o-", color=BLUE, label="Measured mass error")
ax.loglog(x, y[-1] * (x / x[-1])**2, ":", color=GRAY, label="Reference slope 2")
ax.set(xlabel="Radial interval / a", ylabel="Maximum mass error / M")
ax.legend(fontsize=9)
save_figure(fig, "ch03_mass_convergence")

# %% [markdown]
# ## 4. 半質量半径と二分法
# 球全体の半質量半径は$M(<r_h)=M/2$で定義します。計算する関数は
# $$F(r)=\widehat M(<r)/M-1/2.$$
# 線形補間を先取りしないため、試す半径ごとに$[0,r]$を同じ区間数で積分し直します。
# この処理は効率よりも、積分と求根の役割を区別するためのものです。
#
# 二分法は、連続な関数の符号が変わる区間を保ち、幅を半分にします。
# 戻り値は根の近似値と履歴で、履歴の列は`[k, lower, upper, midpoint, F(midpoint), half_width]`です。
# `xtol`は**半区間幅**の許容値で、関数値の許容値ではありません。

# %%
# --8<-- [start:mass-at-radius]
def enclosed_mass_trapezoid(radius, mass, a, n_panels):
    """[0, radius]をn_panels区間で積分します。"""
    check_panels(n_panels)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and nonnegative.")
    if radius == 0.0:
        return 0.0
    r = np.linspace(0.0, radius, n_panels + 1)
    shell = 4.0 * np.pi * r**2 * density_plummer(r, mass, a)
    return cumulative_trapezoid(r, shell)[-1]


def half_mass_residual(radius, n_panels=n_mass):
    return enclosed_mass_trapezoid(radius, mass, a, n_panels) / mass - 0.5
# --8<-- [end:mass-at-radius]


def finite_value(func, x):
    value = float(func(x))
    if not np.isfinite(value):
        raise ValueError("The root function must be finite on the bracket.")
    return value


def check_bisection(func, lower, upper, xtol, max_iter):
    if not np.all(np.isfinite([lower, upper, xtol])) or not lower < upper or xtol <= 0.0:
        raise ValueError("Require finite lower < upper and positive xtol.")
    check_panels(max_iter)
    f_lower, f_upper = finite_value(func, lower), finite_value(func, upper)
    if f_lower != 0.0 and f_upper != 0.0 and np.signbit(f_lower) == np.signbit(f_upper):
        raise ValueError("The endpoints must bracket a root (opposite signs).")
    return f_lower, f_upper


# --8<-- [start:bisection]
def bisection(func, lower, upper, xtol, max_iter=100):
    """半区間幅がxtol以下になるまで二分します。本文 eq:ch3-bisection。"""
    f_lower, f_upper = check_bisection(func, lower, upper, xtol, max_iter)
    if f_lower == 0.0 or f_upper == 0.0:
        endpoint = lower if f_lower == 0.0 else upper
        return endpoint, np.array([[0, endpoint, endpoint, endpoint, 0.0, 0.0]])
    trace = []
    for k in range(max_iter):
        half_width = 0.5 * (upper - lower)
        midpoint = lower + half_width
        f_mid = finite_value(func, midpoint)
        trace.append([k, lower, upper, midpoint, f_mid, half_width])
        if f_mid == 0.0 or half_width <= xtol:
            return midpoint, np.asarray(trace)
        if midpoint == lower or midpoint == upper:
            raise RuntimeError("The bracket cannot be reduced in floating-point arithmetic.")
        ### BEGIN SOLUTION
        if np.signbit(f_mid) == np.signbit(f_lower):
            lower, f_lower = midpoint, f_mid
        else:
            upper, f_upper = midpoint, f_mid
        ### END SOLUTION
    raise RuntimeError("Bisection did not reach xtol within max_iter.")
# --8<-- [end:bisection]

# --8<-- [start:half-mass]
r_half, root_trace = bisection(half_mass_residual, 0.0, 5.0 * a, root_xtol)
r_half_exact = a / np.sqrt(2.0**(2.0 / 3.0) - 1.0)
print(f"Numerical half-mass radius = {r_half:.9f} kpc")
print(f"Analytic half-mass radius  = {r_half_exact:.9f} kpc")
print(f"Residual of numerical F = {half_mass_residual(r_half):.6e}")
print(f"Residual using exact M = {mass_plummer_exact(r_half, mass, a) / mass - 0.5:.6e}")
# --8<-- [end:half-mass]
print(f"Bracket halvings = {int(root_trace[-1, 0])}; midpoint evaluations = {len(root_trace)}")
print(f"Final half width = {root_trace[-1, 5]:.6e} kpc")

# %% [markdown]
# ## 5. 求根の誤差と積分の誤差を分ける
# `xtol`を小さくすると**数値的な関数$F$の根**をよく求められます。
# しかし、$F$に使った質量積分の誤差は残ります。区間数を変える実験と、許容幅を変える実験を分けます。
# 初期区間$[0,5a]$と、対象が球全体の半質量であることは変えません。

# %%
# --8<-- [start:root-convergence]
root_rows = []
for count in [32, 64, 128, 256, 512]:
    def residual(radius):
        return half_mass_residual(radius, n_panels=count)
    root, trace = bisection(residual, 0.0, 5.0 * a, root_xtol)
    actual_error = abs(root - r_half_exact)
    root_rows.append([count, root, actual_error, abs(residual(root)), trace[-1, 5]])
root_convergence = np.asarray(root_rows)
root_orders = np.log2(root_convergence[:-1, 2] / root_convergence[1:, 2])
# --8<-- [end:root-convergence]
print("intervals    root [kpc]    actual error [kpc]    numerical residual")
for row in root_convergence:
    print(f"{int(row[0]):9d}    {row[1]:.9f}    {row[2]:.7e}    {row[3]:.3e}")

def residual_coarse(radius):
    return half_mass_residual(radius, n_panels=64)

tolerance_rows = []
for tol in [1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8]:
    root, trace = bisection(residual_coarse, 0.0, 5.0 * a, tol)
    tolerance_rows.append([tol, trace[-1, 5], abs(root - r_half_exact), abs(residual_coarse(root))])
_, coarse_trace = bisection(residual_coarse, 0.0, 5.0 * a, root_xtol)
print("Fixed 64 intervals; [xtol, bracket half width, actual error, residual]:")
print(np.asarray(tolerance_rows))

fig, ax = plt.subplots(figsize=(5.8, 3.9), layout="constrained")
ax.semilogy(coarse_trace[:, 0], coarse_trace[:, 5] / a, "o-", color=BLUE, ms=3, label="Bracket half-width / a")
true_errors = np.abs(coarse_trace[:, 3] - r_half_exact) / a
mask = true_errors > 0.0
ax.semilogy(coarse_trace[mask, 0], true_errors[mask], "s--", color=VERMILLION, ms=3, label="Error vs. analytic radius / a")
ax.set(xlabel="Number of bracket halvings", ylabel="Dimensionless distance", title="Mass integral: 64 intervals")
ax.legend(fontsize=9)
save_figure(fig, "ch03_bisection_errors")

# %% [markdown]
# ## 6. 重力加速度と円運動速度
# 以下の軌道計算には**解析式のPlummer加速度**を使います。質量表の積分誤差を時間積分誤差に混ぜません。
# 球対称なので$\boldsymbol a=-GM(<r)\boldsymbol r/r^3$で、Plummerモデルでは
# $$\boldsymbol a=-\frac{GM\boldsymbol r}{(r^2+a^2)^{3/2}},\qquad
# v_c^2(r)=\frac{GMr^2}{(r^2+a^2)^{3/2}}.$$
# 原点で加速度は0です。$a$は分母の発散を隠す小さな定数ではなく、密度分布を決めるモデルの長さです。

# %%
# --8<-- [start:acceleration-scalar]
x, y = r0, 0.0
r2 = x**2 + y**2
ax = -G * mass * x / (r2 + a**2)**1.5
ay = -G * mass * y / (r2 + a**2)**1.5
print(f"ax = {ax:.6f}, ay = {ay:.6f} [kpc/Gyr^2]")
# --8<-- [end:acceleration-scalar]

# --8<-- [start:acceleration-plummer]
def acceleration_plummer(pos, mass, a):
    """原点に固定したPlummer球。1粒子の位置・加速度は(1, 2)です。"""
    x, y = pos[0]
    r2 = x**2 + y**2
    ### BEGIN SOLUTION
    ax = -G * mass * x / (r2 + a**2)**1.5
    ay = -G * mass * y / (r2 + a**2)**1.5
    ### END SOLUTION
    return np.array([[ax, ay]])


def circular_speed_plummer(radius, mass, a):
    radius = np.asarray(radius, dtype=float)
    return np.sqrt(G * mass * radius**2 / (radius**2 + a**2)**1.5)
# --8<-- [end:acceleration-plummer]

r_plot = np.linspace(0.0, 5.0 * a, 401)
point_mask = r_plot >= 0.5 * a
fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
ax.plot(r_plot / a, circular_speed_plummer(r_plot, mass, a) / KM_S_TO_KPC_GYR, color=BLUE, label="Plummer")
ax.plot(r_plot[point_mask] / a, np.sqrt(mu / r_plot[point_mask]) / KM_S_TO_KPC_GYR, color=GRAY, ls="--", label="Point mass; same total M")
ax.plot(r0 / a, circular_speed_plummer(r0, mass, a) / KM_S_TO_KPC_GYR, "o", color=VERMILLION, label="Initial radius")
ax.set(xlabel="r / a", ylabel="Circular speed [km/s]", ylim=(0.0, 560.0))
ax.legend(fontsize=9)
save_figure(fig, "ch03_circular_speed")

# %% [markdown]
# ## 7. 初期条件とKDK
# $\boldsymbol r(0)=(r_0,0)$、$\boldsymbol v(0)=(0,\eta v_c(r_0))$とします。
# $\eta=1$が円軌道、$\eta=0.7$は束縛された**非円軌道**です。一般にはKepler楕円ではありません。
# どちらも同じ基準円軌道周期$T_c$の5倍まで計算します。
# KDKの4行と加速度の再利用は前章と同じです。関数には質量とスケール長を渡します。

# %%
# --8<-- [start:orbit-parameters]
v_c = circular_speed_plummer(r0, mass, a)
t_c = 2.0 * np.pi * r0 / v_c
n_periods = 5
steps_per_period = 1000
n_steps = n_periods * steps_per_period
t_end = n_periods * t_c
dt = t_end / n_steps
eta_values = {"Circular": 1.0, "Noncircular": 0.7}
# --8<-- [end:orbit-parameters]
print(f"Circular speed = {v_c / KM_S_TO_KPC_GYR:.6f} km/s")
print(f"Tc = {t_c:.9f} Gyr; duration = {t_end:.9f} Gyr; dt = {1000 * dt:.6f} Myr")

# --8<-- [start:initial-state]
def make_initial_state(radius, eta, mass, a):
    check_model(mass, a)
    if not np.isfinite(radius) or radius <= 0.0 or not np.isfinite(eta):
        raise ValueError("Initial radius must be positive and eta finite.")
    ### BEGIN SOLUTION
    pos = np.array([[radius, 0.0]])
    vel = np.array([[0.0, eta * circular_speed_plummer(radius, mass, a)]])
    ### END SOLUTION
    return np.column_stack((pos, vel))
# --8<-- [end:initial-state]


def prepare_history(state0, t_end, n_steps):
    check_panels(n_steps)
    if not np.isfinite(t_end) or t_end <= 0.0:
        raise ValueError("t_end must be finite and positive.")
    state = np.array(state0, dtype=float, copy=True)
    if state.shape != (1, 4) or not np.all(np.isfinite(state)):
        raise ValueError("Use a finite state array with shape (1, 4).")
    dt = t_end / n_steps
    times = np.arange(n_steps + 1) * dt
    history = np.empty((n_steps + 1, 1, 4))
    history[0] = state
    return state, times, history, dt


# --8<-- [start:leapfrog-step]
def leapfrog_step(acceleration, pos, vel, acc, dt, mass, a):
    """KDK。最後の加速度を次のステップへ渡します。"""
    vel_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_half
    acc_new = acceleration(pos_new, mass, a)
    vel_new = vel_half + 0.5 * dt * acc_new
    return pos_new, vel_new, acc_new
# --8<-- [end:leapfrog-step]

# --8<-- [start:integrate-leapfrog]
def integrate_leapfrog(acceleration, state0, t_end, n_steps, mass, a):
    check_model(mass, a)
    state, times, history, dt = prepare_history(state0, t_end, n_steps)
    pos, vel = state[:, :2].copy(), state[:, 2:].copy()
    acc = acceleration(pos, mass, a)
    for n in range(n_steps):
        pos, vel, acc = leapfrog_step(acceleration, pos, vel, acc, dt, mass, a)
        history[n + 1] = np.column_stack((pos, vel))
        if not np.all(np.isfinite(history[n + 1])):
            raise FloatingPointError("Non-finite state; check the model and time step.")
    return times, history
# --8<-- [end:integrate-leapfrog]

# %%
# --8<-- [start:run-orbits]
cases = {}
start = perf_counter()
for name, eta in eta_values.items():
    initial = make_initial_state(r0, eta, mass, a)
    times, history = integrate_leapfrog(acceleration_plummer, initial, t_end, n_steps, mass, a)
    cases[name] = {"eta": eta, "initial": initial, "times": times, "history": history}
# --8<-- [end:run-orbits]
print(f"Two orbit integrations: {perf_counter() - start:.3f} s (this environment)")
print(f"Force calls per integration: {n_steps + 1}")

fig, ax = plt.subplots(figsize=(5.6, 5.0), layout="constrained")
for name, color, linestyle in [("Circular", BLUE, "-"), ("Noncircular", VERMILLION, "--")]:
    h = cases[name]["history"]
    ax.plot(h[:, 0, 0], h[:, 0, 1], color=color, ls=linestyle, lw=1.1, label=name)
ax.plot([r0], [0.0], "o", color=GREEN, ms=6, label="Common initial position")
ax.plot([0.0], [0.0], "k+", ms=10, label="Plummer center")
ax.set(xlabel="x [kpc]", ylabel="y [kpc]", title="Static Plummer potential; duration = 5 Tc")
ax.set_aspect("equal")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
save_figure(fig, "ch03_orbits")

# %% [markdown]
# ## 8. 保存量を確認する
# 今回は固定した背景場中の**1粒子の比エネルギー**と**比角運動量**です。
# $$\varepsilon=\frac{v^2}{2}+\Phi(r),\qquad \ell_z=xv_y-yv_x.$$
# 球の自己重力エネルギーや、第2章の2体系の全エネルギーとは別です。
# 初期値はこの2例で非零なので、各初期値の絶対値で規格化します。

# %%
# --8<-- [start:diagnostics]
def diagnostics(history, mass, a):
    pos, vel = history[:, 0, :2], history[:, 0, 2:]
    radius = np.sqrt(np.sum(pos**2, axis=1))
    ### BEGIN SOLUTION
    energy = 0.5 * np.sum(vel**2, axis=1) + potential_plummer(radius, mass, a)
    angular = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    ### END SOLUTION
    return radius, energy, angular
# --8<-- [end:diagnostics]

orbit_errors = {}
print("case           max |dE|/|E0|     max |dL|/|L0|     min sampled r [kpc]")
for name, case in cases.items():
    radius, energy, angular = diagnostics(case["history"], mass, a)
    e_error = np.max(np.abs(energy - energy[0])) / abs(energy[0])
    l_error = np.max(np.abs(angular - angular[0])) / abs(angular[0])
    orbit_errors[name] = [e_error, l_error, float(np.min(radius)), float(np.max(radius))]
    print(f"{name:12s}    {e_error:.7e}      {l_error:.7e}    {np.min(radius):.7f}")

fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
for name, color, linestyle in [("Circular", BLUE, "-"), ("Noncircular", VERMILLION, "--")]:
    _, energy, _ = diagnostics(cases[name]["history"], mass, a)
    ax.plot(cases[name]["times"] / t_c, (energy - energy[0]) / abs(energy[0]), color=color, ls=linestyle, label=name)
ax.set(xlabel="Time / Tc", ylabel="[Energy(t) - Energy(0)] / |Energy(0)|")
ax.legend(fontsize=9)
save_figure(fig, "ch03_energy")

# %% [markdown]
# ## 9. 時間刻みを変える
# 円軌道なら、$\Omega=v_c/r_0$として同じ時刻の正解は$r_0(\cos\Omega t,\sin\Omega t)$です。
# 初期条件と最終時刻$T_c$を固定し、解析位置との最大の差を$r_0$で割ります。
# 半径誤差やエネルギー誤差だけでは、軌道上の位置のずれを見落とすことがあります。

# %%
# --8<-- [start:orbit-convergence]
def exact_circle(times, radius, mass, a):
    omega = circular_speed_plummer(radius, mass, a) / radius
    phase = omega * np.asarray(times)
    return np.column_stack((radius * np.cos(phase), radius * np.sin(phase)))


orbit_rows = []
initial_circle = cases["Circular"]["initial"]
for count in [100, 200, 400, 800]:
    t, h = integrate_leapfrog(acceleration_plummer, initial_circle, t_c, count, mass, a)
    exact = exact_circle(t, r0, mass, a)
    position_error = np.max(np.linalg.norm(h[:, 0, :2] - exact, axis=1)) / r0
    orbit_rows.append([count, t_c / count, position_error])
orbit_convergence = np.asarray(orbit_rows)
orbit_orders = np.log2(orbit_convergence[:-1, 2] / orbit_convergence[1:, 2])
# --8<-- [end:orbit-convergence]
print("steps       dt [Myr]        maximum position error/r0   order")
for j, row in enumerate(orbit_convergence):
    order = "--" if j == 0 else f"{orbit_orders[j - 1]:.5f}"
    print(f"{int(row[0]):5d}       {row[1] * 1000:.6f}       {row[2]:.7e}     {order}")

fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
x, y = orbit_convergence[:, 1] / t_c, orbit_convergence[:, 2]
ax.loglog(x, y, "o-", color=BLUE, label="KDK; circular orbit")
ax.loglog(x, y[-1] * (x / x[-1])**2, ":", color=GRAY, label="Reference slope 2")
ax.set(xlabel="Time step / Tc", ylabel="Maximum position error / initial radius")
ax.legend(fontsize=9)
save_figure(fig, "ch03_orbit_convergence")

# %% [markdown]
# ## 10. 非円軌道でも刻みを変えて確かめる
# 非円軌道は一般に閉じません。5$T_c$後に初期位置へ戻らないこと自体を数値誤差とみなしません。
# 同じ時刻の終点を$\Delta t,\Delta t/2,\Delta t/4$で比べます。真の位置との誤差ではなく、
# 数値解どうしの差です。2次の主要誤差が支配するなら、隣接する差の比は約4になります。

# %%
nonc_counts = [1250, 2500, 5000, 10000]
nonc_ends = []
for count in nonc_counts:
    _, h = integrate_leapfrog(acceleration_plummer, cases["Noncircular"]["initial"], t_end, count, mass, a)
    nonc_ends.append(h[-1, 0, :2].copy())
nonc_differences = np.linalg.norm(np.diff(np.asarray(nonc_ends), axis=0), axis=1) / r0
nonc_ratios = nonc_differences[:-1] / nonc_differences[1:]
print("Noncircular endpoint differences / r0:", nonc_differences)
print("Ratios of successive differences:", nonc_ratios)

# %% [markdown]
# ### 演習3.3・3.5：有限範囲と半質量の定義
# 半径の幅を固定して範囲を広げる実験では、区間数も増やします。
# 有限範囲内の半分を求める場合には、何の半分かを関数の右辺で明示します。

# %%
### BEGIN SOLUTION
fixed_dr = a / 32.0
range_rows = []
for factor in [5.0, 10.0, 20.0]:
    outer = factor * a
    count = int(round(outer / fixed_dr))
    value = enclosed_mass_trapezoid(outer, mass, a, count)
    exact_value = mass_plummer_exact(outer, mass, a)
    range_rows.append([factor, count, value / mass, abs(value - exact_value) / mass, 1.0 - exact_value / mass])
cut_mass_numeric = enclosed_mass_trapezoid(5.0 * a, mass, a, n_mass)
cut_fraction_exact = mass_plummer_exact(5.0 * a, mass, a) / (2.0 * mass)
cut_half_exact = a / np.sqrt(cut_fraction_exact**(-2.0 / 3.0) - 1.0)

def cut_half_residual(radius):
    return enclosed_mass_trapezoid(radius, mass, a, n_mass) / mass - cut_mass_numeric / (2.0 * mass)

cut_half, _ = bisection(cut_half_residual, 0.0, 5.0 * a, root_xtol)
### END SOLUTION
print("Fixed dr; [r_tab/a, intervals, M_num/M, quadrature error/M, outside mass/M]:")
print(np.asarray(range_rows))
print(f"Half of M(<5a): numerical radius = {cut_half:.9f} kpc; exact = {cut_half_exact:.9f} kpc")
print("This radius is different from the half-mass radius of the full Plummer sphere.")

# %% [markdown]
# ## 11. 演習：モデルの長さと非円軌道の近点
# **応用：**$M$と初期半径$r_0$を固定し、$a$を変えて円運動速度と半質量半径を調べます。
# 円を計算する場合は、新しい$a$に対応した速度を設定し直します。
# **発展：**動径速度が0になる半径は$\Phi(r)+\ell_z^2/(2r^2)-\varepsilon=0$を満たします。
# 初期遠点$r_0$とは異なる根を二分法で探し、保存された最小半径と比較します。
# これは新しい求根算法ではなく、学習済みの二分法の応用です。

# %%
### BEGIN SOLUTION
scale_rows = []
scale_orbit_rows = []
for scale in [1.5, 3.0, 6.0]:
    vc_scale = circular_speed_plummer(r0, mass, scale)
    rh_scale = scale / np.sqrt(2.0**(2.0 / 3.0) - 1.0)
    scale_rows.append([scale, vc_scale / KM_S_TO_KPC_GYR, rh_scale])
    period_scale = 2.0 * np.pi * r0 / vc_scale
    state_scale = make_initial_state(r0, 1.0, mass, scale)
    times_scale, history_scale = integrate_leapfrog(acceleration_plummer, state_scale, period_scale, 1000, mass, scale)
    exact_scale = exact_circle(times_scale, r0, mass, scale)
    error_scale = np.max(np.linalg.norm(history_scale[:, 0, :2] - exact_scale, axis=1)) / r0
    scale_orbit_rows.append([scale, period_scale, error_scale])
initial = cases["Noncircular"]["initial"]
_, eps_initial, ell_initial = diagnostics(initial[None, :, :], mass, a)
eps0, ell0 = eps_initial[0], ell_initial[0]

def turning_residual(radius):
    return potential_plummer(radius, mass, a) + ell0**2 / (2.0 * radius**2) - eps0

r_peri, peri_trace = bisection(turning_residual, 0.05 * r0, 0.95 * r0, root_xtol)
### END SOLUTION
print("Scale variation [a kpc, vc km/s, rh kpc]:", scale_rows)
print("One-period circle for each scale [a kpc, period Gyr, max position error/r0]:", scale_orbit_rows)
print(f"Pericenter from invariants = {r_peri:.9f} kpc")
print(f"Smallest saved radius = {orbit_errors['Noncircular'][2]:.9f} kpc")
# ここでの差には、時間積分誤差と保存時刻が近点通過時刻に一致しない効果が含まれます。

# %% [markdown]
# ## 12. データと確認結果を保存する
# 本文の図・表・具体的な数値は以下の実行結果から生成します。
# 判定値は標準条件での回帰試験であり、任意のモデルの精度保証ではありません。

# %%
np.savetxt(DATADIR / "mass_profile.csv", np.column_stack((r_grid, rho_grid, mass_numeric, mass_exact)), delimiter=",", header="r_kpc,rho_Msun_kpc3,mass_numeric_Msun,mass_exact_Msun", comments="")
np.savetxt(DATADIR / "mass_convergence.csv", mass_convergence, delimiter=",", header="intervals,dr_kpc,max_mass_error_over_M", comments="")
np.savetxt(DATADIR / "root_trace.csv", root_trace, delimiter=",", header="halvings,lower,upper,midpoint,F_midpoint,half_width", comments="")
np.savetxt(DATADIR / "root_convergence.csv", root_convergence, delimiter=",", header="intervals,root_kpc,actual_error_kpc,numerical_residual,half_width_kpc", comments="")
np.savetxt(DATADIR / "root_tolerances.csv", np.asarray(tolerance_rows), delimiter=",", header="xtol_kpc,half_width_kpc,actual_error_kpc,numerical_residual", comments="")
np.savetxt(DATADIR / "orbit_convergence.csv", orbit_convergence, delimiter=",", header="steps,dt_Gyr,position_error_over_r0", comments="")
np.savetxt(DATADIR / "exercise_range.csv", np.asarray(range_rows), delimiter=",", header="r_tab_over_a,intervals,M_num_over_M,quadrature_error_over_M,outside_mass_over_M", comments="")
np.savetxt(DATADIR / "exercise_scale.csv", np.asarray(scale_orbit_rows), delimiter=",", header="a_kpc,period_Gyr,max_position_error_over_r0", comments="")
for name, case in cases.items():
    radius, energy, angular = diagnostics(case["history"], mass, a)
    columns = np.column_stack((case["times"], case["history"][:, 0], radius, energy, angular))
    np.savetxt(DATADIR / f"orbit_{name.lower()}.csv", columns, delimiter=",", header="t_Gyr,x_kpc,y_kpc,vx_kpc_Gyr,vy_kpc_Gyr,r_kpc,energy_kpc2_Gyr2,ell_kpc2_Gyr", comments="")
validation = {"version": "1.0", "mass_Msun": mass, "a_kpc": a, "r0_kpc": r0, "r_tab_kpc": r_tab,
              "n_mass": n_mass, "xtol_kpc": root_xtol, "density_central": float(density_plummer(0.0, mass, a)),
              "mass_error": float(np.max(np.abs(mass_numeric - mass_exact)) / mass),
              "mass_convergence": mass_convergence.tolist(), "mass_orders": mass_orders.tolist(), "cutoffs": cutoff_rows,
              "r_half": r_half, "r_half_exact": r_half_exact, "root_error": abs(r_half - r_half_exact),
              "root_residual": half_mass_residual(r_half), "true_mass_residual": float(mass_plummer_exact(r_half, mass, a) / mass - 0.5),
              "root_trace": root_trace.tolist(), "root_convergence": root_convergence.tolist(), "root_orders": root_orders.tolist(),
              "tolerances": tolerance_rows, "vc_kms": float(v_c / KM_S_TO_KPC_GYR), "vc_internal": float(v_c),
              "tc_Gyr": float(t_c), "t_end_Gyr": float(t_end), "dt_Myr": float(dt * 1000), "n_steps": n_steps,
              "orbit_errors": orbit_errors, "orbit_convergence": orbit_convergence.tolist(), "orbit_orders": orbit_orders.tolist(),
              "nonc_counts": nonc_counts, "nonc_differences": nonc_differences.tolist(), "nonc_ratios": nonc_ratios.tolist(),
              "scale_rows": scale_rows, "r_peri": r_peri, "force_calls": n_steps + 1,
              "range_rows": range_rows, "cut_half": cut_half, "cut_half_exact": float(cut_half_exact), "scale_orbit_rows": scale_orbit_rows}
(DATADIR / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
assert 1.9 < mass_orders[-1] < 2.1
assert 1.9 < orbit_orders[-1] < 2.1
assert abs(r_half - r_half_exact) < 1.0e-3 * a
assert max(value[1] for value in orbit_errors.values()) < 1.0e-10
assert orbit_errors["Noncircular"][0] < 1.0e-3
assert np.all((nonc_ratios > 3.7) & (nonc_ratios < 4.3))
print("Standard-model checks passed. Data directory:", DATADIR)

# %% [markdown]
# ## 13. アニメーション
# 円軌道と非円軌道を、同じ経過時間で表示します。中心の十字はポテンシャルの中心で、
# 追加した質点ではありません。静止図は上の節で保存しています。

# %%
MAKE_ANIMATION = IN_NOTEBOOK
if MAKE_ANIMATION:
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display
    fig, ax = plt.subplots(figsize=(5.0, 4.6), layout="constrained")
    trails, points = [], []
    for name, color, marker in [("Circular", BLUE, "o"), ("Noncircular", VERMILLION, "s")]:
        trail, = ax.plot([], [], color=color, lw=1.0, alpha=0.65)
        point, = ax.plot([], [], marker, color=color, ms=6, label=name)
        trails.append(trail)
        points.append(point)
    ax.plot([0.0], [0.0], "k+", ms=10)
    stamp = ax.text(0.04, 0.95, "", transform=ax.transAxes)
    ax.set(xlabel="x [kpc]", ylabel="y [kpc]", xlim=(-7.3, 7.3), ylim=(-7.3, 7.3))
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    def update_frame(index):
        for j, name in enumerate(["Circular", "Noncircular"]):
            h = cases[name]["history"]
            trails[j].set_data(h[:index + 1, 0, 0], h[:index + 1, 0, 1])
            points[j].set_data([h[index, 0, 0]], [h[index, 0, 1]])
        stamp.set_text(f"Time = {cases['Circular']['times'][index] / t_c:.2f} Tc")
        return (*trails, *points, stamp)
    frame_ids = np.unique(np.linspace(0, n_steps, 80).astype(int))
    animation = FuncAnimation(fig, update_frame, frames=frame_ids, interval=80, blit=False)
    html = animation.to_jshtml(fps=12)
    plt.close(fig)
    (FIGDIR / "ch03_orbit_animation.html").write_text(html, encoding="utf-8")
    display(HTML(html))
else:
    print("Animation is available in the completed Notebook.")

# %% [markdown]
# ## まとめと演習
# 本文の演習3.1–3.7を参照してください。台形則は密度を質量へ、二分法は質量から半径へ、
# KDKは力から軌道へ進むために使いました。小さな求根残差だけで、真の半質量半径が同じ精度とは限りません。
# また、閉じない非円軌道を、Kepler楕円からの数値誤差とみなさないことが重要です。
# 第4章では指数円盤を作るために累積分布、補間、乱数を組み合わせます。
#
# **参考：** Binney & Tremaine, *Galactic Dynamics*, 2nd ed. (2008), §2.2・§3.1・§3.4.1。
# 杉浦 洋『数値計算の基礎と応用［新訂版］』§3.2–3.3。
# 杉浦洋『数値計算の基礎と応用［新訂版］』第9章の2分法を参照。
