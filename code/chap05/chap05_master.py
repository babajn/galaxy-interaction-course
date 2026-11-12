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
# # 第5章　2つの銀河を接近させる
# **完成版 / 全章改訂 v2.0**
#
# 第2章の重心系、第3章のPlummer重力、第4章のテスト粒子円盤を組み合わせます。
# 時間積分には**加速度を再利用するKDK leapfrog**を使います。新しい数値算法は導入しません。
# 二つの銀河中心は有効な対ポテンシャルで動き、星は移動する二つのPlummer場から重力を受けます。
# 星同士の重力、星から中心への反作用、ガス、動力学的摩擦は含みません。
#
# **実行：** カーネルを再起動して上から実行してください。必要なのはNumPyとMatplotlibです。
# 最後の動画表示のみIPythonを使います。本文PDFが理論の正本、このNotebookが実装・実験の完成版です。
# 本章の具体的なパラメータ、診断量と図は、構成案に沿って設定・計算した教材用の例です。

# %%
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for folder in (ROOT, *ROOT.parents):
    chapter_folder = folder / "code" / "chap05"
    if (chapter_folder / "chap05_master.py").is_file():
        ROOT = chapter_folder
        break
FIGDIR, DATADIR = ROOT / "figures", ROOT / "data"
FIGDIR.mkdir(exist_ok=True)
DATADIR.mkdir(exist_ok=True)
IN_NOTEBOOK = "get_ipython" in globals()
MAKE_ANIMATION = IN_NOTEBOOK
BLUE, VERMILLION, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#777777"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})


def save_figure(fig, name):
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", dpi=170, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)

# %% [markdown]
# ## 1. 標準条件と配列
# $a_k$は各銀河のPlummerスケール長、$b$は中心間の有効対ポテンシャルの長さです。
# $b$を使う中心間力は、広がった二つのPlummer球の厳密な相互力ではありません。
# 標準例だけは$a_1=a_2=b=3\ \mathrm{kpc}$とします。$R_d$と$R_{\max}$は粒子の初期分布を決めます。
# `state[:2]`が銀河中心、`state[2:]`がテスト粒子です。各行は`[x, y, vx, vy]`です。
# 基本単位はkpc, Gyr, $M_\odot$で、内部速度はkpc/Gyrです。

# %%
# --8<-- [start:parameters]
KM_S_TO_KPC_GYR = 1.0227121650537077
G = 4.30091e-6 * KM_S_TO_KPC_GYR**2
masses = np.array([1.0e11, 1.0e11])  # Plummer球の全質量 [Msun]
scales = np.array([3.0, 3.0])        # a1, a2 [kpc]
b = 3.0                             # 中心間力の有効長さ [kpc]
rds = np.array([3.0, 3.0])
rmaxs = np.array([15.0, 15.0])
counts = np.array([1000, 1000])
spins = np.array([+1, +1])           # 今回は軌道も反時計回りなので両方prograde
seed, n_cdf = 42, 4096
d_start, d_peri = 60.0, 8.0         # 初期距離・目標近点距離 [kpc]
t_end, n_steps, save_every = 0.60, 12000, 40
dt = t_end / n_steps                 # 0.05 Myr。出力は2 Myrごとです。
# --8<-- [end:parameters]


def check_positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")


def check_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def check_model(masses, scales, b):
    masses, scales = np.asarray(masses, dtype=float), np.asarray(scales, dtype=float)
    for values in (masses, scales):
        if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Two positive finite masses and two positive Plummer scales are required.")
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("The effective center softening b must be finite and nonnegative.")
    return masses, scales


def check_grid(x, values):
    x, values = np.asarray(x, dtype=float), np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2 or values.shape != x.shape:
        raise ValueError("Nodes and values must be equal-length 1D arrays.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(values)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("Use finite values and strictly increasing nodes.")
    return x, values


check_model(masses, scales, b)

# %% [markdown]
# ## 2. 既習の円盤生成を再利用する
# $p_R(R)\propto R\exp(-R/R_d)$を有限範囲で規格化します。
# 第4章と同じく、**台形則 → CDF → 線形補間 → 乱数 → 円運動速度**の順です。
# このセルの関数は第4章の復習です。粒子には重力計算用の質量を割り当てません。

# %%
# --8<-- [start:disk-cdf]
def cumulative_trapezoid(x, values):
    x, values = check_grid(x, values)
    integral = np.zeros_like(x)
    for j in range(len(x) - 1):
        integral[j + 1] = integral[j] + 0.5 * (values[j] + values[j + 1]) * (x[j + 1] - x[j])
    return integral


def build_radial_cdf(rd, rmax, n_panels):
    check_positive(rd, "rd")
    check_positive(rmax, "rmax")
    check_integer(n_panels, "n_panels")
    nodes = np.linspace(0.0, rmax, n_panels + 1)
    integral = cumulative_trapezoid(nodes, nodes * np.exp(-nodes / rd))
    if not np.isfinite(integral[-1]) or integral[-1] <= 0.0:
        raise ValueError("The CDF normalization must be finite and positive.")
    cdf = integral / integral[-1]
    if np.any(np.diff(cdf) <= 0.0):
        raise ValueError("The inverse CDF needs strictly increasing values.")
    return nodes, cdf
# --8<-- [end:disk-cdf]


def circular_speed_plummer(radius, mass, a):
    radius = np.asarray(radius, dtype=float)
    return np.sqrt(G * mass * radius**2 / (radius**2 + a**2)**1.5)


# --8<-- [start:make-disk]
def make_disk(n_particles, mass, a, rd, rmax, spin, rng, n_panels=4096):
    check_integer(n_particles, "n_particles")
    check_positive(mass, "mass")
    check_positive(a, "a")
    if isinstance(spin, (bool, np.bool_)) or spin not in (-1, +1):
        raise ValueError("spin must be +1 or -1.")
    nodes, values = build_radial_cdf(rd, rmax, n_panels)
    uniform = rng.random((n_particles, 2))
    radius = np.interp(uniform[:, 0], values, nodes)
    phi = 2.0 * np.pi * uniform[:, 1]
    vc = circular_speed_plummer(radius, mass, a)
    pos = np.column_stack((radius * np.cos(phi), radius * np.sin(phi)))
    vel = spin * np.column_stack((-vc * np.sin(phi), vc * np.cos(phi)))
    return np.column_stack((pos, vel)), radius, phi
# --8<-- [end:make-disk]

# %% [markdown]
# ## 3. 移動するPlummer場と、銀河中心の相互力
# テスト粒子では中心からの相対位置$\boldsymbol r-\boldsymbol R_k$を使います。
# 銀河中心では$\boldsymbol D=\boldsymbol R_2-\boldsymbol R_1$から一組の力を作ります。
# $$\boldsymbol A_1=\frac{GM_2\boldsymbol D}{(D^2+b^2)^{3/2}},\qquad
# \boldsymbol A_2=-\frac{GM_1\boldsymbol D}{(D^2+b^2)^{3/2}}.$$
# 第2章と同じく、逆向きで等しいのは加速度でなく$M_1\boldsymbol A_1$と$M_2\boldsymbol A_2$です。

# %%
# --8<-- [start:moving-plummer]
def acceleration_plummer(pos, center, mass, a):
    """任意の中心を持つPlummer場。posは(N, 2)、centerは(2,)です。"""
    ### BEGIN SOLUTION
    relative = pos - center
    radius2 = np.sum(relative**2, axis=1)
    return -G * mass * relative / (radius2 + a**2)[:, None]**1.5
    ### END SOLUTION
# --8<-- [end:moving-plummer]


# --8<-- [start:center-acceleration]
def acceleration_centers(pos, masses, b):
    """共通のbを使う有効対ポテンシャルから、二つの加速度を返します。"""
    dr = pos[1] - pos[0]
    denom2 = np.dot(dr, dr) + b**2
    if not np.isfinite(denom2) or denom2 <= 0.0:
        raise ValueError("The effective force denominator must be positive and finite.")
    ### BEGIN SOLUTION
    common = G * dr / denom2**1.5
    return np.array([masses[1] * common, -masses[0] * common])
    ### END SOLUTION
# --8<-- [end:center-acceleration]


# --8<-- [start:system-acceleration]
def acceleration_system(pos, masses, scales, b):
    """先頭2行は銀河中心、それ以外は質量ゼロのテスト粒子です。"""
    acc = np.empty_like(pos)
    acc[:2] = acceleration_centers(pos[:2], masses, b)
    ### BEGIN SOLUTION
    acc[2:] = acceleration_plummer(pos[2:], pos[0], masses[0], scales[0])
    acc[2:] += acceleration_plummer(pos[2:], pos[1], masses[1], scales[1])
    ### END SOLUTION
    return acc
# --8<-- [end:system-acceleration]

# %% [markdown]
# ## 4. 有効ポテンシャルに合った接近軌道を作る
# $\mu=G(M_1+M_2)$、$\varepsilon=U^2/2-\mu/\sqrt{D^2+b^2}=0$とします。
# これは$b>0$では厳密なKepler放物線ではありません。
# $$h^2=\frac{2\mu D_p^2}{\sqrt{D_p^2+b^2}},\qquad
# U_t=\frac{h}{D_0},\quad U_r=-\sqrt{\frac{2\mu}{\sqrt{D_0^2+b^2}}-U_t^2}.$$
# 負の$U_r$は内向きです。第2章の重心変換を使って各銀河の初期値を決めます。

# %%
# --8<-- [start:encounter-initial]
def make_zero_energy_encounter(masses, d_start, d_peri, b):
    masses = np.asarray(masses, dtype=float)
    check_model(masses, np.ones(2), b)
    if not (np.isfinite(d_start) and np.isfinite(d_peri) and 0.0 < d_peri < d_start):
        raise ValueError("Require finite distances with 0 < d_peri < d_start.")
    mtot = np.sum(masses)
    mu = G * mtot
    h = np.sqrt(2.0 * mu * d_peri**2 / np.sqrt(d_peri**2 + b**2))
    ### BEGIN SOLUTION
    vt = h / d_start
    vr2 = 2.0 * mu / np.sqrt(d_start**2 + b**2) - vt**2
    if vr2 <= 0.0:
        raise ValueError("The incoming radial speed must be real and nonzero.")
    relative_velocity = np.array([-np.sqrt(vr2), vt])
    ### END SOLUTION
    separation = np.array([d_start, 0.0])
    weights = np.array([-masses[1], masses[0]]) / mtot
    pos = weights[:, None] * separation
    vel = weights[:, None] * relative_velocity
    return np.column_stack((pos, vel))
# --8<-- [end:encounter-initial]


centers0 = make_zero_energy_encounter(masses, d_start, d_peri, b)
u0 = centers0[1, 2:] - centers0[0, 2:]
print("Initial relative velocity [km/s]:", u0 / KM_S_TO_KPC_GYR)
print("Relative speed [km/s]:", np.linalg.norm(u0) / KM_S_TO_KPC_GYR)

# %% [markdown]
# ## 5. 円盤へ銀河中心の位置・速度を足す
# 局所円盤の$\boldsymbol s_i,\boldsymbol w_i$を
# $$\boldsymbol r_i=\boldsymbol R_k+\boldsymbol s_i,\qquad
# \boldsymbol v_i=\boldsymbol V_k+\boldsymbol w_i$$
# で慣性系へ移します。速度にも銀河中心の運動を足します。
# `origin`はテスト粒子の**出身銀河**です。後で隣の銀河へ近づいても変えません。
# 一つの乱数生成器から円盤1、円盤2の順に生成するので、二つは同じ点配置ではありません。

# %%
# --8<-- [start:place-disk]
def place_disk(local_state, center_state):
    """局所の位置と速度を慣性系へ移します。入力を上書きしません。"""
    placed = local_state.copy()
    ### BEGIN SOLUTION
    placed[:, :2] += center_state[:2]
    placed[:, 2:] += center_state[2:]
    ### END SOLUTION
    return placed
# --8<-- [end:place-disk]


# --8<-- [start:assemble-state]
def make_initial_state(masses, scales, rds, rmaxs, counts, spins,
                       d_start, d_peri, b, seed, n_cdf=4096):
    check_model(masses, scales, b)
    for values in (rds, rmaxs, counts, spins):
        if np.shape(values) != (2,):
            raise ValueError("Each disk parameter array must have two entries.")
    centers = make_zero_energy_encounter(masses, d_start, d_peri, b)
    rng = np.random.default_rng(seed)
    disks, radii = [], []
    for k in range(2):
        local, radius, _ = make_disk(counts[k], masses[k], scales[k], rds[k], rmaxs[k],
                                     spins[k], rng, n_cdf)
        disks.append(place_disk(local, centers[k]))
        radii.append(radius)
    state0 = np.vstack((centers, disks[0], disks[1]))
    origin = np.repeat(np.arange(2), counts)
    return state0, origin, np.concatenate(radii)
# --8<-- [end:assemble-state]


state0, origin, initial_radius = make_initial_state(masses, scales, rds, rmaxs, counts, spins, d_start, d_peri, b, seed, n_cdf)
np.savez_compressed(DATADIR / "interaction_initial.npz", state=state0, origin=origin, initial_radius=initial_radius)
print(f"state0.shape = {state0.shape}; origin.shape = {origin.shape}")
print("Disk population:", np.bincount(origin))
print("Initial center of mass [kpc]:", np.sum(masses[:, None] * state0[:2, :2], axis=0) / np.sum(masses))

# %% [markdown]
# ## 6. KDKで全行を同時に更新する
# 全中心・全粒子の半kick → 全行のdrift → **新しい中心位置も使って**重力を再計算 → 全行の半kick、です。
# キャッシュした加速度は次のステップへ渡します。中心だけを先に更新し、星には古い中心を使う処理にしません。
# 中心の履歴は毎ステップ、全粒子の履歴は`save_every`ステップごとに保存します。
# どちらも速度は最後の半kickを終えた整数時刻の値です。

# %%
def prepare_history(state0, t_end, n_steps, save_every):
    check_positive(t_end, "t_end")
    check_integer(n_steps, "n_steps")
    check_integer(save_every, "save_every")
    state = np.array(state0, dtype=float, copy=True)
    if state.ndim != 2 or state.shape[1] != 4 or len(state) < 2 or not np.all(np.isfinite(state)):
        raise ValueError("state0 must be finite, have shape (N+2, 4), and include two centers.")
    saved_steps = np.arange(0, n_steps + 1, save_every)
    if saved_steps[-1] != n_steps:
        saved_steps = np.append(saved_steps, n_steps)
    dt = t_end / n_steps
    times = saved_steps * dt
    history = np.empty((len(saved_steps), len(state), 4))
    history[0] = state
    return state, saved_steps, times, history, dt


# --8<-- [start:leapfrog-step]
def leapfrog_step(acceleration, pos, vel, acc, dt, masses, scales, b):
    vel_half = vel + 0.5 * dt * acc
    pos_new = pos + dt * vel_half
    acc_new = acceleration(pos_new, masses, scales, b)
    vel_new = vel_half + 0.5 * dt * acc_new
    return pos_new, vel_new, acc_new
# --8<-- [end:leapfrog-step]


# --8<-- [start:integrate-leapfrog]
def integrate_leapfrog(acceleration, state0, t_end, n_steps, masses, scales, b, save_every=10):
    masses, scales = check_model(masses, scales, b)
    state, saved_steps, times, history, dt = prepare_history(state0, t_end, n_steps, save_every)
    pos, vel = state[:, :2].copy(), state[:, 2:].copy()
    acc = acceleration(pos, masses, scales, b)
    center_times = np.arange(n_steps + 1) * dt
    center_history = np.empty((n_steps + 1, 2, 4))
    center_history[0] = state[:2]
    output_index = 1
    for n in range(n_steps):
        pos, vel, acc = leapfrog_step(acceleration, pos, vel, acc, dt, masses, scales, b)
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            raise FloatingPointError("Non-finite state; check the model and time step.")
        center_history[n + 1] = np.column_stack((pos[:2], vel[:2]))
        if n + 1 == saved_steps[output_index]:
            history[output_index] = np.column_stack((pos, vel))
            output_index += 1
    return times, history, center_times, center_history
# --8<-- [end:integrate-leapfrog]


start = perf_counter()
times, history, center_times, center_history = integrate_leapfrog(acceleration_system, state0, t_end, n_steps, masses, scales, b, save_every)
elapsed = perf_counter() - start
print(f"Integration: {elapsed:.3f} s (this environment only)")
print(f"Force evaluations = {n_steps + 1}; time step = {dt*1000:.3f} Myr")
print(f"Full history: {history.shape}; {history.nbytes/1e6:.6f} MB")
print(f"Center history: {center_history.shape}; {center_history.nbytes/1e6:.6f} MB")

# %% [markdown]
# ## 7. 銀河中心の保存量と近点を確認する
# 中心間の対ポテンシャルに対応する$E_{\rm c}=\sum_k M_k V_k^2/2-GM_1M_2/\sqrt{D^2+b^2}$を使います。
# $\varepsilon(0)=0$なので初期エネルギーでは割りません。$E_*=GM_1M_2/\sqrt{D_0^2+b^2}$で規格化します。
# 中心の毎ステップの履歴で極値を調べます。それでも連続時間の厳密な近点そのものではありません。

# %%
# --8<-- [start:center-diagnostics]
def center_diagnostics(center_history, masses, b):
    pos, vel = center_history[:, :, :2], center_history[:, :, 2:]
    dr, dv = pos[:, 1] - pos[:, 0], vel[:, 1] - vel[:, 0]
    separation = np.linalg.norm(dr, axis=1)
    kinetic = 0.5 * np.sum(masses[None, :] * np.sum(vel**2, axis=2), axis=1)
    ### BEGIN SOLUTION
    potential = -G * masses[0] * masses[1] / np.sqrt(separation**2 + b**2)
    momentum = np.sum(masses[None, :, None] * vel, axis=1)
    r_cm = np.sum(masses[None, :, None] * pos, axis=1) / np.sum(masses)
    ### END SOLUTION
    cross = pos[:, :, 0] * vel[:, :, 1] - pos[:, :, 1] * vel[:, :, 0]
    angular = np.sum(masses[None, :] * cross, axis=1)
    return {"D": separation, "dr": dr, "dv": dv, "E": kinetic + potential,
            "L": angular, "P": momentum, "Rcm": r_cm}
# --8<-- [end:center-diagnostics]


def center_errors(times, center_history, masses, b):
    d = center_diagnostics(center_history, masses, b)
    d0 = d["D"][0]
    vscale = np.sqrt(G * np.sum(masses) / d0)
    escale = G * masses[0] * masses[1] / np.sqrt(d0**2 + b**2)
    pscale = np.sum(masses) * vscale
    lscale = masses[0] * masses[1] / np.sum(masses) * d0 * vscale
    cm_exact = d["Rcm"][0] + times[:, None] * d["P"][0] / np.sum(masses)
    return np.array([np.max(np.abs(d["E"] - d["E"][0])) / escale,
                     np.max(np.abs(d["L"] - d["L"][0])) / lscale,
                     np.max(np.linalg.norm(d["P"] - d["P"][0], axis=1)) / pscale,
                     np.max(np.linalg.norm(d["Rcm"] - cm_exact, axis=1)) / d0])


center_values = center_diagnostics(center_history, masses, b)
center_error = center_errors(center_times, center_history, masses, b)
i_peri = np.argmin(center_values["D"])
print(f"Smallest sampled separation = {center_values['D'][i_peri]:.9f} kpc")
print(f"Time of the smallest sampled separation = {center_times[i_peri]:.6f} Gyr")
print("Center errors [E/Escale, L/Lscale, P/Pscale, Rcm/D0]:", center_error)
np.savetxt(DATADIR / "center_diagnostics.csv", np.column_stack((center_times, center_values["D"], center_values["E"], center_values["L"], center_values["Rcm"])), delimiter=",", header="time_Gyr,D_kpc,E,Lz,Rcm_x_kpc,Rcm_y_kpc", comments="")

fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
ax.plot(center_times, center_values["D"], color=BLUE, label="Effective center orbit")
ax.axhline(d_peri, color=GRAY, ls="--", label="Target pericenter")
ax.plot(center_times[i_peri], center_values["D"][i_peri], "s", color=VERMILLION, label="Sampled minimum")
ax.set(xlabel="Time [Gyr]", ylabel="Center separation [kpc]")
ax.legend(fontsize=9)
save_figure(fig, "ch05_separation")

fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
escale = G * masses[0] * masses[1] / np.sqrt(d_start**2 + b**2)
ax.plot(center_times, (center_values["E"] - center_values["E"][0]) / escale, color=BLUE)
ax.set(xlabel="Time [Gyr]", ylabel=r"$[E_{\rm c}(t)-E_{\rm c}(0)]/E_*$", title="Center orbit: signed energy error")
save_figure(fig, "ch05_center_energy")

# %% [markdown]
# ## 8. 粒子分布の時間発展
# 色とマーカーは出身銀河に固定します。黒い×と＋はPlummer中心であり、追加の質点ではありません。
# 全時刻で同じ軸範囲と縦横比を使います。描画範囲の外へ出ても、粒子は積分から除去しません。
# **問い：** 近点前後で円盤から伸びる構造と、二つの中心の間をつなぐ構造を探してください。

# %%
# --8<-- [start:plot-snapshot]
def plot_snapshot(state, origin, time, limit=90.0):
    fig, ax = plt.subplots(figsize=(5.5, 5.1), layout="constrained")
    for k, (color, marker) in enumerate([(BLUE, "o"), (VERMILLION, "^")]):
        points = state[2 + np.flatnonzero(origin == k), :2]
        ax.scatter(points[:, 0], points[:, 1], s=4, color=color, marker=marker,
                   alpha=0.75, linewidths=0, label=f"Origin: galaxy {k+1}")
    ax.plot(state[0, 0], state[0, 1], "x", color="black", ms=7, mew=1.5, label="Center 1")
    ax.plot(state[1, 0], state[1, 1], "+", color="black", ms=9, mew=1.5, label="Center 2")
    ax.set(xlabel="x [kpc]", ylabel="y [kpc]", title=f"t = {time:.2f} Gyr")
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, markerscale=1.6)
    return fig
# --8<-- [end:plot-snapshot]


snapshot_times = [0.0, 0.20, 0.27, 0.35, 0.50, 0.60]
for target in snapshot_times:
    j = int(np.argmin(np.abs(times - target)))
    if not np.isclose(times[j], target, atol=1e-12, rtol=0.0):
        raise ValueError("Requested snapshot time is not stored; adjust save_every.")
    name = f"ch05_snapshot_{round(100*target):03d}"
    save_figure(plot_snapshot(history[j], origin, times[j]), name)


# 同じ計算の0.35 Gyrを拡大します。全体図とは軸範囲が異なることを本文で明示します。
j_zoom = int(np.argmin(np.abs(times - 0.35)))
fig = plot_snapshot(history[j_zoom], origin, times[j_zoom], limit=38.0)
ax = fig.axes[0]
ax.get_legend().remove()
ax.annotate("Bridge", xy=(0.0, 0.0), xytext=(-30, 22), arrowprops={"arrowstyle": "->", "color": "black"})
ax.annotate("Tail", xy=(23, 8), xytext=(24, 27), arrowprops={"arrowstyle": "->", "color": "black"})
save_figure(fig, "ch05_tidal_zoom")

# %% [markdown]
# ## 9. 外へ移動した粒子の割合
# $s_i=|\boldsymbol r_i-\boldsymbol R_{k(i)}|$は**出身銀河**の中心からの距離です。
# $s_i>30\ \mathrm{kpc}=2R_{\max}$となる粒子を数えます。これは位置で決める指標で、非束縛率ではありません。
# 他の銀河へ移った粒子や、後で戻る粒子も含まれ得ます。`origin`を書き換えません。

# %%
# --8<-- [start:outer-fraction]
def origin_radii(history, origin):
    parent_centers = history[:, origin, :2]
    return np.linalg.norm(history[:, 2:, :2] - parent_centers, axis=2)


def outer_fractions(history, origin, threshold):
    radius = origin_radii(history, origin)
    fractions = np.empty((len(history), 2))
    for k in range(2):
        selected = origin == k
        if not np.any(selected):
            raise ValueError("Both origin groups must contain test particles.")
        fractions[:, k] = np.mean(radius[:, selected] > threshold, axis=1)
    return fractions
# --8<-- [end:outer-fraction]


threshold = 30.0
fractions = outer_fractions(history, origin, threshold)
np.savetxt(DATADIR / "outer_fraction.csv", np.column_stack((times, fractions)), delimiter=",", header="time_Gyr,f_origin1,f_origin2", comments="")
print(f"Final fraction beyond {threshold:g} kpc from the origin center:", fractions[-1])
print("Final maximum origin radii [kpc]:", [np.max(origin_radii(history[-1:], origin)[0, origin == k]) for k in range(2)])
fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
ax.plot(times, fractions[:, 0], color=BLUE, label="Origin: galaxy 1")
ax.plot(times, fractions[:, 1], color=VERMILLION, ls="--", label="Origin: galaxy 2")
ax.set(xlabel="Time [Gyr]", ylabel="Fraction beyond 30 kpc", title="Distance from the center of origin")
ax.legend(fontsize=9)
save_figure(fig, "ch05_outer_fraction")

# %% [markdown]
# ## 10. テスト粒子のエネルギー変化は、そのまま誤差ではない
# 慣性系で$\varepsilon_i=v_i^2/2+\Phi_1+\Phi_2$を計算します。
# 中心が動くため、$d\varepsilon_i/dt=\partial\Phi/\partial t=\sum_k\boldsymbol V_k\cdot\boldsymbol a_{k,i}$で、一般には0ではありません。
# 次の図は初期半径が$3R_d$に最も近い円盤1の1粒子です。個別の例であって全粒子の平均ではありません。

# %%
# --8<-- [start:tracer-energy]
def tracer_energy(history, masses, scales):
    pos, vel = history[:, 2:, :2], history[:, 2:, 2:]
    energy = 0.5 * np.sum(vel**2, axis=2)
    for k in range(2):
        relative = pos - history[:, k:k+1, :2]
        radius2 = np.sum(relative**2, axis=2)
        energy -= G * masses[k] / np.sqrt(radius2 + scales[k]**2)
    return energy
# --8<-- [end:tracer-energy]


candidates = np.flatnonzero(origin == 0)
watch = int(candidates[np.argmin(np.abs(initial_radius[candidates] - 3.0 * rds[0]))])
energy_watch = tracer_energy(history[:, [0, 1, 2 + watch], :], masses, scales)[:, 0]
particle_energy_scale = G * masses[0] / scales[0]
np.savetxt(DATADIR / "tracer_energy_example.csv", np.column_stack((times, energy_watch)), delimiter=",", header="time_Gyr,specific_energy_kpc2_Gyr2", comments="")
fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
ax.plot(times, (energy_watch - energy_watch[0]) / particle_energy_scale, color=BLUE)
ax.set(xlabel="Time [Gyr]", ylabel=r"$[\varepsilon_i(t)-\varepsilon_i(0)]/(GM_1/a_1)$", title=f"One tracer; initial radius = {initial_radius[watch]:.2f} kpc")
save_figure(fig, "ch05_tracer_energy")

# %% [markdown]
# ## 11. 同じ初期粒子で時間刻みを変える
# 相互作用中の粒子には、孤立円盤の円運動の解析位置を使えません。
# 同じ初期配列・終時刻・151個の保存時刻で、隣り合う刻みの解の差を測ります。
# ここで測るのは真の解との誤差でなく**自己収束**です。細かい側で差の比が約4になるかを確認します。
# 時間刻みだけを変え、乱数を引き直しません。

# %%
# --8<-- [start:time-convergence]
step_counts = [3000, 6000, 12000, 24000]
resolution_runs, resolution_rows = [], []
for count in step_counts:
    tt, hh, ct, cc = integrate_leapfrog(acceleration_system, state0, t_end, count,
                                       masses, scales, b, count // 150)
    vv = center_diagnostics(cc, masses, b)
    ff = outer_fractions(hh[-1:], origin, threshold)[0]
    resolution_runs.append(hh)
    resolution_rows.append([count, t_end / count, np.min(vv["D"]),
                            *center_errors(ct, cc, masses, b), *ff])
self_rows = []
for j in range(len(step_counts) - 1):
    difference = resolution_runs[j][:, :, :2] - resolution_runs[j + 1][:, :, :2]
    delta_center = np.max(np.linalg.norm(difference[:, :2], axis=2)) / d_start
    delta_tracer = np.max(np.linalg.norm(difference[:, 2:], axis=2)) / rds[0]
    rms_tracer = np.sqrt(np.mean(np.sum(difference[:, 2:]**2, axis=2))) / rds[0]
    self_rows.append([step_counts[j], step_counts[j+1], t_end / step_counts[j],
                      delta_center, delta_tracer, rms_tracer])
self_convergence = np.asarray(self_rows)
self_ratios = self_convergence[:-1, 4] / self_convergence[1:, 4]
# --8<-- [end:time-convergence]


print("Self-convergence: coarse steps, fine steps, center difference/D0, tracer difference/Rd")
for row in self_convergence:
    print(f"{int(row[0]):5d} {int(row[1]):5d} {row[3]:.7e} {row[4]:.7e}")
print("Ratios of successive tracer differences:", self_ratios)
print("Resolution runs [steps, dt, Dmin, eE, eL, eP, eCM, f1, f2]:")
print(np.asarray(resolution_rows))
np.savetxt(DATADIR / "self_convergence.csv", self_convergence, delimiter=",", header="coarse_steps,fine_steps,coarse_dt_Gyr,center_difference_D0,tracer_difference_Rd,rms_tracer_difference_Rd", comments="")
np.savetxt(DATADIR / "resolution_summary.csv", resolution_rows, delimiter=",", header="n_steps,dt_Gyr,Dmin_kpc,eE,eL,eP,eCM,f1,f2", comments="")
fig, ax = plt.subplots(figsize=(5.8, 3.8), layout="constrained")
ax.loglog(self_convergence[:, 2] * 1000, self_convergence[:, 4], "o-", color=BLUE, label="Maximum tracer difference")
ax.loglog(self_convergence[:, 2] * 1000, self_convergence[:, 5], "s--", color=VERMILLION, label="RMS tracer difference")
reference = self_convergence[-1, 4] * (self_convergence[:, 2] / self_convergence[-1, 2])**2
ax.loglog(self_convergence[:, 2] * 1000, reference, color=GRAY, ls="--", label="Reference slope 2")
ax.set(xlabel="Coarse time step [Myr]", ylabel=r"Position difference / $R_d$")
ax.legend(fontsize=9)
save_figure(fig, "ch05_convergence")

# %% [markdown]
# ## 12. 中心軌道の独立性を確認し、データを保存する
# テスト粒子をすべて外しても、銀河中心の軌道は変わりません。変わった場合は反作用を誤って入れています。
# `history`全体はメモリ上に残し、配布ファイルには初期値、代表的スナップショット、中心軌道を保存します。
# 標準設定から変更した場合も、実行した値が`validation.json`に残ります。

# %%
_, _, ct_only, centers_only = integrate_leapfrog(acceleration_system, state0[:2], t_end, n_steps, masses, scales, b, save_every)
center_independence = np.max(np.abs(centers_only - center_history))
print("Center orbit difference with/without tracers:", center_independence)

snapshot_ids = [int(np.argmin(np.abs(times - t))) for t in snapshot_times]
np.savez_compressed(DATADIR / "interaction_snapshots.npz", times=times[snapshot_ids], state=history[snapshot_ids], origin=origin)
np.savez_compressed(DATADIR / "center_orbit.npz", times=center_times, state=center_history)
summary = {
    "disk_scale_kpc": float(rds[0]),
    "chapter_version": "1.0", "numpy_version": np.__version__,
    "masses_Msun": masses.tolist(), "scales_kpc": scales.tolist(), "b_kpc": b,
    "rds_kpc": rds.tolist(), "rmaxs_kpc": rmaxs.tolist(), "counts": counts.tolist(), "spins": spins.tolist(),
    "seed": seed, "n_cdf": n_cdf, "d_start_kpc": d_start, "d_peri_kpc": d_peri,
    "t_end_Gyr": t_end, "n_steps": n_steps, "dt_Gyr": dt, "save_every": save_every,
    "initial_relative_velocity_kms": (u0 / KM_S_TO_KPC_GYR).tolist(),
    "initial_relative_speed_kms": float(np.linalg.norm(u0) / KM_S_TO_KPC_GYR),
    "sampled_dmin_kpc": float(center_values["D"][i_peri]), "sampled_tperi_Gyr": float(center_times[i_peri]),
    "center_errors": center_error.tolist(), "center_independence": float(center_independence),
    "history_MB": history.nbytes / 1e6, "center_history_MB": center_history.nbytes / 1e6,
    "snapshot_times_Gyr": snapshot_times, "outer_threshold_kpc": threshold,
    "outer_fraction_final": fractions[-1].tolist(), "watch_particle": watch,
    "watch_initial_radius_kpc": float(initial_radius[watch]),
    "watch_final_energy_change": float((energy_watch[-1] - energy_watch[0]) / particle_energy_scale),
    "resolution_summary": resolution_rows, "self_convergence": self_convergence.tolist(),
    "self_ratios": self_ratios.tolist(), "elapsed_seconds_this_environment": elapsed
}
(DATADIR / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
assert np.all(np.isfinite(history))
assert center_independence == 0.0
print("Standard-model checks passed. Data directory:", DATADIR)

# %% [markdown]
# ## 演習
# **確認5.1：** 中心の運動方程式と星の運動方程式を区別し、力の向きと配列の行を説明してください。
# **確認5.2：** 初期速度から比エネルギーと比角運動量を計算し、目標近点8 kpcで条件が成り立つか確認してください。
# **確認5.3：** 円盤の位置にだけ中心位置を足し、速度に中心速度を足さないと何が起きるでしょうか。
# **応用5.4：** 同じ初期配列で時間刻みを変え、151個の同じ時刻で自己収束を確認してください。
# **応用5.5：** 時間刻みを固定し、`save_every`だけを変えてください。中心の毎ステップの履歴と全粒子の終点は変わるでしょうか。
# **応用5.6：** 全中心・全粒子に同じ位置$(5,-3)$ kpc、速度$(30,-20)$ km/sを加え、相対運動を比べてください。
# **発展5.7：** 銀河2の円盤だけ回転を逆にします。中心の初期状態と粒子の初期位置を固定して、同じ時刻の粒子分布と外側割合を比較してください。
#
# 5.7は第6章の比較実験への入口です。ここで変えるのは**円盤の中心に対する速度**だけです。
# 初期値を変えた数値解の差を、時間積分誤差とは呼びません。

# %% [markdown]
# ### 演習5.6・5.7の実装例（完成版）
# 時間刻みの実験とは分け、標準計算を上書きしない変数に結果を入れます。

# %%
# --8<-- [start:exercise-boost]
### BEGIN SOLUTION
boost_pos = np.array([5.0, -3.0])
boost_vel = np.array([30.0, -20.0]) * KM_S_TO_KPC_GYR
boost_initial = state0.copy()
boost_initial[:, :2] += boost_pos
boost_initial[:, 2:] += boost_vel
t_boost, h_boost, _, _ = integrate_leapfrog(acceleration_system, boost_initial, t_end, n_steps, masses, scales, b, save_every)
base_relative = history[:, 2:, :2] - history[:, origin, :2]
boost_relative = h_boost[:, 2:, :2] - h_boost[:, origin, :2]
boost_difference = np.max(np.linalg.norm(boost_relative - base_relative, axis=2)) / rds[0]
print("Uniform-boost relative trajectory difference / Rd:", boost_difference)
### END SOLUTION
# --8<-- [end:exercise-boost]

# --8<-- [start:exercise-retrograde]
### BEGIN SOLUTION
retro_initial = state0.copy()
rows_second = 2 + np.flatnonzero(origin == 1)
retro_initial[rows_second, 2:] = 2.0 * state0[1, 2:] - state0[rows_second, 2:]
t_retro, h_retro, _, c_retro = integrate_leapfrog(acceleration_system, retro_initial, t_end, n_steps, masses, scales, b, save_every)
retro_fractions = outer_fractions(h_retro, origin, threshold)
print("Retrograde experiment: unchanged center trajectory:", np.array_equal(c_retro, center_history))
print("Retrograde experiment: final outer fractions:", retro_fractions[-1])
save_figure(plot_snapshot(h_retro[-1], origin, t_retro[-1]), "ch05_exercise_retrograde")
### END SOLUTION
# --8<-- [end:exercise-retrograde]
summary["boost_difference_Rd"] = float(boost_difference)
summary["retrograde_outer_fraction_final"] = retro_fractions[-1].tolist()
(DATADIR / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

# %% [markdown]
# ## アニメーション
# 保存した履歴を再生します。青・朱色とマーカーは出身銀河に固定します。
# `interval`は画面の再生間隔で、物理的な時間刻み`dt`ではありません。
# 構造の物理的解釈とモデルの限界は本文の5.8–5.12節を参照してください。

# %%
if MAKE_ANIMATION:
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML, display
    plt.rcParams["animation.embed_limit"] = 40
    fig, ax = plt.subplots(figsize=(5.4, 5.1), layout="constrained")
    particles1, = ax.plot([], [], ".", color=BLUE, ms=2, alpha=0.8, label="Origin: galaxy 1")
    particles2, = ax.plot([], [], "^", color=VERMILLION, ms=1.8, alpha=0.7, label="Origin: galaxy 2")
    center1, = ax.plot([], [], "x", color="black", ms=7, mew=1.5)
    center2, = ax.plot([], [], "+", color="black", ms=9, mew=1.5)
    label = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top")
    ax.set(xlim=(-90, 90), ylim=(-90, 90), xlabel="x [kpc]", ylabel="y [kpc]")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    rows1, rows2 = 2 + np.flatnonzero(origin == 0), 2 + np.flatnonzero(origin == 1)
    def update(frame):
        state = history[frame]
        particles1.set_data(state[rows1, 0], state[rows1, 1])
        particles2.set_data(state[rows2, 0], state[rows2, 1])
        center1.set_data([state[0, 0]], [state[0, 1]])
        center2.set_data([state[1, 0]], [state[1, 1]])
        label.set_text(f"t = {times[frame]:.2f} Gyr")
        return particles1, particles2, center1, center2, label
    frame_ids = np.unique(np.r_[np.arange(0, len(times), 3), len(times)-1])
    animation = FuncAnimation(fig, update, frames=frame_ids, interval=60, blit=False)
    html = animation.to_jshtml(fps=16)
    (FIGDIR / "ch05_interaction_animation.html").write_text(html, encoding="utf-8")
    display(HTML(html))
    plt.close(fig)
else:
    print("Animation is available in the completed Notebook.")
