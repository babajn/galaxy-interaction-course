#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minor-merger toy model with Chandrasekhar dynamical friction.

Model
-----
- Fixed Plummer host galaxy.
- A satellite galaxy with mass ratio M_sat / M_host = 0.1.
- The satellite center moves in the host Plummer potential.
- Chandrasekhar dynamical friction acts on the satellite center.
- A massless tracer disk moves in the combined host + satellite Plummer potentials.
- Time integration: classical fourth-order Runge--Kutta (RK4).
- Units: kpc, Gyr, Msun.

This is a teaching model, not a self-consistent N-body merger calculation.
"""

from math import erf
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Units and standard parameters
# ============================================================

KM_S_TO_KPC_GYR = 1.0227121650537077
G = 4.30091e-6 * KM_S_TO_KPC_GYR**2

M_HOST = 1.0e11            # Msun
A_HOST = 3.0               # kpc
M_SAT = 1.0e10             # Msun
A_SAT = 1.5                # kpc

R0 = 20.0                  # kpc
DT = 0.001                 # Gyr = 1 Myr
T_END = 1.50               # Gyr
LN_LAMBDA = np.log1p(M_HOST / M_SAT)

N_TRACER = 500
RD_SAT = 1.0               # kpc
RMAX_SAT = 4.0             # kpc
SPIN_SAT = +1
RANDOM_SEED = 42
SNAPSHOT_TIMES = [0.0, 0.5, 1.0, 1.5]

BLUE = "#0072B2"
VERMILLION = "#D55E00"
BLACK = "#000000"
GRAY = "#777777"

ORBIT_FIGURE = "dynamical_friction_orbit_comparison.png"
SNAPSHOT_FIGURE = "dynamical_friction_satellite_snapshots.png"


# ============================================================
# Plummer model
# ============================================================

def plummer_acceleration(pos, center, mass, a):
    """Acceleration produced by a Plummer potential."""
    dr = pos - center
    r2 = np.sum(dr * dr, axis=-1)
    return -G * mass * dr / (r2 + a * a)[..., None]**1.5


def circular_speed_plummer(radius, mass, a):
    """Circular speed in a spherical Plummer potential."""
    return np.sqrt(G * mass * radius**2 / (radius**2 + a**2)**1.5)


def host_density(radius):
    """Plummer density of the host galaxy."""
    return 3.0 * M_HOST / (4.0 * np.pi * A_HOST**3) * (1.0 + (radius / A_HOST)**2)**(-2.5)


def host_velocity_dispersion(radius):
    """One-dimensional isotropic velocity dispersion of a Plummer sphere."""
    sigma2 = G * M_HOST / (6.0 * A_HOST) * (1.0 + (radius / A_HOST)**2)**(-0.5)
    return np.sqrt(sigma2)


# ============================================================
# Chandrasekhar dynamical friction
# ============================================================

def dynamical_friction_acceleration(pos, vel):
    """Chandrasekhar drag in the local Plummer background."""
    radius = np.linalg.norm(pos)
    speed = np.linalg.norm(vel)

    if speed == 0.0:
        return np.zeros(2)

    rho = host_density(radius)
    sigma = host_velocity_dispersion(radius)
    x = speed / (np.sqrt(2.0) * sigma)
    velocity_factor = erf(x) - 2.0 * x / np.sqrt(np.pi) * np.exp(-x * x)

    prefactor = -4.0 * np.pi * G**2 * M_SAT * rho * LN_LAMBDA * velocity_factor
    return prefactor * vel / speed**3


# ============================================================
# RK4 for the satellite center
# ============================================================

def center_rhs(state, use_friction):
    """RHS for [x, y, vx, vy] of the satellite center."""
    pos = state[0:2]
    vel = state[2:4]
    acc = plummer_acceleration(pos[None, :], np.zeros(2), M_HOST, A_HOST)[0]

    if use_friction:
        acc += dynamical_friction_acceleration(pos, vel)

    return np.concatenate((vel, acc))


def rk4_center_step(state, dt, use_friction):
    k1 = center_rhs(state, use_friction)
    k2 = center_rhs(state + 0.5 * dt * k1, use_friction)
    k3 = center_rhs(state + 0.5 * dt * k2, use_friction)
    k4 = center_rhs(state + dt * k3, use_friction)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_center(use_friction):
    """Integrate only the satellite center."""
    vc0 = circular_speed_plummer(R0, M_HOST, A_HOST)
    state = np.array([R0, 0.0, 0.0, vc0])

    n_steps = int(round(T_END / DT))
    times = np.arange(n_steps + 1) * DT
    history = np.empty((n_steps + 1, 4))
    radius = np.empty(n_steps + 1)
    energy = np.empty(n_steps + 1)
    angular_momentum = np.empty(n_steps + 1)

    for step in range(n_steps + 1):
        pos = state[0:2]
        vel = state[2:4]
        r = np.linalg.norm(pos)

        history[step] = state
        radius[step] = r
        energy[step] = 0.5 * np.dot(vel, vel) - G * M_HOST / np.sqrt(r * r + A_HOST**2)
        angular_momentum[step] = pos[0] * vel[1] - pos[1] * vel[0]

        if step < n_steps:
            state = rk4_center_step(state, DT, use_friction)

    return times, history, radius, energy, angular_momentum


# ============================================================
# Massless tracer disk around the satellite
# ============================================================

def sample_exponential_disk(n, rd, rmax, rng):
    """Sample p(R) proportional to R exp(-R/rd) using trapezoidal CDF + interpolation."""
    r_grid = np.linspace(0.0, rmax, 4097)
    pdf = r_grid * np.exp(-r_grid / rd)

    cdf = np.zeros_like(r_grid)
    dr = np.diff(r_grid)
    cdf[1:] = np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dr)
    cdf /= cdf[-1]

    radius = np.interp(rng.random(n), cdf, r_grid)
    phi = 2.0 * np.pi * rng.random(n)
    pos = np.column_stack((radius * np.cos(phi), radius * np.sin(phi)))
    return pos, radius, phi


def make_initial_state():
    """Satellite center + its massless tracer disk."""
    rng = np.random.default_rng(RANDOM_SEED)

    sat_pos = np.array([R0, 0.0])
    sat_vel = np.array([0.0, circular_speed_plummer(R0, M_HOST, A_HOST)])

    rel_pos, radius, phi = sample_exponential_disk(N_TRACER, RD_SAT, RMAX_SAT, rng)
    vc = circular_speed_plummer(radius, M_SAT, A_SAT)
    tangent = np.column_stack((-np.sin(phi), np.cos(phi)))
    rel_vel = SPIN_SAT * vc[:, None] * tangent

    state = np.empty((1 + N_TRACER, 4))
    state[0, 0:2], state[0, 2:4] = sat_pos, sat_vel
    state[1:, 0:2] = sat_pos + rel_pos
    state[1:, 2:4] = sat_vel + rel_vel
    return state


def full_rhs(state):
    """RHS for satellite center + massless tracer particles."""
    deriv = np.empty_like(state)
    deriv[:, 0:2] = state[:, 2:4]

    sat_pos = state[0, 0:2]
    sat_vel = state[0, 2:4]
    sat_acc = plummer_acceleration(sat_pos[None, :], np.zeros(2), M_HOST, A_HOST)[0]
    sat_acc += dynamical_friction_acceleration(sat_pos, sat_vel)
    deriv[0, 2:4] = sat_acc

    tracer_pos = state[1:, 0:2]
    tracer_acc = plummer_acceleration(tracer_pos, np.zeros(2), M_HOST, A_HOST)
    tracer_acc += plummer_acceleration(tracer_pos, sat_pos, M_SAT, A_SAT)
    deriv[1:, 2:4] = tracer_acc

    return deriv


def rk4_full_step(state, dt):
    k1 = full_rhs(state)
    k2 = full_rhs(state + 0.5 * dt * k1)
    k3 = full_rhs(state + 0.5 * dt * k2)
    k4 = full_rhs(state + dt * k3)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_with_tracers():
    state = make_initial_state()
    n_steps = int(round(T_END / DT))
    save_steps = {int(round(t / DT)): t for t in SNAPSHOT_TIMES}
    snapshots = {}

    for step in range(n_steps + 1):
        if step in save_steps:
            snapshots[save_steps[step]] = state.copy()

        if step < n_steps:
            state = rk4_full_step(state, DT)

    return snapshots


# ============================================================
# Figures
# ============================================================

def plot_orbit_comparison(times, no_df, with_df, radius_no_df, radius_with_df):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)

    axes[0].plot(no_df[:, 0], no_df[:, 1], ls="--", c=GRAY, label="No friction")
    axes[0].plot(with_df[:, 0], with_df[:, 1], c=BLUE, label="With friction")
    axes[0].scatter(0.0, 0.0, s=45.0, c=BLACK, marker="*", label="Host")
    axes[0].set_xlabel("x [kpc]")
    axes[0].set_ylabel("y [kpc]")
    axes[0].set_aspect("equal")
    axes[0].legend()

    axes[1].plot(times, radius_no_df, ls="--", c=GRAY, label="No friction")
    axes[1].plot(times, radius_with_df, c=BLUE, label="With friction")
    axes[1].set_xlabel("Time [Gyr]")
    axes[1].set_ylabel("Satellite radius [kpc]")
    axes[1].legend()

    fig.savefig(ORBIT_FIGURE, dpi=180)
    plt.close(fig)


def plot_snapshots(snapshots):
    fig, axes = plt.subplots(1, len(SNAPSHOT_TIMES), figsize=(13.0, 3.4), constrained_layout=True)

    for ax, time in zip(axes, SNAPSHOT_TIMES):
        state = snapshots[time]
        ax.scatter(state[1:, 0], state[1:, 1], s=3.0, c=BLUE, alpha=0.75, linewidths=0)
        ax.scatter(state[0, 0], state[0, 1], s=32.0, c=VERMILLION, marker="o", label="Satellite")
        ax.scatter(0.0, 0.0, s=42.0, c=BLACK, marker="*", label="Host")
        ax.set_title(f"{time:.1f} Gyr")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_xlim(-25.0, 25.0)
        ax.set_ylim(-25.0, 25.0)
        ax.set_aspect("equal")

    fig.savefig(SNAPSHOT_FIGURE, dpi=180)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    times, history_no_df, radius_no_df, energy_no_df, angular_momentum_no_df = integrate_center(False)
    _, history_with_df, radius_with_df, energy_with_df, angular_momentum_with_df = integrate_center(True)

    snapshots = integrate_with_tracers()
    plot_orbit_comparison(times, history_no_df, history_with_df, radius_no_df, radius_with_df)
    plot_snapshots(snapshots)

    energy_error_no_df = np.max(np.abs((energy_no_df - energy_no_df[0]) / energy_no_df[0]))
    angular_momentum_error_no_df = np.max(np.abs((angular_momentum_no_df - angular_momentum_no_df[0]) / angular_momentum_no_df[0]))

    print(f"Mass ratio M_sat/M_host = {M_SAT / M_HOST:.2f}")
    print(f"ln Lambda = {LN_LAMBDA:.3f}")
    print(f"Initial circular speed = {history_no_df[0, 3] / KM_S_TO_KPC_GYR:.1f} km/s")
    print(f"Final radius without friction = {radius_no_df[-1]:.3f} kpc")
    print(f"Final radius with friction    = {radius_with_df[-1]:.3f} kpc")
    print(f"No-friction max relative energy error = {energy_error_no_df:.3e}")
    print(f"No-friction max relative angular-momentum error = {angular_momentum_error_no_df:.3e}")
    print(f"Saved: {ORBIT_FIGURE}")
    print(f"Saved: {SNAPSHOT_FIGURE}")


if __name__ == "__main__":
    main()
