#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal galaxy-interaction simulation for teaching.

Model
-----
- Two moving Plummer potentials.
- Massless test particles represent stellar disks.
- Test particles feel both galaxies but do not interact with each other.
- The galaxy centers interact through an effective Plummer-softened force.
- Time integration: fourth-order Runge--Kutta (RK4).
- Units: kpc, Gyr, Msun.

The default model is a coplanar, equal-mass, prograde--prograde encounter.
Set SPIN_2 = -1 to make the second disk retrograde.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Units and standard model
# ============================================================

# G = 4.30091e-6 kpc (km/s)^2 Msun^-1, converted to kpc^3 Msun^-1 Gyr^-2
KM_S_TO_KPC_GYR = 1.0227121650537077
G = 4.30091e-6 * KM_S_TO_KPC_GYR**2

M1 = 1.0e11                 # Msun
M2 = 1.0e11                 # Msun
A1 = 3.0                    # kpc, Plummer scale
A2 = 3.0                    # kpc, Plummer scale
A_ORBIT = 3.0               # kpc, effective softening of the center-center force

RD1 = 3.0                   # kpc, exponential-disk scale length
RD2 = 3.0                   # kpc
RMAX1 = 15.0                # kpc
RMAX2 = 15.0                # kpc

N1 = 1000                   # test particles in galaxy 1
N2 = 1000                   # test particles in galaxy 2
SPIN_1 = +1                 # +1: prograde, -1: retrograde
SPIN_2 = +1                 # change to -1 for a prograde-retrograde encounter

R0 = 60.0                   # kpc, initial galaxy-center separation
R_PERI = 8.0                # kpc, target pericenter of the softened zero-energy orbit

DT = 0.001                  # Gyr = 1 Myr
T_END = 0.60                # Gyr
SNAPSHOT_TIMES = [0.00, 0.20, 0.27, 0.35, 0.50, 0.60]
RANDOM_SEED = 42

OUTPUT_FIGURE = "galaxy_interaction_snapshots.png"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
BLACK = "#000000"


# ============================================================
# Gravity
# ============================================================

def plummer_acceleration(pos, center, mass, a):
    """Acceleration at pos from one Plummer potential."""
    dr = pos - center
    r2 = np.sum(dr * dr, axis=-1)
    return -G * mass * dr / (r2 + a * a)[..., None]**1.5


def center_accelerations(r1, r2):
    """Mutual acceleration of the two galaxy centers."""
    dr = r2 - r1
    r2_sep = np.dot(dr, dr)
    denom = (r2_sep + A_ORBIT**2)**1.5
    a1 = G * M2 * dr / denom
    a2 = -G * M1 * dr / denom
    return a1, a2


# ============================================================
# RK4
# ============================================================

def rhs(state):
    """
    Right-hand side of dy/dt = f(y).

    state[:, 0:2] = position (x, y)
    state[:, 2:4] = velocity (vx, vy)

    state[0] and state[1] are galaxy centers.
    state[2:] are massless test particles.
    """
    deriv = np.empty_like(state)
    deriv[:, 0:2] = state[:, 2:4]

    r1 = state[0, 0:2]
    r2 = state[1, 0:2]
    a1, a2 = center_accelerations(r1, r2)
    deriv[0, 2:4] = a1
    deriv[1, 2:4] = a2

    pos = state[2:, 0:2]
    acc = plummer_acceleration(pos, r1, M1, A1) + plummer_acceleration(pos, r2, M2, A2)
    deriv[2:, 2:4] = acc

    return deriv


def rk4_step(state, dt):
    """Advance one time step with classical fourth-order Runge--Kutta."""
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


# ============================================================
# Disk initial conditions
# ============================================================

def sample_exponential_disk(n, rd, rmax, rng):
    """
    Sample R from p(R) proportional to R exp(-R/rd).

    The cumulative distribution is constructed numerically with
    the trapezoidal rule, and inverted with linear interpolation.
    """
    r_grid = np.linspace(0.0, rmax, 4097)
    pdf = r_grid * np.exp(-r_grid / rd)

    cdf = np.zeros_like(r_grid)
    dr = np.diff(r_grid)
    cdf[1:] = np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dr)
    cdf /= cdf[-1]

    u = rng.random(n)
    radius = np.interp(u, cdf, r_grid)
    phi = 2.0 * np.pi * rng.random(n)

    pos = np.column_stack((radius * np.cos(phi), radius * np.sin(phi)))
    return pos, radius, phi


def circular_speed_plummer(radius, mass, a):
    """Circular speed in a spherical Plummer potential."""
    return np.sqrt(G * mass * radius**2 / (radius**2 + a**2)**1.5)


def make_disk(n, center, center_velocity, mass, a, rd, rmax, spin, rng):
    """Create a cold exponential test-particle disk around one galaxy."""
    rel_pos, radius, phi = sample_exponential_disk(n, rd, rmax, rng)
    vc = circular_speed_plummer(radius, mass, a)

    tangent = np.column_stack((-np.sin(phi), np.cos(phi)))
    rel_vel = spin * vc[:, None] * tangent

    pos = center + rel_pos
    vel = center_velocity + rel_vel
    return pos, vel


# ============================================================
# Galaxy-center orbit
# ============================================================

def make_zero_energy_encounter(r0, rperi):
    """
    Initial conditions for a zero-energy relative orbit in the
    effective softened potential

        Phi_rel(r) = -G (M1 + M2) / sqrt(r^2 + A_ORBIT^2).

    The two centers are placed in the center-of-mass frame.
    """
    mu = G * (M1 + M2)

    h2 = 2.0 * mu * rperi**2 / np.sqrt(rperi**2 + A_ORBIT**2)
    h = np.sqrt(h2)

    vrel2 = 2.0 * mu / np.sqrt(r0**2 + A_ORBIT**2)
    vt = h / r0
    vr = -np.sqrt(vrel2 - vt**2)

    separation = np.array([r0, 0.0])
    relative_velocity = np.array([vr, vt])

    mtot = M1 + M2
    r1 = -(M2 / mtot) * separation
    r2 = +(M1 / mtot) * separation
    v1 = -(M2 / mtot) * relative_velocity
    v2 = +(M1 / mtot) * relative_velocity

    return r1, r2, v1, v2


# ============================================================
# Diagnostics
# ============================================================

def center_orbital_diagnostics(state):
    """Specific energy and angular momentum of the relative center orbit."""
    dr = state[1, 0:2] - state[0, 0:2]
    dv = state[1, 2:4] - state[0, 2:4]

    energy = 0.5 * np.dot(dv, dv) - G * (M1 + M2) / np.sqrt(np.dot(dr, dr) + A_ORBIT**2)
    angular_momentum = dr[0] * dv[1] - dr[1] * dv[0]
    separation = np.linalg.norm(dr)
    return energy, angular_momentum, separation


# ============================================================
# Simulation
# ============================================================

def make_initial_state():
    rng = np.random.default_rng(RANDOM_SEED)

    r1, r2, v1, v2 = make_zero_energy_encounter(R0, R_PERI)
    pos1, vel1 = make_disk(N1, r1, v1, M1, A1, RD1, RMAX1, SPIN_1, rng)
    pos2, vel2 = make_disk(N2, r2, v2, M2, A2, RD2, RMAX2, SPIN_2, rng)

    state = np.empty((2 + N1 + N2, 4))
    state[0, 0:2], state[0, 2:4] = r1, v1
    state[1, 0:2], state[1, 2:4] = r2, v2
    state[2:2 + N1, 0:2], state[2:2 + N1, 2:4] = pos1, vel1
    state[2 + N1:, 0:2], state[2 + N1:, 2:4] = pos2, vel2
    return state


def integrate():
    state = make_initial_state()
    n_steps = int(round(T_END / DT))
    save_steps = {int(round(t / DT)): t for t in SNAPSHOT_TIMES}

    snapshots = {}
    times = np.arange(n_steps + 1) * DT
    energy = np.empty(n_steps + 1)
    angular_momentum = np.empty(n_steps + 1)
    separation = np.empty(n_steps + 1)

    for step in range(n_steps + 1):
        energy[step], angular_momentum[step], separation[step] = center_orbital_diagnostics(state)

        if step in save_steps:
            snapshots[save_steps[step]] = state.copy()

        if step < n_steps:
            state = rk4_step(state, DT)

    return snapshots, times, energy, angular_momentum, separation


# ============================================================
# Plot
# ============================================================

def plot_snapshots(snapshots):
    ncols = 3
    nrows = int(np.ceil(len(SNAPSHOT_TIMES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 7.2), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, t in zip(axes, SNAPSHOT_TIMES):
        state = snapshots[t]

        disk1 = state[2:2 + N1]
        disk2 = state[2 + N1:]

        ax.scatter(disk1[:, 0], disk1[:, 1], s=2.0, c=BLUE, alpha=0.8, linewidths=0)
        ax.scatter(disk2[:, 0], disk2[:, 1], s=2.0, c=VERMILLION, alpha=0.8, linewidths=0)
        ax.scatter(state[:2, 0], state[:2, 1], s=22.0, c=BLACK, marker="o")

        ax.set_title(f"{t:.2f} Gyr")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_xlim(-60.0, 60.0)
        ax.set_ylim(-60.0, 60.0)
        ax.set_aspect("equal")

    for ax in axes[len(SNAPSHOT_TIMES):]:
        ax.axis("off")

    fig.savefig(OUTPUT_FIGURE, dpi=180)
    plt.close(fig)


def main():
    snapshots, times, energy, angular_momentum, separation = integrate()
    plot_snapshots(snapshots)

    i_peri = np.argmin(separation)
    e_scale = 0.5 * (G * (M1 + M2) / R0)
    energy_error = np.max(np.abs(energy - energy[0])) / e_scale
    angular_momentum_error = np.max(np.abs(angular_momentum - angular_momentum[0])) / abs(angular_momentum[0])

    print(f"Minimum separation = {separation[i_peri]:.3f} kpc")
    print(f"Pericenter time     = {times[i_peri]:.3f} Gyr")
    print(f"Max energy error    = {energy_error:.3e} (normalized)")
    print(f"Max angular-momentum error = {angular_momentum_error:.3e}")
    print(f"Saved figure: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
