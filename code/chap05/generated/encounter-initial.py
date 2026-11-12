def make_zero_energy_encounter(masses, d_start, d_peri, b):
    masses = np.asarray(masses, dtype=float)
    check_model(masses, np.ones(2), b)
    if not (np.isfinite(d_start) and np.isfinite(d_peri) and 0.0 < d_peri < d_start):
        raise ValueError("Require finite distances with 0 < d_peri < d_start.")
    mtot = np.sum(masses)
    mu = G * mtot
    h = np.sqrt(2.0 * mu * d_peri**2 / np.sqrt(d_peri**2 + b**2))
    vt = h / d_start
    vr2 = 2.0 * mu / np.sqrt(d_start**2 + b**2) - vt**2
    if vr2 <= 0.0:
        raise ValueError("The incoming radial speed must be real and nonzero.")
    relative_velocity = np.array([-np.sqrt(vr2), vt])
    separation = np.array([d_start, 0.0])
    weights = np.array([-masses[1], masses[0]]) / mtot
    pos = weights[:, None] * separation
    vel = weights[:, None] * relative_velocity
    return np.column_stack((pos, vel))
