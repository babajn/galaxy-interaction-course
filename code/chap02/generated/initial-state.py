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
    mtot = np.sum(masses)
    pos1 = r_cm - masses[1] / mtot * separation
    pos2 = r_cm + masses[0] / mtot * separation
    vel1 = v_cm - masses[1] / mtot * relative_velocity
    vel2 = v_cm + masses[0] / mtot * relative_velocity
    return np.column_stack((np.array([pos1, pos2]), np.array([vel1, vel2])))
