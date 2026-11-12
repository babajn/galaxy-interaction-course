def center_diagnostics(center_history, masses, b):
    pos, vel = center_history[:, :, :2], center_history[:, :, 2:]
    dr, dv = pos[:, 1] - pos[:, 0], vel[:, 1] - vel[:, 0]
    separation = np.linalg.norm(dr, axis=1)
    kinetic = 0.5 * np.sum(masses[None, :] * np.sum(vel**2, axis=2), axis=1)
    potential = -G * masses[0] * masses[1] / np.sqrt(separation**2 + b**2)
    momentum = np.sum(masses[None, :, None] * vel, axis=1)
    r_cm = np.sum(masses[None, :, None] * pos, axis=1) / np.sum(masses)
    cross = pos[:, :, 0] * vel[:, :, 1] - pos[:, :, 1] * vel[:, :, 0]
    angular = np.sum(masses[None, :] * cross, axis=1)
    return {"D": separation, "dr": dr, "dv": dv, "E": kinetic + potential,
            "L": angular, "P": momentum, "Rcm": r_cm}
