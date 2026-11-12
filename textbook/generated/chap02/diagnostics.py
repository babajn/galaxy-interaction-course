def diagnostics(history, masses):
    pos, vel = history[:, :, :2], history[:, :, 2:]
    dr, dv = pos[:, 1] - pos[:, 0], vel[:, 1] - vel[:, 0]
    separation = np.linalg.norm(dr, axis=1)
    kinetic = 0.5 * masses[0] * np.sum(vel[:, 0]**2, axis=1)
    kinetic += 0.5 * masses[1] * np.sum(vel[:, 1]**2, axis=1)
    potential = -G * masses[0] * masses[1] / separation
    energy = kinetic + potential
    momentum = masses[0] * vel[:, 0] + masses[1] * vel[:, 1]
    r_cm = (masses[0] * pos[:, 0] + masses[1] * pos[:, 1]) / np.sum(masses)
    angular = masses[0] * (pos[:, 0, 0] * vel[:, 0, 1] - pos[:, 0, 1] * vel[:, 0, 0])
    angular += masses[1] * (pos[:, 1, 0] * vel[:, 1, 1] - pos[:, 1, 1] * vel[:, 1, 0])
    return {"D": separation, "dr": dr, "dv": dv, "K": kinetic, "W": potential,
            "E": energy, "L": angular, "P": momentum, "Rcm": r_cm}
