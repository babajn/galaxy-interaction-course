def tracer_energy(history, masses, scales):
    pos, vel = history[:, 2:, :2], history[:, 2:, 2:]
    energy = 0.5 * np.sum(vel**2, axis=2)
    for k in range(2):
        relative = pos - history[:, k:k+1, :2]
        radius2 = np.sum(relative**2, axis=2)
        energy -= G * masses[k] / np.sqrt(radius2 + scales[k]**2)
    return energy
