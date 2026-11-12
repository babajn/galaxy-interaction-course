def diagnostics(history, mass, a):
    pos, vel = history[:, :, :2], history[:, :, 2:]
    radius = np.sqrt(np.sum(pos**2, axis=2))
    energy = 0.5 * np.sum(vel**2, axis=2) + potential_plummer(radius, mass, a)
    angular = pos[:, :, 0] * vel[:, :, 1] - pos[:, :, 1] * vel[:, :, 0]
    return radius, energy, angular
