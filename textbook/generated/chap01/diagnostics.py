def diagnostics(history, mu):
    pos = history[:, 0, :2]
    vel = history[:, 0, 2:]
    radius = np.sqrt(np.sum(pos**2, axis=1))
    energy = 0.5 * np.sum(vel**2, axis=1) - mu / radius
    angular_momentum = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    return radius, energy, angular_momentum
