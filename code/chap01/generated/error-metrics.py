def error_metrics(times, history, mu, r0):
    pos_exact, _ = exact_circle(times, r0, mu)
    radius, energy, angular_momentum = diagnostics(history, mu)
    position_error = np.linalg.norm(history[:, 0, :2] - pos_exact, axis=1) / r0
    energy_error = np.abs(energy - energy[0]) / abs(energy[0])
    angular_error = np.abs(angular_momentum - angular_momentum[0]) / abs(angular_momentum[0])
    return np.max(position_error), np.max(energy_error), np.max(angular_error)
