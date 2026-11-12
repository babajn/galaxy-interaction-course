def disk_error_metrics(times, history, state0, mass, a, rd, spin):
    radius, energy, angular = diagnostics(history, mass, a)
    exact = exact_disk_positions(times, state0, mass, a, spin)
    position_error = np.max(np.linalg.norm(history[:, :, :2] - exact, axis=2)) / rd
    radius_error = np.max(np.abs(radius - radius[0])) / rd
    energy_error = np.max(np.abs(energy - energy[0]) / np.abs(energy[0]))
    angular_scale = rd * circular_speed_plummer(rd, mass, a)
    angular_error = np.max(np.abs(angular - angular[0])) / angular_scale
    return np.array([position_error, radius_error, energy_error, angular_error])
