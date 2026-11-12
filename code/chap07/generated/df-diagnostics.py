def df_diagnostics(history, params):
    pos, vel = history[:, 0, :2], history[:, 0, 2:]
    radius = np.linalg.norm(pos, axis=1)
    potential = -G * params["m_host"] / np.sqrt(radius**2 + params["a_host"]**2)
    energy = 0.5 * np.sum(vel**2, axis=1) + potential
    angular = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    power = np.zeros(len(history))
    if params["friction"]:
        for j in range(len(history)):
            power[j] = np.dot(vel[j], chandrasekhar_acceleration(pos[j], vel[j], params))
    return radius, energy, angular, power
