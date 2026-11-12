def exact_relative_circle(times, d0, mu):
    omega = np.sqrt(mu / d0**3)
    phase = omega * np.asarray(times)
    return np.column_stack((d0 * np.cos(phase), d0 * np.sin(phase)))
