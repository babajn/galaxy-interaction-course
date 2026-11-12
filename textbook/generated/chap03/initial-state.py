def make_initial_state(radius, eta, mass, a):
    check_model(mass, a)
    if not np.isfinite(radius) or radius <= 0.0 or not np.isfinite(eta):
        raise ValueError("Initial radius must be positive and eta finite.")
    pos = np.array([[radius, 0.0]])
    vel = np.array([[0.0, eta * circular_speed_plummer(radius, mass, a)]])
    return np.column_stack((pos, vel))
