def df_center_rhs(t, state, params):
    if state.shape != (1, 4):
        raise ValueError("The center-only RHS requires shape (1, 4).")
    pos, vel = state[0, :2], state[0, 2:]
    acc = acceleration_plummer(pos[None, :], np.zeros(2), params["m_host"], params["a_host"])[0]
    if params["friction"]:
        acc += chandrasekhar_acceleration(pos, vel, params)
    return np.array([[vel[0], vel[1], acc[0], acc[1]]])
