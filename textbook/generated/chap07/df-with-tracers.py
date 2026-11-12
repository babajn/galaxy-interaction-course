def df_tracer_rhs(t, state, params):
    deriv = np.empty_like(state)
    deriv[:, :2] = state[:, 2:]
    deriv[0] = df_center_rhs(t, state[:1], params)[0]
    pos = state[1:, :2]
    deriv[1:, 2:] = acceleration_plummer(pos, np.zeros(2), params["m_host"], params["a_host"])
    deriv[1:, 2:] += acceleration_plummer(pos, state[0, :2], params["m_sat"], params["a_sat"])
    return deriv
