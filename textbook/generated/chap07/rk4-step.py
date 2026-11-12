def rk4_step(rhs, t, state, dt, params):
    k1 = rhs(t, state, params)
    k2 = rhs(t + 0.5 * dt, state + 0.5 * dt * k1, params)
    k3 = rhs(t + 0.5 * dt, state + 0.5 * dt * k2, params)
    k4 = rhs(t + dt, state + dt * k3, params)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
