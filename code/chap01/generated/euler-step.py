def euler_step(rhs, t, state, dt, mu):
    return state + dt * rhs(t, state, mu)
