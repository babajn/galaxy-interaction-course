def integrate(rhs, stepper, state0, t_end, n_steps, mu):
    """Euler法・RK4用。右辺関数を指定して等間隔で積分します。"""
    state, times, history, dt = prepare_history(state0, t_end, n_steps)
    for n in range(n_steps):
        state = stepper(rhs, times[n], state, dt, mu)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("Non-finite state; check the time step.")
        history[n + 1] = state
    return times, history
