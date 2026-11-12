def integrate_leapfrog(acceleration, state0, t_end, n_steps, mu):
    """初回の力を計算し、以降は前ステップのacc_newを使います。"""
    state, times, history, dt = prepare_history(state0, t_end, n_steps)
    pos, vel = state[:, :2].copy(), state[:, 2:].copy()
    acc = acceleration(pos, mu)  # 初回だけ、ループの前に計算します。
    for n in range(n_steps):
        pos, vel, acc = leapfrog_step(acceleration, pos, vel, acc, dt, mu)
        history[n + 1] = np.column_stack((pos, vel))
        if not np.all(np.isfinite(history[n + 1])):
            raise FloatingPointError("Non-finite state; check the time step.")
    return times, history
