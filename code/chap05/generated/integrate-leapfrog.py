def integrate_leapfrog(acceleration, state0, t_end, n_steps, masses, scales, b, save_every=10):
    masses, scales = check_model(masses, scales, b)
    state, saved_steps, times, history, dt = prepare_history(state0, t_end, n_steps, save_every)
    pos, vel = state[:, :2].copy(), state[:, 2:].copy()
    acc = acceleration(pos, masses, scales, b)
    center_times = np.arange(n_steps + 1) * dt
    center_history = np.empty((n_steps + 1, 2, 4))
    center_history[0] = state[:2]
    output_index = 1
    for n in range(n_steps):
        pos, vel, acc = leapfrog_step(acceleration, pos, vel, acc, dt, masses, scales, b)
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            raise FloatingPointError("Non-finite state; check the model and time step.")
        center_history[n + 1] = np.column_stack((pos[:2], vel[:2]))
        if n + 1 == saved_steps[output_index]:
            history[output_index] = np.column_stack((pos, vel))
            output_index += 1
    return times, history, center_times, center_history
