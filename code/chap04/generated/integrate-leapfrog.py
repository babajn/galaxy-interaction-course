def integrate_leapfrog(acceleration, state0, t_end, n_steps, mass, a, save_every=10):
    check_positive(mass, "mass")
    check_positive(a, "a")
    state, saved_steps, times, history, dt = prepare_history(state0, t_end, n_steps, save_every)
    pos, vel = state[:, :2].copy(), state[:, 2:].copy()
    acc = acceleration(pos, mass, a)
    output_index = 1
    for n in range(n_steps):
        pos, vel, acc = leapfrog_step(acceleration, pos, vel, acc, dt, mass, a)
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vel)):
            raise FloatingPointError("Non-finite state; check the model and time step.")
        if n + 1 == saved_steps[output_index]:
            history[output_index] = np.column_stack((pos, vel))
            output_index += 1
    return times, history
