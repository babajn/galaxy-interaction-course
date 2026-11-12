cases = {}
start = perf_counter()
for name, eta in eta_values.items():
    initial = make_initial_state(masses, [d0, 0.0], [0.0, eta * v_c])
    times, history = integrate_leapfrog(acceleration_two_body, initial, t_end, n_steps, masses)
    cases[name] = {"eta": eta, "initial": initial, "times": times, "history": history}
