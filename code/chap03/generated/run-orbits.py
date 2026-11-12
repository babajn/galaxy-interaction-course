cases = {}
start = perf_counter()
for name, eta in eta_values.items():
    initial = make_initial_state(r0, eta, mass, a)
    times, history = integrate_leapfrog(acceleration_plummer, initial, t_end, n_steps, mass, a)
    cases[name] = {"eta": eta, "initial": initial, "times": times, "history": history}
