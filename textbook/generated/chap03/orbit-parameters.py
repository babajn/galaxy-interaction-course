v_c = circular_speed_plummer(r0, mass, a)
t_c = 2.0 * np.pi * r0 / v_c
n_periods = 5
steps_per_period = 1000
n_steps = n_periods * steps_per_period
t_end = n_periods * t_c
dt = t_end / n_steps
eta_values = {"Circular": 1.0, "Noncircular": 0.7}
