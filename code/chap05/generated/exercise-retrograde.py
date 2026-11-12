retro_initial = state0.copy()
rows_second = 2 + np.flatnonzero(origin == 1)
retro_initial[rows_second, 2:] = 2.0 * state0[1, 2:] - state0[rows_second, 2:]
t_retro, h_retro, _, c_retro = integrate_leapfrog(acceleration_system, retro_initial, t_end, n_steps, masses, scales, b, save_every)
retro_fractions = outer_fractions(h_retro, origin, threshold)
print("Retrograde experiment: unchanged center trajectory:", np.array_equal(c_retro, center_history))
print("Retrograde experiment: final outer fractions:", retro_fractions[-1])
save_figure(plot_snapshot(h_retro[-1], origin, t_retro[-1]), "ch05_exercise_retrograde")
