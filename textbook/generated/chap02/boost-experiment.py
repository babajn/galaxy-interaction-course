r_boost = np.array([5.0, -3.0])
v_boost = np.array([30.0, -20.0]) * KM_S_TO_KPC_GYR
boost_initial = make_initial_state(masses, [d0, 0.0], [0.0, 0.8 * v_c], r_boost, v_boost)
t_boost, h_boost = integrate_leapfrog(acceleration_two_body, boost_initial, t_c, n_steps, masses)
bv = diagnostics(h_boost, masses)
base = diagnostics(cases["Elliptic"]["history"], masses)
boost_relative_error = np.max(np.linalg.norm(bv["dr"] - base["dr"], axis=1)) / d0
boost_errors = conservation_errors(t_boost, h_boost, masses)
