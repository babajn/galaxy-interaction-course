boost_pos = np.array([5.0, -3.0])
boost_vel = np.array([30.0, -20.0]) * KM_S_TO_KPC_GYR
boost_initial = state0.copy()
boost_initial[:, :2] += boost_pos
boost_initial[:, 2:] += boost_vel
t_boost, h_boost, _, _ = integrate_leapfrog(acceleration_system, boost_initial, t_end, n_steps, masses, scales, b, save_every)
base_relative = history[:, 2:, :2] - history[:, origin, :2]
boost_relative = h_boost[:, 2:, :2] - h_boost[:, origin, :2]
boost_difference = np.max(np.linalg.norm(boost_relative - base_relative, axis=2)) / rds[0]
print("Uniform-boost relative trajectory difference / Rd:", boost_difference)
