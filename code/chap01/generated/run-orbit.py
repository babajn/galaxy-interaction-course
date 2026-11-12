times, history = integrate_leapfrog(acceleration_kepler, state0, t_end, n_steps, mu)
_, history_euler = integrate(rhs_kepler, euler_step, state0, t_end, n_steps, mu)
_, history_rk4 = integrate(rhs_kepler, rk4_step, state0, t_end, n_steps, mu)
