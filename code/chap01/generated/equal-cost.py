force_budget = 800
n_lf_cost = force_budget - 1
n_rk_cost = force_budget // 4
t_lf_cost, h_lf_cost = integrate_leapfrog(acceleration_kepler, state0, period, n_lf_cost, mu)
t_rk_cost, h_rk_cost = integrate(rhs_kepler, rk4_step, state0, period, n_rk_cost, mu)
cost_lf_errors = error_metrics(t_lf_cost, h_lf_cost, mu, r0)
cost_rk4_errors = error_metrics(t_rk_cost, h_rk_cost, mu, r0)
