step_counts = [20, 40, 80, 160, 320]
rows_lf, rows_rk4 = [], []
for count in step_counts:
    t, h_lf = integrate_leapfrog(acceleration_kepler, state0, period, count, mu)
    _, h_rk4 = integrate(rhs_kepler, rk4_step, state0, period, count, mu)
    rows_lf.append([count, period / count, *error_metrics(t, h_lf, mu, r0)])
    rows_rk4.append([count, period / count, *error_metrics(t, h_rk4, mu, r0)])
convergence_lf, convergence_rk4 = np.array(rows_lf), np.array(rows_rk4)
orders_lf = np.log2(convergence_lf[:-1, 2] / convergence_lf[1:, 2])
orders_rk4 = np.log2(convergence_rk4[:-1, 2] / convergence_rk4[1:, 2])
