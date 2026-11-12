step_counts = [3000, 6000, 12000, 24000]
resolution_runs, resolution_rows = [], []
for count in step_counts:
    tt, hh, ct, cc = integrate_leapfrog(acceleration_system, state0, t_end, count,
                                       masses, scales, b, count // 150)
    vv = center_diagnostics(cc, masses, b)
    ff = outer_fractions(hh[-1:], origin, threshold)[0]
    resolution_runs.append(hh)
    resolution_rows.append([count, t_end / count, np.min(vv["D"]),
                            *center_errors(ct, cc, masses, b), *ff])
self_rows = []
for j in range(len(step_counts) - 1):
    difference = resolution_runs[j][:, :, :2] - resolution_runs[j + 1][:, :, :2]
    delta_center = np.max(np.linalg.norm(difference[:, :2], axis=2)) / d_start
    delta_tracer = np.max(np.linalg.norm(difference[:, 2:], axis=2)) / rds[0]
    rms_tracer = np.sqrt(np.mean(np.sum(difference[:, 2:]**2, axis=2))) / rds[0]
    self_rows.append([step_counts[j], step_counts[j+1], t_end / step_counts[j],
                      delta_center, delta_tracer, rms_tracer])
self_convergence = np.asarray(self_rows)
self_ratios = self_convergence[:-1, 4] / self_convergence[1:, 4]
