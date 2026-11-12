u_check = np.linspace(0.0, 1.0, 20001)
cdf_rows = []
for count in [32, 64, 128, 256, 512]:
    nodes, values = build_radial_cdf(rd, rmax, count)
    node_error = np.max(np.abs(values - disk_cdf_exact(nodes, rd, rmax)))
    inverted = np.interp(u_check, values, nodes)
    probability_error = np.max(np.abs(disk_cdf_exact(inverted, rd, rmax) - u_check))
    cdf_rows.append([count, rmax / count, node_error, probability_error])
cdf_convergence = np.asarray(cdf_rows)
cdf_orders = np.log2(cdf_convergence[:-1, 3] / cdf_convergence[1:, 3])
