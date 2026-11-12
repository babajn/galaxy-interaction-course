root_rows = []
for count in [32, 64, 128, 256, 512]:
    def residual(radius):
        return half_mass_residual(radius, n_panels=count)
    root, trace = bisection(residual, 0.0, 5.0 * a, root_xtol)
    actual_error = abs(root - r_half_exact)
    root_rows.append([count, root, actual_error, abs(residual(root)), trace[-1, 5]])
root_convergence = np.asarray(root_rows)
root_orders = np.log2(root_convergence[:-1, 2] / root_convergence[1:, 2])
