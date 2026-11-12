def conservation_errors(times, history, masses):
    values = diagnostics(history, masses)
    d_initial = values["D"][0]
    vc_initial = np.sqrt(G * np.sum(masses) / d_initial)
    e_scale = G * masses[0] * masses[1] / d_initial
    p_scale = np.sum(masses) * vc_initial
    r_cm_exact = values["Rcm"][0] + times[:, None] * values["P"][0] / np.sum(masses)
    err_e = np.max(np.abs(values["E"] - values["E"][0])) / e_scale
    l_scale = abs(values["L"][0])
    if l_scale == 0.0:
        raise ValueError("This angular-momentum normalization requires nonzero L(0).")
    err_l = np.max(np.abs(values["L"] - values["L"][0])) / l_scale
    err_p = np.max(np.linalg.norm(values["P"] - values["P"][0], axis=1)) / p_scale
    err_cm = np.max(np.linalg.norm(values["Rcm"] - r_cm_exact, axis=1)) / d_initial
    return np.array([err_e, err_l, err_p, err_cm])
