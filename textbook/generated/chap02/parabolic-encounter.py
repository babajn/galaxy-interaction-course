def make_parabolic_encounter(masses, d_start, d_peri):
    masses = check_masses(masses)
    if not (np.isfinite(d_start) and np.isfinite(d_peri) and 0.0 < d_peri < d_start):
        raise ValueError("Require finite distances with 0 < d_peri < d_start.")
    mu = G * np.sum(masses)
    ell = np.sqrt(2.0 * mu * d_peri)
    vt = ell / d_start
    vr = -np.sqrt(2.0 * mu / d_start - vt**2)
    return make_initial_state(masses, [d_start, 0.0], [vr, vt])

encounter_initial = make_parabolic_encounter(masses, 60.0, 8.0)
t_enc, h_enc = integrate_leapfrog(acceleration_two_body, encounter_initial, 0.60, 1200, masses)
enc_values = diagnostics(h_enc, masses)
i_peri = np.argmin(enc_values["D"])
