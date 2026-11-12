resolution_rows = []
for name in ["PP", "P5", "Q0.1"]:
    coarse = runs[name]
    cfg = coarse["config"]
    args = (cfg["masses"], cfg["scales"], cfg["b"])
    tt, hh, ct, ch = integrate_leapfrog(acceleration_system, coarse["initial"], t_end, 24000, *args, 80)
    assert np.allclose(tt, coarse["times"], rtol=0.0, atol=1e-14)
    difference = hh[:, 2:, :2] - coarse["history"][:, 2:, :2]
    max_difference = np.max(np.linalg.norm(difference, axis=2)) / cfg["rds"][0]
    rms_difference = np.sqrt(np.mean(np.sum(difference**2, axis=2))) / cfg["rds"][0]
    fine_f = outer_fractions(hh[coarse["index"]:coarse["index"]+1], coarse["origin"], threshold)[0]
    resolution_rows.append(dict(name=name, max_difference=max_difference, rms_difference=rms_difference,
                                coarse_f=coarse["fractions"][coarse["index"]], fine_f=fine_f,
                                fine_energy_error=center_errors(ct, ch, cfg["masses"], cfg["b"])[0]))
    del hh, ch, difference
