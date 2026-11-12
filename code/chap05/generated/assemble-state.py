def make_initial_state(masses, scales, rds, rmaxs, counts, spins,
                       d_start, d_peri, b, seed, n_cdf=4096):
    check_model(masses, scales, b)
    for values in (rds, rmaxs, counts, spins):
        if np.shape(values) != (2,):
            raise ValueError("Each disk parameter array must have two entries.")
    centers = make_zero_energy_encounter(masses, d_start, d_peri, b)
    rng = np.random.default_rng(seed)
    disks, radii = [], []
    for k in range(2):
        local, radius, _ = make_disk(counts[k], masses[k], scales[k], rds[k], rmaxs[k],
                                     spins[k], rng, n_cdf)
        disks.append(place_disk(local, centers[k]))
        radii.append(radius)
    state0 = np.vstack((centers, disks[0], disks[1]))
    origin = np.repeat(np.arange(2), counts)
    return state0, origin, np.concatenate(radii)
