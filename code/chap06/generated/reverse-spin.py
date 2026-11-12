retro_config = dict(base_config, spins=np.array([1, -1]))
runs["PR"] = run_case(retro_config)
retro = runs["PR"]
rows2 = 2 + np.flatnonzero(base["origin"] == 1)
manual = base["initial"].copy()
manual[rows2, 2:] = 2.0 * manual[1, 2:] - manual[rows2, 2:]
assert np.allclose(manual, retro["initial"], rtol=0.0, atol=1e-12)
assert np.array_equal(base["center_history"], retro["center_history"])
rows1 = 2 + np.flatnonzero(base["origin"] == 0)
assert np.array_equal(base["history"][:, rows1], retro["history"][:, rows1])
