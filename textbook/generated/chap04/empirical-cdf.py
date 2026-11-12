def empirical_cdf_distance(radius, rd, rmax):
    sorted_radius = np.sort(np.asarray(radius))
    n = len(sorted_radius)
    target = disk_cdf_exact(sorted_radius, rd, rmax)
    right = np.arange(1, n + 1) / n
    left = np.arange(n) / n
    return max(np.max(np.abs(right - target)), np.max(np.abs(left - target)))
