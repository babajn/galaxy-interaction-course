def surface_number_profile(radius, edges):
    """全粒子数で規格化した、円環ごとの数面密度 [kpc^-2]。"""
    counts, _ = np.histogram(radius, bins=edges)
    area = np.pi * (edges[1:]**2 - edges[:-1]**2)
    density = counts / (len(radius) * area)
    return density, counts, area
