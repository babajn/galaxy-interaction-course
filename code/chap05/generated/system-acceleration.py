def acceleration_system(pos, masses, scales, b):
    """先頭2行は銀河中心、それ以外は質量ゼロのテスト粒子です。"""
    acc = np.empty_like(pos)
    acc[:2] = acceleration_centers(pos[:2], masses, b)
    acc[2:] = acceleration_plummer(pos[2:], pos[0], masses[0], scales[0])
    acc[2:] += acceleration_plummer(pos[2:], pos[1], masses[1], scales[1])
    return acc
