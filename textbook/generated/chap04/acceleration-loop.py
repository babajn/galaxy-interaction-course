def acceleration_plummer_loop(pos, mass, a):
    acc = np.empty_like(pos)
    for i in range(len(pos)):
        x, y = pos[i]
        r2 = x**2 + y**2
        acc[i, 0] = -G * mass * x / (r2 + a**2)**1.5
        acc[i, 1] = -G * mass * y / (r2 + a**2)**1.5
    return acc
