KM_S_TO_KPC_GYR = 1.0227121650537077
G = 4.30091e-6 * KM_S_TO_KPC_GYR**2  # kpc^3 / (Msun Gyr^2)
mass = 1.0e11                      # 中心質量 [Msun]
r0 = 10.0                          # 初期半径 [kpc]
mu = G * mass
vc = np.sqrt(mu / r0)               # 円運動速度 [kpc/Gyr]
period = 2.0 * np.pi * r0 / vc       # 軌道周期 [Gyr]
pos0 = np.array([[r0, 0.0]])         # shape = (1, 2)
vel0 = np.array([[0.0, vc]])         # shape = (1, 2)
state0 = np.column_stack((pos0, vel0))  # RK4用：[x, y, vx, vy]
n_steps = 200
t_end = period
dt = t_end / n_steps
