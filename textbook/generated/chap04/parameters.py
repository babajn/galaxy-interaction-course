KM_S_TO_KPC_GYR = 1.0227121650537077
G = 4.30091e-6 * KM_S_TO_KPC_GYR**2  # kpc^3 / (Msun Gyr^2)
mass, a = 1.0e11, 3.0              # Plummer球の質量 [Msun] とスケール長 [kpc]
rd, rmax = 3.0, 15.0               # 粒子分布のスケール長と初期外縁 [kpc]
n_particles = 1000
n_cdf = 4096                      # CDF表の区間数。分点数は4097です。
seed, spin = 42, +1                # +1は反時計回り、-1は時計回りです。
t_end, n_steps = 0.60, 3000        # [Gyr]。dt = 0.2 Myrです。
save_every = 10                    # 時間刻みではなく、履歴の保存間隔です。
dt = t_end / n_steps
