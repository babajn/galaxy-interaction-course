x, y = pos0[0]
r2 = x**2 + y**2
ax = -mu * x / r2**1.5
ay = -mu * y / r2**1.5
print(f"ax = {ax:.6f}, ay = {ay:.6f} [kpc/Gyr^2]")
