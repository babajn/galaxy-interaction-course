x, y = r0, 0.0
r2 = x**2 + y**2
ax = -G * mass * x / (r2 + a**2)**1.5
ay = -G * mass * y / (r2 + a**2)**1.5
print(f"ax = {ax:.6f}, ay = {ay:.6f} [kpc/Gyr^2]")
