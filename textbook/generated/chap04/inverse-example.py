# F -> R と表の役割を交換します。cdfは狭義単調増加でなければなりません。
u_demo = 0.5
radius_demo = linear_interp_one(u_demo, cdf, radius_grid)
radius_numpy = np.interp(u_demo, cdf, radius_grid)
print(f"Radius at CDF=0.5: manual={radius_demo:.9f}, NumPy={radius_numpy:.9f} kpc")
