peri_values = np.array([5.0, 8.0, 12.0])
b_eff = base_config["b"]
force_ratios = peri_values**3 / (peri_values**2 + b_eff**2)**1.5
softening_rows = np.column_stack((peri_values, np.sqrt(peri_values**2 + b_eff**2), force_ratios))
print("D [kpc], effective distance [kpc], acceleration ratio:")
print(softening_rows)
