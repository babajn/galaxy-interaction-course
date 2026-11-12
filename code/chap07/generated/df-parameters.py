df_params = dict(m_host=1.0e11, a_host=3.0, m_sat=1.0e10, a_sat=1.5,
                 ln_lambda=float(np.log(11.0)), friction=True)
df_r0, df_t_end, df_n_steps = 20.0, 1.50, 1500
df_vc = circular_speed_plummer(df_r0, df_params["m_host"], df_params["a_host"])
df_initial = np.array([[df_r0, 0.0, 0.0, df_vc]])
