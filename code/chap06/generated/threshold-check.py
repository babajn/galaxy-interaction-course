threshold_rows = []
for cut in [20.0, 30.0, 40.0]:
    f_pp = outer_fractions(base["history"][base["index"]:base["index"]+1], base["origin"], cut)[0]
    f_pr = outer_fractions(retro["history"][retro["index"]:retro["index"]+1], retro["origin"], cut)[0]
    threshold_rows.append([cut, f_pp[1], f_pr[1], f_pp[1] - f_pr[1]])
