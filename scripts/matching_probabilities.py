# 替代對手的預測與核對共用，完整套用已保存的近況模型。
import numpy as np
import polars as pl

import novice
import ordinal
import temporal_models as tm


def measures(w, b, th, beta):
    """候選對手的局前指標；近況皆缺時回退基準，僅黑方有近況則反向計分。"""
    er, eb, rw, rb = w[:, 0], b[:, 0], w[:, 1], b[:, 1]
    d = pl.DataFrame({"elo": er, "opp_elo": eb, "rd": rw, "opp_rd": rb}).with_columns(
        xr=((pl.col("elo") + pl.col("opp_elo")) / 2 - 1500) / 100,
        xd=(pl.col("elo") - pl.col("opp_elo")) / 100,
        gxd=ordinal.g("rd", "opp_rd") * (pl.col("elo") - pl.col("opp_elo")) / 100,
        sign=pl.lit(1.0),
    )
    d = tm.offset(d, th, "Lichess")
    eta, width = d["eta"].to_numpy(), d["width"].to_numpy()
    delta = beta[0] * (w[:, 2] - b[:, 2]) + beta[1] * (w[:, 3] - b[:, 3]) + beta[2] * (b[:, 4] - w[:, 4]) + beta[3] * (w[:, 5] - b[:, 5])
    p = ordinal.probs(eta, width)
    recent = ordinal.probs(eta + delta, width)
    e = p[:, 2] + p[:, 1] / 2
    es = recent[:, 2] + recent[:, 1] / 2
    return np.column_stack([abs(er - eb), abs(e - .5), abs(es - .5), abs(er - eb) >= 200, abs(er - eb) >= 400,
                            abs(e - .5) >= .25, abs(es - .5) >= .25, (rw >= novice.RD) | (rb >= novice.RD),
                            (w[:, 4] == 1) & (b[:, 4] == 1)])

