# 可行性實測：arena 上一局的對手是隨機指派的，能否據此估計「輸棋對下一局表現」的效果。
# 上一局的期望得分 e（color=False）扣掉期望值 e_bar 後只來自隨機配對，與玩家當天狀態無關。
# 看三件事：它對上一局輸棋的預測力（夠不夠強）、對下一局表現殘差的效果與標準誤（精度夠不夠）、
# 對是否還有下一局的效果（選擇有多大）。下一局限同一場錦標賽，對手同樣是隨機的，基準不受自選混淆。

import sys

sys.path.insert(0, "scripts")
import polars as pl
import pyfixest as pf

from arena_batches import player_sample
from calibrated import probs

d = player_sample().filter(pl.col("layer") != "first_single", pl.col("e_bar").is_not_null())
# 表現的基準要含白方優勢，與匹配品質用的 e（不含）分開
d = probs(d.drop("p_win", "p_draw", "p_loss", strict=False).rename({"e": "e_nc"}), color=True).rename({"e_cal": "e_col", "e_nc": "e"})
d = (
    d.sort("player", "tournament_id", "start", "game_id")
    .with_columns(
        x=(pl.col("e") - pl.col("e_bar")) * 10,
        loss=(pl.col("result") == "loss").cast(pl.Float64),
        resid_next=(pl.col("score") - pl.col("e_col")).shift(-1).over("player", "tournament_id"),
        has_next=pl.col("start").shift(-1).over("player", "tournament_id").is_not_null().cast(pl.Float64),
        gap_next=(pl.col("start").shift(-1) - pl.col("start")).over("player", "tournament_id").dt.total_seconds(),
    )
)
print(d.height, d["has_next"].mean(), d["resid_next"].std())

rows = []
for layer in ["multi", "single", "all"]:
    s = d if layer == "all" else d.filter(pl.col("layer") == layer)
    pdf = s.select("x", "e_bar", "elo_c", "log_games_in_tour", "log_remaining", "novice_games", "white", "loss", "has_next", "resid_next", "speed", "player").to_pandas()
    for y in ["loss", "has_next", "resid_next"]:
        m = pf.feols(f"{y} ~ x + e_bar + elo_c + log_games_in_tour + log_remaining + novice_games + white | speed", pdf, vcov={"CRV1": "player"})
        t = m.tidy()
        rows.append(dict(layer=layer, y=y, coef_per_0_1_e=t.loc["x", "Estimate"], se=t.loc["x", "Std. Error"], n=m._N))
r = pl.DataFrame(rows)
# 輸棋效果 = 縮減式 / 第一階段（Wald 比值），標準誤以 delta method 粗估（忽略兩者的共變）
fs = r.filter(pl.col("y") == "loss").select("layer", fs="coef_per_0_1_e", fs_se="se")
rf = r.filter(pl.col("y") == "resid_next").select("layer", rf="coef_per_0_1_e", rf_se="se")
w = fs.join(rf, on="layer").with_columns(loss_effect=pl.col("rf") / pl.col("fs"), loss_effect_se=(pl.col("rf_se") / pl.col("fs")).abs())
with pl.Config(tbl_cols=-1, tbl_width_chars=200):
    print(r)
    print(w)
r.write_csv("outputs/tables/explore_prev_opponent.csv")
w.write_csv("outputs/tables/explore_prev_opponent_wald.csv")
