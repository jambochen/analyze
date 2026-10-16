# 以月檔 Event 的時間類別（time_class）分組，與以對局當時等級分所屬的速度類別（speed）
# 分組，兩者差多少：受影響的對局數，與各分析主要估計值的前後對照。
# 修正前的輸出由重跑前手動複製到 outputs/tables/before_speed_fix/，那時的類別欄名為 time_class。

import polars as pl

BEFORE = "outputs/tables/before_speed_fix"
AFTER = "outputs/tables"

g = pl.scan_parquet("data/processed/lichess_games.parquet").select("month", "time_control", "time_class", "speed", "base_seconds")
cross = g.group_by("time_class", "speed").agg(n=pl.len()).sort("time_class", "speed").collect()
cross.write_csv("outputs/tables/speed_fix_effect_cross.csv")
print(cross)

# 後續分析都只用有鐘的對局，比例以此為分母
month = (
    g.filter(pl.col("base_seconds").is_not_null())
    .group_by("month")
    .agg(n=pl.len(), n_mismatch=(pl.col("time_class") != pl.col("speed")).sum())
    .with_columns(share=pl.col("n_mismatch") / pl.col("n"))
    .sort("month")
    .collect()
)
month.write_csv("outputs/tables/speed_fix_effect_month.csv")
with pl.Config(tbl_rows=-1):
    print(month)

tc = (
    g.filter(pl.col("time_class") != pl.col("speed"))
    .group_by("time_control", "time_class", "speed")
    .agg(n=pl.len(), first_month=pl.col("month").min(), last_month=pl.col("month").max())
    .sort("n", descending=True)
    .collect()
)
tc.write_csv("outputs/tables/speed_fix_effect_time_control.csv")
print(tc.head(20))

# 每張表要比較的量與其標準誤（無則 None）；其餘的字串與布林欄即為列的鍵
TABLES = {
    "calibration_coef": [("est", "se_dyadic")],
    "calibration_tests": [("wald_dyadic", None), ("p_dyadic_holm", None)],
    "calibration_cox": [("slope", "slope_se"), ("intercept", "intercept_se")],
    "calibration_prediction": [("diff_vs_elo", "diff_se_dyadic")],
    "sessions_threshold": [("threshold_seconds", None), ("ratio_threshold", None)],
    "sessions_rates": [("continue_session", None), ("leave_8h", None), ("churn_14d", None), ("tour_left", None)],
    "match_quality_leave_ri": [("slope", "null_sd"), ("p_holm", None)],
    "match_quality_leave_clogit": [("coef", "se_cluster_tournament")],
    "match_quality_leave_fe": [("coef", "se_cluster_player")],
    "match_quality_duration": [("coef", "se_hetero")],
    "match_quality_duration_leave": [("coef_q", "se_q")],
    "eomm_assumption_wald": [("stat", None), ("p", None)],
    "eomm_assumption_c": [("est", "se")],
    "eomm_assumption_coef": [("coef", "se_cluster_player")],
    "eomm_assumption_bootstrap": [("delta", "delta_method_se")],
    "eomm_assumption_ovb": [("ame_q", "ame_q_se"), ("k_explain_away", None)],
    "player_days_sample": [("player_days", None), ("churn_rate", None)],
    "fixed_effects_churn_coef": [("est", "se"), ("ase", "ase_se")],
    "fixed_effects_churn_diff": [("diff_lobby_minus_arena", "diff_se")],
    "fixed_effects_churn_hausman": [("chi2", None), ("p", None)],
    "fixed_effects_churn_stage_tests": [("chi2", None), ("p", None)],
}

rows = []
for name, qs in TABLES.items():
    b = pl.read_csv(f"{BEFORE}/{name}.csv", infer_schema_length=None)
    b = b.rename({"time_class": "speed"}) if "time_class" in b.columns else b
    a = pl.read_csv(f"{AFTER}/{name}.csv", infer_schema_length=None)
    keys = [c for c in a.columns if a.schema[c] in (pl.String, pl.Boolean)]
    m = b.join(a, on=keys, how="full", suffix="_after", nulls_equal=True, validate="1:1", coalesce=True)
    key = pl.concat_str([pl.format("{}={}", pl.lit(k), pl.col(k).cast(pl.String).fill_null("")) for k in keys], separator="; ")
    for q, se in qs:
        rows.append(m.select(
            table=pl.lit(name), key=key, quantity=pl.lit(q),
            before=pl.col(q).cast(pl.Float64), after=pl.col(f"{q}_after").cast(pl.Float64),
            se_before=pl.col(se).cast(pl.Float64) if se else pl.lit(None, pl.Float64),
        ))
est = pl.concat(rows).with_columns(diff=pl.col("after") - pl.col("before")).with_columns(diff_in_se=pl.col("diff") / pl.col("se_before"))
est.write_csv("outputs/tables/speed_fix_effect_estimates.csv")

# 結論是否改變看三件事：符號、以 0.05 判的顯著與否（有標準誤或 p 值者），以及差距相對於標準誤的大小
sig = lambda c: pl.when(pl.col("quantity").is_in(["p", "p_holm", "p_dyadic_holm"])).then(pl.col(c) < 0.05).otherwise((pl.col(c) / pl.col("se_before")).abs() > 1.96)
summary = (
    est.with_columns(
        sign_flip=(pl.col("before").sign() * pl.col("after").sign()) < 0,
        sig_change=sig("before") != sig("after"),
    )
    .group_by("table", "quantity")
    .agg(
        rows=pl.len(),
        missing_before=pl.col("before").is_null().sum(),
        missing_after=pl.col("after").is_null().sum(),
        max_abs_diff=pl.col("diff").abs().max(),
        max_abs_diff_in_se=pl.col("diff_in_se").abs().max(),
        median_abs_diff_in_se=pl.col("diff_in_se").abs().median(),
        sign_flips=pl.col("sign_flip").sum(),
        sig_changes=pl.col("sig_change").sum(),
    )
    .sort("table", "quantity")
)
summary.write_csv("outputs/tables/speed_fix_effect_summary.csv")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(summary)
