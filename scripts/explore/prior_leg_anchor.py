import sys

import polars as pl

sys.path.insert(0, "scripts")
import stratum

# clean.py 對不適用者填 0，壓力指標須限定確有前一航段者才有意義。
b = stratum.load().filter("prior_leg_available")

# 排定周轉是班表給的緩衝，上一段實際晚到吃掉它。剩餘緩衝即本班承受的壓力，
# 兩個輸入皆外生：排定值來自班表，上一段實際到達由 gate 保證早於本班的
# info_cutoff_utc。
b = b.with_columns(
    (pl.col("prior_leg_turnaround_min") - pl.col("prior_leg_arr_delay_min")).alias("slack")
)

# 過夜停場的排定周轉長達數百分鐘，壓力無從談起，且會主導連續指標的尺度。
b = b.filter(pl.col("prior_leg_turnaround_min") <= 240)

# 剩餘緩衝恆為正：緩衝若真被吃穿，飛機尚未到場，本班不可能由同一架執飛，
# 且發布者的 gate 另要求上一段實際到達早於本班 cutoff。故壓力無法以「緩衝
# 耗盡」界定，改取當日轉接樣本中緩衝最低的五分之一。
cut = b["slack"].quantile(0.2)
b = b.with_columns((pl.col("slack") <= cut).cast(pl.Float64).alias("squeezed"))
print("當日轉接樣本", b.height, "緩衝門檻", cut, "低於門檻者比例", round(float(b["squeezed"].mean()), 4))

# prior_leg_arr_delay_min 與 slack 機械相關，prior_leg_available 在此恆為真，兩者皆不進控制。
wx = [c for c in stratum.WX if c not in ("prior_leg_arr_delay_min", "prior_leg_available")]
b = b.drop_nulls(wx)

# 緩衝被吃掉時本班延誤已有外部成因，late-aircraft 欄應吸收之。以它確認共變量本身有效。
print(b.group_by("squeezed").agg(
    pl.len().alias("n"),
    pl.col("y1_attributed").mean().alias("y1"),
    (pl.col("reported_late_aircraft_delay_min") > 0).mean().alias("late_ac"),
).sort("squeezed"))

carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != stratum.REF)
# 主效應與交互同時入模：主效應是該公司的水準差異，交互才是對共變量的反應差異。
extra = [pl.col("squeezed")] + [
    ((pl.col("operating_carrier") == c).cast(pl.Float64) * pl.col("squeezed")).alias(f"sq_x_{c}") for c in carriers
]

rows = []
for col, label in [("y1_attributed", "y1"), ("y3_wx_vs_nas", "y3")]:
    out = stratum.fit(b, col, controls=wx, extra=extra).with_columns(pl.lit(label).alias("y"))
    print(out.filter(pl.col("term").str.starts_with("sq_x_")).sort("coef", descending=True))
    rows.append(out)

pl.concat(rows).write_csv("outputs/tables/prior_leg_anchor.csv")
