# Heliyon 模型複製的共用部分：玩家日的樣本、固定效應 logit 與堆疊模型。
# fixed_effects_churn.py 與 fixed_effects_churn_bootstrap.py 共用。
import numpy as np
import polars as pl
import pyfixest as pf

N_SUB = 20_000
EXP_RAW = ["gap_mean", "gap_sd", "win_rate", "win_streak_rate", "loss_streak_rate"]
EXP_ECAL = ["ecal_mean", "ecal_sd", "win_rate", "win_streak_rate", "loss_streak_rate"]
EXP_SCORE = ["gap_mean", "gap_sd", "score_rate", "win_streak_rate", "loss_streak_rate"]
STAGES = ["s1", "s2", "s3", "s4", "s5"]
SPECS = {
    "raw": EXP_RAW + ["n"],
    "ecal": EXP_ECAL + ["n"],
    "raw_score": EXP_SCORE + ["n"],
    # 不含表現變數：分差經由勝負的部分不被控制，對應隨機配對分析的總效果
    "raw_total": ["gap_mean", "gap_sd", "n"],
    "ecal_total": ["ecal_mean", "ecal_sd", "n"],
    "q_total": ["q_mean", "n"],
    "q": ["q_mean", "win_rate", "win_streak_rate", "loss_streak_rate", "n"],
    # 隨機配對下分差的期望仍依自身等級分在對手群中的位置而定；玩家固定效應吸收不了當日等級分的變動，故另以自身等級分控制
    "raw_elo": EXP_RAW + ["n", "elo_c", "elo_c2"],
    "raw_total_elo": ["gap_mean", "gap_sd", "n", "elo_c", "elo_c2"],
    "q_total_elo": ["q_mean", "n", "elo_c", "elo_c2"],
    # 分差等五個變數在各階段各有一個係數（等價於原文的主效果加交互作用），階段本身以虛擬變數進入
    "raw_stage": [f"{x}_x_{s}" for s in STAGES for x in EXP_RAW] + ["n"] + STAGES[1:],
    "ecal_stage": [f"{x}_x_{s}" for s in STAGES for x in EXP_ECAL] + ["n"] + STAGES[1:],
}


def load():
    """2014 年的玩家日，加上單位換算、遊戲階段與新帳號標記。"""
    pdays = pl.read_parquet("data/processed/player_days.parquet").with_columns(
        # 分差以 100 等級分為單位；期望得分以百分點為單位，與勝率同尺度。對手較強時 gap 為正、ecal 較低
        gap_mean=pl.col("gap_mean") / 100,
        gap_sd=pl.col("gap_sd") / 100,
        ecal_mean=pl.col("ecal_mean") * 100,
        ecal_sd=pl.col("ecal_sd") * 100,
        q_mean=pl.col("q_mean") * 100,
        elo_c=(pl.col("elo_mean") - 1500) / 100,
        elo_c2=((pl.col("elo_mean") - 1500) / 100) ** 2,
        churn=pl.col("churn").cast(pl.Float64),
        n=pl.col("n").cast(pl.Float64),
        # 遊戲階段以當日最後一局的累計局數分段；第一段的上限 30 局即校準所用的新手期門檻
        stage=pl.col("game_no").cut([30, 100, 300, 1000], labels=["s1", "s2", "s3", "s4", "s5"], left_closed=False),
    )
    new_players = pdays.filter(pl.col("first_game_no") == 1)["player"].unique().implode()
    pdays = pdays.with_columns(new_player=pl.col("player").is_in(new_players))
    return pdays


def heliyon_sample(d):
    # 原文只留至少兩個活躍日、且期間內流失狀態有變動的玩家；conditional logit 本來就不使用其他玩家。
    # 某日全無流失（或全流失）時日期的係數發散，該日的觀察對其他係數沒有資訊，與 fixest 的做法相同一併排除，
    # 排除後玩家的條件可能不再成立，故交替篩到不變為止
    while True:
        h = d.height
        d = (
            d.with_columns(nd=pl.len().over("player"), k=pl.col("churn").sum().over("player"))
            .filter(pl.col("nd") >= 2, pl.col("k") > 0, pl.col("k") < pl.col("nd"))
            .with_columns(nd=pl.len().over("day"), k=pl.col("churn").sum().over("day"))
            .filter(pl.col("k") > 0, pl.col("k") < pl.col("nd"))
            .drop("nd", "k")
        )
        if d.height == h:
            break
    return (
        d.sort("player", "day")
        .with_columns([(pl.col("stage") == s).cast(pl.Float64).alias(s) for s in STAGES])
        .with_columns([(pl.col(x) * pl.col(s)).alias(f"{x}_x_{s}") for s in STAGES for x in set(EXP_RAW + EXP_ECAL)])
    )


def ybar_var(d):
    # 流失率的玩家 cluster 變異數，供平均半彈性的 delta method
    g = d.group_by("player").agg(y=pl.col("churn").sum(), n=pl.len())
    yb = g["y"].sum() / g["n"].sum()
    G = g.height
    return yb, float(((g["y"] - yb * g["n"]) ** 2).sum()) / g["n"].sum() ** 2 * G / (G - 1)


def stacked(dl, da, xcols, family):
    # 兩樣本疊起來的完全交互作用模型：玩家與日期的固定效應依樣本分開，係數為「大廳」與「arena 減大廳」。
    # 概似可分離，點估計與分開估計相同；同一玩家在兩樣本的 score 進同一個 cluster，差距的標準誤因此含兩估計值的共變
    st = pl.concat([dl.with_columns(arena=pl.lit(0.0)), da.with_columns(arena=pl.lit(1.0))], how="diagonal").with_columns(
        ps=pl.col("player").cast(pl.String) + "_" + pl.col("arena").cast(pl.String), ds=pl.col("day").cast(pl.String) + "_" + pl.col("arena").cast(pl.String)
    )
    st = st.with_columns([(pl.col(x) * pl.col("arena")).alias(f"{x}__arena") for x in xcols])
    terms = xcols + [f"{x}__arena" for x in xcols]
    pdf = st.select("player", "ps", "ds", "churn", *terms).to_pandas()
    del st
    fml = "churn ~ " + " + ".join(terms) + " | ps + ds"
    # 只用係數與 CRV1 共變異，不保留資料與設計矩陣，否則兩樣本疊起來的模型超過記憶體上限
    lean = dict(lean=True, store_data=False, copy_data=False)
    if family == "logit":
        m = pf.feglm(fml, data=pdf, family="logit", vcov={"CRV1": "player"}, demeaner_backend="rust", **lean)
    else:
        m = pf.feols(fml, data=pdf, vcov={"CRV1": "player"}, demeaner_backend="rust", **lean)
    names = list(m.coef().index)
    return m.coef().to_numpy(), m._vcov, names


def diff_rows(dl, da, spec, family, label):
    xcols = SPECS[spec]
    b, V, names = stacked(dl, da, xcols, family)
    ybl, vyl = ybar_var(dl)
    yba, vya = ybar_var(da)
    out = []
    for x in xcols[:2]:
        i, j = names.index(x), names.index(f"{x}__arena")
        bl, dlt = b[i], b[j]
        ba = bl + dlt
        se_b = np.sqrt(V[j, j])
        if family == "logit":
            cl_, ca_ = 1 - ybl, 1 - yba
            dvy = bl**2 * vyl + ba**2 * vya
        else:
            cl_, ca_ = 1 / ybl, 1 / yba
            dvy = bl**2 * vyl / ybl**4 + ba**2 * vya / yba**4
        # ASE 差 = β_L c_L - (β_L + δ) c_A，對 (β_L, δ) 的梯度
        gr = np.array([cl_ - ca_, -ca_])
        Vs = V[np.ix_([i, j], [i, j])]
        out.append({
            "label": label, "spec": spec, "estimator": "logit_fe" if family == "logit" else "lpm_fe", "term": x,
            "beta_lobby": bl, "beta_arena": ba, "se_lobby": np.sqrt(V[i, i]), "se_arena": np.sqrt(V[i, i] + V[j, j] + 2 * V[i, j]),
            "diff_lobby_minus_arena": -dlt, "diff_se": se_b,
            "ase_lobby": bl * cl_, "ase_arena": ba * ca_, "ase_diff": bl * cl_ - ba * ca_, "ase_diff_se": np.sqrt(gr @ Vs @ gr + dvy),
        })
    return out


def player_subsample(samples):
    # 大廳隨機抽 N_SUB 位玩家，arena 玩家數本來就少，全數保留
    lob_players = samples[False]["player"].unique().sort().sample(N_SUB, seed=1)
    return {False: heliyon_sample(samples[False].filter(pl.col("player").is_in(lob_players.implode()))), True: samples[True]}
