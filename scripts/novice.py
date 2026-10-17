# 新手期的界定，依據見 docs/novice-period.md。依新手分組或控制新手的分析都從這裡取門檻；
# 欄名標明界定方式（novice_rd、novice_games），同名不同義的欄位會讓報告的界定對不到程式。

import polars as pl

# Lichess 自 2015 年 6 月起標示 provisional 的門檻；100 只存在一天，作敏感度
RD = 110
RD_ALT = 100
GAMES = 30


def novice_rd(rd, opp_rd):
    """新手對局：任一方對局前重放的 RD 達門檻。"""
    return pl.max_horizontal(rd, opp_rd) >= RD
