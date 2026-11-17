# 使用 Lichess 的預測與行為研究

本地全文及方法評讀見 [論文保存清單](competitive-game-methods.md)；研究目標整理見 [一對一對戰文獻](competitive-game-research-goals.md)。同一平台或同一官方月檔來源，不等於本專案的 arena 清理後樣本。

## 可直接引用的預測比較

Tang、Wang、Jin（2025）的 *Is Elo Rating Reliable? A Study Under Model Misspecification* 使用官方 2014 年 standard 對局，沒有 arena 限制。Table 2 的 Chess 列中，TrueSkill 的平均 cross entropy 最低，其後為 Glicko、Elo2k、Elo；Pairwise 沒有 Chess 結果。該損失以和局記半分，評估期望得分，不能與本報告三類 log loss 直接排名。原文按累積損失曲線挑選超參數，未交代獨立調參 holdout；表中未列不確定性區間。[原文 Table 2、Appendix A、E、F.2](https://arxiv.org/html/2502.10985v1)。

本地 TrueSkill 採相同測試樣本的三類 log loss 與 score MSE，實際結果見 [預測比較](../results/trueskill-prediction.md)；原文數值僅保留作背景，未與本地損失混排。其 Gaussian skill 與 draw 模型的原始來源為 [Herbrich、Minka、Graepel（2006）](https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/NIPS2006_0688.pdf)。

本地 [PDF](../../ref/competitive-games/tang.pdf) 與 [文字](../../ref/competitive-games/tang.txt) 已保存；下載腳本 `scripts/fetch/competitive_papers.sh` 固定版本，`ref/competitive-games/manifest.json` 保存雜湊。`scripts/literature_rating_report.py` 摘錄 Table 2 至 `outputs/tables/literature_rating_report.csv`，明標為引用結果。

## 其他實際使用 Lichess 的研究

| 論文 | 資料關係與方法 | 在本題的用途 |
|---|---|---|
| [Gee 等（2025）](https://arxiv.org/html/2503.21713v2) | 抽取不同評分群體的玩家歷史；correlated random intercept/slope 與評分軌跡檢查 | 本專案已完成雙方 ordinal 延伸及時間外比較，結果見 [玩家差異模型](../results/hierarchical-prediction.md)；原文未作同條件的三類損失比較 |
| [Chowdhary 等（2023）](https://www.nature.com/articles/s41598-023-27735-9) | 官方 2013 至 2016 年資料，聚焦 Blitz；序列重排、連勝連敗與開局多樣性 | 描述行為結構；不提供局前模型預測勝出的證據。控制對手及評分後的序列檢查，不能直接以無條件重排替代 |
| [Sanjaya 等（2022）](https://doi.org/10.3390/a15050152) | 官方多個月份的 Lichess 資料；按 rating 分箱的 payoff matrix、Nash clustering 與循環計數 | 描述 non-transitivity；未證明個人相剋特徵可提高未來對局預測 |
| [Pav，Inferring Piece Value in Chess and Chess Variants](https://arxiv.org/html/2509.04691) | 官方 2013 至 2014 年資料；評分校準、measurement error 與局中棋子價值 | 可補充評分不確定性的動機；主題與局前勝和負預測不同，沒有本題可直接套用的預測改善證據 |

學習與經驗多樣性的研究已收於研究目標筆記，其全文評讀未完成，不據摘要決定本報告的方法。
