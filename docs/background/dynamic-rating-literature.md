# 動態評分與情境對戰模型

以下依原始論文核對資料與評估設計；不將各文的模型表現跨資料直接排名。

| 文獻 | 實際資料與模型 | 預測證據及界線 |
|---|---|---|
| Coulom（2008），Whole-History Rating | KGS 圍棋。收斂實驗使用 rated 對局歷史；預測比較使用無讓子、固定 komi 的 rated 對局。以 dynamic Bradley–Terry、Wiener process prior 與 Newton’s method 求整段 rating history 的 MAP。 | 早期資料調整參數、後期資料測試，採 incremental updates 預測對局結果，並非僅回顧歷史排名。原文使用二元勝負 formulation，評估主要是 accuracy；不直接提供西洋棋勝和負的 calibration 證據。來源：[作者原文](https://www.remi-coulom.fr/WHR/WHR.pdf)，模型、Incremental Updates 與 Prediction Ability 小節。 |
| Dangauthier、Herbrich、Minka、Graepel（2007），TrueSkill Through Time | ChessBase 歷史西洋棋對局，選用 1850 至 2006 年資料。以 latent skill time series、Gaussian dynamics 與 expectation propagation 做 smoothing；TTT-D 加入 player-specific draw margin。 | 原文以全歷史 factor graph 重建 skill curves，使用前後年份資訊；模型比較採 log-evidence，未提供按時間保留未來對局的 test。因此其歷史實力推論不能直接視為賽前預測驗證。來源：[NeurIPS 原文](https://papers.nips.cc/paper_files/paper/2007/file/9f53d83ec0691550f7d2507d57f4f5a2-Paper.pdf)，Models for Ranking through Time 與 Experiments and Results。 |
| Chen、Joachims（2016），Predicting Matchups and Preferences in Context | 實證含 ATP／WTA 網球、Aligulac 的 StarCraft II 職業賽，以及由 Food、Tijuana restaurant、DePaul movie rating 轉出的 pairwise preferences；亦使用 synthetic data。將 player 與 context features 映射為 blade/chest vectors，以 logistic probability 表達可有 intransitivity 的對戰關係。 | 網球以較早年份 train／validate、較晚年份 test；StarCraft II 與 preference 採 random split。網球 features 包含 betting odds，不能把全部增益歸因於近期戰績或 context；原模型預測二元選擇，亦非 latent skill dynamics model。來源：[作者原文](https://www.cs.cornell.edu/people/tj/publications/chen_joachims_16b.pdf)，模型、Real-World Datasets 與 Empirical results。 |

## 對 Lichess 局前勝和負預測的意義

以下是依上述設計作出的本題判斷。WHR 提供重新利用過去對手關係以更新目前實力的候選方法；TTT 提供同時表達 skill uncertainty 與 draw propensity 的候選方法。若用於局前預測，每個預測時點都只能使用此前已知結果；不能先用完整期間做 smoothing，再把歷史 rating 當作當時可見特徵。

Chen 與 Joachims 提供檢查 context interaction 與 intransitivity 的理由，但其資料、features 與 binary outcome 都不同於本題。是否值得採用，仍須在相同局前資訊、相同時間外驗證及明確的勝和負 probability model 下比較；三篇文獻都未直接證明它們在本專案 Lichess 資料優於既有基準。
