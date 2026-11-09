#import "theme.typ": *
#import "@preview/fletcher:0.5.8": diagram, node, edge
#import "@preview/cetz:0.4.2"
#import "@preview/oxifmt:1.0.0": strfmt

#show: deck.with(config-info(
  title: title-font[競技對局的預測與決策],
  subtitle: [分析過程報告｜20 分鐘],
  author: [陳力福、陳政博、李宜哲],
))

#let tab(name) = csv("../outputs/tables/" + name + ".csv", row-type: dictionary)
#let num(x) = strfmt("{}", int(float(x)), fmt-thousands-separator: ",")
#let f0(x) = strfmt("{:.0}", float(x))
#let f2(x) = strfmt("{:.2}", float(x))
#let f3(x) = strfmt("{:.3}", float(x))
#let f4(x) = strfmt("{:.4}", float(x))
#let pct(x) = strfmt("{:.1}", 100 * float(x)) + "%"
#let pv(x) = { let v = float(x); if v < 0.001 [< 0.001] else [#f3(v)] }
#let fig(name, ..args) = image("../outputs/figures/" + name + ".svg", ..args)
#let flow = (node-stroke: 0.6pt + slate, node-inset: 0.6em, node-corner-radius: 3pt, spacing: (2.2em, 1.6em), edge-stroke: 0.6pt + slate)
#let rule-top = table.hline(stroke: 1pt + slate)
#let rule-mid = table.hline(stroke: 0.5pt + rule)
#let speeds = ("Bullet", "Blitz", "Classical")
#let two(a, b) = stack(spacing: 0.35em, a, b)
#let logit = math.op("logit")
#let sigmoid = math.op("sigmoid")

#let base-tests = tab("expected_score_baselines_tests")
#let lrt-size = tab("baseline_slides_lrt_size")
#let perm = tab("arena_random_pairing_permutation")
#let ri = tab("arena_crossover_ri").filter(r => r.null == "main" and r.stat == "beta_lag1" and r.weighting == "player")
#let contrast = tab("arena_crossover_contrast").filter(r => r.lag == "1" and r.sample == "multi" and r.effect == "direct")
#let rf-coef = tab("rating_filter_error_coef").filter(r => r.model == "dr")
#let pred = tab("state_adjusted_probs_prediction").filter(r => r.group == "all" and r.metric == "log_loss" and r.vs == "baseline" and r.model == "opp_sym")
#let pv-log = tab("prospective_validation_log_score").filter(r => r.metric == "log_loss")
#let pv-total = tab("prospective_validation_total_effect").find(r => r.speed == "Blitz")
#let posthoc = tab("prospective_validation_posthoc")
#let nov-cox = tab("novice_period_cox")
#let ri-all = tab("arena_crossover_ri").filter(r => r.null == "main")
#let steps = tab("state_adjusted_probs_steps")
#let pred-all = tab("state_adjusted_probs_prediction").filter(r => r.metric == "log_loss" and r.vs == "baseline")
#let mark(x) = if x == "1.0" or x == "true" [保留] else [不保留]
#let nb-tests = tab("novice_break_tests")
#let channel = tab("next_opponent_channel")
#let lobby-steps = tab("state_adjusted_lobby_steps").filter(r => r.gap_min == "60")
#let lobby-pred = tab("state_adjusted_lobby_prediction").filter(r => r.gap_min == "60" and r.metric == "log_loss")

#let temporal-periods = tab("temporal_slides_periods")
#title-slide()

== 本次的分析問題

#table(columns: (auto, 1fr), rule-top,
[問題], [分析與診斷], rule-mid,
[預測基礎], [等級分、近期戰績與時間外驗證],
[效果如何解讀], [配對管道、續下選擇與推論界線],
[平台與玩家用途], [替換配對的取捨、近期表現提醒],
[適用條件], [首局、開局間隔與資料缺失], rule-top,
)

Lichess：線上西洋棋。Bullet、Blitz、Classical 為不同速度，各自估計。

2014 年探索與流程檢查；2015 年原登記驗證另列。

各條分析路線分別檢查方法假設，據此決定後續問題。


== 等級分與不確定性

#text(size: 0.8em, fill: mute)[建模｜出發點：Glickman 的動態評分 @glickman1999]

#table(columns: (auto, 1fr), rule-top,
[原問題], [從反覆對戰追蹤會變動的實力], rule-mid,
[等級分 $r$], [目前實力的估計；每局後更新],
[RD], [rating deviation：等級分的不確定程度],
[本題資料], [局前分數＋重放得到的局前 RD＋歷史], rule-top,
)

Lichess 的勝、和、負分別記 1、0.5、0。

期望得分 $E = P("勝") + 0.5 P("和")$；尚未給出三類機率。


== 勝和負的有序模型

#text(size: 0.8em, fill: mute)[建模｜主要做法：有序 logistic @mccullagh1980]

$ P("勝") = F(eta-w), quad P("不輸") = F(eta+w) $
$ P("和") = P("不輸") - P("勝"), quad P("負") = 1-P("不輸") $

#table(columns: (auto, 1fr), rule-top,
[$F(z) = 1 / (1+e^(-z))$], [logistic 累積分布], rule-mid,
[$eta$：局前優勢], [變大時，提高勝與不輸的機率],
[$w > 0$：和局帶寬的一半], [變大時，提高和局機率], rule-top,
)

兩條累積 logit 共用優勢項；以三類結果的概似估計。


== 不確定性進入基準

#text(size: 0.8em, fill: mute)[建模｜主要做法：含 RD 的有序 logistic]

$ eta = a(macron(r)) + [beta_1 + beta_2 macron(r) + beta_3 g(sigma)] d $

#table(columns: (auto, 1fr), rule-top,
[$d, macron(r)$], [白方減黑方的分差、雙方平均等級分], rule-mid,
[$a(macron(r))$], [同分時的先手優勢],
[$sigma = sqrt("RD"_W^2 + "RD"_B^2)$], [雙方不確定性；$g(sigma)$ 隨它增大而下降],
[$w$], [隨平均等級分改變的和局帶寬], rule-top,
)

無 RD 項與 Glicko 形式作對照；檢定與預測支持 RD 基準。


== 多局相依的推論

#text(size: 0.8em, fill: mute)[推論｜主要做法：依共享玩家修正變異 @aronow2015]

#grid(columns: (1fr, 1fr), gutter: 1em, align: horizon,
stack(spacing: 0.65em,
[同一玩家反覆出現，局數不等於獨立資訊量],
[$hat(V) = A^(-1) B A^(-1)$：sandwich 變異估計],
[$A$：概似的曲率；$B$：估計方程的變異],
[$B$ 納入共用任一玩家的對局間共變異],
),
text(size: 0.75em, table(columns: (auto, auto, auto), rule-top,
[有效玩家數], [當獨立], [修正後], rule-mid,
..lrt-size.map(r => ([#num(r.effective_players)], [#pct(r.at("naive_0.05"))], [#pct(r.at("adjusted_0.05"))])).flatten(), rule-top,
)),
)

概似比改用加權卡方分布 @rotnitzky1990；表為名目 5% 的模擬拒絕率。


== 新手門檻的診斷

#grid(columns: (1.45fr, 1fr), gutter: 1em, align: horizon,
figure(fig("novice_break_curve", width: 100%), caption: [Elo calibration around the first game with pre-game RD below 110. Bars: 95% intervals]),
text(size: 0.85em, stack(spacing: 0.65em,
[檢查：門檻附近的校準迴歸],
[$logit mu = alpha + beta logit E$],
[$mu$：平均得分；$E$：基準期望],
[校準時 $alpha=0, beta=1$],
[以 $sum (S-mu) X=0$ 配適 @papke1996],
[$S$：得分；$X=(1,logit E)$],
[和局保留半分，不用二元概似],
)),
)

未檢出門檻跳躍；RD 110 作分組約定，建模仍用連續 RD。

== 前局結果與校準偏差

#figure(fig("baseline_slides_after_result", height: 62%), caption: [Calibration intercept of the main baseline after each previous result. Bars: 95% intervals])

截距為正表示高於基準期望；比較前局結果與 RD 分層的偏差。

近況仍有預測資訊；無法直接區分評分追蹤誤差與心理影響。


== 配對機制與因果問題

#text(size: 0.8em, fill: mute)[識別｜檢查：依配對規則建立置換分布 @fisher1935]

#table(columns: (auto, 1fr), rule-top,
[arena], [限時錦標賽；同時空閒者形成配對批次], rule-mid,
[研究的處理], [$h = macron(e)-e$；$e$ 為實際配對的去先手校準期望，$macron(e)$ 為批內機制平均],
[置換的做法], [保留玩家資料，依原規則重新配對，再算統計量],
[虛無分布], [無前局效果時，統計量可由配對偶然性變動的範圍], rule-top,
)

單局批沒有替代配法；小批須保留避開重複對手等規則。

隨機的是對手指派；輸贏與是否續下都不是隨機指派。


== 配對機制的診斷

#figure(table(columns: (auto, auto, auto), rule-top,
[2014 年的批], [雙方等級分相關], [置換分布 95%], rule-mid,
..(("random", [超過 12 人]), ("smart", [12 人以下])).map(((k, lab)) => {
let r = perm.find(x => x.group == "2014|" + k)
(lab, [#f3(r.corr)], [[#f3(r.corr_null_lo), #f3(r.corr_null_hi)]])
}).flatten(), rule-top,
))

- 相關落在置換範圍；小批仍有等待時間與避開重複對手的限制
- Bullet 的局序有不平衡；純隨機子樣本另作檢查
- 純隨機子樣本中，Blitz 的續下者總效果方向相同

診斷用來檢查識別條件；未拒絕不能證明隨機性。

== 配對管道的排查

#table(columns: (auto, 1fr, 1fr), rule-top,
[發現／候選], [檢查], [對分析的影響], rule-mid,
[強對手後的對手較弱], [Blitz 續下者的下一局低於期望], [總效果混合後續配對管道],
[輸得快、較早空閒], [控制前局結果與局長，關係仍在], [不足以說明管道],
[不能再配上局對手], [前對手不在下一批時仍有關係], [不足以說明管道],
[同批其餘成員較弱], [關係集中於下局對手來自原批], [描述上符合批次組成機制], rule-top,
)

主要檢定未確立固定下一局對手後的效果；仍需檢查誰會續下。


== 續下選擇的修剪界限

#figure(text(size: 14pt, cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // 兩組的殘差分布；強對手組續下較少，弱對手組多出的續下者無法辨認，假設在上端或下端修剪
  for (y, lab, n) in ((1.4, [弱對手組：續下較多], 10), (0, [強對手組：續下較少], 8.5)) {
    rect((0, y), (n, y + 0.8), fill: sq-light, stroke: 0.5pt + sq-dark)
    content((-0.3, y + 0.4), lab, anchor: "east")
  }
  rect((8.5, 1.4), (10, 2.2), fill: sq-dark.lighten(20%), stroke: 0.5pt + sq-dark)
  content((9.25, 2.6), text(fill: mute)[假設多出者在上端])
  line((0, -0.4), (10, -0.4), mark: (end: ">"), stroke: 0.5pt + mute)
  content((5, -0.8), text(fill: mute)[依下一局殘差排序])
})))

- 只有續下者有下一局；修剪比例由兩組續下率差決定
- 修剪上下界 @lee2009：多出的續下者全在上端或全在下端，修剪後的差即上下界
- 假設：每位玩家的續下不會因對手更強而增加；群體趨勢無法驗證

界限之內為「兩種對手強弱下都會續下的玩家」的效果。


== 續下選擇的效果界限

#grid(
  columns: (1.3fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  figure(fig("baseline_slides_bounds", width: 100%), caption: [First next game, batches with two or more games, player-weighted. Bars: 95% intervals]),
  stack(
    spacing: 0.8em,
    [Blitz：續下者差異與加權區間不含 0],
    [加權：以續下機率的倒數補足較少被觀察的玩家],
    [修剪上下界：Bullet 與 Blitz 皆含 0],
    [加權依賴可觀測變數足以解釋選擇；修剪依賴個體單調性],
  ),
)

續下者的差異不足以確立「兩種情況都續下者」的因果效果。


== 近況修正的模型

$ eta_"adj" = eta_"base" + f(s_i) - f(s_j) $

#table(
  columns: (auto, 1fr), rule-top,
  [符號], [意義], rule-mid,
  [$eta_"base"$], [校準基準的線性預測式],
  [$s_i, s_j$], [玩家 $i$ 與對手 $j$ 的上一局結果、$Delta r$ 與歷史缺失指標],
  [$Delta r$], [本局與上一局的局前等級分之差，本局開始前已知],
  [$f$], [雙方共用的近況修正函數], rule-top,
)

固定基準作 offset，只估近況係數；$f$ 為近況變數的線性組合。


== 大廳對局的移轉檢查

#text(size: 0.85em, table(columns: (auto, auto, auto), rule-top,
[log loss 改善], [大廳重新估計], [直接套用 arena 係數], rule-mid,
..speeds.map(s => {
let g(m) = f4(-float(lobby-pred.find(r => r.speed == s and r.model == m and r.group == "all").diff))
([#s], [#g("opp_sym")], [#g("arena_model")])
}).flatten(), rule-top,
))

大廳由玩家選對手；近況限同速度、一小時內的上一局。

同一模型形式可重估；arena 的係數直接移植反而較差。


== 時間外驗證的流程

#figure(text(size: 15pt, diagram(
  ..flow, spacing: (2em, 1.5em),
  node((0, 0), two[訓練前段][估計基準與近況], name: <fit>),
  node((2, 0), two[訓練後段][選擇候選項], name: <select>),
  node((0, 1), two[完整訓練段][固定欄位後重估], name: <refit>),
  node((2, 1), two[後續測試段][只評估固定模型], name: <test>),
  edge(<fit>, <select>, "-|>"), edge(<select>, <refit>, "-|>"), edge(<refit>, <test>, "-|>"),
)))

基準也限訓練期；選定候選後重估，最終測試不再選模型。


== 預測比較的判準

#text(size: 0.8em, fill: mute)[驗證｜主要做法：後續對局的對數損失]

$ L = -1/n sum_(k=1)^n log p_k (y_k), quad Delta L = L_"候選" - L_"基準" $

$n$：對局觀測數；$p_k(y_k)$：模型給實際結果的機率。

#table(columns: (auto, 1fr), rule-top,
[損失較低], [對實際結果給較高機率；懲罰過度自信的錯誤], rule-mid,
[內部選擇], [相依修正區間的上界低於零才加入候選],
[最終測試], [固定模型後，比較全體、新手、老手及校準], rule-top,
)

區間條件於已訓練模型，不含重新訓練與選項的變異。


== 近況修正的時間外檢查

#figure(fig("temporal_slides_lichess_recent", height: 64%), caption: [Recent model minus RD baseline. Conditional 95% intervals; lower is better])

Bullet、Blitz 在兩個測試時段均改善。Classical 未保留近況項。


== 新手誤差的交互作用

$x_i = Delta r_i / 100$，$u_i = "RD"_i / 100$；$i$ 為玩家、$j$ 為對手

#table(
  columns: (auto, 1fr), rule-top,
  [候選], [回答的問題], rule-mid,
  [$x_i u_i - x_j u_j$], [分數變動的修正是否隨 RD 改變],
  [$abs(x_i) u_i - abs(x_j) u_j$], [上升與下降是否需要不同斜率], rule-top,
)

- 依序加入，以內部選擇段的新手預測判定
- 保留交互作用時一併保留分數變動項

預測增量無法區分情緒影響（tilt）與等級分追蹤實力的誤差。


== 新手延伸的時間外檢查

#figure(fig("temporal_slides_lichess_novice_extension", height: 64%), caption: [RD extension minus recent model. Conditional 95% intervals; lower is better])

Blitz 兩段及 Bullet 後段有額外改善。零差值表示未選入延伸項。


== 前瞻與時間外的證據

#table(columns: (auto, 1fr), rule-top,
[驗證], [設計與用途], rule-mid,
[2015 年登記驗證], [啟用資料前固定模型、方向與判準；近況預測改善重現],
[完整時間外流程], [已用資料重新按時間估計與選項；檢查流程穩定性],
[新增 RD 交互作用], [看過資料後提出；尚未獲新的確認性證據], rule-top,
)

總效果的重現不解除續下選擇問題；新候選不能借用既有驗證。


== 替換配對的分析設計

#text(size: 0.8em, fill: mute)[配對探索｜主要做法：固定候選集合的最佳化]

#table(columns: (auto, 1fr), rule-top,
[延伸問題], [同批玩家能否配得較不懸殊？], rule-mid,
[固定條件], [玩家、開局時間、執棋顏色與局數],
[可行限制], [每人恰好一局、避免立即重配],
[比較對象], [歷史配法與同批的合法替代配法],
[求解方式], [白黑雙方各為一組，尋找成本最小的完整配對], rule-top,
)

規則歧義或沒有合法替代的批次維持原狀。

結果是同批靜態比較；尚未觀察換對手後的賽果與行為。

== 配對目標的比較

#text(size: 0.8em, fill: mute)[配對探索｜候選目標與共同限制]

#table(columns: (auto, 1fr), rule-top,
[候選目標], [要減少的量], rule-mid,
[平均分差], [全批對局的等級分差總和],
[機率不平衡], [模型期望得分偏離一半的程度],
[尾端優先], [超過指定分差門檻的局數],
[最差配對優先], [批內最大的分差],
[個人限制], [每人相對原對手的分差增量], rule-top,
)

主要目標同為最優時，再減少分差惡化的玩家局與換對手次數。

各配法均檢查平均分差、極端對局與個人分差變化。

== 可替換配對的範圍

#figure(fig("matching_policy_coverage", height: 64%), caption: [Batch feasibility and coverage of alternative matching])

後續多數批次沒有替代空間；開賽批較有改善空間。

全體評估保留無法替換的對局，避免高估可實現的改善。

== 配對目標與代價

#figure(fig("matching_tradeoffs_modifiable", height: 64%), caption: [Objectives and player-game costs within modifiable batches])

平均、尾端與最差配對的改善並不等價。

尾端與最差配對優先仍保留大部分平均改善。

== 個人保護與整體改善

#figure(fig("matching_tradeoffs_all", height: 64%), caption: [Tradeoffs across all games, including unchanged batches])

同樣平均改善可有較少玩家局惡化；個人限制同時約束配對雙方。

分差不增加，仍不保證模型不平衡或其他體驗不惡化。

== 配對結果的分組診斷

#figure(fig("matching_tradeoffs_uncertainty", height: 64%), caption: [Player-game costs grouped by the player's own pre-game RD])

機率目標下，高 RD 玩家局的分差惡化較多。

RD 大使機率接近一半，不等於真實實力接近；等待與動態效果仍待分析。

== 連勝連敗的比較基準

#grid(columns: (1.45fr, 1fr), gutter: 1em, align: horizon,
figure(fig("baseline_slides_streaks_lichess", width: 100%), caption: [After four losses or wins. Mean score minus expected score; 95% intervals]),
stack(spacing: 0.7em,
[只看前四局全負或全勝後的下一局],
[比較實際得分與各模型的期望],
[Elo 高估強方，連敗者多為弱方；校準會改變偏差方向],
),
)

近況修正吸收部分超額；結果不能直接判定配對操控或心理機制。


== 玩家提醒的分析設計

#text(size: 0.8em, fill: mute)[玩家探索｜主要做法：歷史提醒與後續偏差比較]

$ z = (sum_(k in H) (S_k-E_k)) / sqrt(sum_(k in H) V_k) $

$H$：同場前五局；$S_k$：實際得分；$E_k, V_k$：局前基準的得分期望與變異。

#table(columns: (auto, 1fr), rule-top,
[連敗提示], [近期連輸的長度], rule-mid,
[偏差提示], [標準化得分差低於固定門檻],
[等頻率比較], [訓練期選門檻，測試期固定], rule-top,
)

$z$ 是提醒分數，不作常態檢定；當局對手只用於評估。

== 提醒選中的玩家情境

#figure(fig("player_alerts_overlap", height: 64%), caption: [Alert overlap and difficulty of preceding games])

只有連敗提示者，先前對局較難；只有偏差提示者，先前預期較高。

兩種提醒回答不同問題，不能把彼此不重疊直接當成誤報。

== 提醒後的偏差與權重

#figure(fig("player_alerts_evaluation", height: 64%), caption: [Alert frequency and subsequent residuals under two weighting schemes])

後續偏差：實際得分減近況模型期望；提醒與未提醒組分別加權。

高頻玩家影響整體結果；提醒尚未穩定辨識下一局的負偏差。

== 玩家提醒的診斷與界線

#table(columns: (auto, 1fr), rule-top,
[診斷], [對解讀的影響], rule-mid,
[月份分層], [部分負偏差集中於年底，仍有資料口徑疑慮],
[等玩家權重], [降低高頻玩家影響，但兩組玩家組成仍不同],
[歷史與續下條件], [只涵蓋已有足夠歷史且繼續開局者],
[後續輸贏], [含單局隨機性，無法作失常的真值], rule-top,
)

提醒可供解讀戰績；休息是否改善表現仍需介入設計。

== 首局與間隔的分析設計

#text(size: 0.8em, fill: mute)[適用性探索｜主要做法：固定模型與新增情境比較]

#table(columns: (auto, 1fr), rule-top,
[比較模型], [要區分的資訊], rule-mid,
[原固定模型], [既有 RD 與近況預測],
[重新校準], [一般機率修正的收益],
[首局／間隔／合併], [新情境相對重新校準的增量],
[直接比較原模型], [新方案是否真的優於既有做法], rule-top,
)

首局指本場首次出現；間隔指兩次開局之間的時間。

訓練期固定模型與分類，測試期同時檢查局與玩家權重。

== 新情境的時間外比較

#figure(fig("entry_context_recent", height: 64%), caption: [Context models minus the fixed recent model; lower is better])

首局標籤沒有穩定增量；間隔修正隨速度、期間與權重而異。

新方案須同時與重新校準及原固定模型比較。

== 缺少歷史的分層診斷

#figure(fig("entry_context_history", height: 64%), caption: [Interval model versus fixed recent model by history availability])

缺歷史群組在 Blitz 改善、Bullet 惡化；已知間隔者沒有同樣模式。

缺歷史不等於久未遊玩；開局間隔包含上一局局長，不能識別休息效果。

== 適用範圍與排除代價

#figure(fig("entry_context_scope", height: 64%), caption: [Retained coverage and calibration on the same retained samples])

訓練支持範圍由訓練期平均分數與分差的分位數界定。

校準未一致改善；篩選規則仍屬診斷，尚非通用的可信度規則。

== 跨遊戲的資料條件

#table(columns: (auto, 1fr, 1fr), rule-top,
[資料], [能沿用的流程], [限制與取捨], rule-mid,
[OGS 圍棋], [另估基準、近況與 RD 延伸], [只有結束時間，局前歷史未確認],
[AoE2 世紀帝國], [另估基準與近況], [無 RD、僅單月；以累計局數分組], rule-top,
)

兩份樣本皆為勝負結果，有序模型簡化為二元 logistic。

OGS 保留探索定位；AoE2 的改善需檢查跨期間穩定性。


== 外部資料的預測檢查

#grid(columns: (1fr, 1fr), gutter: 1em,
figure(fig("temporal_slides_ogs_recent", width: 100%), caption: [OGS: end-time ordering only]),
figure(fig("temporal_slides_aoe2_recent", width: 100%), caption: [AoE2: one-month snapshot]),
)

兩圖均為近況減校準基準的對數損失差，越低越好。

改善並非各群一致；先辨認資料可用性，再討論能否推廣。


== 目前證據與待解問題

#table(columns: (auto, 1fr, 1fr), rule-top,
[分析路線], [目前證據], [下一步需要], rule-mid,
[近況與機制], [預測增量；因果仍受選擇限制], [新資料與可辨認的機制],
[替換配對], [固定集合內的目標取捨], [隊列事件與動態評估],
[玩家提醒], [對手難度改變戰績解讀], [跨期間穩定性與介入設計],
[適用條件], [缺歷史與權重影響判斷], [新樣本驗證與維護策略], rule-top,
)

新增分析均屬探索，尚待新的確認性驗證。

討論重點：方法假設是否合適，哪些缺口最值得繼續分析。


== 方法出處

#text(size: 11pt, columns(2, gutter: 1.5em, bibliography("refs.bib", title: none, style: "apa")))
