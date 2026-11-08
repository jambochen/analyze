#import "theme.typ": *
#import "@preview/fletcher:0.5.8": diagram, node, edge
#import "@preview/cetz:0.4.2"
#import "@preview/oxifmt:1.0.0": strfmt

#show: deck.with(config-info(
  title: title-font[等級分與對局結果預測],
  subtitle: [機率校準、時間外驗證與跨遊戲分析進度],
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

#let y2014 = tab("data_overview_scale").find(r => r.year == "2014")
#let speed-n = tab("data_overview_speed")
#let holdout = tab("data_overview_holdout").first()
#let base-tests = tab("expected_score_baselines_tests")
#let lrt-size = tab("baseline_slides_lrt_size")
#let rd-cross = tab("expected_score_slides_rd_cross")
#let batch = tab("arena_random_pairing_batch_size").filter(r => r.period == "2014")
#let single-share = batch.filter(r => r.n_games == "1" and r.first == "false").map(r => int(r.games)).sum() / batch.map(r => int(r.games)).sum()
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

== 資料

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  text(size: 0.8em, table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    rule-top,
    [2014 年], [局數], [其中 arena],
    rule-mid,
    ..speeds.map(s => {
      let a = speed-n.find(r => r.speed == s and r.arena == "true").len
      let o = speed-n.find(r => r.speed == s and r.arena == "false").len
      ([#s], [#num(int(a) + int(o))], [#num(a)])
    }).flatten(),
    rule-top,
  )),
  stack(
    spacing: 0.8em,
    [Lichess 的 rated 對局；玩家 #num(y2014.players) 位],
    [Lichess 建立方法，OGS、AoE2 作外部比較],
    [2015 年 1/1 至 4/19 的 arena：#num(holdout.games) 局，登記的前瞻驗證已完成],
    [欄位：對局前的 `WhiteElo`、`BlackElo`，`Event`、開局時間與 `Result`；不用賽後的 `RatingDiff`],
  ),
)

三種速度類別各有等級分，模型分開估計。

== 前次的結論

#figure(text(size: 0.85em, table(
  columns: (auto, auto),
  align: left,
  rule-top,
  [前次], [結論],
  rule-mid,
  [Elo 的期望得分], [斜率 $hat(beta)$ 皆低於 1：高估強方；截距 $hat(alpha) > 0$：白方優勢],
  [主基準], [加入 RD 的有序 logistic，勝、和、負三類同時建模],
  [留下的問題], [多局相依、新帳號、上一局被分到強弱對手的效果與續下],
  rule-top,
)))

本次：補上留下的問題，修正基準，再以它判斷連勝連敗。

== 分析目標

#define[等級分在哪些情境下不足以預測對局結果？]

#table(
  columns: (auto, 1fr), rule-top,
  [交付對象], [競技遊戲的評分與配對系統分析者], rule-mid,
  [失準診斷], [分差、不確定性與近期戰績下的偏差],
  [預測修正], [勝、和、負機率及時間外預測表現],
  [適用範圍], [不同遊戲、配對來源與新手／老手], rule-top,
)

期望得分將和局記為半分，與勝率分開解讀。

== 分析架構

#figure(text(size: 0.85em, table(
  columns: (auto, auto, auto), rule-top,
  [問題], [主要做法], [輸出], rule-mid,
  [基準失準], [依等級分與 RD 校準], [期望得分與結果機率],
  [近況增量], [訓練期間內選項], [固定測試段的分數差],
  [新手誤差], [RD 與分數變動的交互作用], [新手與老手的預測比較],
  [跨遊戲適用], [各遊戲重新估計], [可用情境與資料限制], rule-top,
)))

RD：等級分的不確定性（rating deviation）。

因果結果與連勝連敗診斷列於附錄。

== 同一玩家的多局不獨立

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  stack(
    spacing: 0.8em,
    [GEE 的 sandwich：$hat(V) = A^(-1) B A^(-1)$ @liang1986],
    [$A$：information；$B$：score 的變異，依群加總],
    [一局兩位玩家：依玩家、對手兩方向分群 @aronow2015],
    [LRT 在相依下為加權卡方和 $sum_j lambda_j chi^2_(1, j)$ @imhof1961],
  ),
  text(size: 0.8em, table(
    columns: (auto, auto, auto, auto),
    align: (right, right, right, right),
    rule-top,
    [玩家數], [有效玩家數], [當獨立], [修正後],
    rule-mid,
    ..lrt-size.map(r => ([#num(r.n_player)], [#num(r.effective_players)], [#pct(r.at("naive_0.05"))], [#pct(r.at("adjusted_0.05"))])).flatten(),
    table.cell(colspan: 4, text(size: 0.85em, fill: mute)[模擬的拒絕率，名目 5%]),
    rule-top,
  )),
)

$lambda_j$ 由 sandwich 估計；預測差值的區間亦處理雙方玩家的相依。

== 校準斜率的分層

#figure(fig("expected_score_slides_strata", height: 66%), caption: [Calibration slope of Elo's expected score by stratum. Bars: 95% intervals])

斜率在每個等級分、來源與時段都低於 1：高估強方不是少數子樣本造成的。

== 三個基準的比較

#figure(table(
  columns: (auto, auto, auto), align: (left, right, right), rule-top,
  [受限模型的檢定], [有序 logistic], [Glicko 形式], rule-mid,
  ..speeds.map(s => {
    let g(m) = base-tests.find(r => r.speed == s and r.sample == "all" and r.restricted == m)
    ([#s], [#pv(g("ordinal").p)], [#pv(g("glicko").p)])
  }).flatten(), rule-top,
))

對立模型：分差斜率隨雙方 RD 改變的有序 logistic。

相依修正的 LRT 拒絕兩個特例，預測另由時間外資料評估。

== 新手期

#grid(
  columns: (1.3fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  fig("expected_score_slides_rdslope", width: 100%),
  stack(
    spacing: 0.8em,
    [新手：任一方 RD ≥ 110，即 Lichess 標示 provisional 的門檻],
    [半數帳號 RD 跌破 110 的局數：#rd-cross.map(r => [#r.speed #r.games 局]).join("、")],
    [新手對局留在樣本，由主基準的 RD 項吸收],
  ),
)

偏差隨 RD 連續變化，門檻處沒有斷點。


== RD 隨已下局數

#figure(fig("expected_score_slides_rd", height: 64%))

RD 幾乎只由已下的局數決定：Lichess 的 RD 不因閒置增加。

== 兩種新手界定

#figure(table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  rule-top,
  [校準斜率], [RD ≥ 110：新手], [老手], [未滿 30 局：新手], [老手],
  rule-mid,
  ..speeds.map(s => {
    let g(t, k) = f3(nov-cox.find(r => r.speed == s and r.stratum_type == t and r.stratum == k).slope)
    ([#s], [#g("def_rd110", "novice")], [#g("def_rd110", "established")], [#g("def_games30", "novice")], [#g("def_games30", "established")])
  }).flatten(),
  rule-top,
))

兩種界定下老手的斜率相近；主分析取系統自己的 RD 門檻。


== 門檻處的斷點

#grid(
  columns: (1.6fr, 1fr),
  gutter: 1em,
  align: horizon,
  figure(fig("novice_break_curve", width: 100%), caption: [Calibration of Elo's $E$ by game relative to the first game with pre-game RD < 110. Bars: 95% intervals]),
  stack(
    spacing: 0.7em,
    [截距、斜率皆為 $a + b k + c dot 1{k >= 0}$],
    [$H_0$：兩個 $c$ 皆為 0；修正的 LRT],
    text(size: 0.8em, table(
      columns: (auto, auto),
      align: (left, right),
      rule-top,
      [], [Holm 校正的 $p$],
      rule-mid,
      ..speeds.map(s => ([#s], [#pv(nb-tests.find(r => r.speed == s).p_holm)])).flatten(),
      rule-top,
    )),
    text(size: 0.8em, fill: mute)[第 −1 局的階梯：使 RD 跌破 110 的局分差較小],
  ),
)

門檻處沒有跳躍：偏差逐局收斂，110 是約定而非轉折點。

== 剛贏、剛輸之後

#figure(fig("baseline_slides_after_result", height: 62%), caption: [Calibration intercept of the main baseline after each previous result. Bars: 95% intervals])

剛贏之後各組皆好於期望；剛輸之後只有 RD 最高的一組差於期望。

== 等級分調得不夠

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  stack(
    spacing: 0.8em,
    [$Delta r$：本局與上一局的對局前等級分之差，本局開始前已知],
    [剛贏之後好於期望；剛輸之後只有 RD 最高的五分之一差於期望],
    [給定上一局結果，$Delta r$ 越大下一局越好],
  ),
  text(size: 0.8em, table(
    columns: (auto, auto),
    align: (left, right),
    rule-top,
    [], [$Delta r$ 每 100 分的效果（等級分）],
    rule-mid,
    ..speeds.map(s => ([#s], [#f0(rf-coef.find(r => r.speed == s).elo_per_100)])).flatten(),
    rule-top,
  )),
)

方向是等級分調得不夠；與 tilt 在本資料分不開。


== 近況修正的模型

$ eta_"adj" = eta_"base" + f(s_i) - f(s_j) $

#table(
  columns: (auto, 1fr), rule-top,
  [符號], [意義], rule-mid,
  [$eta_"base"$], [校準基準的線性預測式],
  [$s_i, s_j$], [雙方上一局結果、等級分變動與歷史缺失指標],
  [$f$], [雙方共用的近況修正係數], rule-top,
)

候選是否保留由訓練期間內的預測比較決定。

== 時間外驗證的流程

#figure(text(size: 15pt, diagram(
  ..flow, spacing: (2em, 1.5em),
  node((0, 0), two[訓練前段][估計基準與近況], name: <fit>),
  node((2, 0), two[訓練後段][選擇候選項], name: <select>),
  node((0, 1), two[完整訓練段][固定欄位後重估], name: <refit>),
  node((2, 1), two[後續測試段][只評估固定模型], name: <test>),
  edge(<fit>, <select>, "-|>"), edge(<select>, <refit>, "-|>"), edge(<refit>, <test>, "-|>"),
)))

主基準與近況模型使用相同的時間邊界。

== Lichess 的時間切分

#figure(text(size: 0.85em, table(
  columns: (auto, auto, auto, auto), rule-top,
  [初次估計], [內部選項], [重估], [最終測試], rule-mid,
  ..temporal-periods.filter(r => r.game == "Lichess").map(r => ([#r.fit], [#r.select], [#r.refit], [#r.test])).flatten(), rule-top,
)))

- 全站資料估計 RD 基準，arena 本場有上一局者估計近況
- 測試段不重疊，訓練段隨時間擴大
- 已用資料的穩健性檢查，不是新的前瞻確認

== 預測比較的判準

$ Delta L = L_"候選" - L_"目前模型" $

#table(
  columns: (auto, 1fr), rule-top,
  [對數損失 log loss], [實際結果的預測機率越高，損失越低], rule-mid,
  [內部選項], [平均損失差的相依修正 95% 區間上界低於零],
  [最終測試], [全體、新手、老手分別比較損失與校準],
  [區間範圍], [條件於已訓練模型，不含訓練與選項的不確定性], rule-top,
)

Lichess 的 Elo 只有期望得分，另以得分誤差比較。

== Lichess 的近況增量

#figure(fig("temporal_slides_lichess_recent", height: 64%), caption: [Recent model minus RD baseline. Conditional 95% intervals; lower is better])

Bullet、Blitz 在兩個測試時段均改善。Classical 未保留近況項。

== 新手誤差的候選

$x_i = Delta r_i / 100$，$u_i = "RD"_i / 100$

#table(
  columns: (auto, 1fr), rule-top,
  [候選], [回答的問題], rule-mid,
  [$x_i u_i - x_j u_j$], [分數變動的修正是否隨 RD 改變],
  [$abs(x_i) u_i - abs(x_j) u_j$], [上升與下降是否需要不同斜率], rule-top,
)

- 依序加入，以內部選擇段的新手預測判定
- 保留交互作用時一併保留分數變動項

探索預測結構，不能據此區分 tilt 與實力追蹤誤差。

== 新手誤差的預測增量

#figure(fig("temporal_slides_lichess_novice_extension", height: 64%), caption: [RD extension minus recent model. Conditional 95% intervals; lower is better])

Blitz 兩段及 Bullet 後段有額外改善。零差值表示未選入延伸項。

== 跨遊戲的資料條件

#figure(text(size: 0.8em, table(
  columns: (auto, auto, auto, auto), rule-top,
  [資料], [結果], [不確定性資訊], [用途與限制], rule-mid,
  [Lichess], [勝、和、負], [重放 RD], [方法建立、arena 與大廳],
  [OGS 圍棋], [勝、負], [重放 RD], [只有結束時間，探索性參考],
  [AoE2 1v1], [勝、負], [局前累計局數], [排隊配對，單月快照], rule-top,
)))

各遊戲分別估計係數，新手定義不強作等價。

== 外部資料的時間切分

#figure(text(size: 0.75em, table(
  columns: (auto, auto, auto, auto, auto), rule-top,
  [遊戲], [初次估計], [內部選項], [重估], [測試], rule-mid,
  ..temporal-periods.filter(r => r.game != "Lichess").map(r => ([#r.game], [#r.fit], [#r.select], [#r.refit], [#r.test])).flatten(), rule-top,
)))

OGS 排除自我對局。AoE2 的歷史須已結束且等級分接續。

== AoE2 的近況增量

#figure(fig("temporal_slides_aoe2_recent", height: 64%), caption: [Recent model minus calibrated baseline. Conditional 95% intervals])

改善集中於新手。老手的區間含零，不解讀為確立的傷害。

== OGS 的近況增量

#figure(fig("temporal_slides_ogs_recent", height: 64%), caption: [End-time ordering only. Recent model minus calibrated baseline; conditional 95% intervals])

近況改善全體與新手。缺少開局時間，不能確認歷史在局前可用。

== OGS 的新手延伸

#figure(fig("temporal_slides_ogs_novice_extension", height: 64%), caption: [End-time ordering only. RD extension minus recent model; conditional 95% intervals])

兩個速度類別都有新手增量，仍限探索性外部參考。

== 驗證證據的區別

#figure(text(size: 0.85em, table(
  columns: (auto, auto, auto), rule-top,
  [證據], [模型與資料], [可支持的結論], rule-mid,
  [登記的前瞻驗證], [固定模型，未見的 2015 年樣本], [登記模型的外推表現],
  [完整時間外流程], [已用資料，訓練內選項], [流程的時間穩定性],
  [新手交互作用], [事後假設，固定後續測試], [探索性預測增量], rule-top,
)))

既有前瞻驗證不能替新增交互作用提供確認性證據。

== 界線與限制

- OGS 缺少開局時間，僅作結束時間排序的探索
- AoE2 只有單月，Elo 期望為代理基準
- 新手延伸尚需未使用資料的確認性驗證
- 區間條件於已訓練模型，重疊訓練段不作獨立實驗
- 預測改善不直接推出配對公平性或留存效果

== 目前的發現

+ RD 校準提供較合適的基準，近況另帶來預測資訊
+ Lichess 的近況增量通過完整時間外檢查，新手仍有可修正的結構
+ 跨遊戲的改善集中於新手，老手須依遊戲與情境評估

交付機率診斷、修正與適用範圍。

== 未完成與下一步

#figure(table(
  columns: (auto, 1fr), rule-top,
  [項目], [下一步], rule-mid,
  [新手延伸], [另留未使用資料，先登記再確認],
  [OGS 資訊可用性], [取得開局時間才能驗證局前歷史],
  [最終報告], [整合預測修正、時間外證據與適用範圍],
  [選做分析], [首局與空檔時間的預測增量], rule-top,
))

停止規則暫停延伸。

== 參考文獻

#text(size: 11pt, columns(2, gutter: 1.5em, bibliography("refs.bib", title: none, style: "apa")))



#show: appendix

== arena 的隨機配對

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  stack(
    spacing: 0.8em,
    [同一時刻空閒的玩家組成一批，批內配對不看等級分],
    [一局一批占 2014 年 arena 局的 #pct(single-share)：只有一種配法，不提供隨機性],
    [批內打亂對手的置換分布，比較雙方等級分的批內相關],
  ),
  text(size: 0.8em, table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    rule-top,
    [2014 年的批], [實際相關], [置換分布 95%],
    rule-mid,
    ..(("random", [超過 12 人]), ("smart", [12 人以下])).map(((k, lab)) => {
      let r = perm.find(x => x.group == "2014|" + k)
      (lab, [#f3(r.corr)], [[#f3(r.corr_null_lo), #f3(r.corr_null_hi)]])
    }).flatten(),
    rule-top,
  )),
)

批內的對手強弱與玩家當下的狀態無關。


== 配對批次的大小

#figure(fig("baseline_slides_batch_size", height: 62%), caption: [2014 arena games by number of games created in the same pairing batch])

開賽第一批之後，多數批只有一局：只有一種配法，不提供隨機性。

== 置換分布的分層

#figure(text(size: 0.75em, table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  rule-top,
  [2014 年的批], [局數], [實際相關], [置換分布 95%],
  rule-mid,
  ..perm.filter(r => r.group.split("|").len() == 3 and r.group.starts-with("2014")).map(r => {
    let (per, rule, k) = r.group.split("|")
    ([#(if rule == "random" [超過 12 人] else [12 人以下])，#k 局], [#num(r.games)], [#f3(r.corr)], [[#f3(r.corr_null_lo), #f3(r.corr_null_hi)]])
  }).flatten(),
  rule-top,
)))

各分層的相關皆落在置換分布之內；12 人以下的批分差略小，見限制。

== 批內重新配對的檢定

#figure(text(size: 15pt, diagram(
  ..flow,
  spacing: (2.4em, 1.2em),
  node((0, 0), [實際的配對], name: <act>),
  edge("-|>"),
  node((1, 0), [統計量 $hat(beta)$], name: <stat>),
  node((0, 1), two[依原規則在批內][重新配對 2000 次], name: <re>),
  edge("-|>"),
  node((1, 1), [虛無分布], name: <null>),
  edge(<stat>, <null>, "-", stroke: (paint: mute, dash: "dashed"), label: text(fill: mute)[比較], label-side: left),
)))

- 處理 $h$：對手比批內候選平均強多少
- $hat(beta)$：下一局殘差對 $h$ 的 score 除以 information，各速度類別分開
- 區間：反轉檢定，取不被拒絕的效果值
- 不依賴模型的誤差分布；前次的批內迴歸依賴各局獨立

只用到「批內配對隨機」這一個假設 @fisher1935。


== 平衡與安慰劑

#figure(table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  rule-top,
  [檢查（Holm 校正的 $p$）], [Bullet], [Blitz],
  rule-mid,
  ..(("balance_elo_c", [自身等級分對 $h$ 的斜率]), ("balance_rd", [自身 RD]), ("balance_novice_games", [是否新帳號]), ("balance_log_games_in_tour", [本場已下局數])).map(((k, lab)) => {
    let g(sp) = ri-all.find(r => r.stat == k and r.weighting == "player" and r.batches == "all" and r.speed == sp).p_holm_diag
    (lab, [#pv(g("Bullet"))], [#pv(g("Blitz"))])
  }).flatten(),
  rule-mid,
  ..(([安慰劑：下一局的 $h$ 對本局殘差],),).map(((lab,)) => {
    let g(sp) = ri-all.find(r => r.stat == "placebo" and r.weighting == "player" and r.batches == "all" and r.speed == sp).p_holm_placebo
    (lab, [#pv(g("Bullet"))], [#pv(g("Blitz"))])
  }).flatten(),
  rule-top,
))

只有 Bullet 的本場已下局數不平衡，符合「優先配對等待最久者」；它在直接效果的模型內。

== 剛被分到強對手之後

#figure(table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  rule-top,
  [], [總效果：每 0.1 的 $h$（等級分）], [重新配對 $p$（Holm）], [直接效果 $p$（Holm）],
  rule-mid,
  ..("Bullet", "Blitz").map(s => {
    let r = ri.find(x => x.speed == s and x.batches == "all")
    let c = contrast.find(x => x.speed == s)
    ([#s], [#f2(r.elo_per_0_1)], [#pv(r.p_holm)], [#pv(c.p_holm)])
  }).flatten(),
  rule-top,
))

- 總效果：殘差取主基準，含「上一局影響下一局對手」的管道
- 直接效果：固定下一局的對手，玩家內的配對 $t$ 檢定
- 只用純隨機規則的批，Blitz 的總效果結論相同

Blitz：下一局低於期望，只延續一局；直接效果方向相同，未能確立。


== 劑量反應

#figure(fig("baseline_slides_dose", height: 62%), caption: [Player-weighted. Bars: 95% intervals])

Blitz 的效果集中在 $h$ 的兩端。

== 各局與子樣本

#figure(text(size: 0.8em, table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, left, right, right, right),
  rule-top,
  [], [加權與批], [第一局], [第二局], [第三局],
  rule-mid,
  ..("Bullet", "Blitz").map(sp => (("player", "all", [玩家，全部批]), ("game", "all", [局，全部批]), ("player", "random_rule", [玩家，純隨機規則])).map(((w, b, lab)) => {
    let g(k) = pv(ri-all.find(r => r.stat == "beta_lag" + str(k) and r.weighting == w and r.batches == b and r.speed == sp).p)
    ([#sp], lab, g(1), g(2), g(3))
  }).flatten()).flatten(),
  rule-top,
)))

#text(size: 0.8em, fill: mute)[重新配對檢定的 $p$，未校正]

Blitz 的第一局在各組合下皆拒絕；第二、三局皆不拒絕。

== 下一局的對手

#figure(table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  rule-top,
  [], [下一局對手等級分（百分）對 $h$ 的斜率], [重新配對 $p$],
  rule-mid,
  ..("Bullet", "Blitz").map(sp => {
    let r = ri-all.find(x => x.stat == "next_opp_elo" and x.weighting == "player" and x.batches == "all" and x.speed == sp)
    ([#sp], [#f2(r.est)], [#pv(r.p)])
  }).flatten(),
  rule-top,
))

被分到強對手之後，下一局的對手較弱：總效果含這條 arena 特有的管道。



== 下一局對手較弱的原因

#figure(text(size: 0.8em, table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  rule-top,
  [對 $h$ 的係數（批內去平均，玩家分群的 95% 區間）], [Bullet], [Blitz],
  rule-mid,
  ..(("step1_score", "score_lag1", [上一局的得分]), ("step1_duration", "log_dur", [上一局局長的對數]), ("step2_h_only", "opp_elo_next", [下一局對手的等級分]), ("step2_with_mediators", "opp_elo_next", [同上，加入上一局結果與局長])).map(((m, y, lab)) => {
    let g(sp) = { let r = channel.find(x => x.speed == sp and x.model == m and x.y == y and x.x == "h_lag1"); [#f2(r.est) #text(size: 0.85em, fill: mute)[[#f2(r.lo), #f2(r.hi)]]] }
    (lab, g("Bullet"), g("Blitz"))
  }).flatten(),
  rule-top,
)))

- 輸得快、較早空閒：強對手使人較常輸，局長反而較長；加入後係數不縮小
- 上一局的對手同批而不重配：對手不在同一批時係數仍遠離 0
- 兩者皆不成立

== 批內其餘成員

#figure(text(size: 0.8em, table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  rule-top,
  [下一局對手等級分對 $h$ 的係數], [Bullet], [Blitz],
  rule-mid,
  ..(("split_opp_from_prev_batch", [下一局的對手來自同一個上一批]), ("split_opp_from_other_batch", [來自其他批])).map(((m, lab)) => {
    let g(sp) = { let r = channel.find(x => x.speed == sp and x.model == m); [#f2(r.est) #text(size: 0.85em, fill: mute)[[#f2(r.lo), #f2(r.hi)]]] }
    (lab, g("Bullet"), g("Blitz"))
  }).flatten(),
  rule-top,
)))

- 同一批的等級分總和固定：自己配到越強，其餘的人平均越弱
- 其餘的人約略同時下完，再落入同一個下一批
- 第三個候選，執行前寫定並訂為最後一個

下一局的管道是配對規則的機械效果；直接效果以下一批的組成控制它。

== 續下率隨對手強弱

#figure(fig("baseline_slides_continue_rate", height: 62%), caption: [Within bins of $macron(e)$; first next game, batches with two or more games, player-weighted. Bars: 95% intervals])

對手越強，同場再下一局的比例越低，且單調：修剪上下界的前提成立。

== 續下的選擇

#figure(text(size: 14pt, cetz.canvas(length: 1cm, {
  import cetz.draw: *
  // 兩組的殘差分布；強對手組續下較少，弱對手組多出的續下者無法辨認，假設在上端或下端修剪
  for (y, lab, n) in ((1.4, [弱對手組：續下較多], 10), (0, [強對手組：續下較少], 8.5)) {
    rect((0, y), (n, y + 0.8), fill: sq-light, stroke: 0.5pt + sq-dark)
    content((-0.3, y + 0.4), lab, anchor: "east")
  }
  rect((8.5, 1.4), (10, 2.2), fill: sq-dark.lighten(20%), stroke: 0.5pt + sq-dark)
  content((9.25, 2.6), text(fill: mute)[多出的續下者])
  line((0, -0.4), (10, -0.4), mark: (end: ">"), stroke: 0.5pt + mute)
  content((5, -0.8), text(fill: mute)[依下一局殘差排序])
})))

- 只有續下者有下一局；強對手使人較少續下
- 修剪上下界 @lee2009：多出的續下者全在上端或全在下端，修剪後的差即上下界
- 假設：對手強弱對續下的影響方向一致；資料上續下率隨 $h$ 單調下降

界限之內為「兩種對手強弱下都會續下的玩家」的效果。

== 續下者之外未能確立

#grid(
  columns: (1.3fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  figure(fig("baseline_slides_bounds", width: 100%), caption: [First next game, batches with two or more games, player-weighted. Bars: 95% intervals]),
  stack(
    spacing: 0.8em,
    [只用續下者與倒數機率加權：Blitz 不含 0],
    [修剪上下界：Bullet 與 Blitz 皆含 0],
    [續下率的差僅數個百分點，殘差分布寬],
  ),
)

Blitz 的效果只在續下者上成立。


== 各局的上下界

#figure(fig("baseline_slides_bounds_lag", height: 62%), caption: [Thick: trimming bounds; thin: 95% interval covering the effect])

之後各局的界限皆含 0，且隨局數變寬。


== 登記模型的選項

#figure(table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  rule-top,
  [加入的項], ..speeds.map(s => [#s]),
  rule-mid,
  ..(("arena", [arena 的整體校正]), ("prev_result", [自身的上一局結果]), ("dr", [自身的 $Delta r$]), ("opp_sym", [對手的近況])).map(((k, lab)) => (lab, ..speeds.map(s => mark(steps.find(r => r.speed == s and r.step == k).kept)))).flatten(),
  rule-top,
))

此設計用下半年選項、全年估計 offset；模型的確認依 2015 年登記驗證。

== 修正後的基準

#grid(
  columns: (1.5fr, 1fr),
  gutter: 1em,
  align: horizon,
  figure(fig("baseline_slides_shift", width: 100%), caption: [Established players. Bars: 95% intervals, delta method with dyadic sandwich]),
  stack(
    spacing: 0.8em,
    [$eta + f(s_"自己") - f(s_"對手")$：雙方共用係數],
    [$f$：上一局結果；Blitz、Classical 另含 $Delta r$],
    [修正量除以主基準的斜率，折成等級分],
  ),
)

修正等於依雙方近況之差平移等級分。


== 新手與老手

#figure(table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  rule-top,
  [log loss 的改善], [全部], [新手], [老手],
  rule-mid,
  ..speeds.map(s => {
    let g(gr) = f4(-float(pred-all.find(r => r.speed == s and r.group == gr and r.model == "opp_sym").diff))
    ([#s], [#g("all")], [#g("novice")], [#g("established")])
  }).flatten(),
  rule-top,
))

改善集中在新手：新手的等級分離實力最遠。

== 修正前後的校準

#figure(fig("baseline_slides_adjusted_curve", height: 68%))

修正後剛贏與剛輸的曲線大致重合。

== 登記模型的預測比較

#figure(table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  rule-top,
  [], [主基準的 log loss], [2014 下半年的改善], [2015 年的改善],
  rule-mid,
  ..speeds.map(s => {
    let a = pred.find(r => r.speed == s)
    let b = pv-log.find(r => r.speed == s)
    let base = float(a.mean) - float(a.diff)
    ([#s], [#f4(base)], [#f4(-float(a.diff))], [#f4(-float(b.diff))])
  }).flatten(),
  rule-top,
))

- 加入的項依序檢驗，下半年的 log loss 改善且 dyadic 區間不含 0 才保留
- 對手的近況帶來的改善大於自身
- Elo 公式換成主基準的改善遠大於近況的修正

近況是主基準之上幅度小、穩定的修正。


== 大廳對局

#figure(text(size: 0.8em, table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, right, right, right),
  rule-top,
  [], [近況三項], [log loss 改善：大廳估計], [新手], [arena 模型直接套用],
  rule-mid,
  ..speeds.map(s => {
    let k = lobby-steps.filter(r => r.speed == s and r.step != "lobby").all(r => r.kept == "1.0")
    let g(m, gr) = f4(-float(lobby-pred.find(r => r.speed == s and r.model == m and r.group == gr).diff))
    ([#s], if k [皆保留] else [部分], [#g("opp_sym", "all")], [#g("opp_sym", "novice")], [#g("arena_model", "all")])
  }).flatten(),
  rule-top,
)))

- 上一局：同一速度類別、1 小時內的大廳對局；30 分鐘與 2 小時方向相同
- 大廳的整體校正不保留，與 arena 相同
- 負的改善即比主基準差：arena 的係數套到大廳，老手組變差

近況的方向在兩種配對來源相同，大小不同，須依來源分別估計。

== 前瞻驗證

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  text(size: 0.8em, table(
    columns: (auto, auto),
    align: (left, right),
    rule-top,
    [登記的檢驗量], [2015 年],
    rule-mid,
    [Blitz 的總效果：單尾 $p$], [#pv(pv-total.p_one_sided_2015)],
    [兩年同質性 $p$], [#pv(pv-total.p_homogeneity)],
    ..("Bullet", "Blitz").map(s => ([#s 修正後勝率的 log loss 改善], [#f4(-float(pv-log.find(r => r.speed == s).diff))])).flatten(),
    rule-top,
  )),
  text(size: 0.8em, table(
    columns: (auto, auto, auto),
    align: (left, center, center),
    rule-top,
    [事後型態], [Bullet], [Blitz],
    rule-mid,
    ..(("dr_coefficient", [$Delta r$ 的係數為正]), ("after_win", [剛贏之後好於期望]), ("high_rd_after_loss", [高 RD 剛輸之後差於期望])).map(((k, lab)) => {
      let d(s) = if posthoc.find(r => r.pattern == k and r.speed == s).direction_holds == "true" [方向相同] else [不同]
      (lab, d("Bullet"), d("Blitz"))
    }).flatten(),
    rule-top,
  )),
)

登記於動用 2015 年資料之前；事後型態只檢驗方向。


== 兩年的總效果

#figure(table(
  columns: (auto, auto, auto),
  align: (left, right, right),
  rule-top,
  [Blitz 的總效果（每 0.1 的 $h$，等級分）], [2014], [2015],
  rule-mid,
  [估計], [#f2(pv-total.elo_per_0_1_2014)], [#f2(pv-total.elo_per_0_1_2015)],
  [樣本：#num(pv-total.rows) 列、#num(pv-total.players) 位玩家], [], [],
  rule-top,
))

兩年的差在同質性檢定下不拒絕；差值的區間仍寬。

== 連勝連敗：Lichess

#grid(
  columns: (1.5fr, 1fr),
  gutter: 1em,
  align: horizon,
  figure(fig("baseline_slides_streaks_lichess", width: 100%), caption: [Jul–Dec 2014 arena. Bars: 95% intervals, dyadic sandwich]),
  stack(
    spacing: 0.8em,
    [連輸四局之後的第五局：得分減期望的平均],
    [一局是否納入只依之前的結果，期望正確時平均為 0],
    [連輸者多為弱方；Elo 低估弱方],
  ),
)

以 Elo 為基準看不出連敗，Bullet 甚至相反；以主基準為準，連敗與連勝都多於期望。


== 連串中的弱方

#figure(fig("baseline_slides_underdog", height: 62%), caption: [Share of streak games in which the player had the lower Elo expected score])

Lichess 的連敗多由弱方構成；AoE2 強弱各半，故兩種基準相近。

== 連勝連敗：排隊配對的 AoE2

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  align: horizon,
  figure(fig("baseline_slides_streaks_aoe2", width: 100%), caption: [AoE2 1v1 ranked, one month. Bars: 95% intervals]),
  stack(
    spacing: 0.8em,
    [系統依等級分排隊配對，最接近電競],
    [系統的期望近乎校準，兩種基準相近],
    [納入上一局的修正之後，超額消失],
    text(size: 0.8em, fill: mute)[Lichess Blitz 的超額縮小但仍在],
  ),
)

連敗的超額可由等級分調得不夠解釋，不需要操控。
