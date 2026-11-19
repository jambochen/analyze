#import "theme.typ": *
#import "@preview/oxifmt:1.0.0": strfmt

#show: deck.with(config-info(
  title: title-font[局前評分與近期戰績],
  subtitle: [一對一競技對局的預測模型與結論],
  author: [陳力福、陳政博、李宜哲],
))

#let tab(name) = csv("../outputs/tables/" + name + ".csv", row-type: dictionary)
#let num(x) = strfmt("{}", int(float(x)), fmt-thousands-separator: ",")
#let f2(x) = strfmt("{:.2}", float(x))
#let fig(name, ..args) = image("../outputs/figures/" + name + ".svg", ..args)
#let rt = table.hline(stroke: 1pt + slate)
#let rm = table.hline(stroke: 0.5pt + rule)
#let flow(top, bottom) = block(width: 100%, stroke: rule, inset: 10pt, stack(dir: ttb, spacing: 5pt, top, text(size: 0.8em, fill: mute, bottom)))
#let stage(body) = block(below: 0.7em, text(size: 0.78em, fill: slate, body))
#let samples = tab("temporal_prediction_sample").filter(r => r.game == "Lichess" and r.segment == "test")
#let periods = tab("temporal_slides_periods")
#let time-labels = (Bullet: [短用時], Blitz: [中用時], Classical: [長用時])
#let time-order = (Bullet: "1", Blitz: "2", Classical: "3")
#let f3(x) = strfmt("{:.3}", float(x))
#let pval(x) = if float(x) < 0.001 { [< 0.001] } else { [#f3(x)] }
#let pooled-results = tab("baseline_report_comparison")
#let estimate-cell(r) = {
  let fill = if float(r.hi) < 0 {
    rgb("#8bcba4")
  } else if float(r.lo) > 0 {
    rgb("#ee9996")
  } else {
    rgb("#f2f2f2")
  }
  table.cell(fill: fill, stroke: 1.5pt + white)[#f3(float(r.diff) * 1000)#linebreak()#text(size: 0.85em)[[#f3(float(r.lo) * 1000), #f3(float(r.hi) * 1000)]]]
}

#set par(leading: 0.55em)
#set table(inset: (x: 0.65em, y: 0.55em))
#show figure: set block(spacing: 0.55em)
#show figure.caption: set text(size: 0.65em)

#title-slide()

== 資料與評分不確定性
#stage[Lichess 西洋棋平台｜2014 年標準棋]

#text(size: 0.9em, table(columns: (auto, 1fr), rt,
[樣本], [arena 限時錦標賽，玩家可連續下多局], rm,
[rated], [標記為計分對局；不依事後是否更新評分篩選],
[用時類別], [短：Bullet；中：Blitz；長：Classical，各自估計],
[RD], [rating deviation，評分的不確定程度 @glickman1999；以歷史對局重建，模型使用連續值],
rt,
))

== 資料與預測單位
#stage[兩段測試樣本｜2014 年]

#text(size: 0.9em, table(columns: (auto, auto, auto, auto), rt,
[用時類別], [測試期間], [實際對局], [觀測筆數], rm,
..samples.sorted(key: r => time-order.at(r.speed) + r.fold).map(r => ([#time-labels.at(r.speed)], [#r.fold], [#num(r.games)], [#num(r.n)])).flatten(), rt,
))

評估限於同場已有前局的玩家。雙方都符合時，同一局留下兩筆紀錄。

== 評分基準與近期戰績
#stage[ordinal logistic regression｜白方結果依序為負、和、勝 @mccullagh1980]

#table(columns: (auto, 1fr), rt,
[上次的評分基準], [$eta_0=a(macron(r))+beta_1 d+beta_2 d macron(r)+beta_3 g(sigma)d$], rm,
[一階歷史模型], [$eta_1=eta_0+f(h_W)-f(h_B)$], rt,
)

#text(size: 0.82em)[$a$ 為同分時的先手優勢，$d$ 是白減黑的分差，$macron(r)$ 是平均評分，$g(sigma)$ 隨雙方 RD 增加而下降。$h_W,h_B$ 是白、黑雙方在同場錦標賽的上一局資訊，所有玩家共用同一組歷史係數。]

#text(size: 0.85em)[以 logistic CDF 求出白勝與白方不輸的機率。$P("draw") = P("win or draw") - P("win")$。#linebreak()白方的預測得分為 $hat(s) = P("win") + 0.5 P("draw")$。]

== 訓練期選入的上一局資訊
#stage[各用時在兩個訓練期分別以相依修正 BIC 選模。]

同場上一局的候選資訊：只用結果、只用評分變動、兩者都用，與純評分基準比較。四個候選在同一批訓練對局重估全部參數。

#let selected = tab("baseline_report_selection")
#text(size: 0.8em, table(columns: (auto, auto, 1fr, 1fr, 1fr), rt,
[訓練期], [測試期], [短用時選入], [中用時選入], [長用時選入], rm,
..periods.filter(r => r.game == "Lichess").map(r => (
  [#r.refit], [#r.test],
  ..("Bullet", "Blitz", "Classical").map(speed => [#selected.find(v => v.speed == speed and v.fold == r.fold).label]),
)).flatten(), rt,
))

#text(size: 0.8em)[每格選出的模型固定係數，用於該列測試期。含上一局結果的模型也標記對手紀錄缺失。]

== 選定模型的測試期誤差
#stage[$Delta = "MSE"_"selected" - "MSE"_"baseline"$]

#text(size: 0.75em, table(columns: (0.7fr, 1fr, 1.6fr, 1fr), rt,
[用時], [基準 MSE], [MSE 差（$times 10^(-3)$）#linebreak()95% CI], [相對誤差下降], rm,
..("Bullet", "Blitz", "Classical").map(speed => {
  let r = pooled-results.find(r => r.speed == speed and r.model == "recent" and r.vs == "baseline")
  ([#time-labels.at(speed)], [#f3(r.reference_mean)], estimate-cell(r), [#f3(r.reduction_percent)%])
}).flatten(), rt,
))
#text(size: 0.8em)[各期選出的「一階歷史模型」作為後續比較基準。]

== 歷史長度與賽果機率

#text(size: 0.9em)[固定評分基準與兩個切點間的距離，以該玩家同場前一、二或三局的結果組合修正預測。]


#let history = tab("markov_order_prediction").filter(r => r.metric == "log_loss")
#text(size: 0.8em)[單局 log loss 為 $ell_(t,k)=-ln hat(p)_(t,k)(y_t)$。下表為下半年平均配對差（高階 − 低階，$times 10^(-3)$）與 95% CI]
#text(size: 0.85em, table(columns: (1.15fr, 1fr, 1fr, 1fr), rt,
[回看長度], [短用時], [中用時], [長用時], rm,
..("2", "3").map(k => (
  [#(int(k) - 1) 局 → #k 局],
  ..("Bullet", "Blitz", "Classical").map(speed => {
    let r = history.find(r => r.k == k and r.speed == speed)
    let d = float(r.diff)
    let margin = 1.96 * float(r.diff_se_dyadic)
    estimate-cell((diff: d, lo: d - margin, hi: d + margin))
  }),
)).flatten(), rt,
))

#text(size: 0.8em)[多看第二局僅中用時有明確改善。模型建議另依期望得分的 MSE 決定。]

== 廣義線性混合模型：玩家截距與上一局斜率
#stage[Generalized Linear Mixed Model（GLMM）｜允許上一局戰績對得分預測的修正幅度因人而異。 @gee2025]

$ eta_H = eta_"fixed" + (a_i-a_j) + (b_i t_i-b_j t_j) $

#table(columns: (auto, 1fr), rt,
[$a_i-a_j$], [random intercept：控制評分與近期資訊後，白方與黑方的個人優勢之差。], rm,
[$b_i t_i-b_j t_j$], [random slope：兩位玩家各自對近期結果的額外修正，再取白減黑。], rt,
)

$ t_i = bold(1) {"previous arena result" = "win"} - macron(p)_i $
#text(size: 0.8em)[$macron(p)_i$ 是該玩家同用時先前計分對局的累積勝率。黑方同理。若缺上一局紀錄，取零並另設缺失指標。]

== GLMM：個人係數如何估計
#stage[利用全體玩家的資料估計個人斜率。]

$ theta_i = (a_i,b_i)^T, quad theta_i | Sigma ~ cal(N)_2(0,Sigma) $

$ underbrace(p(theta_i | "data", "other parameters"), "conditional posterior") prop underbrace(L_i(theta_i), "likelihood for games involving " i) times underbrace(p(theta_i | Sigma), "shared prior") $

#table(columns: (auto, 1fr), rt,
[訓練中見過的玩家], [依個人係數的後驗分布對預測機率取平均。], rm,
[訓練中未見的玩家], [對玩家係數的共同分布積分。], rt,
)

== GLMM：整體模型與個人斜率的預測增益
#stage[含個人截距與斜率的 GLMM 減去各欄對照｜兩期合併 MSE 差（單位 $10^(-3)$）與 95% CI，負值表示改善。]

#table(columns: (auto, 1fr, 1fr), rt,
[用時], [加入個人斜率#linebreak()#text(size: 0.8em)[對照：只有個人截距的 GLMM]], [改用含個人截距與斜率的 GLMM#linebreak()#text(size: 0.8em)[對照：一階歷史模型]], rm,
..("Bullet", "Blitz", "Classical").map(speed => (
  [#time-labels.at(speed)],
  ..("intercept", "recent").map(ref => estimate-cell(pooled-results.find(r => r.speed == speed and r.model == "slope" and r.vs == ref))),
)).flatten(), rt,
)

短用時加入個人斜率後，誤差低於只有個人截距的 GLMM，仍高於一階歷史模型。

== TrueSkill 的勝和負機率
#stage[假設對局｜A 的實力估計較低、不確定性較高 @herbrich2006]

$ S_i ~ cal(N)(mu_i,sigma_i^2), quad X_i | S_i ~ cal(N)(S_i,beta^2) $
#text(size: 0.85em)[$D=X_A-X_B ~ cal(N)(mu_A-mu_B,c^2)$；$c^2=sigma_A^2+sigma_B^2+2beta^2$。]
#figure(fig("trueskill_illustration_probabilities", width: 100%, height: 55%, fit: "contain"), caption: [Hypothetical example · pre-game outcome probabilities])

#text(size: 0.8em)[$sigma_i$ 描述實力估計的不確定性，$beta$ 描述當局表現的波動。兩個分界為 $±epsilon$。]

== TrueSkill 的賽後更新
#stage[同一假設例子｜觀察到 A 勝]

#text(size: 0.83em)[$p(S_A,S_B | D > epsilon) prop P(D > epsilon | S_A,S_B) p(S_A) p(S_B)$]
#figure(fig("trueskill_illustration_update", width: 100%, height: 58%, fit: "contain"), caption: [Hypothetical A win · prior and Normal approximation to posterior marginals])

#text(size: 0.78em)[以相同平均數與變異的常態分布近似各玩家的邊際後驗。]
#text(size: 0.72em)[下一局的實力變異：$sigma_("next prior")^2 = sigma_("current posterior")^2 + tau^2$。]




== TrueSkill 的參數選擇
#stage[各用時、各測試期分開選參數｜固定起始平均 $mu_0=0$ 與當局表現波動 $beta=1$]

#let tuning = tab("trueskill_tuning")
#let candidates(key) = tuning.map(r => float(r.at(key))).dedup().sorted().map(str).join("、")
#table(columns: (auto, 1fr, auto), rt,
[參數], [控制什麼], [候選值], rm,
[$sigma_0$], [玩家剛出現時，實力估計有多不確定], [#candidates("sigma")],
[$tau$], [兩局之間的實力波動，每局前變異增加 $tau^2$], [#candidates("tau")],
[$p_"draw"$], [換算成和局門檻 $epsilon$，值越大，和局區間越寬], [#candidates("draw_probability")], rt,
)

== TrueSkill 的測試期誤差
#stage[同場已有前局的玩家｜兩期合併得分 MSE；下表負值表示 TrueSkill 較準。]

#table(columns: (auto, 2fr), rt,
[用時], [TrueSkill − 一階歷史模型#linebreak()MSE 差與 95% CI（單位 $10^(-3)$）], rm,
..("Bullet", "Blitz", "Classical").map(speed => (
  [#time-labels.at(speed)],
  estimate-cell(pooled-results.find(r => r.speed == speed and r.model == "trueskill" and r.vs == "recent")),
)).flatten(), rt,
)
#text(size: 0.8em)[長用時 TrueSkill 的誤差較高，和局參數達本次搜尋上界。]

== 各用時的模型選擇
#stage[預測同場續下玩家的期望得分｜依兩期合併的測試 MSE 選擇]

#table(columns: (auto, 1fr, 1.5fr), rt,
[用時], [建議模型], [選擇依據], rm,
[短], [一階歷史模型], [上一局結果與評分變動帶來小幅改善，改用 GLMM 後誤差上升。],
[中], [個人截距與斜率 GLMM], [相較一階歷史模型，誤差略降，改善主要出現在後期。],
[長], [一階歷史模型], [加入上一局結果後誤差略降，GLMM 的額外改善不明確。], rt,
)

#text(size: 0.8em)[本次 TrueSkill 在三種用時皆未顯示額外增益。]

== 方法來源
#text(size: 11pt, columns(2, gutter: 1.5em,
  bibliography("refs.bib", title: none, style: "apa")
))
