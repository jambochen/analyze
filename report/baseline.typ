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
#let recent-results = tab("likelihood_selection_prediction").filter(r => r.model == "recent" and r.vs == "baseline")
#let result-table(metric, data: recent-results) = {
  let rows = data.filter(r => r.metric == metric)
  let cells = ()
  for r in rows.filter(r => r.group == "all").sorted(key: r => time-order.at(r.speed) + r.fold) {
    cells.push([#time-labels.at(r.speed) / #(if r.fold == "Jul-Sep" { [7–9 月] } else { [10–12 月] })])
    for group in ("all", "novice", "established") {
      let v = rows.find(v => v.speed == r.speed and v.fold == r.fold and v.group == group)
      let fill = if float(v.hi) < 0 { rgb("#e6f2eb") } else if float(v.lo) > 0 { rgb("#fff0e3") } else { rgb("#f0f1f3") }
      cells.push(table.cell(fill: fill, stroke: 1.5pt + white)[
        #f3(float(v.diff) * 1000)
        #linebreak()
        #text(size: 0.85em)[[#f3(float(v.lo) * 1000), #f3(float(v.hi) * 1000)]]
      ])
    }
  }
  text(size: 0.65em)[score MSE 差（$times 10^(-3)$）；方括號為 95% CI]
  text(size: 0.7em, table(columns: (1.05fr, 1fr, 1fr, 1fr), inset: (x: 0.45em, y: 0.35em), rt,
    [用時 / 測試期間], [全體], [任一方高 RD], [雙方低 RD], rm,
    ..cells, rt,
  ))
}

#let ts-example = tab("trueskill_illustration")
#let ts-a = ts-example.find(r => r.player == "A")
#let ts-b = ts-example.find(r => r.player == "B")
#let trueskill-comparison = tab("trueskill_report_comparison")
#let trueskill-periods = tab("trueskill_report_periods")
#let hierarchical-periods = tab("hierarchical_report_periods")
#let pooled-results = tab("baseline_report_comparison")
#let legend = text(size: 0.65em)[負值：新模型誤差較小。綠：CI 全低於零；灰：含零；橘：全高於零。]
#let estimate-cell(r) = {
  let fill = if float(r.hi) < 0 { rgb("#e6f2eb") } else if float(r.lo) > 0 { rgb("#fff0e3") } else { rgb("#f0f1f3") }
  table.cell(fill: fill, stroke: 1.5pt + white)[#f3(float(r.diff) * 1000)#linebreak()#text(size: 0.85em)[[#f3(float(r.lo) * 1000), #f3(float(r.hi) * 1000)]]]
}

#set par(leading: 0.55em)
#set table(inset: (x: 0.65em, y: 0.55em))
#show figure: set block(spacing: 0.55em)
#show figure.caption: set text(size: 0.65em)

#title-slide()

== 分析目標與研究問題
#text(size: 1.05em, font: "Noto Serif CJK TC")[評分之外的近期資訊，能否改善下一局的預測？]

#text(size: 0.85em, table(columns: (1.2fr, 2fr), rt,
[研究問題], [本次比較], rm,
[上一局資訊有無增量？], [評分基準與加入上一局資訊的模型],
[玩家差異是否改善預測？], [共用係數與玩家 random intercept／slope],
[TrueSkill 能否取代現有做法？], [相同測試紀錄與權重的預測比較], rt,
))

#text(size: 0.8em)[模型輸出勝、和、負機率；本次以期望得分的 score MSE 比較表現。]

== 資料與評分不確定性
#stage[Lichess 西洋棋平台｜2014 年標準棋]

#text(size: 0.9em, table(columns: (auto, 1fr), rt,
[樣本], [arena 限時錦標賽，玩家可連續下多局], rm,
[rated], [標記為計分對局；不依事後是否更新評分篩選],
[用時類別], [短：Bullet；中：Blitz；長：Classical，各自估計],
[RD], [rating deviation，評分的不確定程度；模型使用連續值],
[高 RD 對局], [任一方 RD ≥ 110；依平台後來採用的門檻分組], rt,
))

#text(size: 0.8em)[短、中、長為本報告的用時簡稱；圖中對應 Short、Medium、Long。]
#text(size: 0.65em)[圖中 novice／established：高 RD／雙方低 RD，非帳號年齡。#linebreak() 資料：#link("https://database.lichess.org/")[Lichess 官方資料庫]；RD 由過去對局重建：@glickman1999 @glickman2001。]

== 資料與預測單位
#stage[兩段測試樣本｜2014 年]

#text(size: 0.9em, table(columns: (auto, auto, auto, auto), rt,
[用時類別], [測試期間], [實際對局], [觀測筆數], rm,
..samples.sorted(key: r => time-order.at(r.speed) + r.fold).map(r => ([#time-labels.at(r.speed)], [#r.fold], [#num(r.games)], [#num(r.n)])).flatten(), rt,
))



#text(size: 0.8em)[每筆觀測是一位玩家在一局中的紀錄。#linebreak() 納入條件：該玩家在這場錦標賽已有先前紀錄；對手可無。#linebreak() 每筆紀錄的權重相同；雙方都符合條件時，同一局保留兩筆紀錄。]

#text(size: 0.8em)[輸贏可能影響繼續下棋；結果限於實際繼續下棋且符合條件者。]

== 續下選擇與預測對象
#stage[上次的轉移機率只包含留下來繼續下棋的人]

#text(size: 0.85em, table(columns: (1fr, 2.3fr), rt,
[玩家的下一步], [可觀察的資料], rm,
[繼續下棋], [能觀察下一局結果，納入預測比較],
[離開], [沒有下一局結果，無法觀察若續下會如何], rt,
))

#text(size: 0.85em)[上一局結果可能影響續下；留下來的人並非所有玩家的隨機樣本。]
#text(size: 0.85em)[基準與新增模型使用相同續下者，評估他們下一局的勝、和、負機率。]

== 上次的預測模型
#stage[含 RD 的 ordinal logistic regression｜以局前評分資訊預測勝、和、負 @mccullagh1980]

$ eta_0 = a(x)+(beta_1+beta_2 x)d+beta_3 g(sigma)d $
$ log w=c_0+c_1 x+c_2 x^2 $

#text(size: 0.95em, table(columns: (auto, 1fr), rt,
[$eta_0, 2w$], [白方相對黑方的整體優勢、兩個 cutpoint 的距離], rm,
[$x,d$], [中心化並縮放的平均評分、白方減黑方的分差],
[$a(x)$], [二次式；容許先手優勢隨平均評分變化],
[$sigma=sqrt("RD"_W^2+"RD"_B^2)$], [$g(sigma)$ 隨雙方不確定性增大而下降], rt,
))

#text(size: 0.75em)[$g(sigma)=(1+3q^2 sigma^2/pi^2)^(-1/2)$，$q=ln(10)/400$；$g$ 固定，$beta$ 由資料估計。]

#text(size: 0.8em)[沿用模型形式；BIC 比較的各候選在相同錦標賽訓練資料上重估全部參數。]

== 勝和負機率與評估
#stage[ordinal logistic 的輸出與本次比較目標]

$ p_"勝"=F(eta-w), quad p_"負"=F(-eta-w) $
$ p_"和"=1-p_"勝"-p_"負", quad F(z)=1/(1+exp(-z)) $

#text(size: 0.85em)[$eta$：白方相對優勢；$w>0$：cutpoint 半間距。]

$ hat(s)=p_"勝"+0.5p_"和", quad s in {0,0.5,1} $
#text(size: 0.85em)[score MSE：每筆紀錄的 $(s-hat(s))^2$ 取平均。]
#text(size: 0.8em)[評估期望得分；無法區分期望得分相同的不同三類機率分布。]

== 加入雙方上一局資訊
#stage[在評分基準上加入雙方上一局資訊]

$ eta_1=eta_0+f(h_W)-f(h_B) $

#text(size: 0.9em, table(columns: (auto, 1fr), rt,
[$h_W,h_B$], [白黑雙方在這場錦標賽各自上一局的資訊], rm,
[$f$], [雙方共用係數的線性函數],
[候選資訊], [上一局結果、紀錄缺失的 indicator variable、評分變動],
[評分變動], [本局與這場錦標賽上一局的局前評分之差], rt,
))

#text(size: 0.8em)[對手無上一局紀錄，以 indicator variable 表示；當局的 #raw("RatingDiff") 不作輸入。#linebreak() 各候選在相同訓練資料重估基準、上一局資訊與 cutpoint 參數。]

== 後續時段的預測評估
#stage[訓練期共同估計與選模 → 固定後測試]

#for r in periods.filter(r => r.game == "Lichess") [
  #grid(columns: (1.5fr, auto, 1fr), gutter: 0.7em,
    flow([訓練：#r.refit], [以相同資料估計參數#linebreak() 相依修正 BIC 選模]), [→],
    flow([測試：#r.test], [固定模型與係數#linebreak() score MSE]),
  )
  #v(0.9em)
]

#text(size: 0.7em)[資料已使用過，屬探索比較。CI 條件於已配適模型，不含重新訓練與選模變動。]

== 上一局資訊的選擇
#stage[各訓練期以相依修正 BIC 選擇；完整重估基準與新增項]
#text(size: 0.82em, table(columns: (0.65fr, 1.6fr, 1.6fr), rt,
[用時], [7–9 月測試所用模型], [10–12 月測試所用模型], rm,
..("Bullet", "Blitz", "Classical").map(speed => (
  [#time-labels.at(speed)],
  ..("Jul-Sep", "Oct-Dec").map(fold => [#tab("baseline_report_selection").find(r => r.speed == speed and r.fold == fold).label]),
)).flatten(), rt,
))
#text(size: 0.8em)[候選：評分基準、加結果、加評分變動、兩者都加。]
#text(size: 0.8em)[含結果的候選一併包含紀錄缺失指標。]
#text(size: 0.75em)[「上一局模型」指各次訓練選出的模型；使用的資訊可隨時段不同。]

== 加入上一局資訊的 score MSE
#stage[加入上一局資訊的模型 − 評分基準模型｜score MSE 差]
#result-table("score_mse")
#legend

== 上一局資訊的合併結果
#stage[上一局模型 − 評分基準｜依觀測筆數合併兩段測試資料]
#text(size: 0.75em, table(columns: (0.65fr, 1fr, 1.6fr, 1fr), rt,
[用時], [基準 MSE], [MSE 差（$times 10^(-3)$）#linebreak()95% CI], [相對誤差下降], rm,
..("Bullet", "Blitz", "Classical").map(speed => {
  let r = pooled-results.find(r => r.speed == speed and r.model == "recent" and r.vs == "baseline")
  ([#time-labels.at(speed)], [#f3(r.reference_mean)], estimate-cell(r), [#f3(r.reduction_percent)%])
}).flatten(), rt,
))
#legend
#text(size: 0.8em)[三種用時的合併誤差均下降；相對改善幅度小。]
#text(size: 0.75em)[長用時兩段平均差均為負；合併 CI 低於零，前段單獨看仍含零。]

== 含玩家隨機效果的 GLMM
#cite(<gee2025>)

$ eta_H = eta_"fixed" + a_i-a_j+b_i t_i-b_j t_j $

#text(size: 0.75em)[$i$：本局白方玩家；$j$：本局黑方玩家；下列定義適用於雙方。]

#text(size: 0.85em, table(columns: (1fr, 2.5fr), rt,
[$eta_"fixed"$], [共同估計評分、上一局結果、評分變動與紀錄缺失的固定項], rm,
[$a_i$：random intercept], [玩家的額外優勢；正值表示控制評分等變數後，表現仍高於模型預期],
[$t_i$：近期結果], [上一局是否獲勝，減去局前可觀察的累積勝率],
[$b_i$：random slope], [近期結果係數的個人偏差；正值表示個人係數高於共用係數], rt,
))

#text(size: 0.75em)[random：玩家的係數來自共同分布；同一玩家在此模型內共用係數。#linebreak() 累積勝率限同用時的先前計分對局；原文為 binary 模型；此處保留和局，並納入雙方玩家效果。]

== GLMM 的階層貝氏估計

$ (a_i,b_i)^T ~ "Normal"_2(0, Sigma) $

#text(size: 0.75em)[$a_i$：玩家 $i$ 的額外 intercept；$b_i$：近期結果 slope 的個人偏差。#linebreak()
零平均：共用關係已納入 $eta_"fixed"$，此處描述玩家相對共用關係的差異。]

#text(size: 0.82em, table(columns: (1fr, 2.8fr), rt,
[$Sigma$：covariance matrix], [玩家差異的尺度與相關係數，由訓練資料共同估計], rm,
[prior], [觀察這批資料前，對係數分布的假設],
[partial pooling], [玩家的資料較少時，個人係數的估計較靠近共同平均],
[posterior], [結合 prior 與訓練資料後的係數分布],
[已見／未見玩家], [訓練期出現過的玩家用個人 posterior；未見玩家則對共同分布積分], rt,
))

#text(size: 0.75em)[依 posterior 計算預測機率，再取平均；固定項採 Normal prior，玩家尺度採 Half-Normal prior。#linebreak() 每局只進 likelihood 一次；測試仍保留相同的全部觀測與權重。]

== 玩家差異的比較設計
#stage[區分整套模型的預測表現與 random slope 的增量]
#text(size: 0.8em, table(columns: (1fr, 1.8fr, 1.8fr), rt,
[面向], [共用係數的上一局模型], [玩家差異模型], rm,
[近期固定項], [訓練內以 BIC 選擇], [結果、評分變動與缺失全部納入],
[估計], [maximum likelihood], [階層 Bayesian；包含 prior],
[訓練權重], [每筆玩家紀錄等權], [每個實際對局只進 likelihood 一次],
[測試權重], [相同紀錄、每筆等權], [相同紀錄、每筆等權], rt,
))
#text(size: 0.8em)[整體差異不能全部歸因於 random slope。]
#text(size: 0.75em)[random-intercept-only 對照維持相同固定項與每局一次的 likelihood。]

== 玩家差異模型的比較
#stage[允許玩家係數不同的模型 − 共用係數的上一局模型｜合併 7–9 月與 10–12 月測試資料]

#text(size: 0.75em, table(columns: (0.65fr, 1fr, 1.6fr, 1fr), rt,
[用時], [上一局模型 MSE], [MSE 差（$times 10^(-3)$）#linebreak()95% CI], [相對誤差下降], rm,
..("Bullet", "Blitz", "Classical").map(speed => {
  let r = pooled-results.find(r => r.speed == speed and r.model == "slope" and r.vs == "recent")
  ([#time-labels.at(speed)], [#f3(r.reference_mean)], estimate-cell(r), [#f3(r.reduction_percent)%])
}).flatten(), rt,
))

#text(size: 0.8em)[與共用係數模型相比，中用時對局的預測誤差較小，短用時較大，長用時差異不明確。]
#text(size: 0.7em)[相對誤差下降為負表示誤差增加；中用時平均改善幅度小。]
#text(size: 0.7em)[比較使用已分析過的資料，屬探索性分析。#linebreak() CI 考慮同一玩家跨時段出現的相依，不含重新訓練的不確定性。]

== Random slope 的增量
#stage[含 random intercept／slope − 僅含 random intercept｜合併測試資料]
#text(size: 0.78em, table(columns: (1fr, 2fr), rt,
[用時], [score MSE 差（$times 10^(-3)$）#linebreak()95% CI], rm,
..("Bullet", "Blitz", "Classical").map(speed => {
  let r = pooled-results.find(r => r.speed == speed and r.model == "slope" and r.vs == "intercept")
  ([#time-labels.at(speed)], estimate-cell(r))
}).flatten(), rt,
))
#legend
#text(size: 0.8em)[在此階層模型內，加入 random slope 後三種用時的合併誤差皆下降。]
#text(size: 0.75em)[相對共用係數模型，整套階層模型僅在中用時有明確改善。]

== 玩家差異模型的分組結果
#stage[允許玩家係數不同的模型 − 共用係數的上一局模型]
#result-table("score_mse", data: hierarchical-periods)
#legend


== 機率調整幅度
#stage[玩家差異模型 − 評分基準｜勝率變化的絕對值]
#figure(fig("hierarchical_report_changes", width: 100%, height: 72%, fit: "contain"), caption: [All player-game rows · percentage points])

平均與 90th percentile 描述調整幅度；預測誤差另以 score MSE 評估。

== TrueSkill 的實力與表現
#stage[當局表現分為實力與隨機誤差 @herbrich2006]

$ X_i = S_i + e_i, quad e_i ~ cal(N)(0,beta^2) $
$ S_i ~ cal(N)(mu_i,sigma_i^2) $

#text(size: 0.85em, table(columns: (0.8fr, 2.5fr), rt,
[$S_i$、$mu_i$], [玩家實力及其目前估計；$i$ 表示玩家], rm,
[$sigma_i$], [實力估計的不確定性；累積對局資訊後可縮小],
[$X_i$、$beta$], [當局表現及其隨機波動；相同實力也可能發揮不同], rt,
))

#text(size: 0.85em)[資料只觀察到勝、和、負，沒有直接觀察 $S_i$ 或 $X_i$。]
#text(size: 0.8em)[預測同時考慮實力估計的不確定性 $sigma_i$ 與當局表現的波動 $beta$。]

== TrueSkill 的勝和負機率
#stage[假設例子｜A：$mu_A=#f2(ts-a.prior_mu)$、$sigma_A=#f2(ts-a.prior_sigma)$；B：$mu_B=#f2(ts-b.prior_mu)$、$sigma_B=#f2(ts-b.prior_sigma)$；$beta=#f2(ts-a.beta)$、$epsilon=#f2(ts-a.epsilon)$]

#text(size: 0.85em)[$D=X_A-X_B ~ cal(N)(mu_A-mu_B,c^2)$；$c^2=sigma_A^2+sigma_B^2+2beta^2$。]
#figure(fig("trueskill_illustration_probabilities", width: 100%, height: 55%, fit: "contain"), caption: [Hypothetical example · pre-game outcome probabilities])

#text(size: 0.85em, table(columns: (1fr, 1fr, 1fr), rt,
[A 負：$D < -epsilon$], [和局：$-epsilon <= D <= epsilon$], [A 勝：$D > epsilon$], rt,
))
#text(size: 0.7em)[$epsilon$：draw margin；採獨立 Normal skill 與獨立表現誤差。面積為三類預測機率。]

== TrueSkill 的賽後更新
#stage[同一假設例子｜觀察到 A 勝]

#text(size: 0.83em)[$p(S_A,S_B | "A 勝") prop P("A 勝" | S_A,S_B) p(S_A) p(S_B)$]
#figure(fig("trueskill_illustration_update", width: 100%, height: 58%, fit: "contain"), caption: [Hypothetical A win · prior and Normal approximation to posterior marginals])

#text(size: 0.78em)[A 的估計上升，B 的估計下降；本例中 A 原有不確定性較大，更新幅度也較大。]
#text(size: 0.72em)[posterior 通常不是 Normal；TrueSkill 保留各玩家的平均數與 variance，作 Normal approximation。]

== TrueSkill 的逐局更新
#stage[前一局的更新結果，形成下一局的 prior]

#text(size: 0.85em, grid(columns: (1fr, auto, 1fr, auto, 1fr), gutter: 0.5em,
  flow([局前分布], [預測勝、和、負]), [→],
  flow([觀察賽果], [更新雙方分布]), [→],
  flow([Normal approximation], [保存平均數與 variance]),
))

#v(0.5em)
#text(size: 0.85em)[容許實力隨對局變動：下一局前加入獨立的 Normal 變動。]
$ mu_(i,"next") = mu_(i,"post"), quad sigma_(i,"next")^2 = sigma_(i,"post")^2 + tau^2 $

#text(size: 0.8em, table(columns: (0.9fr, 2.5fr), rt,
[$tau$], [每次更新前實力變動的 standard deviation；不確定性隨之增加], rm,
[$beta$], [當局表現的波動；不直接累積到下一局的實力], rt,
))
#text(size: 0.72em)[只用已可用的賽果向前更新；各玩家保留獨立 Normal approximation。]

== 兩種模型的比較
#stage[目前的 ordinal logistic regression 與一對一 TrueSkill]

#text(size: 0.77em, table(columns: (0.7fr, 1.65fr, 1.65fr), rt,
[比較面向], [ordinal logistic regression], [TrueSkill], rm,
[模型輸入], [平台評分、RD 與局前特徵], [玩家身分、過去對手與賽果],
[玩家 $i$ 勝率], [$F(eta-w)$], [$Phi((mu_i-mu_j-epsilon)/c)$],
[估計與更新], [訓練期估係數；測試期輸入隨對局改變], [賽後更新雙方實力的近似 posterior],
[不確定性], [RD 作為 predictor；係數另有估計誤差], [對雙方 skill distribution 積分], rt,
))

#text(size: 0.72em)[$eta$：玩家 $i$ 的線性預測值；$w$：cutpoint 半間距。#linebreak() $F(z)=1/(1+exp(-z))$；$Phi$：standard Normal CDF。#linebreak() $c^2=sigma_i^2+sigma_j^2+2beta^2$，此處 $sigma_i^2$ 為加入 $tau^2$ 後的局前 variance。]
#text(size: 0.75em)[兩者皆以兩個門檻分出勝、和、負；差異也包含資料輸入與更新規則。]

== TrueSkill 的比較條件
#stage[本地實驗｜比較兩套具體預測流程]
#text(size: 0.8em, table(columns: (0.9fr, 1.8fr, 1.8fr), rt,
[條件], [ordinal logistic], [本次 TrueSkill], rm,
[歷史範圍], [平台評分包含 arena 外歷史], [僅 arena 對手與賽果；含未納入評估的對局],
[初始資訊], [平台局前評分與重建 RD], [所有玩家從相同 Normal prior 開始],
[白方優勢], [由訓練資料估計], [不另設參數],
[和局範圍], [隨平均評分變化], [每種用時、每段固定 draw margin], rt,
))
#text(size: 0.8em)[測試紀錄與權重相同；輸入資訊、模型結構與更新規則不同。]
#text(size: 0.75em)[結果不能單獨歸因於 logistic 與 Normal CDF。]

== TrueSkill 的本地估計
#stage[訓練內選參數，測試期固定參數並逐局更新實力]

#text(size: 0.76em, table(columns: (1fr, 1fr, 0.65fr, 0.65fr, 0.7fr), rt,
[用時], [測試期間], [$sigma_0$], [$tau$], [$epsilon$], rm,
..tab("trueskill_parameters").sorted(key: r => time-order.at(r.speed) + r.fold).map(r => (
  [#time-labels.at(r.speed)], [#if r.fold == "Jul-Sep" { [7–9 月] } else { [10–12 月] }],
  [#f2(r.sigma)], [#f2(r.tau)], [#f3(r.draw_margin)]
)).flatten(), rt,
))
#text(size: 0.75em)[$sigma_0$：初始不確定性；$tau$：每次更新前新增實力變動的 standard deviation。#linebreak() $epsilon$：draw margin；固定 $mu_0=#tab("trueskill_parameters").first().mu$、$beta=#tab("trueskill_parameters").first().beta$。]
#text(size: 0.75em)[以六月／九月的三類 log loss 選擇參數；測試樣本、權重與 logistic 相同。#linebreak() 以 API 最後下棋時間作賽果可用時間代理，僅納入早於開局的紀錄。]

#text(size: 0.7em)[測試比較用 score MSE；長用時的和局參數選到候選上界，未依測試結果擴大搜尋。]

== TrueSkill 的預測比較
#stage[TrueSkill − 共用係數的上一局模型｜合併 7–9 月與 10–12 月測試資料]

#text(size: 0.78em, table(columns: (1fr, 2fr), rt,
[用時], [score MSE 差（$times 10^(-3)$）#linebreak() 95% CI], rm,
..("Bullet", "Blitz", "Classical").map(speed => {
  let mse = trueskill-comparison.find(r => r.speed == speed and r.vs == "recent" and r.metric == "score_mse")
  ([#time-labels.at(speed)], estimate-cell(mse))
}).flatten(), rt,
))
#text(size: 0.75em)[未見 TrueSkill 的期望得分誤差較小；長用時誤差較大。]
#text(size: 0.7em)[與評分基準相比，TrueSkill 也未顯示優勢。此比較屬探索性分析。]

== TrueSkill 的分組結果
#stage[TrueSkill − 共用係數的上一局模型]
#result-table("score_mse", data: trueskill-periods)

#text(size: 0.72em)[在兩個測試時段中，TrueSkill 對長用時高 RD 與雙方低 RD 對局的預測均較差。#linebreak() TrueSkill 僅由 arena 賽果估計；logistic 的平台評分包含更廣的對局歷史。]

== 單局預測的用途界線
#stage[連敗與累積得分偏差｜固定局前條件、只重抽賽果的模擬]

#table(columns: (1fr, 2.3fr), rt,
[已完成的檢查], [結果與限制], rm,
[連敗與玩家權重], [每段連續對局或每位玩家等權，結論不同],
[累積得分偏差與對局數], [較長的連續紀錄集中於少數玩家，不作一般異常門檻],
[模擬設定], [重抽賽果後，評分、上一局資訊與是否繼續下棋仍固定], rt,
)

單局預測改善，尚不足以宣告個人失常或建議停手。

== 模型取捨的證據
#stage[相同測試紀錄與權重｜兩段合併 score MSE 差（$times 10^(-3)$）及個別 95% CI]
#text(size: 0.72em, table(columns: (0.65fr, 1.35fr, 1.35fr, 1.35fr), rt,
[用時], [上一局模型#linebreak()− 評分基準], [GLMM#linebreak()− 上一局模型], [TrueSkill#linebreak()− 上一局模型], rm,
..("Bullet", "Blitz", "Classical").map(speed => (
  [#time-labels.at(speed)],
  ..(("recent", "baseline"), ("slope", "recent"), ("trueskill", "recent")).map(pair => estimate-cell(
    pooled-results.find(r => r.speed == speed and r.model == pair.at(0) and r.vs == pair.at(1)))),
)).flatten(), rt,
))
#legend
#text(size: 0.75em)[依合併增量提出探索性建議；分段結果說明穩定性，不要求每段 CI 均排除零。]
#text(size: 0.7em)[模型取捨依已看過的資料提出，未事前登記；CI 未作多重比較校正。]

== 模型建議與依據
#text(size: 0.78em, table(columns: (0.65fr, 1.5fr, 2fr), rt,
[用時類別], [建議採用], [選擇理由], rm,
[短用時], [加入上一局資訊的 ordinal logistic regression], [兩個測試時段的預測均較評分基準準確；加入玩家隨機效果後，整體誤差反而較大],
[中用時], [含玩家隨機效果的 GLMM], [合併測試資料後，預測比共用係數模型準確；改善主要來自後一時段],
[長用時], [加入上一局結果的 ordinal logistic regression], [兩段平均誤差皆下降，合併 CI 位於零以下；前段單獨看仍不明確；GLMM 未見額外改善], rt,
))

#text(size: 0.8em)[目前的比較結果不支持改用 TrueSkill。]
#text(size: 0.75em)[依已使用資料提出的模型選擇，仍須以新資料驗證。#linebreak() 適用對象限於在同一場錦標賽已有先前對局紀錄的玩家。]

== 方法來源
#text(size: 11pt, columns(2, gutter: 1.5em,
  bibliography("refs.bib", title: none, style: "apa")
))
