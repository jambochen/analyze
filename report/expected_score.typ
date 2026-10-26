#import "theme.typ": *
#import "@preview/fletcher:0.5.8": diagram, node, edge
#import "@preview/cetz:0.4.2"
#import "@preview/oxifmt:1.0.0": strfmt

#show: deck.with(config-info(
  title: title-font[線上對戰的勝率預估],
  subtitle: [期望勝率與實際得分],
  author: [陳力福、陳政博、李宜哲],
))

#let tab(name) = csv("../outputs/tables/" + name + ".csv", row-type: dictionary)
#let num(x) = strfmt("{}", int(float(x)), fmt-thousands-separator: ",")
#let f2(x) = strfmt("{:.2}", float(x))
#let f3(x) = strfmt("{:.3}", float(x))
#let pct(x) = strfmt("{:.1}", 100 * float(x)) + "%"
#let fig(name, ..args) = image("../outputs/figures/" + name + ".svg", ..args)
#let flow = (node-stroke: 0.6pt + slate, node-inset: 0.6em, node-corner-radius: 3pt, spacing: (2.2em, 1.6em), edge-stroke: 0.6pt + slate)
#let rule-top = table.hline(stroke: 1pt + slate)
#let rule-mid = table.hline(stroke: 0.5pt + rule)
#let speeds = ("Bullet", "Blitz", "Classical")
#let logit = math.op("logit")
#let sigmoid = math.op("sigmoid")
#let two(a, b) = stack(spacing: 0.35em, a, b)

#let speed-n = tab("data_overview_speed")
#let y2014 = tab("data_overview_scale").find(r => r.year == "2014")
#let implied = tab("expected_score_slides_implied")
#let cal-score = tab("expected_score_baselines_calibration_score")
#let cal-imp = tab("calibration_implied")
#let cox = tab("calibration_cox")
#let pred = tab("calibration_prediction")
#let era = tab("glicko_replay_era")
#let cross = tab("expected_score_slides_rd_cross")
#let share = tab("novice_period_share")
#let base-tests = tab("expected_score_baselines_tests")

#title-slide()

== 資料來源

#grid(
  columns: (1fr, auto),
  gutter: 1.2em,
  align: horizon,
  stack(
    spacing: 0.8em,
    [Lichess：非營利組織經營的免費、開源線上西洋棋平台],
    [官方資料庫的月檔：每局一筆，含局前等級分與結果],
    [等級分系統為 Glicko-2，規則與原始碼公開],
  ),
  image("assets/lichess-logo.svg", height: 3.5em),
)

#table(
  columns: (auto, auto, auto),
  rule-top,
  [配對方式], [誰決定對手], [等級分的作用],
  rule-mid,
  [大廳], [玩家自列表選擇], [可設等級分範圍],
  [指定邀戰], [玩家指定], [不適用],
  [arena], [系統指派], [不看等級分],
  rule-top,
)

arena 的對手與玩家當下的狀態無關。

== 資料與樣本

#figure(table(
  columns: (auto, auto, auto, auto),
  rule-top,
  [速度類別], [估計時長], [2014 年局數], [其中 arena],
  rule-mid,
  ..speeds.zip(([未滿 180 秒], [180 至 479 秒], [480 秒以上])).map(((s, t)) => {
    let a = speed-n.find(r => r.speed == s and r.arena == "true").len
    let o = speed-n.find(r => r.speed == s and r.arena == "false").len
    ([#s], t, [#num(int(a) + int(o))], [#num(a)])
  }).flatten(),
  rule-top,
))

#text(size: 0.8em)[估計時長 = 底秒 + 30 × 每步加秒]

#figure(text(size: 13pt, cetz.canvas({
  import cetz.draw: *
  // 一個月一單位；4 月 19 日約在 2015 年的第 3.6 個月
  let u = 0.9
  rect((0, 0), (12 * u, 0.5), fill: sq-dark, stroke: none)
  rect((12 * u, 0), (15.6 * u, 0.5), fill: sq-light, stroke: none)
  rect((15.6 * u, 0), (18 * u, 0.5), fill: rgb("#eef0f1"), stroke: none)
  for (x, t) in ((0, [2014-01]), (12, [2015-01]), (15.6, [04-19]), (18, [2015-06])) {
    line((x * u, -0.1), (x * u, 0.6), stroke: 0.6pt + slate)
    content((x * u, -0.45), text(fill: mute, t))
  }
  content((6 * u, 0.7), [全部 rated 對局], anchor: "south")
  content((13.8 * u, 0.7), two[arena][留作驗證], anchor: "south")
  content((16.8 * u, 0.7), text(fill: mute, two[改配對][規則後]), anchor: "south")
})))

- 2014 年玩家：#num(y2014.players) 位

三種速度各有等級分，模型分開估計。

== 欄位的用途

#figure(table(
  columns: (auto, auto, auto),
  align: left,
  rule-top,
  [用途], [欄位], [取出的量],
  rule-mid,
  [計算], [`WhiteElo`、`BlackElo`], [分差 $d$、雙方平均等級分],
  [], [`TimeControl`], [速度類別],
  [], [`Event`], [是否為 arena],
  [], [雙方帳號], [累計局數、重算的 RD],
  rule-mid,
  [對照], [`Result`], [實際得分 $S$],
  rule-top,
))

== Glicko 的期望得分

#v(1fr)
#align(center, text(size: 1.2em)[$E = 1 / (1 + 10^(-g(sigma) d \/ 400)) = sigmoid(q g(sigma) d)$])
#v(0.8em)
#align(center, stack(
  spacing: 0.7em,
  [$d$：白方減黑方的等級分差；$q = ln 10 \/ 400$],
  [$g(sigma) = 1 \/ sqrt(1 + 3 q^2 sigma^2 \/ pi^2) <= 1$；$sigma = sqrt("RD"_W^2 + "RD"_B^2)$],
  text(size: 0.8em, fill: mute)[RD（rating deviation）：等級分的不確定程度；新帳號 1500、RD 350],
))
#v(1fr)

Lichess 賽後更新等級分用的是考慮 RD 的 Glicko 公式。

== Elo 的期望得分

#grid(
  columns: (1fr, 1.1fr),
  gutter: 1.2em,
  align: horizon,
  stack(
    spacing: 0.9em,
    $E = 1 / (1 + 10^(-d \/ 400)) = sigmoid(q d)$,
    [Glicko 取 RD = 0],
    text(size: 0.8em, table(
      columns: (auto, auto, auto),
      rule-top,
      [], [用到], [用在],
      rule-mid,
      [Glicko], [分差、RD], [賽後更新等級分],
      [Elo], [分差], [配對、玩家判斷對手],
      rule-top,
    )),
  ),
  fig("expected_score_slides_elo_glicko", width: 100%),
)

== 得分偏差

#grid(
  columns: (1.1fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  fig("expected_score_slides_calcurve", width: 100%),
  stack(
    spacing: 0.8em,
    [依 $E$ 每 0.05 一組，平均的 $S - E$],
    [],
    [弱方得分高於期望，強方低於期望。],
  ),
)

== 以 logistic 迴歸估計 Elo 的參數

#align(center, stack(
  spacing: 0.7em,
  [$p_i = EE[S_i] = sigmoid(alpha + beta q d_i)$；Elo 正確即 $alpha = 0$、$beta = 1$],
))

#v(0.3em)
有和局，$S_i in {0, 0.5, 1}$ 不是 0、1，故用 quasi-likelihood @wedderburn1974 @papke1996：

#align(center, $Q(alpha, beta) = sum_i [S_i log p_i + (1 - S_i) log(1 - p_i)]$)

#v(0.3em)
#stack(
  spacing: 0.7em,
  [最大化：$partial Q \/ partial (alpha, beta) = sum_i (S_i - p_i)(1, q d_i) = 0$，以牛頓法數值求解],
  [檢定：score test，$U^top I^(-1) U$ 對照卡方；$U$、$I$ 為虛無假設下 $Q$ 的梯度與 Fisher information],
)

== 估計的 $alpha$ 與 $beta$

#grid(
  columns: (1fr, 1.1fr),
  gutter: 1.2em,
  align: horizon,
  stack(
    spacing: 0.8em,
    [$alpha$：白方優勢；$beta$：斜率],
    [期望正確：$alpha = 0$、$beta = 1$],
    text(size: 0.8em, fill: mute)[老手對局：雙方 RD < 110],
    text(size: 0.8em, {
      let row(f) = speeds.map(s => f(cal-score.find(r => r.speed == s)))
      let ci(a, k) = text(size: 0.85em, fill: mute)[[#f3(a.at(k + "_lo")), #f3(a.at(k + "_hi"))]]
      table(
        columns: (auto, auto, auto, auto),
        align: (left, right, right, right),
        rule-top,
        [], ..speeds.map(s => [#s]),
        rule-mid,
        [$hat(alpha)$], ..row(a => [#f3(a.alpha)]),
        [], ..row(a => ci(a, "alpha")),
        [$hat(beta)$], ..row(a => [#f3(a.beta)]),
        [], ..row(a => ci(a, "beta")),
        table.cell(colspan: 4, text(size: 0.85em, fill: mute)[score test 的 95% 區間]),
        rule-top,
      )
    }),
  ),
  fig("expected_score_slides_calfit", width: 100%),
)

截距皆大於 0、斜率皆明顯低於 1：白方有優勢，強方被高估。

== 三類結果的有序模型

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  align: horizon,
  stack(
    spacing: 0.8em,
    $P("白勝") = F(eta - w)$,
    $P("白不輸") = F(eta + w)$,
    [$eta$：白方的潛在優勢；$w$：判為和局的區間半寬],
    [$F$：logistic 分布函數],
    [$eta$、$w$ 皆隨雙方平均等級分 $macron(r)$ 變化],
    text(size: 0.8em, fill: mute)[檢定皆為概似比檢定，同一表內以 Holm 法校正],
  ),
  text(size: 14pt, cetz.canvas({
    import cetz.draw: *
    // 潛在優勢 η + ε，ε 為 logistic；門檻在 ±w
    let (m, w, h) = (0.8, 0.9, 12)
    let f(x) = h * calc.exp(-(x - m)) / calc.pow(1 + calc.exp(-(x - m)), 2)
    let seg(a, b) = range(0, 31).map(i => { let x = a + (b - a) * i / 30; (x, f(x)) })
    for (a, b, c) in ((-5, -w, sq-light), (-w, w, rgb("#eef0f1")), (w, 5, sq-dark.lighten(30%))) {
      line((a, 0), ..seg(a, b), (b, 0), close: true, fill: c, stroke: none)
    }
    line(..seg(-5, 5), stroke: 1pt + slate)
    line((-5, 0), (5.2, 0), stroke: 0.8pt + slate, mark: (end: ">"))
    for x in (-w, w) { line((x, -0.2), (x, f(x)), stroke: 1pt + slate) }
    line((m, 0), (m, f(m) + 0.3), stroke: (paint: mute, thickness: 0.6pt, dash: "dashed"))
    content((m, f(m) + 0.35), text(fill: mute)[$eta$], anchor: "south")
    content((-2.5, -0.5), [白負])
    content((0, -0.5), [和])
    content((3, -0.5), [白勝])
    content((-w, -1.1), text(fill: mute)[$-w$])
    content((w, -1.1), text(fill: mute)[$w$])
  })),
)

白方優勢由 $eta$ 承擔，和局由 $w$ 承擔，可分開檢定。

== 和局與白方優勢

#figure(
  fig("expected_score_slides_level", width: 74%),
  caption: [分差 25 分以內、雙方皆已下滿 30 局的對局；點為描述值，線為有序模型],
)

#let draw-at(s, r) = pct(cal-imp.find(x => x.speed == s and x.rbar == r).p_draw_d0)
#let adv = cal-imp.filter(x => x.rbar in ("1200", "1500", "1800", "2100")).map(x => float(x.white_adv_rating_points))
- 勢均力敵的和局率，1200 → 2100 分：#speeds.map(s => [#s #draw-at(s, "1200") → #draw-at(s, "2100")]).join[；]
- 白方先手約當 #calc.round(calc.min(..adv)) 至 #calc.round(calc.max(..adv)) 分

和局率隨等級上升：$w$ 須隨 $macron(r)$ 變化。

== 斜率 $beta$ 與累計局數

#grid(
  columns: (1.4fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  figure(fig("expected_score_slides_experience", width: 100%), caption: [Elo 的 $E$]),
  text(size: 0.85em, table(
    columns: (auto, auto, auto),
    rule-top,
    [$hat(beta)$], [第 0 局], [200 局以上],
    rule-mid,
    ..speeds.map(s => {
      let g(v) = cox.find(r => r.speed == s and r.stratum_type == "experience" and r.stratum == v).slope
      ([#s], [#f2(g("0"))], [#f2(g(">200"))])
    }).flatten(),
    rule-top,
  )),
)

第 0 局偏得最多；數十局後趨穩，仍低於 1。兩者分開處理。

== 依 Glicko-2 重算 RD

#grid(
  columns: (1fr, 1.2fr),
  gutter: 1.2em,
  align: horizon,
  stack(
    spacing: 0.8em,
    text(size: 14pt, diagram(
      ..flow,
      spacing: (1em, 1.2em),
      node((0, 0), [月檔表頭的等級分]),
      edge("-|>"),
      node((0, 1), [依 lila 原始碼逐局重算]),
      edge("-|>"),
      node((0, 2), [每局對局前的 RD], stroke: 1.2pt + sq-dark),
    )),
    [表頭不連續處以表頭校正],
  ),
  fig("expected_score_slides_replay", width: 100%),
)

#let e14 = era.filter(r => r.era.starts-with("recomputed") or r.era.starts-with("online (2014-08"))
雙方歷史皆完整時，相符比例逾 #pct(calc.min(..e14.map(r => float(r.share_point_both_clean))))；重算的 RD 可用。

== 新手期的界定

新手對局：任一方對局前 RD ≥ 110

#text(size: 0.7em, fill: mute)[門檻：Lichess 自 2015 年 6 月起標示 provisional 的值；2014 年尚無此標示，此處為事後套用]

#grid(
  columns: (1.3fr, 1fr),
  gutter: 1em,
  align: horizon,
  figure(
    fig("expected_score_slides_rdslope", width: 100%),
    caption: [Elo 的 $E$；排除含 2014 年以前已有對局之帳號的對局；區間把各局當成獨立],
  ),
  text(size: 0.75em, table(
    columns: (auto, auto, auto, auto),
    rule-top,
    table.header([], [半數跌破 110], table.cell(colspan: 2)[新手對局占比]),
    [], [的局數], [arena], [大廳],
    rule-mid,
    ..speeds.map(s => {
      let nov(src) = pct(share.find(r => r.speed == s and r.source == src).nov_main)
      ([#s], [#cross.find(r => r.speed == s).games], [#nov("arena")], [#nov("lobby")])
    }).flatten(),
    rule-top,
  )),
)

斜率隨 RD 連續上升，110 處未見跳躍：新手對局不排除，模型加入 RD 項。

== 老手的斜率為何低於 1

#grid(
  columns: (1fr, 1.5fr),
  gutter: 1em,
  align: horizon,
  stack(
    spacing: 0.8em,
    text(size: 0.8em, table(
      columns: (auto, auto, auto),
      rule-top,
      [誤差], [關係], [斜率],
      rule-mid,
      [古典量測誤差], [$r = theta + e$], [衰減],
      [Berkson 型，線性], [$theta = r + u$], [不衰減],
      [Berkson 型，logistic], [$theta = r + u$], [約乘上 $g(sigma)$],
      rule-top,
    )),
    text(size: 0.75em, fill: mute)[$theta$：真實實力；$r$：等級分；$sigma = sqrt("RD"_W^2 + "RD"_B^2)$],
  ),
  figure(fig("expected_score_slides_elo_vs_glicko", width: 100%), caption: [老手對局：雙方 RD < 110]),
)

Glicko 的 $E$ 已含 $g(sigma)$，斜率仍低於 1：RD 項之外，分差的斜率仍須保留。

== 三個模型

#text(size: 0.9em, table(
  columns: (auto, auto),
  rule-top,
  [模型], [線性預測式],
  rule-mid,
  [有序 logistic], [$beta_1 d + beta_2 macron(r) + beta_3 d macron(r)$],
  [Glicko 形式], [$beta_4 g(sigma) d$],
  [加入 RD 的有序 logistic], [$beta_1 d + beta_4 g(sigma) d + beta_2 macron(r) + beta_3 d macron(r)$],
  rule-top,
))

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  align: horizon,
  text(size: 0.85em)[$g(sigma) = 1 \/ sqrt(1 + 3 q^2 sigma^2 \/ pi^2)$ \ $sigma = sqrt("RD"_W^2 + "RD"_B^2)$；$q = ln 10 \/ 400$],
  text(size: 13pt, diagram(
    ..flow,
    spacing: (1.2em, 2em),
    node((1, 0), [加入 RD 的有序 logistic], stroke: 1.2pt + sq-dark),
    edge((1, 0), (0, 1), "-|>", text(fill: mute)[$beta_4 = 0$], label-side: right),
    edge((1, 0), (2, 1), "-|>", text(fill: mute)[$beta_1 = beta_2 = beta_3 = 0$], label-side: left),
    node((0, 1), [有序 logistic]),
    node((2, 1), [Glicko 形式]),
  )),
)

前兩者皆為第三者的特例，以巢狀概似比檢定比較。

== 三個模型的比較

#let elo-gain(s) = strfmt("{:.4}", -float(pred.find(r => r.speed == s and r.exp_group == "all" and r.model == "ordinal_rating" and r.metric == "log_loss").diff_vs_elo))
#grid(
  columns: (auto, 1fr),
  gutter: 1.2em,
  align: horizon,
  stack(
    spacing: 0.7em,
    $"LS" = -1/n sum_i log hat(p)(y_i)$,
    text(size: 0.8em)[越小越好],
    text(size: 0.8em)[1–6 月估計、7–12 月評估],
    text(size: 0.8em)[新手對局：任一方 RD ≥ 110],
    text(size: 0.8em, table(
    columns: (auto, auto, auto),
    rule-top,
    [對完整模型], [有序 logistic], [Glicko 形式],
    rule-mid,
    ..speeds.map(s => {
      let g(m) = base-tests.find(r => r.speed == s and r.sample == "all" and r.restricted == m)
      ([#s], [#num(g("ordinal").lr)（#g("ordinal").df）], [#num(g("glicko").lr)（#g("glicko").df）])
    }).flatten(),
    rule-top,
    table.cell(colspan: 3, text(size: 0.85em, fill: mute)[概似比統計量（自由度）]),
  )),
  ),
  fig("expected_score_slides_baselines", width: 100%),
)

#text(size: 0.7em, fill: mute)[有序 logistic 相對 Elo，每局 LS 減少：#speeds.map(s => [#s #elo-gain(s)]).join[、]]

兩個受限模型皆被拒絕；之後的期望勝率以加入 RD 的有序 logistic 計算。

== 本次的結論

$eta = beta_1 d + beta_3 d macron(r) + beta_4 g(sigma) d + beta_2 macron(r)$，$w$ 隨 $macron(r)$ 變化

#table(
  columns: (auto, auto),
  rule-top,
  [項], [對應的錯],
  rule-mid,
  [$beta_1 d + beta_3 d macron(r)$], [高估強方，各分層方向一致],
  [三類結果、$w$ 隨 $macron(r)$], [和局隨等級上升],
  [$eta$ 的先手項], [白方的固定優勢],
  [$beta_4 g(sigma) d$], [新帳號偏差最大，隨 RD 連續收斂],
  [$beta_1 d$ 與 $beta_4 g(sigma) d$ 並存], [老手乘上 $g(sigma)$ 後斜率仍低於 1],
  rule-top,
)

兩個受限模型皆被拒；拼成的模型即之後的期望勝率。

== 本次的限制

#table(
  columns: (auto, auto, auto),
  rule-top,
  [], [本次], [下一次],
  rule-mid,
  [對局間的相依], [當成獨立], [處理相依後重新檢定],
  [大廳的對局來源], [未區分指定邀戰], [以 API 的 `source` 分開後重估],
  rule-top,
)

受影響的是檢定與區間；模型的選擇另有樣本外的依據。

== 下一次的問題

#text(size: 0.85em, table(
  columns: (auto, auto),
  rule-top,
  [問題], [用到本次的],
  rule-mid,
  [剛被分到強對手之後，下一局是否低於期望], [加入 RD 的有序 logistic],
  [剛輸的人較少續下，下一局的樣本經過選擇], [加入 RD 的有序 logistic],
  [剛輸、剛贏之後等級分是否失準], [老手的斜率低於 1],
  rule-top,
))

前提：arena 的對手隨機指派，下一次先驗證。

之後：修正後的勝和負機率、停止規則的回測、以 2015 年 arena 驗證。

== 參考文獻

#text(size: 11pt, bibliography("refs.bib", title: none, style: "apa"))
