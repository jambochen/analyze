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
#let cox = tab("calibration_cox")
#let pred = tab("calibration_prediction")
#let base-tests = tab("expected_score_baselines_tests")
#let base-aic = tab("expected_score_baselines_aic")
#let leave = tab("descriptive_slides_streak_leave")
#let leave-at(s, r, k) = leave.find(x => x.speed == s and x.result == r and x.streak == str(k)).leave_rate

#title-slide()

== 資料來源

#grid(
  columns: (1fr, auto),
  gutter: 1.2em,
  align: horizon,
  stack(
    spacing: 0.8em,
    [Lichess：非營利組織經營的免費、開源線上西洋棋平台],
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
  [好友對戰], [玩家指定], [不適用],
  [arena], [系統指派], [不看等級分],
  rule-top,
)

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
  [計算], [`WhiteElo`、`BlackElo`], [分差 $d$、雙方平均等級分 $macron(r)$],
  [], [`TimeControl`], [速度類別],
  [], [`Event`], [是否為 arena、屬於哪一場],
  [], [雙方帳號、開局時間], [累計局數、重算的 RD、上一局結果、是否續下],
  rule-mid,
  [對照], [`Result`], [實際得分 $S$],
  rule-top,
))

== Glicko 的期望得分

#align(center, text(size: 1.2em)[$E = 1 / (1 + 10^(-g(sigma) d \/ 400)) = sigmoid(q g(sigma) d)$])
#v(0.8em)
#align(center, stack(
  spacing: 0.7em,
  [$d$：白方減黑方的等級分差；$q = ln 10 \/ 400$],
  [$g(sigma) = 1 \/ sqrt(1 + 3 q^2 sigma^2 \/ pi^2) <= 1$；$sigma = sqrt("RD"_W^2 + "RD"_B^2)$],
  text(size: 0.8em, fill: mute)[RD（rating deviation）：等級分的不確定程度；新帳號 1500、RD 350],
))

RD 取 0 即 Elo 的 $E = sigmoid(q d)$，配對只看分差。

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

白方視角的結果 $Y$：白負 $(j = 1)$、和 $(j = 2)$、白勝 $(j = 3)$

$ logit P(Y <= j) = log P(Y <= j) / (1 - P(Y <= j)) = theta_j (macron(r)) - eta, quad j = 1, 2 $

#grid(
  columns: (1.3fr, 1fr),
  gutter: 1em,
  align: horizon,
  stack(
    spacing: 0.6em,
    text(size: 0.9em, grid(
      columns: 2,
      align: (right, left),
      column-gutter: 0.8em,
      row-gutter: 0.75em,
      [$macron(r)$], [雙方的平均等級分],
      [$eta = a(macron(r)) + (beta_1 + beta_2 macron(r)) d$], [白方的潛在優勢],
      [$a$], [先手優勢：同分時的 $eta$],
      [$theta_(1, 2) = minus.plus w(macron(r))$], [切點],
      [$w > 0$], [和局帶寬：$2 w$ 決定和局率],
    )),
    text(size: 0.75em, fill: mute)[每分的效果隨 $macron(r)$ 變化；$a$、$log w$ 皆為 $macron(r)$ 的二次式],
  ),
  text(size: 14pt, cetz.canvas({
    import cetz.draw: *
    // 潛在優勢 η + ε，ε 為 logistic；切點 θ1、θ2 畫在 ±w
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
    content((-w, -1.1), text(fill: mute)[$theta_1$])
    content((w, -1.1), text(fill: mute)[$theta_2$])
    line((-w, -1.7), (w, -1.7), stroke: 0.6pt + mute, mark: (start: ">", end: ">"))
    content((0, -2.1), text(fill: mute)[$2 w$])
  })),
)

== 斜率 $beta$ 與累計局數

#grid(
  columns: (1.4fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  figure(fig("expected_score_slides_experience", width: 100%), caption: [Elo's $E$. Bars: score-test 95% intervals]),
  stack(
    spacing: 0.8em,
    [各組分別估 $p_i = sigmoid(alpha + beta q d_i)$],
    text(size: 0.8em, fill: mute)[quasi-likelihood、score test 區間同前；不限 RD],
    text(size: 0.8em, {
      let g(s, v) = cox.find(r => r.speed == s and r.stratum_type == "experience" and r.stratum == v)
      let ci(a) = text(size: 0.85em, fill: mute)[[#f2(a.slope_lo), #f2(a.slope_hi)]]
      table(
        columns: (auto, auto, auto),
        align: (left, right, right),
        rule-top,
        [$hat(beta)$], [第 0 局], [200 局以上],
        rule-mid,
        ..speeds.map(s => ([#s], [#f2(g(s, "0").slope)], [#f2(g(s, ">200").slope)], [], ci(g(s, "0")), ci(g(s, ">200")))).flatten(),
        table.cell(colspan: 3, text(size: 0.85em, fill: mute)[score test 的 95% 區間]),
        rule-top,
      )
    }),
  ),
)

== 三個模型

#text(size: 0.9em, table(
  columns: (auto, auto, auto),
  rule-top,
  [模型], [線性預測式], [納入 Elo 沒考慮的],
  rule-mid,
  [有序 logistic], [$a(macron(r)) + beta_1 d + beta_2 d macron(r)$], [先手、和局],
  [Glicko 形式], [$a + beta_3 g(sigma) d$], [水平波動],
  [加入 RD 的有序 logistic], [$a(macron(r)) + beta_1 d + beta_2 d macron(r) + beta_3 g(sigma) d$], [兩者],
  rule-top,
))

#v(0.6em)
#align(center, text(size: 15pt, diagram(
  ..flow,
  spacing: (2.5em, 3em),
  node((1, 0), [加入 RD 的有序 logistic], stroke: 1.2pt + sq-dark),
  edge((1, 0), (0, 1), "-|>", text(fill: mute)[$beta_3 = 0$], label-side: right, label-sep: 1em),
  edge((1, 0), (2, 1), "-|>", text(fill: mute, two[$beta_1 = beta_2 = 0$][$a$、$w$ 不隨 $macron(r)$]), label-side: left, label-sep: 1em),
  node((0, 1), [有序 logistic]),
  node((2, 1), [Glicko 形式]),
)))
#v(0.6em)

兩個特例各與加入 RD 的有序 logistic 以概似比檢定比較。

== 三個模型的比較

#let elo-gain(s) = strfmt("{:.4}", -float(pred.find(r => r.speed == s and r.exp_group == "all" and r.model == "ordinal_rating" and r.metric == "log_loss").diff_vs_elo))
#align(center, text(size: 0.9em, table(
    columns: (auto, auto, auto, auto),
    rule-top,
    [$H_0$], [有序 logistic], [Glicko 形式], [AIC 差],
    rule-mid,
    ..speeds.map(s => {
      let g(m) = base-tests.find(r => r.speed == s and r.sample == "all" and r.restricted == m)
      let a(m) = float(base-aic.find(r => r.speed == s and r.sample == "all" and r.model == m).aic)
      let pv(m) = { let x = float(g(m).p_naive); if x < 0.001 [< 0.001] else [#f3(x)] }
      ([#s], pv("ordinal"), pv("glicko"), [#num(a("glicko") - a("ordinal"))])
    }).flatten(),
    rule-top,
  )))

#text(size: 0.7em, fill: mute)[表：LRT $p$ 值，對立假設為加入 RD 的有序 logistic；AIC 差為 Glicko 形式減有序 logistic。]

有序 logistic 與 Glicko 形式皆被拒絕；之後的期望勝率以加入 RD 的有序 logistic 計算。

== 樣本外的校準

#figure(
  fig("expected_score_slides_holdout", width: 100%),
  caption: [Games from Jul–Dec 2014. Solid: ordinal logistic with RD; faded dashed: Elo. Bars: 95% intervals],
)

期望得分已貼近實際；但帳號的前幾局仍不準，和局機率也略低於實際。

== 上一局與本局的轉移

#stack(
  spacing: 0.7em,
  [一階 Markov：本局結果的分布只依上一局；$Y$：負、和、勝],
  [$p_(i j) = P(Y_t = j | Y_(t - 1) = i)$，估計為 $hat(p)_(i j) = n_(i j) \/ n_(i +)$：轉移次數除以列和],
  text(size: 0.8em, fill: mute)[出處：@anderson1957],
)

#figure(
  fig("descriptive_slides_markov", height: 52%),
  caption: [2014 arena games with a previous game in the same arena; each bar is one row of the transition matrix],
)

輸後較常再輸，贏後較常再贏；這也包含玩家本身的強弱。

== 連輸、連贏之後的離開

#grid(
  columns: (1.3fr, 1fr),
  gutter: 1.2em,
  align: horizon,
  figure(fig("descriptive_slides_streak_leave", width: 100%), caption: [2014 arena games after which the arena still had time for another game; draws excluded]),
  text(size: 0.85em, stack(
    spacing: 0.9em,
    [離開：錦標賽仍有時間，同一場內不再下],
    [連輸 $k$ 局：到本局為止連續輸 $k$ 局，和局中斷],
    table(
      columns: (auto, auto, auto),
      align: (left, right, right),
      rule-top,
      [Blitz 離開], [輸], [贏],
      rule-mid,
      ..(1, 3).map(k => ([連續 #k 局], [#pct(leave-at("Blitz", "loss", k))], [#pct(leave-at("Blitz", "win", k))])).flatten(),
      rule-top,
    ),
    text(size: 0.9em, fill: mute)[連輸 $k$ 局的人，都是前 $k - 1$ 次輸後留下的人],
  )),
)

輸了的玩家較常離開；上一頁的轉移只含續下者。

== 本次的結論與限制

加入 RD 的有序 logistic 能根據雙方賽前的等級分與 RD，比 Elo 更準地估計勝、和、負的機率。

限制：

- 檢定與區間假設各局獨立，同一玩家的多局可能彼此相關
- 新帳號的最初幾局仍不準
- 上一局的影響只往前看一局，且只含續下者

#show: appendix

== 參考文獻

#text(size: 11pt, bibliography("refs.bib", title: none, style: "apa"))


== 切點隨平均等級分的形式

#figure(
  fig("expected_score_slides_threshold", height: 68%),
  caption: [$a$ and $log w$ estimated separately in 20 equal-count bins of mean rating. Bars: iid 95% intervals],
)

