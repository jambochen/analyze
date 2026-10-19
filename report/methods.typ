#import "theme.typ": *
#import "@preview/fletcher:0.5.8": diagram, node, edge
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot

#show: deck.with(config-info(
  title: title-font[線上對戰的勝率預估],
  subtitle: [可能使用的統計方法],
  author: [陳力福、陳政博、李宜哲],
))

#let fig(name, ..args) = image("../outputs/figures/" + name + ".svg", ..args)
#let flow = (node-stroke: 0.6pt + slate, node-inset: 0.6em, node-corner-radius: 3pt, spacing: (2.2em, 1.6em), edge-stroke: 0.6pt + slate)
#let logit = math.op("logit")
#let rule-top = table.hline(stroke: 1pt + slate)
#let rule-mid = table.hline(stroke: 0.5pt + rule)
// 對局方塊：勝深、和中、負淺，只表結果，不帶方向的價值
#let sq(kind, faded: false) = {
  let c = if kind == "w" { slate } else if kind == "d" { mute.lighten(30%) } else { white }
  box(width: 0.9em, height: 0.9em, stroke: 0.4pt + if faded { rule } else { mute }, fill: if faded { c.lighten(75%) } else { c })
}
#let row(..ks, faded: false) = stack(dir: ltr, spacing: 0.15em, ..ks.pos().map(k => sq(k, faded: faded)))

// 方法介紹頁：左為通用例子的圖，右為問題、資料、例子
#let intro(name, rows) = figure(grid(
  columns: (1.15fr, 1fr),
  gutter: 1.5em,
  align: horizon,
  fig(name, width: 100%),
  // 這張表沒有表頭，取消主題對第一列的加粗
  {
    show table.cell.where(y: 0): set text(weight: 350, fill: ink)
    table(
      columns: (auto, 1fr),
      align: (left + top, left + top),
      rule-top,
      ..rows.map(((k, v)) => (text(fill: slate, weight: 500, k), v)).flatten(),
      rule-top,
    )
  },
))
#let wtab(..rows) = table(
  columns: (auto, auto, auto),
  align: center,
  rule-top,
  [結果], [$Y$], [權重],
  rule-mid,
  ..rows,
  rule-top,
)

#title-slide()

== 對局結果

#figure(grid(
  columns: (auto, 1fr),
  gutter: 2.5em,
  align: horizon,
  table(
    columns: (auto, auto),
    align: center,
    rule-top,
    [結果], [白方得分 $Y$],
    rule-mid,
    [白勝], [1], [和局], [0.5], [黑勝], [0],
    rule-top,
  ),
  align(left)[
    - 分差 $d$ = 白方等級分 − 黑方等級分。
    - 系統依 $d$ 給出白方的期望得分 $p(d)$，不給和局機率。
  ],
))

問題：$p(d)$ 是否等於實際的平均得分 $bb(E)[Y | d]$。

== 系統的期望得分

#let q = calc.ln(10) / 400
#figure(grid(
  columns: (1.1fr, 1fr),
  gutter: 1.5em,
  align: horizon,
  text(size: 13pt, cetz.canvas({
    plot.plot(
      size: (9, 5.5),
      x-label: [$d$],
      y-label: [$p(d)$],
      x-tick-step: none,
      x-ticks: ((-400, [−400]), (0, [0]), (200, [200]), (400, [400])),
      y-tick-step: none,
      y-ticks: ((0, [0]), (0.5, [0.5]), (0.76, [0.76]), (1, [1])),
      y-min: 0,
      y-max: 1,
      axis-style: "left",
      {
        plot.add(domain: (-800, 800), x => 1 / (1 + calc.pow(10, -x / 400)), style: (stroke: 1.5pt + slate), label: [Elo])
        // 示意的收縮 g = 0.7
        plot.add(domain: (-800, 800), x => 1 / (1 + calc.pow(10, -0.7 * x / 400)), style: (stroke: (paint: sq-dark, dash: "dashed", thickness: 1.5pt)), label: [Glicko])
        plot.add(((200, 0), (200, 0.76), (-800, 0.76)), style: (stroke: (paint: mute, dash: "dotted")))
      },
    )
  })),
  align(left)[
    $ p(d) = 1 / (1 + 10^(-d slash 400)) $
    #text(size: 16pt)[
      - odds $= 10^(d slash 400)$：每 400 分乘 10
      - Glicko-2：$d$ 收縮為 $g(sigma) d$，$g <= 1$ @glickman2001 @glickman1999
    ]
  ],
))


== 平均相同，結果不同

#figure(grid(
  columns: (auto, auto, auto),
  column-gutter: 1.2em,
  row-gutter: 0.8em,
  align: (right + horizon, left + horizon, left + horizon),
  text(fill: mute)[對局組 1], row("d", "d", "d", "d", "d", "d"), text(fill: mute)[平均得分 0.5],
  text(fill: mute)[對局組 2], row("w", "l", "w", "l", "w", "l"), text(fill: mute)[平均得分 0.5],
  [], text(size: 14pt, fill: mute)[方塊：深為勝、中為和、淺為負], [],
))

問題二：勝、和、負各自的機率如何隨 $d$ 變化。

== logistic 迴歸

#intro("methods_intuition_logistic_example", (
  ([問題], [二元結果的機率如何隨解釋變數變化]),
  ([資料], [每個觀察值一個 0 或 1 的結果與一個解釋變數]),
  ([例子], [讀書時數與是否通過]),
  ([做法], [機率取 logit 後為解釋變數的線性函數]),
))

== 機率、odds 與 logit

#figure(grid(
  columns: (auto, 1fr),
  gutter: 2.5em,
  align: horizon,
  table(
    columns: (auto, auto, auto),
    align: center,
    rule-top,
    [$mu$], [odds], [logit],
    rule-mid,
    [0.1], [1/9], [−2.2],
    [0.5], [1], [0],
    [0.9], [9], [2.2],
    rule-top,
  ),
  align(left)[
    $ Y_i | x_i tilde "Bernoulli"(mu_i) $
    $ logit mu_i = log mu_i / (1 - mu_i) = beta_0 + beta_1 x_i $
    #text(size: 16pt)[
      - $mu_i$：$P(Y_i = 1 | x_i)$
      - $logit: (0, 1) -> RR$，反函數 $mu_i = 1 slash (1 + e^(-(beta_0 + beta_1 x_i)))$
      - $e^(beta_1)$：$x$ 增加一單位，odds 的倍數
      - 最大概似：$ell = sum_i [y_i log mu_i + (1 - y_i) log(1 - mu_i)]$
    ]
  ],
))

Elo：$logit p(d) = q d$，$q = ln 10 slash 400$。

== logistic 迴歸的潛在變數

#let lgt(z) = 1 / (1 + calc.exp(-z))
#let logis(z) = calc.exp(-z) / calc.pow(1 + calc.exp(-z), 2)
// 示意的參數：eta = 0.8
#figure(grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  align: horizon,
  text(size: 13pt, cetz.canvas({
    plot.plot(
      size: (8, 3.5),
      x-label: [$Y^*$],
      y-label: [density],
      x-tick-step: none,
      x-ticks: ((0, [0]), (0.8, [$eta$])),
      y-tick-step: none,
      y-ticks: (),
      y-min: 0,
      y-max: 0.3,
      axis-style: "left",
      {
        plot.add(domain: (0, 5), z => logis(z - 0.8), fill: true, style: (stroke: none, fill: sq-light))
        plot.add(domain: (-5, 5), z => logis(z - 0.8), style: (stroke: 1.5pt + slate))
        plot.add(((0, 0), (0, 0.3)), style: (stroke: (paint: mute, dash: "dashed")))
      },
    )
    cetz.draw.content((2.4, 3.1), text(fill: mute)[$Y = 0$])
    cetz.draw.content((6.2, 3.1), text(fill: mute)[$Y = 1$])
  })),
  align(left)[
    $ Y^* = eta + epsilon, quad epsilon tilde "logistic" $
    $ P(Y = 1) = P(Y^* > 0) = F(eta) $
    $ logit P(Y = 1) = eta = beta_0 + beta_1 x $
    #text(size: 16pt)[
      - $F$：logistic 分布函數
      - 無和局時，Elo 即 $eta = q d$
    ]
  ],
))


// 三類結果的方塊；faded 表示這條 logit 不用到的類別
#let cat(s, faded: false) = box(width: 2.2em, height: 1.4em, stroke: 0.6pt + if faded { rule } else { slate }, align(center + horizon, text(fill: if faded { rule } else { ink }, s)))
#let cut = box(width: 0.1em, height: 1.8em, stroke: (left: 1.5pt + sq-dark))
#let cats(..xs) = stack(dir: ltr, spacing: 0.3em, ..xs)

== 三類：以一類為基準

#figure(grid(
  columns: (auto, auto, auto),
  column-gutter: 1.5em,
  row-gutter: 1em,
  align: (right + horizon, left + horizon, left + horizon),
  text(fill: mute)[第一條], cats(cat[$0$], cat[$0.5$], cat(faded: true)[$1$]), $log P(Y = 0.5) / P(Y = 0) = beta_(0 1) + beta_(1 1) x$,
  text(fill: mute)[第二條], cats(cat[$0$], cat(faded: true)[$0.5$], cat[$1$]), $log P(Y = 1) / P(Y = 0) = beta_(0 2) + beta_(1 2) x$,
))

- baseline-category logit @agresti2013

兩條斜率各自估計，未用到負、和、勝的順序。

== 三類：依順序切

#figure(grid(
  columns: (auto, auto, auto),
  column-gutter: 1.5em,
  row-gutter: 1em,
  align: (right + horizon, left + horizon, left + horizon),
  text(fill: mute)[第一刀], cats(cat[$0$], cut, cat[$0.5$], cat[$1$]), $logit P(Y >= 0.5) = eta + w$,
  text(fill: mute)[第二刀], cats(cat[$0$], cat[$0.5$], cut, cat[$1$]), $logit P(Y = 1) = eta - w$,
))

- 兩刀沿同一個順序切，共用 $eta$
- $w > 0$：兩條 logit 的間距
- 累積 logit @mccullagh1980

分差只經由 $eta$ 作用：強的一方同時少輸、多贏。

== 勝、和、負的有序 logistic

// 示意的參數，不是估計值：w = 0.6，eta = 0.006 d
#figure(grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  align: horizon,
  text(size: 13pt, cetz.canvas({
    plot.plot(
      size: (8, 3),
      x-label: [$Y^*$],
      y-label: [density],
      x-tick-step: none,
      x-ticks: ((-0.6, [$-w$]), (0.6, [$w$])),
      y-tick-step: none,
      y-ticks: (),
      y-min: 0,
      y-max: 0.3,
      axis-style: "left",
      {
        plot.add(domain: (-5, 5), z => logis(z - 0.5), style: (stroke: 1.5pt + slate))
        plot.add(((-0.6, 0), (-0.6, 0.3)), style: (stroke: (paint: mute, dash: "dashed")))
        plot.add(((0.6, 0), (0.6, 0.3)), style: (stroke: (paint: mute, dash: "dashed")))
      },
    )
    cetz.draw.content((1.5, 2.7), text(fill: mute)[$Y = 0$])
    cetz.draw.content((4, 2.7), text(fill: mute)[$Y = 0.5$])
    cetz.draw.content((6.5, 2.7), text(fill: mute)[$Y = 1$])
  })),
  text(size: 13pt, cetz.canvas({
    plot.plot(
      size: (8, 3),
      x-label: [$d$],
      y-label: [probability],
      x-tick-step: none,
      x-ticks: ((-400, [−400]), (0, [0]), (400, [400])),
      y-tick-step: none,
      y-ticks: ((0, [0]), (0.5, [0.5]), (1, [1])),
      y-min: 0,
      y-max: 1,
      axis-style: "left",
      {
        plot.add(domain: (-800, 800), x => 1 - lgt(0.006 * x + 0.6), style: (stroke: (paint: mute, dash: "dashed", thickness: 1.5pt)))
        plot.add(domain: (-800, 800), x => lgt(0.006 * x + 0.6) - lgt(0.006 * x - 0.6), style: (stroke: 1.5pt + sq-dark))
        plot.add(domain: (-800, 800), x => lgt(0.006 * x - 0.6), style: (stroke: 1.5pt + slate))
      },
    )
    cetz.draw.content((0.7, 2.3), text(fill: mute)[$Y = 0$])
    cetz.draw.content((5.2, 1.2), text(fill: sq-dark)[$Y = 0.5$])
    cetz.draw.content((7.3, 2.3), text(fill: slate)[$Y = 1$])
  })),
))

- $Y^* > w$：$Y = 1$；$Y^* < -w$：$Y = 0$；其間 $Y = 0.5$
- $P(Y = 0.5) = F(eta + w) - F(eta - w)$
- $eta = a + b x_d$，$x_d = d slash 100$

== 隨等級分變化的設定

#figure(table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  rule-top,
  [部分], [式子], [意義],
  rule-mid,
  [白方優勢], [$a_0 + a_1 x_r + a_2 x_r^2$], [同分時白方的潛在優勢],
  [分差], [$(b_0 + b_1 x_r) x_d$], [每 100 分使 $eta$ 移動多少],
  [RD 收縮], [$b_4 g(sigma) x_d$], [等級分不確定時效果變弱],
  [和局帶寬], [$w = exp(c_0 + c_1 x_r + c_2 x_r^2)$], [和局率隨等級分變化],
  rule-top,
))

- $x_r = (macron(r) - 1500) slash 100$；最大概似估計，和局為獨立的一類
- 期望得分：$bb(E)[Y] = P(Y = 1) + 0.5 P(Y = 0.5)$

== 系統的預測是特例

#figure(grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  align: horizon,
  text(size: 16pt, diagram(
    ..flow,
    spacing: (1.5em, 2.5em),
    node((0.5, 0), stack(spacing: 0.4em, [完整模型], [上表全部]), name: <f>),
    node((0, 1), stack(spacing: 0.4em, [有序 logistic], [$b_4 = 0$]), name: <o>),
    node((1, 1), stack(spacing: 0.4em, [Glicko 形式], [只留 $a_0, b_4, c_0$]), name: <g>),
    edge(<f>, <o>, "-|>"),
    edge(<f>, <g>, "-|>"),
  )),
  align(left)[
    $ 2 (ell_1 - ell_0) approx sum_j lambda_j chi^2_1 $
    #text(size: 16pt)[
      - $lambda_j$：受檢參數那一塊 $I^(-1) V$ 的特徵值 @rotnitzky1990
      - $V$：同一玩家的多局相依
      - proportional odds：勝的一刀另加 $delta x_d$，檢定 $delta = 0$ @peterson1990
    ]
  ],
))

Glicko 形式的斜率自由估計。


== 玩家何時停

#figure(grid(
  columns: (auto, auto),
  column-gutter: 1em,
  row-gutter: 0.5em,
  align: (right + horizon, left + horizon),
  ..(
    ("1", ("w", "w", "l", "w", "w", "d")),
    ("2", ("w", "l")),
    ("3", ("d", "w", "w", "l", "w", "w")),
    ("4", ("l",)),
    ("5", ("w", "w", "w", "l")),
  ).map(((p, ks)) => (text(fill: mute)[玩家 #p], row(..ks))).flatten(),
))

- 經驗規則：輸了就停、連輸兩局就停

問題：依過去的結果停下，能否避開表現低於期望的局。

== 公平賭局與停止策略

#intro("methods_intuition_gambler_example", (
  ([問題], [公平的賭局中，選擇何時停手能不能贏錢]),
  ([資料], [一連串的結果，給定過去，下一步的期望為 0]),
  ([例子], [擲公平硬幣，贏到 3 元就停：常小贏、偶爾大輸，平均仍為 0]),
  ([結論], [只依過去決定的停手時點，改變不了期望]),
))

每局的殘差即一步，停止規則即停手策略。

== 累積殘差是 martingale

#figure(fig("methods_intuition_paths", height: 50%))

- 殘差 $e_t = y_t - hat(mu)_t$，$hat(mu)_t$：有序 logistic 的期望得分
- $M_t = sum_(s <= t) e_s$，$bb(E)[M_(t+1) | cal(F)_t] = M_t$
- 等價於 $bb(E)[e_(t+1) | cal(F)_t] = 0$：已校準，且無延續

== 停止規則不改變期望

#figure(fig("methods_intuition_stopping", height: 62%))

- $tau$ 只依過去、有上限
- optional stopping theorem：$bb(E)[M_tau] = bb(E)[M_0] = 0$ @doob1953

規則有用 $<=>$ 某些過去之後 $bb(E)[e_(t+1) | cal(F)_t] < 0$。

== 回測

#figure(grid(
  columns: (auto, auto),
  column-gutter: 1em,
  row-gutter: 0.8em,
  align: (right + horizon, left + horizon),
  text(fill: mute)[當天], stack(dir: ltr, spacing: 0.15em, row("w", "w", "d", "w", "l"), box(width: 0.3em, height: 1.2em, stroke: (left: 1.5pt + sq-dark)), row("w", "l", "l", "w", faded: true), h(0.5em), text(size: 15pt, fill: mute)[刪去]),
  text(fill: mute)[隔天], stack(dir: ltr, spacing: 0.15em, box(stroke: 1.2pt + sq-dark, inset: 1pt, sq("l")), row("w", "d", "w")),
  [], text(size: 14pt, fill: mute)[方塊：深為勝、中為和、淺為負；豎線：規則「輸了就停」觸發],
))

- $hat(delta) = macron(e)_"drop"$；規則有用 $<=>$ $delta < 0$
- 單樣本 $t$ 檢定；2 條規則 × 2 個速度類別，Holm 校正 @holm1979
- 輔助：刪去的局對隔天第一局

== 輸後表現差的兩種解釋

#figure(grid(
  columns: (1.3fr, 1fr),
  gutter: 2em,
  align: horizon,
  text(size: 16pt, diagram(
    ..flow,
    spacing: (3em, 2em),
    node((1, 0), [當天的狀態], name: <s>, stroke: (paint: mute, dash: "dashed")),
    node((0, 1), [上一局輸贏], name: <r>),
    node((2, 1), [下一局的殘差], name: <e>),
    edge(<s>, <r>, "-|>"),
    edge(<s>, <e>, "-|>"),
    edge(<r>, <e>, "-|>", label: [輸棋本身], label-side: right, stroke: (paint: sq-dark, thickness: 1.2pt)),
  )),
  table(
    columns: (auto, auto, auto),
    align: center,
    rule-top,
    [], [輸給強敵], [輸給弱敵],
    rule-mid,
    [輸棋本身], [該停], [該停],
    [當天狀態], [不必停], [該停],
    rule-top,
  ),
))

問題四：輸後表現差，是輸棋本身，還是當天狀態？

== Arena 的隨機配對

// 四位玩家的對局沿時間排開；同一時刻空閒者組成一批，批內不看等級分配對
#figure(cetz.canvas(length: 1.35cm, {
  import cetz.draw: *
  let games = (
    ("A", ((0, 2.2), (2.6, 5.2), (5.6, 8.4))),
    ("B", ((0, 2.2), (2.6, 4.6), (5.6, 8.4))),
    ("C", ((0, 3.4), (3.8, 6.4), (6.8, 9.6))),
    ("D", ((0, 3.4), (3.8, 6.0), (6.8, 9.6))),
  )
  for (i, (who, gs)) in games.enumerate() {
    let y = -i * 0.9
    content((-0.6, y), text(size: 16pt, who))
    for (a, b) in gs {
      rect((a, y - 0.3), (b, y + 0.3), fill: sq-light, stroke: 0.5pt + sq-dark)
    }
  }
  for (x, top, bot) in ((2.6, 0, 1), (3.8, 2, 3), (5.6, 0, 1)) {
    rect((x - 0.25, -top * 0.9 + 0.45), (x + 0.25, -bot * 0.9 - 0.45), stroke: (paint: slate, dash: "dashed"), radius: 0.1)
  }
  content((4.5, -3.75), text(size: 14pt, fill: mute)[虛線框：同一時刻空閒的玩家，組成一個配對批次])
  line((0, -3.0), (10, -3.0), mark: (end: ">"), stroke: 0.5pt + mute)
  content((10, -3.35), text(size: 14pt, fill: mute)[時間])
}))

- 2014 年 arena：批內配對不看等級分，與狀態無關
- 大廳：對手由玩家挑選，不用

比較限於批內。

== 處理：對手比期望強

// 以 Elo 形式舉例的數值：A 1500 分，候選 1600、1700、1400
#figure(grid(
  columns: (1.2fr, 1fr),
  gutter: 1.5em,
  align: horizon,
  text(size: 16pt, diagram(
    ..flow,
    spacing: (3em, 1.2em),
    node((0, 1), [A：1500], name: <a>),
    node((2, 0), [B：1600，$hat(p)_B = 0.36$], name: <b>),
    node((2, 1), [C：1700，$hat(p)_C = 0.24$], name: <c>, stroke: 1.2pt + sq-dark),
    node((2, 2), [D：1400，$hat(p)_D = 0.64$], name: <d>),
    edge(<a>, <b>, "-", stroke: (paint: mute, dash: "dashed")),
    edge(<a>, <c>, "-|>", label: [實際配到], label-side: center, label-fill: white),
    edge(<a>, <d>, "-", stroke: (paint: mute, dash: "dashed")),
  )),
  align(left)[
    $ macron(p) = (0.36 + 0.24 + 0.64) / 3 = 0.41 $
    $ hat(p) = hat(p)_C = 0.24 $
    $ h = macron(p) - hat(p) = 0.17 $
  ],
))

- $hat(p)$：對實際對手；$macron(p)$：對批內候選的平均
- $h > 0$：抽到的對手比平均強

估計目標：上一局的 $h$ 對下一局殘差的效果。

== 分層比較

#intro("methods_intuition_simpson_example", (
  ([問題], [比較處理的效果，而資料來自許多水準不同的層]),
  ([資料], [每層各有處理程度不同的觀測值]),
  ([例子], [腎結石的兩種療法 @charig1986：分層看 A 都較好，合併後 B 反而較好]),
  ([做法], [模型含區集效果，只用層內的差異；即隨機完全區集設計 @fisher1935]),
))

每個配對批次為一個區集。

== 批內的迴歸

// 示意的資料，不是估計值：三批的水準不同、斜率相同
#let batches = (
  (0.12, ((-0.15, 0.03), (-0.05, -0.04), (0.02, 0.05), (0.1, -0.06), (0.18, 0.02))),
  (-0.1, ((-0.2, -0.02), (-0.08, 0.04), (0.0, -0.05), (0.06, 0.03), (0.14, -0.01))),
  (0.0, ((-0.12, -0.04), (-0.02, 0.05), (0.05, -0.03), (0.12, 0.04), (0.2, -0.02))),
)
#let slope = -0.3
#figure(grid(
  columns: (1.1fr, 1fr),
  gutter: 1.5em,
  align: horizon,
  text(size: 13pt, cetz.canvas({
    plot.plot(
      size: (9, 5.5),
      x-label: [$h$, previous game],
      y-label: [$e$, next game],
      x-tick-step: none,
      x-ticks: ((0, [0]),),
      y-tick-step: none,
      y-ticks: ((0, [0]),),
      axis-style: "left",
      {
        for (i, (lv, pts)) in batches.enumerate() {
          let c = (slate, sq-dark, mute).at(i)
          plot.add(pts.map(((x, u)) => (x, lv + slope * x + u)), mark: "o", mark-size: 0.15, style: (stroke: none), mark-style: (fill: c, stroke: none))
          plot.add(domain: (-0.22, 0.22), x => lv + slope * x, style: (stroke: 1pt + c))
        }
      },
    )
  })),
  align(left)[
    $ e_i = a_s + gamma h_i + epsilon_i $
    #text(size: 16pt)[
      - $a_s$：批 $s$ 的區集效果
      - $gamma$：各批共同的斜率
    ]
  ],
))

- 殘差對 $h$ 與批次虛擬變數的線性迴歸；$t$ 檢定 $gamma = 0$

== 識別的假設

#figure(table(
  columns: (auto, auto, auto),
  rule-top,
  [假設], [在本資料上], [檢查或處理],
  rule-mid,
  [批內配對隨機], [成立：arena 的配對規則], [一局一批的批 $h = 0$，不貢獻],
  [下一局的對手不受影響], [不成立：輸得快者較早空閒], [殘差已扣除對手強弱；其餘視為總效果],
  [續下與 $h$ 無關], [未知], [logistic 迴歸檢驗 $h$ 能否預測續下],
  [$h$ 只經由輸贏作用], [不一定：$h$ 也改變等級分的更新], [只報告抽到強敵的效果],
  rule-top,
))

決定 $hat(gamma)$ 能否解讀為上一局的效果。

== 迴歸的假設

#figure(table(
  columns: (auto, auto, auto),
  rule-top,
  [假設], [在本資料上], [檢查或處理],
  rule-mid,
  [效果對 $h$ 為線性], [未知], [加入 $h^2$ 項，以 $F$ 檢定],
  [各批的斜率相同], [未知], [依速度類別、等級分組各估 $hat(gamma)$],
  [殘差的變異數相同], [不成立：和局率隨等級變動], [以 White 穩健標準誤對照],
  rule-top,
))

影響標準誤與解讀，不影響識別。

== 前瞻驗證

#figure(grid(
  columns: (12fr, 3.6fr, 2.4fr),
  align: center,
  row-gutter: 0.3em,
  text(size: 14pt, fill: slate)[探索與估計], text(size: 14pt, fill: slate)[前瞻驗證], text(size: 14pt, fill: slate)[改制後],
  box(width: 100%, height: 0.9em, fill: sq-dark), box(width: 100%, height: 0.9em, fill: sq-light), box(width: 100%, height: 0.9em, fill: rule),
  text(size: 12pt, fill: mute)[2014], text(size: 12pt, fill: mute)[2015/1/1 至 4/19], text(size: 12pt, fill: mute)[4/20 起],
))

- 驗證：$hat(delta)$ 與 $hat(gamma)$
- 動用前 commit：假設、模型、主基準、判定規則、最小可偵測效果
- 判定：單尾，$H_0$：效果為 0 或與 2014 年反向；附兩年差的區間

#show: appendix

== 只對平均建模的校準

#figure(grid(
  columns: (1fr, 1.1fr),
  gutter: 1.5em,
  align: horizon,
  fig("methods_intuition_calibration", height: 55%),
  align(left)[
    $ logit bb(E)[Y | d] = alpha + beta q d $
    $ H_0: alpha = 0, beta = 1 $
    #text(size: 15pt)[
      - Cox 的校準迴歸 @cox1958
      - $Y in {0, 0.5, 1}$ 代入 Bernoulli 對數概似 @papke1996
      - 點比對角線平 $<=>$ $beta < 1$
    ]
  ],
))

== 參考文獻

#text(size: 11pt, columns(2, gutter: 1.5em, bibliography("refs.bib", title: none, style: "apa")))
