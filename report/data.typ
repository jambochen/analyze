#import "theme.typ": *
#import "@preview/fletcher:0.5.8": diagram, node, edge
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot
#import "@preview/oxifmt:1.0.0": strfmt

#show: deck.with(config-info(
  title: title-font[線上對戰的勝率預估],
  author: [陳力福、陳政博、李宜哲],
))

#let num(x) = strfmt("{}", int(x), fmt-thousands-separator: ",")
#let scale = csv("../outputs/tables/data_overview_scale.csv", row-type: dictionary)
#let y2014 = scale.find(r => r.year == "2014")
#let holdout = csv("../outputs/tables/data_overview_holdout.csv", row-type: dictionary).first()
#let per-player = csv("../outputs/tables/data_overview_per_player.csv", row-type: dictionary).first()
#let missing = csv("../outputs/tables/data_overview_missing.csv", row-type: dictionary).first()
#let batch = csv("../outputs/tables/arena_random_pairing_batch_size.csv", row-type: dictionary).filter(r => r.period == "2014")
#let single-share = batch.filter(r => r.n_games == "1" and r.first == "false").map(r => int(r.games)).sum() / batch.map(r => int(r.games)).sum()

#let fig(name, ..args) = image("../outputs/figures/" + name + ".svg", ..args)
#let flow = (node-stroke: 0.6pt + slate, node-inset: 0.6em, node-corner-radius: 3pt, spacing: (2.2em, 1.6em), edge-stroke: 0.6pt + slate)

#title-slide()
== 連勝連敗的爭議

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1.2em,
    align: horizon,
    {
      // 截圖已遮去發文者帳號與投票數
      let shot(name) = block(stroke: 0.5pt + rule, image("screenshots/" + name + ".png", width: 100%))
      stack(
        spacing: 0.4em,
        shot("marvelrivals_rigged"),
        shot("overwatch2_rigged"),
        shot("valorant_losers_queue"),
      )
    },
    fig("streak_chance", width: 100%),
  ),
  caption: [以投入度為目標的配對研究：EOMM @chen2017eomm、OptMatch @gong2020optmatch、EnMatch @wang2024enmatch],
)

配對規則與每局的期望勝率皆不公開，玩家與開發商的說法皆無從檢驗。

== 配對時的期望勝率

#let two(a, b) = stack(spacing: 0.35em, a, b)
#figure(text(size: 15pt, diagram(
  ..flow,
  spacing: (1.6em, 1.4em),
  node((0, 0), [排隊]),
  edge("-|>"),
  node((1, 0), two[依雙方等級分][算出期望勝率], name: <est>, stroke: 1.2pt + sq-dark),
  edge("-|>"),
  node((2, 0), text(fill: mute, two[勝率近一半][才配對]), name: <match>, stroke: (paint: mute, dash: "dashed")),
  edge("-|>"),
  node((3, 0), [對局], name: <play>),
  edge("-|>"),
  node((4, 0), [輸了], name: <lost>),
  edge("-|>"),
  node((5, 0), [停下], name: <stop>),
  edge(<lost>, <est>, "-|>", bend: -55deg, label: [再開一局], label-side: right),
  node((1, 1), text(fill: sq-dark)[研究對象], stroke: none),
  edge(<est>, (1, 1), "-", stroke: (paint: mute, dash: "dotted")),
  // 配對目標無論求公平或求留住玩家，都以期望勝率為輸入，故只檢驗輸入
  node((2, -0.9), text(size: 13pt, fill: mute)[求公平或留住玩家], stroke: none),
  edge(<match>, (2, -0.9), "-", stroke: (paint: mute, dash: "dotted")),
  node((3.5, 1.6), two[當下狀態：][剛輸、連輸、已連下幾局], name: <state>),
  edge(<state>, <play>, "-|>"),
  edge(<state>, <est>, "-|>", stroke: (paint: mute, dash: "dashed"), label: text(fill: mute)[未納入], label-side: left),
  node((5, 1), text(fill: sq-dark)[該停嗎？], stroke: none),
  edge(<stop>, (5, 1), "-", stroke: (paint: mute, dash: "dotted")),
)))

#define(stack(
  spacing: 0.6em,
  [期望勝率在哪些情況下系統性地錯？],
  [依對局條件：分差、先手、等級分的不確定性],
  [依玩家狀態：剛輸、連輸、已連下幾局],
))

== 分析目標

#figure(table(
  columns: (auto, auto),
  table.hline(stroke: 1pt + slate),
  [目標], [內容],
  table.hline(stroke: 0.5pt + rule),
  [檢驗], [期望勝率在哪些對局條件與玩家狀態下偏離實際結果],
  [預測], [依玩家當下狀態修正的勝、和、負機率],
  [建議], [輸棋之後該不該再開一局],
  table.hline(stroke: 1pt + slate),
))

結論另以預留的 2015 年資料檢驗一次。

== 資料來源與規模

#figure(fig("data_overview_timeline", height: 62%))

- Lichess 的等級分規則、配對規則與局前等級分皆公開。
- Lichess 官方資料庫的月檔，授權 CC0。
- 2013 年 1 月至 2015 年 6 月月檔中的標準棋 rated 對局，共 #num(scale.map(r => int(r.games)).sum()) 局。

== 等級分系統

#figure(
  table(
    columns: (auto, auto, auto, auto),
    table.hline(stroke: 1pt + slate),
    [系統], [記錄的量], [對手 RD 大時], [使用例],
    table.hline(stroke: 0.5pt + rule),
    [Elo], [等級分], [不考慮], [FIDE],
    [Glicko], [等級分、RD], [期望得分往 0.5 收縮], [Chess.com],
    [Glicko-2], [另加波動度], [同 Glicko], [Lichess],
    [MMR], [遊戲內部的實力分數], [多不公開], [Overwatch 2、Valorant],
    table.hline(stroke: 1pt + slate),
  ),
  caption: [出處：@elo1978、@glickman1999、@glickman2001；MMR 與可見段位分開：@riot2021rr],
)

- RD（rating deviation）：等級分的不確定程度，新帳號較大。

各系統皆以局前等級分估出期望得分，再以實際結果更新。

== 期望得分與賽後更新

#figure(table(
  columns: (auto, auto, auto),
  table.hline(stroke: 1pt + slate),
  [系統], [期望得分 $E$], [賽後更新],
  table.hline(stroke: 0.5pt + rule),
  [Elo], [$E = 1 \/ (1 + 10^(-d \/ 400))$], [$r' = r + K (S - E)$],
  [Glicko], [$E = 1 \/ (1 + 10^(-g d \/ 400))$], [調整幅度隨自身 RD 變大],
  [Glicko-2], [同 Glicko], [同 Glicko，另以波動度調整 RD],
  table.hline(stroke: 1pt + slate),
))

- $r$、$r'$：自己賽前、賽後的等級分。
- $d$：等級分差（自己 − 對手）。
- $S$：實際得分，勝 1、和 0.5、負 0。
- $K$：每局調整幅度的常數，由系統設定。
- $g = 1 \/ sqrt(1 + 3 q^2 "RD"_"對手"^2 \/ pi^2)$，$q = ln 10 \/ 400$。

RD 為 0 時 $g = 1$，期望得分與 Elo 相同。

== 期望得分曲線與和局

#figure(grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  align: horizon,
  text(size: 13pt, cetz.canvas({
    plot.plot(
      size: (9, 5.5),
      x-label: none,
      y-label: [期望得分],
      x-tick-step: none,
      x-ticks: ((-400, [−400]), (0, [0]), (400, [400])),
      y-tick-step: none,
      y-ticks: ((0, [0]), (0.5, [0.5]), (1, [1])),
      y-min: 0,
      y-max: 1,
      axis-style: "left",
      legend: "inner-north-west",
      {
        plot.add(domain: (-800, 800), x => 1 / (1 + calc.pow(10, -x / 400)), style: (stroke: 1.5pt + slate), label: [對手 RD = 0])
        // Glicko 的 g(RD)：RD = 350 為新帳號的初始值
        let g = 1 / calc.sqrt(1 + 3 * calc.pow(350 / 173.7178, 2) / calc.pow(calc.pi, 2))
        plot.add(domain: (-800, 800), x => 1 / (1 + calc.pow(10, -g * x / 400)), style: (stroke: (paint: mute, thickness: 1.5pt, dash: "dashed")), label: [對手 RD = 350])
      },
    )
    cetz.draw.content((4.5, -1.2), [等級分差（自己 − 對手）])
  })),
  fig("data_overview_result_by_gap", width: 100%),
))

各分差下皆有和局，和局的實際得分記 0.5。

== 速度類別

#figure(
  table(
    columns: (auto, auto, auto),
    table.hline(stroke: 1pt + slate),
    [類別], [估計時長], [時間制例],
    table.hline(stroke: 0.5pt + rule),
    [Bullet], [未滿 180 秒], [1+0、2+1],
    [Blitz], [180 至 479 秒], [3+0、3+2、5+0],
    [Classical], [480 秒以上], [10+0、15+15],
    table.hline(stroke: 1pt + slate),
  ),
  caption: [出處：scalachess `Speed.scala`],
)

- 時間制 3+2：每方 3 分鐘，每走一步加 2 秒。
- 估計時長 = 底秒 + 30 × 每步加秒。

三類各有一套等級分。

== 四種配對方式

#figure(table(
  columns: (auto, auto, auto, auto),
  table.hline(stroke: 1pt + slate),
  [配對方式], [誰決定對手], [依等級分篩選], [期間],
  table.hline(stroke: 0.5pt + rule),
  [大廳], [玩家自列表選擇], [可設等級分範圍], [全年],
  [指定邀戰], [玩家指定], [不適用], [全年],
  [arena], [系統指派], [不看等級分], [全年],
  [pool], [系統指派], [先看近期戰績，再看等級分], [2014 年 6 至 10 月],
  table.hline(stroke: 1pt + slate),
))

- 等級分依速度類別分開，同一類別內各配對方式共用一個等級分。
- 邀戰不進配對的分析。
- pool 的等級分自成一套，不進分析。

系統指派且不看等級分者，僅 arena。

== arena 的配對規則

#figure(
  text(size: 17pt, diagram(
    ..flow,
    node((0, 0), [加入錦標賽]),
    edge("-|>"),
    node((1, 0), [等待區], name: <wait>),
    edge("-|>"),
    node((2, -0.6), [等待逾 12 人：\ 打亂後兩兩一組], name: <many>),
    node((2, 0.6), [12 人以下：\ 避開剛對過的對手], name: <few>),
    edge(<wait>, <few>, "-|>"),
    node((3, 0), [下棋], name: <play>),
    edge(<many>, <play>, "-|>"),
    edge(<few>, <play>, "-|>"),
    edge(<play>, <wait>, "-|>", bend: 70deg, label: [下完即回到等待區，直到時間結束], label-side: center, label-fill: white),
    node(enclose: (<many>, <few>), stroke: (paint: sq-dark, dash: "dashed"), inset: 0.8em, corner-radius: 4pt),
    node((2, -1.3), text(fill: sq-dark)[不看等級分], stroke: none, fill: white),
  )),
  caption: [出處：lila 原始碼 `PairingSystem.scala`],
)

- 規則沿用至 2015 年 4 月 19 日，期間等級分照常更新。
- 2014 年 #calc.round(single-share * 100)% 的 arena 對局，配對當下只有兩人空閒。

對手是誰，取決於誰恰好同時下完。

== 自選與指派的分差

#figure(fig("data_overview_rating_gap", height: 70%))

arena 的分差比自選對手的大廳對局寬。

== 欄位與資訊時點

// 以開打為界分左右，欄位落在哪一側即決定能否用於預測
#figure(stack(
  spacing: 1em,
  grid(
    columns: (1fr, 1.5em, 1.5em, 1fr),
    row-gutter: 0.6em,
    align: (right, center, center, left),
    grid.cell(colspan: 4, align: center, text(fill: mute, size: 15pt)[開打]),
    text(fill: slate, weight: 500)[對局前已知], [], [], text(fill: slate, weight: 500)[結果之後],
    [雙方帳號], [], [], [`Result`、`Termination`],
    [開局時間、`TimeControl`], [], [], text(fill: mute)[`WhiteRatingDiff`],
    [`WhiteElo`、`BlackElo`], [], [], text(fill: mute)[`BlackRatingDiff`],
    caption[預測與基準], [], [], caption[前兩者為反應變數],
    grid.vline(x: 2, start: 1, stroke: 1pt + slate),
  ),
  block(stroke: (paint: rule, dash: "dashed"), inset: 0.6em)[月檔未含，另行取得：`source` 由 Lichess API 逐局查詢\ rating deviation 依 Glicko-2 規則自資料起點逐局重算],
))

- RatingDiff：該局使雙方等級分各加減多少。
- `WhiteElo`、`BlackElo`：該局所屬速度類別的等級分。
- 速度類別依時間制重新判定，不用月檔 `Event` 的標籤。
RatingDiff 是結果的函數，一律不用。

== 分析單位與樣本切分

// 段寬依月數比例：12、12、3.6、2.4
#figure(grid(
  columns: (12fr, 12fr, 3.6fr, 2.4fr),
  align: center + bottom,
  row-gutter: 0.3em,
  text(size: 14pt, fill: slate)[補對局歷史], text(size: 14pt, fill: slate)[全部 rated 對局], text(size: 14pt, fill: slate, two[arena][留作驗證]), text(size: 14pt, fill: slate, two[改配對][規則後]),
  box(width: 100%, height: 0.9em, fill: rule), box(width: 100%, height: 0.9em, fill: sq-dark), box(width: 100%, height: 0.9em, fill: sq-light), box(width: 100%, height: 0.9em, fill: rule),
  text(size: 12pt, fill: mute)[2013], text(size: 12pt, fill: mute)[2014], text(size: 12pt, fill: mute)[1/1–4/19], text(size: 12pt, fill: mute)[4/20–6/30],
))
#figure(text(size: 15pt, diagram(
  ..flow,
  spacing: (2.5em, 2em),
  node((2, 0), [等待區], name: <pool>),
  node((1, 1), [第 1 局], name: <g1>),
  node((2, 1), [第 2 局], name: <g2>),
  node((3, 1), [第 3 局], name: <g3>),
  node((0, 1), [玩家], stroke: none),
  edge(<g1>, <g2>, "-|>"),
  edge(<g2>, <g3>, "-|>"),
  edge(<pool>, <g1>, "-|>", label: [每局由系統指派對手], label-side: right),
  edge(<pool>, <g2>, "-|>"),
  edge(<pool>, <g3>, "-|>"),
  node((4, 1), [⋯], stroke: none),
  edge(<g3>, (4, 1), "-|>"),
  node((2, 2), text(fill: mute)[停下：任一局之後皆可], name: <stop>, stroke: (paint: mute, dash: "dashed")),
  edge(<g1>, <stop>, "-|>", stroke: (paint: mute, dash: "dashed")),
  edge(<g2>, <stop>, "-|>", stroke: (paint: mute, dash: "dashed")),
  edge(<g3>, <stop>, "-|>", stroke: (paint: mute, dash: "dashed")),
)))

- 分析單位：一位玩家的一局。
- 連續對局：同一場 arena 內的接連數局。
- 留作驗證：2015 年 1/1–4/19 的 arena，#num(holdout.games) 局、#num(holdout.players) 位玩家。

== 2014 年樣本概況

#figure(grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  align: horizon,
  fig("data_overview_per_player", width: 100%),
  fig("data_overview_composition", width: 100%),
))

- #num(y2014.players) 位玩家，其中 #num(per-player.arena_players) 位下過 arena。
- 每人局數的中位數為 #num(float(per-player.q50)) 局，前 1% 的玩家每人逾 #num(float(per-player.q99)) 局。
- 缺失：等級分 #num(missing.elo_missing) 局、未下完 #num(missing.unfinished) 局、RatingDiff #num(missing.rating_diff_missing) 局。

同一玩家的對局彼此相依。

== 限制與類推的界線

#figure(table(
  columns: (auto, auto, auto),
  table.hline(stroke: 1pt + slate),
  [], [Lichess 2014 arena], [電競排位],
  table.hline(stroke: 0.5pt + rule),
  [對手來源], [系統指派，不看等級分], [依實力排隊],
  [分差], [寬], [窄],
  [配對依據], [等待時間、上一個對手], [可能含留住玩家],
  table.hline(stroke: 1pt + slate),
))

- 資料僅含下完的對局，不含等候中放棄與取消的配對。

結論僅適用於對手不依等級分指派的情境。

#show: appendix

== 參考文獻

#text(size: 14pt, bibliography("refs.bib", title: none, style: "apa"))
