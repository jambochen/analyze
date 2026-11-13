#import "theme.typ": *
#import "@preview/oxifmt:1.0.0": strfmt

#show: deck.with(config-info(
  title: title-font[局前評分與近期戰績],
  subtitle: [一對一競技對局的預測資訊｜分析過程報告],
  author: [陳力福、陳政博、李宜哲],
))

#let tab(name) = csv("../outputs/tables/" + name + ".csv", row-type: dictionary)
#let num(x) = strfmt("{}", int(float(x)), fmt-thousands-separator: ",")
#let f1(x) = strfmt("{:.1}", float(x))
#let f2(x) = strfmt("{:.2}", float(x))
#let pct(x) = strfmt("{:.1}", 100 * float(x)) + "%"
#let fig(name, ..args) = image("../outputs/figures/" + name + ".svg", ..args)
#let rt = table.hline(stroke: 1pt + slate)
#let rm = table.hline(stroke: 0.5pt + rule)
#let flow(top, bottom) = block(width: 100%, stroke: rule, inset: 10pt, stack(dir: ttb, spacing: 5pt, top, text(size: 0.8em, fill: mute, bottom)))
#let stage(body) = block(below: 0.7em, text(size: 0.78em, fill: slate, body))
#let samples = tab("temporal_prediction_sample").filter(r => r.game == "Lichess" and r.segment == "test")
#let periods = tab("temporal_slides_periods")
#let scenarios = tab("prediction_value_scenarios").filter(r => r.speed == "Blitz" and r.fold == "Oct-Dec" and r.group == "novice" and r.previous == "loss" and r.opponent_previous == "win")
#let selection = tab("temporal_prediction_selection").filter(r => r.game == "Lichess")
#let models = tab("temporal_prediction_models").filter(r => r.game == "Lichess")

#set par(leading: 0.55em)
#set table(inset: (x: 0.65em, y: 0.55em))
#show figure: set block(spacing: 0.55em)
#show figure.caption: set text(size: 0.65em)

#title-slide()

== 評分與本次問題
#text(size: 1.25em, font: "Noto Serif CJK TC")[已知局前評分，#linebreak() 再看上一局能否預測得更準？]

#grid(columns: (1fr, auto, 1fr, auto, 1fr), gutter: 0.5em,
flow([評分與 RD], [基準模型]), [→],
flow([加入近況], [雙方上一局]), [→],
flow([RD 交互作用], [不確定性與近況]))

比較後續對局的誤差、改善幅度與分組差異。

== 資料與評分不確定性
#stage[Lichess｜2014 年標準棋]

#text(size: 0.9em, table(columns: (auto, 1fr), rt,
[rated 對局], [會納入平台評分更新的對局], rm,
[基準／近況樣本], [可辨識來源的全站資料／arena 限時錦標賽],

[速度], [Bullet、Blitz、Classical：用時由短到長，各自估計],
[RD], [rating deviation；評分不確定程度；重建時受初始歷史與更新順序限制],
[分組], [任一方 RD ≥ 110：高 RD 對局；其餘為雙方低 RD], rt,
))

#text(size: 0.75em, grid(columns: (1fr, auto, 1fr), gutter: 0.5em,
[一場實際對局], [→], [自身須有同場上一局#linebreak() 對手可缺歷史，以指標表示]))

#text(size: 0.75em)[novice／established：高 RD／雙方低 RD，非帳號年齡。]

== 資料與預測單位
#stage[兩段時間外測試樣本｜2014 年]

#text(size: 0.9em, table(columns: (auto, auto, auto, auto), rt,
[速度], [測試期間], [實際對局], [玩家局], rm,
..samples.sorted(key: r => r.speed + r.fold).map(r => ([#r.speed], [#r.fold], [#num(r.games)], [#num(r.n)])).flatten(), rt,
))



玩家局：自身在本場有上一局者，同局可含雙方視角。

沒有續下者不補造下一局，預測只涵蓋實際對局。

== 模型與預測輸出
#stage[有序 logistic｜勝、和、負]

$ p_"勝"=F(eta-w), quad p_"不輸"=F(eta+w), quad p_"和"=p_"不輸"-p_"勝" $
#text(size: 0.7em)[$F(z)=1/(1+e^(-z))$；$eta$：白方優勢；$w>0$：和局帶寬的一半。]

#text(size: 0.9em, table(columns: (auto, 1fr), rt,
[評分基準], [分差、平均評分、先手與雙方 RD], rm,
[近況修正], [固定基準，加雙方前局結果、缺歷史與評分變動候選],
[RD 延伸], [允許評分變動的修正幅度隨 RD 改變], rt,
))

固定優勢時，$w$ 越大，和局機率越高；以概似估計參數。

== 評分資訊的基準
#stage[白方視角｜評分與 RD]

$ eta_0 = a(x)+(beta_1+beta_2 x)d+beta_3 g(sigma)d $
$ log w=c_0+c_1 x+c_2 x^2 $

#text(size: 0.95em, table(columns: (auto, 1fr), rt,
[$x,d$], [中心化並縮放的平均評分、白方減黑方的分差], rm,
[$a(x)$], [二次式；容許先手優勢隨平均評分變化],
[$sigma=sqrt("RD"_W^2+"RD"_B^2)$], [$g(sigma)$ 隨雙方不確定性增大而下降], rt,
))

#text(size: 0.75em)[$g(sigma)=(1+3q^2 sigma^2/pi^2)^(-1/2)$，$q=ln(10)/400$；$g$ 固定，$beta$ 由資料估計。]



== 雙方近況的增量
#stage[基準作 offset｜只估近況係數]

$ eta_1=eta_0+f(h_W)-f(h_B) $

#table(columns: (auto, 1fr), rt,
[$h_W,h_B$], [白黑雙方的同場上一局結果、評分變動與缺歷史標記], rm,
[$f$], [雙方共用係數的線性函數],
[候選順序], [先加入結果與缺歷史，再考慮評分變動],
[評分變動], [本局與同場上一局的局前評分之差], rt,
)

當局 RatingDiff 含本局結果，不作輸入；和局帶寬固定。

== RD 與近況的交互作用
#stage[RD × 評分變動｜再考慮升降不對稱]

$ eta_2=eta_1+gamma_1 (u_W s_W-u_B s_B)
 +gamma_2 (abs(u_W)s_W-abs(u_B)s_B) $

#table(columns: (auto, 1fr), rt,
[$u$：近期評分變動／100], [兩次局前評分之差], rm,
[$s$：RD／100], [連續不確定性，不切成固定能力群體],
[$gamma_1$ 項], [評分變動的作用隨 RD 改變],
[$gamma_2$ 項], [評分上升與下降可不對稱], rt,
)

W、B：白方與黑方；$eta_1$ 為近況模型優勢。

#text(size: 0.8em)[依序檢查；交互作用連同必要主項加入。]

== 匿名對局的預測輸出
#stage[Blitz 後段模型｜訓練期選例、白方視角]
#let example = scenarios.find(r => r.model == "baseline")

#text(size: 0.9em)[白黑評分：#num(example.elo)／#num(example.opp_elo)；RD：#f1(example.rd)／#f1(example.opp_rd)。]

#text(size: 0.9em)[前局：白方負、黑方勝；評分變動：#f1(float(example.dr)*100)／#f1(float(example.opp_dr)*100)。]

#table(columns: (auto, auto, auto, auto), rt,
[模型], [負], [和], [勝], rm,
..(("baseline", "評分與 RD"), ("recent", "加入近況"), ("novice_extension", "加入 RD 交互作用")).map(pair => {
let r=scenarios.find(r => r.model == pair.at(0))
([#pair.at(1)], [#pct(r.p_loss)], [#pct(r.p_draw)], [#pct(r.p_win)])
}).flatten(), rt,
)

#text(size: 0.9em)[固定同一局前條件；非前局輸贏的因果比較。]

== 時間外驗證
#stage[估計 → 選擇變數 → 重估 → 測試]

#for r in periods.filter(r => r.game == "Lichess") [
  #text(size: 0.9em, grid(columns: (1fr, auto, 1fr, auto, 1fr), gutter: 0.4em,
    block(width: 100%, stroke: rule, inset: 6pt)[估計係數#linebreak() #r.fit], [→],
    block(width: 100%, stroke: rule, inset: 6pt)[選擇變數#linebreak() #r.select], [→],
    block(width: 100%, stroke: slate, inset: 6pt)[固定模型測試#linebreak() #r.test],
  ))
  #grid(columns: (2fr, 1fr), gutter: 0.8em, block(stroke: (bottom: 0.7pt + rule), inset: 4pt)[#text(size: 0.7em)[合併 #r.refit：先估基準，再固定估近況]], [])
  #v(0.6em)
]

#text(size: 0.9em)[訓練資料重疊，兩次結果分開呈現。#linebreak() 已用資料的時間外探索；區塊寬度不代表期間長度。]

== 模型選擇與評估
#stage[同一批對局｜成對比較損失]

$ L=-1/n sum_(k=1)^n log p_k(y_k), quad Delta L=L_"候選"-L_"參考" $

$n$：玩家局數；$p_k(y_k)$：給實際勝、和、負結果的機率。

#text(size: 0.95em, table(columns: (auto, 1fr), rt,
[近況變數], [訓練內部全體對數損失差的區間上界低於零], rm,
[RD 延伸變數], [同樣判準，評估訓練內部的高 RD 組],
[最終測試], [分組測試；log loss、得分 MSE 與校準], rt,
))

#text(size: 0.9em)[共享任一玩家者納入共變異；區間不含重訓與選項變異。]

== 實際選入項目
#stage[內部驗證選定｜測試期不再選項]

#text(size: 0.95em, table(columns: (1.4fr, 1fr, 1fr, 1fr, 1fr), rt,
[速度／期間], [#stack(dir: ttb, spacing: 5pt, [前局結果], [缺歷史])], [評分變動], [RD × 變動], [升降#linebreak() 不對稱], rm,
..models.filter(r => r.model == "novice_extension").sorted(key: r => r.speed + r.fold).map(r => {
  let recent = models.find(m => m.speed == r.speed and m.fold == r.fold and m.model == "recent")
  let cols = r.columns.split(",")
  let rc = recent.columns.split(",")
  let rd = selection.find(m => m.speed == r.speed and m.fold == r.fold and m.step == "rd_interaction")
  ([#r.speed / #r.fold],
   [#if cols.contains("d_prev_win") { [保留] } else { [—] }],
   [#if rc.contains("d_dr") { [近況選入] } else if cols.contains("d_dr") { [隨延伸加入] } else { [—] }],
   [#if rd.kept == "true" { [該步選入] } else if cols.contains("rd_dr") { [隨後項加入] } else { [—] }],
   [#if cols.contains("rd_abs_dr") { [保留] } else { [—] }])
}).flatten(), rt,
))

#text(size: 0.9em)[隨項加入：保留主項與較低階項；非單獨通過選擇。]

== 近況的時間外結果
#stage[近況 − 基準｜負值表示改善]
#figure(fig("prediction_report_metrics", width: 100%, height: 72%, fit: "contain"), caption: [Paired differences · conditional 95% intervals])
#text(size: 0.75em)[$Y in {0,1/2,1}$；$hat(e)=p_"勝"+p_"和"/2$；得分 MSE：$"mean"((Y-hat(e))^2)$。]
Classical 未選入候選，零代表模型相同。

== RD 延伸的結果
#stage[RD 延伸 − 近況｜負值表示改善]
#figure(fig("temporal_slides_lichess_novice_extension", width: 100%, height: 72%, fit: "contain"), caption: [Log-loss differences · conditional 95% intervals])

== 改善的實際幅度
#stage[相對損失下降｜正值表示改善]
#figure(fig("prediction_value_lichess_gain", width: 100%, height: 72%, fit: "contain"), caption: [Relative log-loss reduction · fixed reference denominator])

#text(size: 0.9em)[$(L_"參考"-L_"候選")/L_"參考" times 100%$；越高越好，非玩家勝率增加。]

== 機率調整幅度
#stage[近況相對基準｜勝率變化的絕對值]
#figure(fig("prediction_report_changes", width: 100%, height: 72%, fit: "contain"), caption: [All player-game rows · percentage points])

平均與第九十分位數描述調整幅度，不代表預測改善。

== 校準檢查
#stage[Blitz 全體｜依基準預測共同分箱]
#figure(fig("prediction_report_calibration", width: 100%, height: 72%, fit: "contain"), caption: [Observed minus predicted mean score · common baseline bins])

零線表示組內平均相符；此圖不取代勝、和、負各自的校準。

== 缺歷史的適用範圍
#stage[納入首局｜間隔模型相對固定近況模型]
#figure(fig("entry_context_history", width: 100%, height: 72%, fit: "contain"), caption: [Interval minus fixed recent · log-loss differences])

#text(size: 0.7em)[放寬自身須有同場上一局；間隔取資料內前次開局，可跨場。#linebreak() 找不到前次開局另設缺失類別，非長間隔。]

== 跨遊戲的資料條件
#stage[各遊戲另估二元 logistic]

#table(columns: (auto, 1fr, 1fr), rt,
[資料], [分組依據], [資料限制], rm,
[OGS 圍棋], [重建 RD], [只有結束時間],
[AoE2 世紀帝國], [累計局數；無 RD], [單月；已核對上一局完成], rt,)

#text(size: 0.75em)[OGS 可能情境（示意，非實際時間比例）：]
#text(size: 0.9em, grid(columns: (1fr, auto, 1fr, auto, 1fr), gutter: 0.4em,
[本局開始], [→], [「上一局」結束], [→], [本局結束]))

依結束時間排序，不能保證開局時已知前局結果。

== 跨遊戲的預測結果
#stage[近況 − 各遊戲基準｜負值表示改善]
#grid(columns: (1fr, 1fr), gutter: 0.8em,
figure(fig("temporal_slides_ogs_recent", width: 100%), caption: [OGS: end-time ordering only]),
figure(fig("temporal_slides_aoe2_recent", width: 100%), caption: [AoE2: one-month snapshot]),
)

OGS 高 RD 組與 AoE2 低累計局數組的改善較集中。

== 序列診斷的設計
#stage[固定特徵、重抽賽果｜每次生成整批對局]

#grid(columns: (1fr, auto, 1fr, auto, 1fr), gutter: 0.4em,
flow([固定局前機率], [對手、近況與續下]), [→],
flow([每局抽一個結果], [換成雙方視角]), [→],
flow([計算視窗摘要], [重複模擬得參考範圍]))

#text(size: 0.9em, grid(columns: (1fr, auto, 1fr), gutter: 0.5em,
[本局模擬：輸改為贏], [↛], [下一局評分與近況仍固定]))

和局中斷連敗；視窗等權與玩家等權分別計算。

人工參考範圍，不是信賴區間或失常檢定。

== 連敗與玩家權重
#stage[五局連敗｜觀察比例 ÷ 固定機率參考比例]
#figure(fig("prediction_report_sequences", width: 100%, height: 72%, fit: "contain"), caption: [Faint bars: artificial reference ranges, not confidence intervals])

#text(size: 0.75em)[同一玩家、同場連續五局，可重疊；五局全輸記為連敗。#linebreak() 參考線一：觀察與參考比例相同。]

== 累積偏差的定義
#stage[固定長度視窗｜總得分偏離預期的程度]

$ Q_(i,K) = (sum_(t=i)^(i+K-1)(Y_t-e_t))^2 / (sum_(t=i)^(i+K-1)v_t) $
$ e_t=p_(t,"勝")+p_(t,"和")/2, quad v_t=p_(t,"勝")+p_(t,"和")/4-e_t^2 $

$Y_t$：得分；$K$：視窗長度；$e_t,v_t$：局前模型的均值與變異。

固定機率且人工獨立時，$E(Q_(i,K))=1$。

同一次模擬共用整批賽果，保留重疊視窗的相依。

== 累積偏差與歷史支持
#stage[近況模型｜偏差與可用玩家數一起看]
#figure(fig("prediction_report_support", width: 100%, height: 72%, fit: "contain"), caption: [Recent model · descriptive summaries])
#text(size: 0.9em)[上：累積偏差，參考值一；下：可用玩家數，對數刻度。]

== 目前判斷與下一步
#table(columns: (1fr, 2.5fr), rt,
[已完成的比較], [目前判斷], rm,
[預測], [部分速度與時段改善；高 RD 組較集中],
[適用情境], [首局修正不一致；跨遊戲須核對資訊時間],
[連續戰績], [權重與歷史長度改變摘要，尚不足以判定異常], rt,
)

#text(size: 0.9em)[下一步：新假設另留未用資料；序列模型處理評分回饋與續下。]
