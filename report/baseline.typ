#import "theme.typ": *
#import "@preview/oxifmt:1.0.0": strfmt

#show: deck.with(config-info(
  title: title-font[局前評分與近期戰績],
  subtitle: [一對一競技對局的預測模型與結論],
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
#let scenarios = tab("likelihood_selection_scenarios").filter(r => r.speed == "Blitz" and r.fold == "Oct-Dec" and r.group == "novice" and r.previous == "loss" and r.opponent_previous == "win")
#let models = tab("likelihood_selection_report_models")
#let lrts = tab("likelihood_selection_lrt").filter(r => r.step == "recent_to_asymmetric")
#let f3(x) = strfmt("{:.3}", float(x))
#let pval(x) = if float(x) < 0.001 { [< 0.001] } else { [#f3(x)] }

#set par(leading: 0.55em)
#set table(inset: (x: 0.65em, y: 0.55em))
#show figure: set block(spacing: 0.55em)
#show figure.caption: set text(size: 0.65em)

#title-slide()

== 分析目標與研究問題
#text(size: 1.05em, font: "Noto Serif CJK TC")[評分已隨戰績更新，近期結果還有額外預測資訊嗎？]

#text(size: 0.9em, table(columns: (1.3fr, 2fr), rt,
[比較], [回答的問題], rm,
[評分基準 → 簡單近況], [雙方上一局資訊是否改善預測？],
[簡單近況 → RD 延伸], [更彈性的評分不確定性修正是否值得？],
[評分基準 → 選定模型], [整套選模流程能否改善後續預測？], rt,
))

交付局前勝、和、負機率，檢查改善幅度與適用範圍。

== 資料與評分不確定性
#stage[Lichess 西洋棋平台｜2014 年標準棋]

#text(size: 0.9em, table(columns: (auto, 1fr), rt,
[樣本], [arena 限時錦標賽，玩家可連續下多局], rm,
[rated], [標記為計分對局；不依事後是否更新評分篩選],
[速度], [Bullet、Blitz、Classical：用時由短到長，各自估計],
[RD], [rating deviation，評分的不確定程度；由歷史重建],
[高 RD 對局], [任一方 RD ≥ 110；依平台後來採用的門檻分組], rt,
))

#text(size: 0.8em)[模型使用連續 RD；分組僅用於檢查不同情境。]
#text(size: 0.65em)[圖中 novice／established：高 RD／雙方低 RD，非帳號年齡。#linebreak() 資料：#link("https://database.lichess.org/")[Lichess 官方資料庫]；RD 重建：@glickman1999 @glickman2001。]

== 資料與預測單位
#stage[兩段測試樣本｜2014 年]

#text(size: 0.9em, table(columns: (auto, auto, auto, auto), rt,
[速度], [測試期間], [實際對局], [玩家局], rm,
..samples.sorted(key: r => r.speed + r.fold).map(r => ([#r.speed], [#r.fold], [#num(r.games)], [#num(r.n)])).flatten(), rt,
))



#text(size: 0.8em)[玩家局：自身在同場錦標賽有上一局；對手可缺歷史。#linebreak() 每筆玩家局等權；雙方都符合時，同局貢獻兩筆。]

#text(size: 0.8em)[輸贏可能影響續下；結果限於實際續下且符合條件者。]

== 模型與預測輸出
#stage[有序 logistic｜負、和、勝為有順序的結果 @mccullagh1980]

$ p_"勝"=F(eta-w), quad p_"不輸"=F(eta+w) $
$ p_"和"=p_"不輸"-p_"勝", quad p_"負"=1-p_"不輸" $

#text(size: 0.8em)[$F(z)=1/(1+e^(-z))$；$eta$：白方優勢；$w>0$；兩個切點的距離 $2w$ 稱為和局帶寬。]

#table(columns: (1fr, 2fr), rt,
[優勢 $eta$], [評分、先手、RD 與近況的迴歸式], rm,
[共用優勢], [同時影響「獲勝」與「不輸」的機率],
[和局帶寬 $2w$], [優勢固定時，帶寬越大，和局機率越高], rt,
)

#text(size: 0.8em)[以賽果概似估計參數；黑方玩家局交換勝負機率。]

== 評分資訊的基準
#stage[評分基準｜白方視角，尚未加入近期戰績]

$ eta_0 = a(x)+(beta_1+beta_2 x)d+beta_3 g(sigma)d $
$ log w=c_0+c_1 x+c_2 x^2 $

#text(size: 0.95em, table(columns: (auto, 1fr), rt,
[$x,d$], [中心化並縮放的平均評分、白方減黑方的分差], rm,
[$a(x)$], [二次式；容許先手優勢隨平均評分變化],
[$sigma=sqrt("RD"_W^2+"RD"_B^2)$], [$g(sigma)$ 隨雙方不確定性增大而下降], rt,
))

#text(size: 0.75em)[$g(sigma)=(1+3q^2 sigma^2/pi^2)^(-1/2)$，$q=ln(10)/400$；$g$ 固定，$beta$ 由資料估計。]

== 雙方近況的預測修正
#stage[簡單近況｜在評分基準上加入雙方上一局資訊]

$ eta_1=eta_0+f(h_W)-f(h_B) $

#text(size: 0.9em, table(columns: (auto, 1fr), rt,
[$h_W,h_B$], [白黑雙方同場上一局的資訊], rm,
[$f$], [雙方共用係數的線性函數],
[候選資訊], [前局結果與缺歷史、評分變動，或兩者],
[評分變動], [本局與同場上一局的局前評分之差], rt,
))

#text(size: 0.8em)[對手缺歷史以指標表示；當局的 #raw("RatingDiff") 不作輸入。#linebreak() 各候選在相同訓練玩家局重估基準、近況與帶寬參數。]

== 匿名對局的預測輸出
#stage[Blitz｜訓練期實例，相同局前條件]
#let example = scenarios.find(r => r.model == "baseline")

#text(size: 0.9em)[白黑評分：#num(example.elo)／#num(example.opp_elo)；RD：#f1(example.rd)／#f1(example.opp_rd)。]

#text(size: 0.9em)[前局：白方負、黑方勝；評分變動：#f1(float(example.dr)*100)／#f1(float(example.opp_dr)*100)。]

#table(columns: (auto, auto, auto, auto), rt,
[模型], [負], [和], [勝], rm,
..(("baseline", "評分基準"), ("recent", "簡單近況")).map(pair => {
let r=scenarios.find(r => r.model == pair.at(0))
([#pair.at(1)], [#pct(r.p_loss)], [#pct(r.p_draw)], [#pct(r.p_win)])
}).flatten(), rt,
)

#text(size: 0.9em)[例子取自訓練期；整批測試才能判斷機率調整是否更準。]

== 後續時段的預測評估
#stage[訓練期共同估計與選模 → 固定後測試]

#for r in periods.filter(r => r.game == "Lichess") [
  #grid(columns: (1.5fr, auto, 1fr), gutter: 0.7em,
    flow([訓練：#r.refit], [相同玩家局共同估計#linebreak() 相依修正 BIC；AIC 對照]), [→],
    flow([測試：#r.test], [固定模型與係數#linebreak() 對數損失、得分 MSE、校準]),
  )
  #v(0.9em)
]

#text(size: 0.75em)[資訊準則採相依修正 @varin2011；損失差區間亦處理共享玩家相依。#linebreak() 訓練資料重疊；已用資料上的探索，測試結果不參與選模。]

== 近況的樣本外結果
#stage[簡單近況 − 評分基準｜負值表示改善]
#figure(fig("likelihood_selection_report_metrics", width: 100%, height: 72%, fit: "contain"), caption: [Paired differences · conditional 95% intervals])
#text(size: 0.8em)[Bullet、Blitz 全體兩段皆改善；Classical 前段不明確。]

== 歷史長度與隱藏狀態
#stage[既有探索｜作為挑選候選模型的依據]
#figure(fig("prediction_alternatives_history", width: 100%, height: 60%, fit: "contain"), caption: [Paired log-loss differences · conditional 95% intervals])

#text(size: 0.7em)[Markov：前幾局結果組合；HMM：未觀測狀態逐局轉移並影響賽果。#linebreak() 共用全年基準；Blitz 上上局有額外預測改善，HMM 未優於上一局模型。]

== RD 與近況的交互作用
#stage[RD 延伸｜評分變動的作用隨不確定程度改變]

$ eta_2=eta_1+gamma_1 (u_W s_W-u_B s_B)
 +gamma_2 (abs(u_W)s_W-abs(u_B)s_B) $

#table(columns: (auto, 1fr), rt,
[$u$：近期評分變動／100], [兩次局前評分之差], rm,
[$s$：RD／100], [以連續值表示評分不確定性],
[$gamma_1$ 項], [評分變動的作用隨 RD 改變],
[$gamma_2$ 項], [評分上升與下降可不對稱], rt,
)

W、B：白方與黑方；$eta_1$ 為近況模型優勢。

#text(size: 0.8em)[保留評分變動主項；不對稱項連同對稱交互作用納入。]

== RD 樣條模型
#stage[RD 延伸｜RD 本身以平滑曲線修正優勢 @hastie1986]

$ eta_S = eta_1 + sum_(j=1)^4 theta_j (B_j("RD"_W)-B_j("RD"_B)) $

#text(size: 0.85em, table(columns: (auto, 1fr), rt,
[自然三次樣條], [以平滑相接的分段曲線描述 RD 的作用], rm,
[$B_j$], [把 RD 轉成迴歸變數；節點只取訓練資料],
[雙方共用曲線], [白方修正減黑方修正；自由度固定],
[候選範圍], [Bullet、Blitz 與各簡單近況設定組合；Classical 未比較], rt,
))

#text(size: 0.8em)[交互作用調整評分變動的作用；此處樣條直接修正 RD 的作用。]

== BIC 選定的模型
#stage[選定模型｜全候選中相依修正 BIC 最小者]

#text(size: 0.9em, table(columns: (1.3fr, 1fr, 1fr, 1.3fr), rt,
[速度／測試期], [前局結果#linebreak() 與缺歷史], [評分變動], [RD 形式], rm,
..models.sorted(key: r => r.speed + r.fold).map(r => {
let cols = r.columns.split(",")
([#r.speed / #r.fold],
 [#if cols.contains("d_prev_win") { [保留] } else { [—] }],
 [#if cols.contains("d_dr") { [保留] } else { [—] }],
 [#if int(r.spline_parameters) > 0 { [樣條] } else if cols.contains("rd_dr") { [交互作用] } else { [—] }])
}).flatten(), rt,
))

#text(size: 0.8em)[Bullet、Blitz 選入 RD 樣條；Classical 選入前局結果。#linebreak() Classical 未列入樣條候選；「—」表示未選入其餘候選項。]

== 巢狀比較的證據
#stage[結果＋評分變動 → 加 RD 交互作用與不對稱｜LRT @rotnitzky1990]

#text(size: 0.8em, table(columns: (1.4fr, 1fr, auto, 1fr), rt,
[速度／測試期], [LR], [新增參數], [Holm p 值], rm,
..lrts.sorted(key: r => r.speed + r.fold).map(r => ([#r.speed / #r.fold], [#f2(r.lr)], [#num(r.df)], [#pval(r.p_holm)])).flatten(), rt,
))

#text(size: 0.75em)[虛無分布採相依修正的加權卡方；Holm 校正多重比較，極小 p 值為上界。#linebreak() 增項證據不等於 BIC 最佳；交互作用與樣條非巢狀。]

== RD 候選的樣本外比較
#stage[交互作用、樣條各自選定後 − 簡單近況｜負值表示改善]
#figure(fig("likelihood_selection_report_branches", width: 100%, height: 65%, fit: "contain"), caption: [Conditional 95% intervals · panels use different x-axis scales])

#text(size: 0.75em)[加入樣條後的預測表現隨速度與時段改變；Bullet 前段較簡單近況差。]

== 選定模型的樣本外結果
#stage[選定模型 − 評分基準｜負值表示改善]
#figure(fig("likelihood_selection_report_selected", width: 100%, height: 65%, fit: "contain"), caption: [Conditional 95% intervals · panels use different x-axis scales])

#text(size: 0.75em)[Bullet 前段惡化；後段三種速度的全體損失皆改善。]

== AIC 與 BIC 的敏感度
#stage[選模取捨｜AIC 作對照，主準則固定為 BIC]

#text(size: 0.68em, table(columns: (1.1fr, 1.5fr, 1.5fr, 0.8fr), rt,
[速度／測試期], [BIC 最小模型], [AIC 最小模型], [AIC 模型的 ΔBIC], rm,
..models.sorted(key: r => r.speed + r.fold).map(r => ([#r.speed / #r.fold], [#r.label], [#r.aic_label], [#f2(r.aic_choice_delta_bic)])).flatten(), rt,
))

#text(size: 0.75em)[ΔBIC：AIC 選中模型減去 BIC 最小值。#linebreak() Bullet 前段以玩家數計 BIC，會多保留前局結果。]

#text(size: 0.8em)[兩套準則選出的模型在 Bullet 前段皆變差；不依測試更換準則。]

== 改善的實際幅度
#stage[簡單近況與選定模型相對評分基準｜正值表示改善]
#figure(fig("likelihood_selection_report_gain", width: 100%, height: 72%, fit: "contain"), caption: [Relative log-loss reduction · fixed reference denominator])

#text(size: 0.9em)[$(L_"參考"-L_"候選")/L_"參考" times 100%$；越高越好，非玩家勝率增加。]

== 機率調整幅度
#stage[選定模型相對基準｜勝率變化的絕對值]
#figure(fig("likelihood_selection_report_changes", width: 100%, height: 72%, fit: "contain"), caption: [All player-game rows · percentage points])

平均與第九十分位數描述調整幅度，不代表預測改善。

== 校準檢查
#stage[Blitz 全體｜依基準預測共同分箱]
#figure(fig("likelihood_selection_report_score", width: 100%, height: 72%, fit: "contain"), caption: [Observed minus predicted mean score · common baseline bins])

零線表示組內平均相符；勝、和、負的校準須分開檢查。

== 勝和負的校準
#stage[全體玩家局｜各模型依自身預測分箱]
#figure(fig("likelihood_selection_report_classes", width: 100%, height: 65%, fit: "contain"), caption: [Descriptive binned calibration gaps · percentage points])

#text(size: 0.75em)[各類機率的分箱偏差未同步改善；Bullet 前段的勝負偏差變大。]

== 固定模型的前瞻驗證
#stage[2015 年 arena｜依前瞻登記固定模型後評估]
#figure(fig("prediction_alternatives_prospective", width: 100%, height: 58%, fit: "contain"), caption: [Fixed-model log-loss differences · conditional 95% intervals])

#text(size: 0.75em)[Bullet、Blitz 的近況模型通過登記判準；Classical 僅作描述。#linebreak() 驗證對象不包含後續 RD 交互作用或樣條候選。]

== 缺歷史的情境實驗
#stage[另納入本場首次出現的玩家局｜兩項探索使用不同參考]

#text(size: 0.85em, table(columns: (1fr, 1.4fr, 1.5fr), rt,
[增加的資訊], [比較參考], [結果], rm,
[是否為本場首局], [一般重新校準], [玩家局等權、玩家等權皆未穩定改善],
[距前次開局的間隔], [固定近況模型], [缺同場歷史組：Blitz 改善、Bullet 變差], rt,
))

#text(size: 0.8em)[間隔可跨場；查無前次開局另設缺失類別。#linebreak() 間隔包含上一局遊玩時間，不能解讀為休息效果。]

缺歷史情境尚未得到跨速度穩定的額外修正。

== 不同配對來源的預測
#stage[既有探索｜大廳下半年；同一基準與相同測試玩家局]
#figure(fig("prediction_alternatives_lobby", width: 100%, height: 60%, fit: "contain"), caption: [Lobby predictions vs RD baseline · conditional 95% intervals])

#text(size: 0.7em)[大廳：玩家自選對手；近況限同速度且開局間隔不超過一小時。#linebreak() 大廳另估有改善，直接套用 arena 係數較差；基準使用全年資料。]

== 跨遊戲的資料條件
#stage[各遊戲另估二元 logistic｜依訓練內驗證誤差選模]

#text(size: 0.85em, table(columns: (auto, 1fr, 1.5fr), rt,
[遊戲], [分組依據], [估計／選擇／測試], rm,
..periods.filter(r => r.game != "Lichess").map(r => ([#r.game], [#if r.game == "OGS" { [重建 RD] } else { [累計局數；無 RD] }], [#r.fit / #r.select / #r.test])).flatten(), rt,
))

#text(size: 0.8em)[OGS 僅有結束時間，可能出現下列順序：]
#text(size: 0.9em, grid(columns: (1fr, auto, 1fr, auto, 1fr), gutter: 0.4em,
[本局開始], [→], [「上一局」結束], [→], [本局結束]))

#text(size: 0.8em)[OGS 的前局結果未必在局前已知；AoE2 已核對上一局完成。#linebreak() 各自比較原始評分、校準基準與簡單近況，不直接套用 Lichess 係數。]

== OGS 的預測結果
#stage[圍棋｜校準與近況分開比較，負值表示改善]
#figure(fig("temporal_slides_ogs_comparisons", width: 100%, height: 60%, fit: "contain"), caption: [OGS log-loss differences · conditional 95% intervals · different x-axis scales])

#text(size: 0.8em)[校準基準改善預測；簡單近況再改善全體與高 RD 組。#linebreak() 只有結束時間排序，結果限於探索性比較。]
#text(size: 0.6em)[資料：#link("https://github.com/online-go/goratings/blob/master/data/ogs-data.db")[OGS goratings]。]

== AoE2 的預測結果
#stage[世紀帝國｜Elo 參考機率由局前分差代入 Elo 公式計算]
#figure(fig("temporal_slides_aoe2_comparisons", width: 100%, height: 60%, fit: "contain"), caption: [AoE2 log-loss differences · conditional 95% intervals · different x-axis scales])

#text(size: 0.8em)[校準相對 Elo 參考機率：全體差異不明確，低累計局數組較差。#linebreak() 簡單近況相對校準基準：全體與低累計局數組改善。]
#text(size: 0.6em)[一次擷取的單月資料；#link("https://aoe2-guide.vercel.app/data")[AoE2 粉絲網站資料]（維護者與授權條款未明）。]

== 單局預測的用途界線
#stage[序列診斷｜固定局前條件、只重抽賽果的模擬]

#table(columns: (1fr, 2.3fr), rt,
[已完成的檢查], [結果與限制], rm,
[連敗與玩家權重], [每個視窗等權、每位玩家等權的結論不同],
[累積偏差與歷史長度], [長視窗集中於少數玩家，不作一般異常門檻],
[模擬設定], [重抽賽果後仍固定評分、近況與續下，未重建回饋], rt,
)

單局預測改善，尚不足以宣告個人失常或建議停手。

== 預測模型與研究結論
#table(columns: (1fr, 2.5fr), rt,
[研究問題], [結論], rm,
[近期資訊], [Bullet、Blitz 的簡單近況有小幅預測改善；Classical 依時段而異],
[模型複雜度], [BIC 選定模型未穩定跨期改善；Bullet 前段惡化],
[適用範圍], [缺歷史修正未穩定；換來源或遊戲須重新評估], rt,
)

#text(size: 0.9em)[輸入局前評分、RD 與雙方近況，輸出勝、和、負機率。]
#text(size: 0.8em)[已保存各速度與時間切分的模型，可重現歷史驗證預測。#linebreak() 主模型限自身有同場上一局者，用於現行平台的局前預測仍待評估。]

== 方法來源
#text(size: 11pt, columns(2, gutter: 1.5em,
  bibliography("refs.bib", title: none, style: "apa")
))
