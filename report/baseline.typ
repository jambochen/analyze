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
#let time-labels = (Bullet: [短用時], Blitz: [中用時], Classical: [長用時])
#let time-order = (Bullet: "1", Blitz: "2", Classical: "3")
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
[評分基準 → 加入上一局資訊], [雙方上一局資訊是否改善預測？],
[再加交互作用或樣條], [交互作用或樣條能否再改善預測？],
[評分基準 → 選定模型], [整套選模流程能否改善後續預測？], rt,
))

交付局前勝、和、負機率，檢查改善幅度與適用範圍。

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



#text(size: 0.8em)[每筆觀測是一位玩家在一局中的紀錄。#linebreak() 納入條件：該玩家在這場錦標賽已有先前紀錄；對手可無。#linebreak() 每筆紀錄等權；雙方都符合時，同局貢獻兩筆。]

#text(size: 0.8em)[輸贏可能影響繼續下棋；結果限於實際繼續下棋且符合條件者。]

== 上次的預測模型
#stage[含 RD 的有序 logistic｜以局前評分資訊預測勝、和、負 @mccullagh1980]

$ eta_0 = a(x)+(beta_1+beta_2 x)d+beta_3 g(sigma)d $
$ log w=c_0+c_1 x+c_2 x^2 $

#text(size: 0.95em, table(columns: (auto, 1fr), rt,
[$eta_0, 2w$], [白方優勢、兩個切點的距離], rm,
[$x,d$], [中心化並縮放的平均評分、白方減黑方的分差],
[$a(x)$], [二次式；容許先手優勢隨平均評分變化],
[$sigma=sqrt("RD"_W^2+"RD"_B^2)$], [$g(sigma)$ 隨雙方不確定性增大而下降], rt,
))

#text(size: 0.75em)[$g(sigma)=(1+3q^2 sigma^2/pi^2)^(-1/2)$，$q=ln(10)/400$；$g$ 固定，$beta$ 由資料估計。]

#text(size: 0.8em)[沿用模型形式；本次各候選在相同錦標賽訓練資料上重估全部參數。]

== 加入雙方上一局資訊
#stage[在評分基準上加入雙方上一局資訊]

$ eta_1=eta_0+f(h_W)-f(h_B) $

#text(size: 0.9em, table(columns: (auto, 1fr), rt,
[$h_W,h_B$], [白黑雙方在這場錦標賽各自上一局的資訊], rm,
[$f$], [雙方共用係數的線性函數],
[候選資訊], [上一局結果、紀錄缺失指示變數、評分變動],
[評分變動], [本局與這場錦標賽上一局的局前評分之差], rt,
))

#text(size: 0.8em)[對手無上一局紀錄，以指示變數表示；當局的 #raw("RatingDiff") 不作輸入。#linebreak() 各候選在相同訓練資料重估基準、上一局資訊與切點參數。]

== 匿名對局的預測輸出
#stage[中用時｜訓練期實例，相同局前條件]
#let example = scenarios.find(r => r.model == "baseline")

#text(size: 0.9em)[白黑評分：#num(example.elo)／#num(example.opp_elo)；RD：#f1(example.rd)／#f1(example.opp_rd)。]

#text(size: 0.9em)[前局：白方負、黑方勝；評分變動：#f1(float(example.dr)*100)／#f1(float(example.opp_dr)*100)。]

#table(columns: (auto, auto, auto, auto), rt,
[模型], [負], [和], [勝], rm,
..(("baseline", "評分基準"), ("recent", "加入上一局資訊")).map(pair => {
let r=scenarios.find(r => r.model == pair.at(0))
([#pair.at(1)], [#pct(r.p_loss)], [#pct(r.p_draw)], [#pct(r.p_win)])
}).flatten(), rt,
)

#text(size: 0.9em)[例子取自訓練期；整批測試才能判斷機率調整是否更準。]

== 後續時段的預測評估
#stage[訓練期共同估計與選模 → 固定後測試]

#for r in periods.filter(r => r.game == "Lichess") [
  #grid(columns: (1.5fr, auto, 1fr), gutter: 0.7em,
    flow([訓練：#r.refit], [以相同資料估計參數#linebreak() 相依修正 BIC 選模]), [→],
    flow([測試：#r.test], [固定模型與係數#linebreak() 對數損失、得分 MSE、校準]),
  )
  #v(0.9em)
]

#text(size: 0.75em)[資訊準則採相依修正 @varin2011；損失差區間亦處理同一玩家多次出現造成的相依。#linebreak() 訓練資料重疊；已用資料上的探索，測試結果不參與選模。]

== 加入上一局資訊的預測結果
#stage[加入上一局資訊後的模型 − 評分基準｜負值表示改善]
#figure(fig("likelihood_selection_report_metrics", width: 100%, height: 72%, fit: "contain"), caption: [Paired differences · conditional 95% intervals])
#text(size: 0.8em)[短用時、中用時全體兩段皆改善；長用時前段不明確。]

== 更早的賽果與隱藏狀態
#stage[既有探索｜作為挑選候選模型的依據]
#figure(fig("prediction_alternatives_history", width: 100%, height: 60%, fit: "contain"), caption: [Paired log-loss differences · conditional 95% intervals])

#text(size: 0.7em)[Markov：前幾局結果組合；HMM：未觀測狀態逐局轉移並影響賽果。#linebreak() 共用全年基準；中用時上上局有額外預測改善，HMM 未優於上一局模型。]

== RD 與評分變動的交互作用
#stage[評分變動的作用隨不確定程度改變]

$ eta_2=eta_1+gamma_1 (u_W s_W-u_B s_B)
 +gamma_2 (abs(u_W)s_W-abs(u_B)s_B) $

#table(columns: (auto, 1fr), rt,
[$u$：近期評分變動／100], [兩次局前評分之差], rm,
[$s$：RD／100], [以連續值表示評分不確定性],
[$gamma_1$ 項], [評分變動的作用隨 RD 改變],
[$gamma_2$ 項], [評分上升與下降可不對稱], rt,
)

W、B：白方與黑方；$eta_1$ 為加入上一局資訊後的白方優勢。

#text(size: 0.8em)[保留評分變動主項；不對稱項連同對稱交互作用納入。]

== RD 樣條模型
#stage[RD 本身以平滑曲線修正優勢 @hastie1986]

$ eta_S = eta_1 + sum_(j=1)^4 theta_j (B_j("RD"_W)-B_j("RD"_B)) $

#text(size: 0.85em, table(columns: (auto, 1fr), rt,
[自然三次樣條], [以平滑相接的分段曲線描述 RD 的作用], rm,
[$B_j$], [把 RD 轉成迴歸變數；節點只取訓練資料],
[雙方共用曲線], [白方修正減黑方修正；自由度固定],
[候選範圍], [短用時、中用時與各種上一局資訊組合；長用時未比較], rt,
))

#text(size: 0.8em)[交互作用調整評分變動的作用；此處樣條直接修正 RD 的作用。]

== BIC 選定的模型
#stage[選定模型｜全候選中相依修正 BIC 最小者]

#text(size: 0.9em, table(columns: (1.3fr, 1fr, 1fr, 1.3fr), rt,
[用時類別／測試期], [上一局結果#linebreak() 與紀錄缺失], [評分變動], [RD 形式], rm,
..models.sorted(key: r => time-order.at(r.speed) + r.fold).map(r => {
let cols = r.columns.split(",")
([#time-labels.at(r.speed) / #r.fold],
 [#if cols.contains("d_prev_win") { [保留] } else { [—] }],
 [#if cols.contains("d_dr") { [保留] } else { [—] }],
 [#if int(r.spline_parameters) > 0 { [樣條] } else if cols.contains("rd_dr") { [交互作用] } else { [—] }])
}).flatten(), rt,
))

#text(size: 0.8em)[短用時、中用時選入 RD 樣條；長用時選入前局結果。#linebreak() 長用時未列入樣條候選；「—」表示未選入其餘候選項。]

== 巢狀比較的證據
#stage[結果＋評分變動 → 加 RD 交互作用與不對稱｜LRT @rotnitzky1990]

#text(size: 0.8em, table(columns: (1.4fr, 1fr, auto, 1fr), rt,
[用時類別／測試期], [LR], [新增參數], [Holm p 值], rm,
..lrts.sorted(key: r => time-order.at(r.speed) + r.fold).map(r => ([#time-labels.at(r.speed) / #r.fold], [#f2(r.lr)], [#num(r.df)], [#pval(r.p_holm)])).flatten(), rt,
))

#text(size: 0.75em)[虛無分布採相依修正的加權卡方；Holm 校正多重比較，極小 p 值為上界。#linebreak() 增項證據不等於 BIC 最佳；交互作用與樣條非巢狀。]

== 交互作用與樣條的預測結果
#stage[再加交互作用或樣條 − 只加上一局資訊｜負值表示改善]
#figure(fig("likelihood_selection_report_branches", width: 100%, height: 65%, fit: "contain"), caption: [Conditional 95% intervals · panels use different x-axis scales])

#text(size: 0.75em)[加入樣條後的預測表現隨用時類別與時段改變；短用時前段比只加入上一局資訊的模型差。]

== 選定模型的樣本外結果
#stage[選定模型 − 評分基準｜負值表示改善]
#figure(fig("likelihood_selection_report_selected", width: 100%, height: 65%, fit: "contain"), caption: [Conditional 95% intervals · panels use different x-axis scales])

#text(size: 0.75em)[短用時前段惡化；後段三種用時類別的全體損失皆改善。]

== 改善的實際幅度
#stage[兩種模型都與評分基準比較｜正值表示改善]
#figure(fig("likelihood_selection_report_gain", width: 100%, height: 72%, fit: "contain"), caption: [Relative log-loss reduction · fixed reference denominator])

#text(size: 0.9em)[$(L_"參考"-L_"候選")/L_"參考" times 100%$；越高越好，非玩家勝率增加。]

== 機率調整幅度
#stage[選定模型相對基準｜勝率變化的絕對值]
#figure(fig("likelihood_selection_report_changes", width: 100%, height: 72%, fit: "contain"), caption: [All player-game rows · percentage points])

平均與第九十分位數描述調整幅度，不代表預測改善。

== 校準檢查
#stage[中用時全體｜依基準預測共同分箱]
#figure(fig("likelihood_selection_report_score", width: 100%, height: 72%, fit: "contain"), caption: [Observed minus predicted mean score · common baseline bins])

零線表示組內平均相符；勝、和、負的校準須分開檢查。

== 勝和負的校準
#stage[全部紀錄｜各模型依自身預測分箱]
#figure(fig("likelihood_selection_report_classes", width: 100%, height: 65%, fit: "contain"), caption: [Descriptive binned calibration gaps · percentage points])

#text(size: 0.75em)[各類機率的分箱偏差未同步改善；短用時前段的勝負偏差變大。]

== 固定模型的前瞻驗證
#stage[2015 年 arena｜依前瞻登記固定模型後評估]
#figure(fig("prediction_alternatives_prospective", width: 100%, height: 58%, fit: "contain"), caption: [Fixed-model log-loss differences · conditional 95% intervals])

#text(size: 0.75em)[短用時、中用時加入上一局資訊後，通過登記判準；長用時僅作描述。#linebreak() 驗證對象不包含後續 RD 交互作用或樣條候選。]

== 沒有先前對局紀錄時的預測
#stage[另納入玩家在這場錦標賽首次出現的對局紀錄｜兩項探索使用不同參考]

#text(size: 0.85em, table(columns: (1fr, 1.4fr, 1.5fr), rt,
[增加的資訊], [比較參考], [結果], rm,
[是否為玩家在這場錦標賽首次出現], [重新估計校準參數], [每筆紀錄或每位玩家等權，皆未穩定改善],
[距前次開局的間隔], [固定、含上一局資訊的模型], [無這場錦標賽先前紀錄者：中用時改善、短用時變差], rt,
))

#text(size: 0.8em)[前次開局不限這場錦標賽；查無紀錄另設缺失類別。#linebreak() 間隔包含上一局遊玩時間，不能解讀為休息效果。]

沒有先前紀錄時，兩種用時類別未得到一致的預測改善。

== 不同配對來源的預測
#stage[既有探索｜大廳下半年；同一基準與相同測試資料]
#figure(fig("prediction_alternatives_lobby", width: 100%, height: 60%, fit: "contain"), caption: [Lobby predictions vs RD baseline · conditional 95% intervals])

#text(size: 0.7em)[大廳：玩家自選對手；上一局須為同用時類別，開局間隔不超過一小時。#linebreak() 大廳另估有改善，直接套用 arena 係數較差；基準使用全年資料。]

== 跨遊戲的資料條件
#stage[各遊戲另估二元 logistic｜依訓練內驗證誤差選模]

#text(size: 0.85em, table(columns: (auto, 1fr, 1.5fr), rt,
[遊戲], [分組依據], [估計／選擇／測試], rm,
..periods.filter(r => r.game != "Lichess").map(r => ([#r.game], [#if r.game == "OGS" { [重建 RD] } else { [累計局數；無 RD] }], [#r.fit / #r.select / #r.test])).flatten(), rt,
))

#text(size: 0.8em)[OGS 僅有結束時間，可能出現下列順序：]
#text(size: 0.9em, grid(columns: (1fr, auto, 1fr, auto, 1fr), gutter: 0.4em,
[本局開始], [→], [「上一局」結束], [→], [本局結束]))

#text(size: 0.8em)[OGS 的前局結果未必在局前已知；AoE2 已核對上一局完成。#linebreak() 各自比較校準前後、加入上一局資訊前後，不直接套用 Lichess 係數。]

== OGS 的預測結果
#stage[圍棋另分短、長用時｜圖中 Short、Long，負值表示改善]
#figure(fig("temporal_slides_ogs_comparisons", width: 100%, height: 60%, fit: "contain"), caption: [OGS log-loss differences · conditional 95% intervals · different x-axis scales])

#text(size: 0.8em)[校準後的模型改善預測；再加入上一局資訊後，全體與高 RD 組改善。#linebreak() 只有結束時間排序，結果限於探索性比較。]
#text(size: 0.6em)[資料：#link("https://github.com/online-go/goratings/blob/master/data/ogs-data.db")[OGS goratings]。]

== AoE2 的預測結果
#stage[世紀帝國｜Elo 參考機率由局前分差代入 Elo 公式計算]
#figure(fig("temporal_slides_aoe2_comparisons", width: 100%, height: 60%, fit: "contain"), caption: [AoE2 log-loss differences · conditional 95% intervals · different x-axis scales])

#text(size: 0.8em)[校準相對 Elo 參考機率：全體差異不明確，低累計局數組較差。#linebreak() 校準後再加入上一局資訊：全體與低累計局數組改善。]
#text(size: 0.6em)[一次擷取的單月資料；#link("https://aoe2-guide.vercel.app/data")[AoE2 粉絲網站資料]（維護者與授權條款未明）。]

== 單局預測的用途界線
#stage[連敗與累積得分偏差｜固定局前條件、只重抽賽果的模擬]

#table(columns: (1fr, 2.3fr), rt,
[已完成的檢查], [結果與限制], rm,
[連敗與玩家權重], [每段連續對局或每位玩家等權，結論不同],
[累積得分偏差與對局數], [較長的連續紀錄集中於少數玩家，不作一般異常門檻],
[模擬設定], [重抽賽果後，評分、上一局資訊與是否繼續下棋仍固定], rt,
)

單局預測改善，尚不足以宣告個人失常或建議停手。

== 預測模型與研究結論
#table(columns: (1fr, 2.5fr), rt,
[研究問題], [結論], rm,
[近期資訊], [短用時、中用時加入上一局資訊後有小幅改善；長用時依時段而異],
[模型複雜度], [BIC 選定模型未穩定跨期改善；短用時前段惡化],
[適用範圍], [無先前紀錄時改善不一致；換來源或遊戲須重新評估], rt,
)

#text(size: 0.9em)[輸入局前評分、RD 與雙方上一局資訊，輸出勝、和、負機率。]
#text(size: 0.8em)[已保存各用時類別與時間切分的模型，可重現歷史驗證預測。#linebreak() 評估限於在這場錦標賽已有先前紀錄的玩家；現行平台仍待驗證。]

== 方法來源
#text(size: 11pt, columns(2, gutter: 1.5em,
  bibliography("refs.bib", title: none, style: "apa")
))
