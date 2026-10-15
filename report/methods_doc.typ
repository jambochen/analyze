#import "@preview/cetz:0.4.2"

#set document(title: "美國航班延誤成因申報的責任歸因稽核：方法與分析計畫")
#set page(paper: "a4", margin: (x: 2.6cm, y: 2.4cm), numbering: "1")
#set text(font: ("Noto Sans CJK TC",), lang: "zh", region: "tw", size: 11pt)
#set par(justify: true, leading: 0.85em, first-line-indent: 0em)
#show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans CJK TC"), size: 9.5pt)
#set heading(numbering: "1.1")
#show heading: it => block(above: 1.4em, below: 0.7em)[#it]
#show heading.where(level: 1): it => block(above: 1.8em, below: 0.9em)[#it]
#set figure(gap: 0.8em)
#set figure(supplement: [Figure])
#show figure.where(kind: table): set figure(supplement: [Table])
#set math.equation(numbering: "(1)")
#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading {
    let n = numbering(el.numbering, ..counter(heading).at(el.location()))
    link(el.location(), if el.level == 1 [第 #n 章] else [第 #n 節])
  } else if el != none and el.func() == math.equation {
    link(el.location(), numbering(el.numbering, ..counter(math.equation).at(el.location())))
  } else { it }
}
#show figure.caption: set text(size: 9.5pt)
#show table: set par(justify: false)
// 原始碼的換行在中文字之間會排成一個空格。
#show regex("[\\p{Han}，。、；：「」（）][ \\n]+[\\p{Han}，。、；：「」（）]"): it => it.text.replace(regex("[ \\n]+"), "")

#let rows(name) = csv("../outputs/tables/" + name + ".csv").slice(1)
#let pick(tbl, key, col: 1) = float(tbl.find(r => r.at(0) == key).at(col))
#let pct(x) = str(calc.round(x * 100, digits: 0)) + "%"
#let thou(x) = {
  let c = str(int(x)).clusters()
  c.enumerate().map(((i, d)) => if i > 0 and calc.rem(c.len() - i, 3) == 0 { "," + d } else { d }).join()
}
#let tc(..keys) = keys.pos().map(k => cite(k, form: "prose")).join([、])
#let hline = (x, y) => if y == 0 { (bottom: 0.7pt) }
#let dims = rows("sample_dims")

#align(center)[
  #text(size: 17pt, weight: "bold")[美國航班延誤成因申報的責任歸因稽核]

  #v(0.3em)
  #text(size: 12pt)[方法與分析計畫]

  #v(0.6em)
  #text(size: 10.5pt)[統計研究所　學期分析報告]
]

#v(1.2em)

#block(fill: rgb("#f7fafc"), inset: 1em, radius: 3pt, width: 100%)[
  #text(size: 10pt)[
    *摘要。* 本研究以實測氣象及航班條件為基準，比較各航空公司將延誤分鐘數
    記入 `carrier` 或其他成因欄位的傾向，據此提出監理查核的優先順序。
    主分析將五個申報欄位合為兩類，比較 `carrier` 欄與其餘欄的申報配置；
    四欄成因模型與天氣欄啟用率用於辨認差異的形式。另以設定真值的
    plasmode simulation 評估排序、證據標記與區間的表現。
  ]
]

= 研究問題、資料回顧與研究框架 <ch:frame>

本研究要回答：在調整起點與目的地的月份環境、實測氣象及航班條件後，
哪些航空公司將延誤分鐘數記入 `carrier` 以外欄位的傾向與其他可比公司不同？
差異的方向與程度為何？查核清單納入兩個方向的偏離；單憑偏離，
尚不能認定申報不實。

資料來源與清理方式已於前次報告說明，此處僅回顧分析所需的樣本定義。
稽核樣本為到達延誤至少十五分鐘的已執飛航班，共
#thou(pick(rows("methods_sample"), "航班數")) 班，涵蓋
#thou(pick(rows("methods_sample"), "執飛航空公司")) 家航空公司；起點或目的地在起飛前
是否屬 IFR 或 LIFR 天氣是樣本內的變數，不作篩選條件。未達十五分鐘申報門檻的航班
沒有填報義務，成因欄位的空值不能視為零。
下文以 $bold(x)$ 概括兩端氣象與航班條件；三種氣象資訊設定及其時點見
@sec:weather-sets。

本題的資料形狀決定方法選擇：一班航班可把非負分鐘數記入五個*無序*成因欄，
各欄合計為固定的申報總分鐘；許多欄位為零，但零分鐘與未達門檻的結構性缺值
不同。資料也沒有經獨立查核的逐班真實成因。因此需要處理固定總量、多個零值
與缺乏驗證標籤，並將公司差異與違規認定分開。

*單筆理賠的異常分數。* #tc(<crawford2022erclaims>) 研究的每筆急診理賠
有診斷代碼及由低到高的嚴重程度代碼。作者比較同一診斷下的病例：對某筆
理賠，計算其他病例中嚴重程度不低於該筆的比例；比例愈低，代表該筆的
代碼相對偏高。罕見診斷的比較樣本不足，因此作者依各診斷的平均嚴重程度
作階層分群，用兩折交叉驗證比較分群後預測嚴重程度的能力，
同時避免群內病例過少。
這種分數描述相對罕見程度，不能單憑分數判定編碼錯誤。

同一篇論文另有一項不同用途的分析：以已審核理賠的審核前後金額差，
訓練 random forest 預測待審案件可避免的支出，據此排列審核順序。
本研究沒有逐班查核後的正確成因，也沒有相應的「審核前後差額」，
因此沒有可供訓練這類排序模型的標籤。嚴重程度代碼有高低順序，
但本題的延誤成因沒有這種順序；一班航班還可將分鐘分到多欄，
且偏高與偏低都可能值得查核。本文因而不用該尾端比例作主分數，
而比較可比條件下各公司的分鐘配置。

*醫院層級的多訊號分析。* #tc(<shekhar2022fraud>) 以三種資料訊號找出值得
查核的醫院。第一種在支出迴歸中控制病人既往病史與地區，以醫院固定效應
衡量尚未被這些條件解釋的支出差異。第二種把各醫院的診斷代碼使用頻率
化為高維資料，以 Isolation Forest 等方法尋找少數代碼上的不尋常模式。
第三種按服務類型或病人慢性病分布，以 Hellinger 距離找相似醫院；
再比較給付代碼的使用頻率及其對應的平均支出。作者以即時決選投票法（IRV）
整合各方法的排序，並用已公開的反詐欺案件評估結果；
這些案件只提供部分已知個案，沒有完整的正確申報名單。

#block(breakable: false)[
  本研究可借用第三種方法先界定可比對象、再比較申報行為的思路，
  但須依機場月份、航線、氣象及航班條件界定可比公司，
  不能用正在稽核的成因申報來決定比較對象。
  Hellinger 距離可輔助檢查公司面對的氣象或航班條件分布是否相近；
  連續條件的分布估計或分組方式會影響距離，分布相近也不能保證航線相同。
  因此主排序仍依調整機場月份、氣象與航班條件後的公司申報差異，
  不以 Hellinger 距離決定比較對象或公司排序。
  醫院支出與高維診斷代碼在本資料中沒有直接對應物：本題要解釋的是
  五欄延誤分鐘如何分配，而非總支出或數千種代碼的稀有組合。
  本文保留一個明確的主清單估計對象，再以其他成因分析解釋偏離型態；
  不將回答不同問題的排序合成單一分數。兩篇研究與本題的對照見
  @tbl:framework-differences。
]

#figure(
  table(
    columns: (1.35fr, 1.05fr, 1.65fr),
    stroke: hline,
    align: left,
    row-gutter: 0.45em,
    [本研究的資料特徵], [文獻中的做法], [採用的方法],
    [一班航班可分配分鐘到多欄；成因沒有高低順序], [#cite(<crawford2022erclaims>, form: "prose")：同診斷群內找嚴重程度偏高的單筆理賠], [以二類及四欄 Poisson 配置模型比較分鐘份額，偏高與偏低均列入排序],
    [沒有逐班真實成因或審核前後差額], [#cite(<crawford2022erclaims>, form: "prose")：以已審核案件訓練 random forest], [不以已審核案件預測查核收益；用設定真值的模擬檢查清單表現],
    [航線、機隊、天氣及航管環境可能影響真實延誤成因], [#cite(<shekhar2022fraud>, form: "prose")：按病人與服務特徵尋找相似醫院], [控制兩端機場月份、氣象及航班條件，並檢查同航線競飛者],
    [本題主清單與輔助分析回答不同問題], [#cite(<shekhar2022fraud>, form: "prose")：以多種異常訊號合併排序], [以 `carrier` 與非 `carrier` 的公司差異排序，其餘分析解釋偏離型態],
    [實測氣象不是逐班成因真值，亦無已查核案件], [#cite(<shekhar2022fraud>, form: "prose")：以公開案件作部分驗證], [以 plasmode simulation 評估排序與誤標；清單僅供查核],
  ),
  caption: [Differences from the cited claims studies and the corresponding analysis methods.],
) <tbl:framework-differences>

本題主要比較 `carrier` 與其餘四欄的申報份額；`weather` 與 NAS 的對比則用於
觀察這兩欄的相對申報差異。其餘四欄合稱*非 `carrier` 申報*，
並不等於真正的外部成因。例如，`late-aircraft` 表示前一班同機尾航班晚到，
其根本原因仍可能與航空公司有關，因此另將它與 `carrier` 同組，
檢查主清單對分組方式是否敏感。

這個類比還有識別上的界限。令 $C_i$ 表示「延誤實際由航司可控制的事由造成」，
它在資料中不可觀測；$Y_i^A$ 表示本班至少有一分鐘記入 `carrier`。給定實測
氣象與航班條件 $bold(x)$，公司 $c$ 的觀測申報率可拆為

$ P(Y_i^A = 1 | c, bold(x))
  &= P(C_i = 1 | c, bold(x)) P(Y_i^A = 1 | C_i = 1, c, bold(x)) \
  &+ P(C_i = 0 | c, bold(x)) P(Y_i^A = 1 | C_i = 0, c, bold(x)). $ <eq:mixture>

第一項含實際可歸責成因的比例與記入 `carrier` 的敏感度，第二項含其他成因
被記入 `carrier` 的比例。只有申報與氣象觀測，無法分別估出三個機率。
分鐘份額模型同樣受此限制：它先估*條件化後的申報差異*；要把差異讀作
申報規則不同，還須假設給定 $bold(x)$ 與兩端機場 × 年月後，各公司實際可歸責
成因的比例可比。這是假設，不是氣象欄位已證明的事實。

機隊對低雲與強風的耐受度，以及航線與班表差異，都可能破壞上述可比性；
氣象測站也無法觀測航路天氣與航管指令。#ref(<ch:sim>) 以真實成因因公司而異、
申報規則卻相同的模擬情境，評估清單因此產生的名次偏移與誤標。
@fig:attribution-identification 用兩種情況說明：相同的申報份額差異，可能來自真實成因或申報規則。

#figure(
  table(
    columns: (1.15fr, 3.2fr),
    stroke: hline,
    align: left,
    row-gutter: 0.45em,
    [假設觀察], [相同觀測條件下，甲公司的非 `carrier` 申報份額高於乙公司],
    [可能情況一], [甲真正較少遇到可歸責於公司的延誤；兩家公司申報規則相同],
    [可能情況二], [兩家公司真實成因相近；甲的分類規則把較多分鐘記入非 `carrier`],
    [資料能判定], [申報配置有差異，但無法只靠係數區分上述兩種原因],
  ),
  caption: [Two possible causes of the same observed allocation difference. Carriers A and B are hypothetical.],
) <fig:attribution-identification>

#figure(
  image("../outputs/figures/methods_overlap_levels.png", width: 82%),
  caption: [Multi-carrier overlap at origin airport-month and directed route-month levels. Orange: share of flights in groups with at least two carriers; blue: share of carrier pairs that ever share a group; grey: share of carrier pairs linked directly or through other carriers.],
) <fig:methods-overlap>

@fig:methods-overlap 的兩排使用同一批航班，分別按*起點機場 × 年月*與*同向航線 × 年月*
分組。橘柱的分子是落在至少兩家公司執飛的組合內的航班數，分母是圖中全部航班數。
藍柱與灰柱的分母是全部公司對：藍柱的分子是曾在同一組合出現的公司對，灰柱的
分子是直接或經由其他公司連得起來的公司對。只有一家公司的組合不因間接比較而
提供資訊，間接比較增加的是可比的公司對。
在前一口徑，相關組合涵蓋 #pct(pick(rows("methods_overlap_levels"), "airport-month", col: 6))
的航班；限同向航線與同年月後，有多家公司的組合只占全部航線月份組合的
#pct(pick(rows("methods_overlap_levels"), "route-month", col: 5))，這些組合涵蓋
#pct(pick(rows("methods_overlap_levels"), "route-month", col: 6)) 的航班。
因此同場同月不能視為同航線競飛；即使同航線同月，這張圖仍不測量氣象條件是否相近。主模型分別控制起點與目的地的
機場月份效果，另以同航線、同月份的競飛者作較嚴格的檢查。同層公司若
面對不同的氣象條件，估計仍可能依賴模型外插。

= 主分析：`carrier` 與非 `carrier` 的分鐘數邊界 <ch:binary>

== 二類結果與估計對象

令 $a_i$ 是 `reported_carrier_delay_min`，$e_i$ 是同班 `weather`、NAS、
`security`、`late-aircraft` 四欄分鐘數之和，$L_i=a_i+e_i$。
只對有申報義務且 $L_i>0$ 的已執飛航班定義 $q_i=e_i/L_i$。
這個份額量的是*申報分鐘中有多少記於 `carrier` 以外欄位*，不等於真實不可控的
延誤比例。航班等權的 $bb(E)[q_i]$ 與分鐘加權的
$(sum_i e_i) / (sum_i L_i)$ 回答不同問題；主清單採後者的條件化版本，
前者用於檢查長延誤航班是否主導結果。

樣本中可見全記 `carrier`、全記非 `carrier` 及兩類並記的航班；多數有
多家公司都有航班的機場月份，也呈現公司間的份額差異。這表示結果變數有足夠的
觀測變異可供估計，卻無法據此衡量不實申報率或模型的偵測力。
@fig:cause-groups 顯示同一組申報欄位如何分別進入二類主分析與四欄診斷。

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let border = rgb("#4b6277")
    let arrow = (stroke: (paint: border, thickness: 0.9pt), mark: (end: ">"))

    rect((0, 0.2), (4.0, 3.8), fill: rgb("#f1f4f8"), stroke: border)
    content((2.0, 3.45), text(size: 9pt, weight: "bold")[五個申報欄位])
    content((2.0, 2.9), text(size: 8.5pt)[carrier])
    content((2.0, 2.35), text(size: 8.5pt)[weather])
    content((2.0, 1.8), text(size: 8.5pt)[NAS])
    content((2.0, 1.25), text(size: 8.5pt)[security])
    content((2.0, 0.7), text(size: 8.5pt)[late-aircraft])

    rect((6.0, 2.1), (14.3, 3.8), fill: rgb("#e7f1fb"), stroke: border)
    content((10.15, 3.35), text(size: 9pt, weight: "bold")[主分析：二類])
    content((10.15, 2.78), text(size: 8.5pt)[carrier  |  weather + NAS + security + late-aircraft])
    content((10.15, 2.32), text(size: 8pt)[公司效果決定查核排序])

    rect((6.0, 0.2), (14.3, 1.9), fill: rgb("#e9f4e9"), stroke: border)
    content((10.15, 1.45), text(size: 9pt, weight: "bold")[輔助診斷：四欄])
    content((10.15, 0.88), text(size: 8.5pt)[carrier  |  weather  |  NAS  |  late-aircraft])
    content((10.15, 0.42), text(size: 8pt)[security 不計入四欄分母])

    line((4.0, 2.75), (6.0, 2.75), ..arrow)
    line((4.0, 1.35), (6.0, 1.35), ..arrow)
  }),
  caption: [五個申報欄位分別用於二類主分析與四欄成因診斷。],
) <fig:cause-groups>

== 固定總分鐘數下的二類模型

令 $u=(o,d,t,c)$ 為起點 $o$、目的地 $d$、年月 $t$ 與執飛公司 $c$ 的
單元；$m_(u,E)$ 與 $m_(u,A)$ 是該單元分別記入非 `carrier` 與
`carrier` 的分鐘數。$M_u=m_(u,E)+m_(u,A)$，只納入 $M_u>0$ 的單元；
兩家公司若在同一起點或目的地的機場 × 年月分層中都有航班，便有直接
比較機會。例如一月在 X 機場有甲、丙，二月在 Y 機場有丙、乙；兩組分別
提供甲丙和丙乙的比較資訊，模型可能據此估計甲乙的差異。這是區集設計的
連通性（connectedness）：把公司看成處理、機場月份看成區集，線性加法模型下
設計連通時，所有處理對比才可估計 @eccleston1974connected；分類超過兩個方向時
連通的判定見 @weeks1964connectedness。@abowd2002connected 在人員與雇主的線性
模型中討論同一性質，稱為連通群組。這些都是線性模型的結果；本題使用 PPML 與
兩端機場月份效果，連通只是必要條件，仍須檢查公司效果能否辨識，
並確認氣象共變數分布有共同支持。
設

$ mu_(u,j)=bb(E)[m_(u,j)|M_u,o,d,t,c,overline(bold(x))_u]
  =exp(eta_u+lambda_(o,t,j)+kappa_(d,t,j)+theta_(c,j)
  +overline(bold(x))_u^top bold(psi)_j), quad j in {A,E}. $ <eq:binary-poisson>

$eta_u$ 吸收單元總量；$lambda_(o,t,j)$ 與 $kappa_(d,t,j)$ 分別吸收起點、
目的地每月的基準歸因配置。兩端是交叉的固定效應；此加法設定控制兩端
的平均環境，仍可能留下航線特有差異。同航線 × 同月的公司比較作較嚴格的
敏感度分析。
氣象與航班條件須和類別交互，否則在同一單元兩列中相同，會被 $eta_u$ 吸收。
以 A 為基準成因、DL 為基準公司，$theta_(c,E)$ 是公司 $c$ 相對 DL，
把分鐘記到非 `carrier` 而非 `carrier` 的比值之比的對數。DL 只是參照點，
清單仍須以全部可比公司的等權平均為基準比較各公司。

單元固定效應的估計條件使 $mu_(u,A)+mu_(u,E)=M_u$；二類 fit 的份額
$pi_(u,E)=mu_(u,E)/M_u$ 因而落在零到一。模型使用 Poisson 的估計方程式，
不把連續的一段延誤分鐘當成互相獨立的 Bernoulli 試驗。估計時須檢查 separation、
少數大單元的影響，以及聚合後的氣象差異。推論先按起點 × 年月與目的地 × 年月
雙向聚類，再由 #ref(<ch:sim>) 檢查區間涵蓋率；單元聚合未保留機尾號，
須以逐班檢查確認機尾號相依沒有使區間失準。模型變異數不能直接作為標準誤。

== 分組與航班權重的檢查

`late-aircraft` 只說明前段同機尾航班晚到，沒有指出那一班為何晚到。
另將 `late-aircraft` 與 `carrier` 同組，重估 `weather + NAS + security`
對 `carrier + late-aircraft`；也可排除 `late-aircraft` 分鐘，比較直接外因
與 `carrier`。#tc(<bilotkach2022causes>) 亦將前段晚到視為根因不明的
半內生類別。若優先清單高度依賴這一欄，應將前段根因列為首要查核資料。
`security` 在四欄診斷模型中因稀少而剔除，在五欄二分主模型中仍納入；
兩個模型的分母因此不同，不能直接比較係數大小。

一般的 logistic regression 以零或一的結果估計事件機率；逐班 fractional logit 使用相同的
logistic 平均數形式，但以 $q_i$ 這個零到一的份額為結果，估計條件平均份額。
兩者最大化同一個 Bernoulli log-likelihood 形式的目標函數。二元結果下，
每筆可解讀為 Bernoulli log-likelihood；份額不是一次二元試驗，
同一式子不能當成份額的真實 Bernoulli likelihood，因此在此稱為準概似。
後者須正確設定條件平均數，
並以穩健方式估計標準誤。
此模型保留零與一端點，並納入逐班氣象條件
@papke1996fractional。
每班等權版檢查單元聚合與長延誤權重是否改變公司排序；若以 $L_i$ 加權且
使用相同 logit 線性預測式，它與二類 binomial 配置模型具有相同分數方程式，
不構成獨立的識別證據。兩種估計皆只給出申報差異，不能據此判定動機。

= 成因診斷：分鐘數如何分配 <ch:simplex>

== 由四個欄位到一個配置模型

天氣欄申報較高時，減少的可能是 NAS 分鐘，也可能是 `carrier` 分鐘；
兩種情況的監理意義不同。二類主模型比較責任邊界，四欄模型則辨認
分鐘數在各成因間的配置。同一航班的申報分鐘數由各欄共同分配，
某欄增加時，其餘欄的合計必須減少。逐欄獨立建模無法保留這項約束，
也難以區分對天氣欄的偏好與整體非 `carrier` 申報的增加。

令 $u=(o,d,t,c)$ 表示同一航線 × 年月內的一家公司，$m_(u k)$ 是該單元所有
航班在成因 $k$ 上的分鐘數合計。$k$ 取 `carrier`、`weather`、`NAS`、
`late-aircraft`。`security` 在稽核樣本極少，剔除後四欄定義分析用的
總分鐘數 $M_u=sum_k m_(u k)$。只納入 $M_u>0$ 的單元，並檢查公司間
可比的航線及兩端機場月份重疊。
成因為零是有效的分類結果，不能為了取對數而加常數或刪除。

#figure(
  image("../outputs/figures/methods_cause_mix.png", width: 84%),
  caption: [Unadjusted reported-minute shares by cause. Security is excluded from the four-cause model.],
) <fig:methods-cause-mix>

@fig:methods-cause-mix 呈現未調整的成因分鐘數分布，不能取代控制兩端
機場月份與氣象後的公司比較。@fig:methods-simplex 則呈現各成因欄位的零值情形。

#figure(
  image("../outputs/figures/methods_simplex_zeros.png", width: 98%),
  caption: [Zero structure in four-cause reports with positive total and complete weather. Left: positive causes per flight; right: zero-minute route-month-carrier units, with the grey bar counting units where at least one cause is zero.],
) <fig:methods-simplex>

圖中可見，多數航班只啟用少數成因；右圖以航線 × 年月 × 公司為單元
計算零值比例，與模型的單元相同。即使聚合後各欄仍有零值，故要求四個份額
皆為正的對數比模型會丟掉有效申報。
四欄又共享同一筆總分鐘數，成因診斷仍在固定總量內聯合估計。

聚合後，每個單元以四列記錄成因分鐘數，既保留聯合配置，也讓高維固定效應
的估計可行。不過，逐班氣象差異會被單元平均值取代；後續驗證將檢查
這項近似對估計的影響。

== Poisson 表達如何維持總量約束

設 $zeta_(u k)$ 為成因 $k$ 的相對強度，模型寫成

$ zeta_(u k) = lambda_(o(u),t(u),k) + kappa_(d(u),t(u),k)
  + theta_(c(u), k)
  + overline(bold(x))_u^top bold(psi)_k, $ <eq:composition>

$ mu_(u k) = bb(E)[m_(u k) | M_u, o(u), d(u), t(u), c(u), overline(bold(x))_u]
  = exp(eta_u + zeta_(u k)). $ <eq:poisson>

其中 $eta_u$ 是單元固定效應，$lambda_(o,t,k)$ 與 $kappa_(d,t,k)$ 允許
兩端機場月份各有自己的基準成因分配；$overline(bold(x))_u$ 是單元內
氣象與航班特徵的平均，並與
成因交互。若只放共變量的共同主效應，它在四列中相同，會被 $eta_u$ 吸收。

Poisson pseudo log-likelihood 對 $eta_u$ 的一階條件是
$sum_k(m_(u k)-mu_(u k))=0$。因此 fit 的四欄期望值必須加總為 $M_u$。
把這個條件代回 @eq:poisson，可得

$ mu_(u k) = M_u pi_(u k), quad
  pi_(u k) = exp(zeta_(u k)) / (sum_l exp(zeta_(u l))). $ <eq:softmax>

消去 $eta_u$ 後，與模型參數有關的目標函數等於
$ sum_(u,k) m_(u k) log pi_(u k)
  = sum_u M_u sum_k (m_(u k)/M_u) log pi_(u k). $
因此方法實際比較的是*固定總量下的成因份額*，並以單元總分鐘數
$M_u$ 加權；$M_u=0$ 的單元不提供配置資訊。
二類模型若令 $p_u=pi_(u,E)$，此準則為
$sum_u [m_(u E) log p_u + m_(u A) log(1-p_u)]$。
在相同總量與預測式下，它與總分鐘加權的 logistic regression 有相同的點估計；
這是估計準則的代數關係，不表示分鐘是獨立的二項試驗。
#tc(<baker1994multinomial>) 的 multinomial–Poisson transformation 給出
*計數資料*在固定類別總量時兩種係數表示的精確關係。本題的分鐘數是時長，
即使記為整數也不是獨立事件的計數，因此不把此轉換解讀成分鐘數的
multinomial 或 Poisson 分布假設。借用的是固定總量下條件平均數的代數形式。
Poisson 寫法讓單元與層 × 成因固定效應
得以用高維固定效應工具估計，而不須逐一展開大量虛擬變數。

混合二項模型另假設單元來自數個未觀測群體，各群有不同的二項配置機率
@grun2024mixture。它可把零值頻率與配置差異歸於不同群體，但需指定群體數、
群體分布，並說明以分鐘作二項試驗的理由。即使估出群體，也不能只憑申報資料
將其辨認為真實天氣、系統預設或刻意改報。PPML 的優點是估計對象限於條件
平均分鐘，所需的分布假設較少；代價是不能單獨解釋零值的發生機制，
也不能辨認未觀測群體。零值的啟用率另作分析。

== 公司係數與申報欄位對比

為使係數有固定的參照點，設 $theta_(c, #raw("carrier"))=0$、
$theta_("DL",k)=0$。在相同兩端機場月份效果與單元共變量下，
$exp(theta_(c,k))$ 是公司 $c$ 相對 DL、成因 $k$ 相對 `carrier` 的
*比值之比*。下式的 $pi_(c,k)$ 表示在相同兩端月份效果與共變量下，把
@eq:softmax 評估於公司 $c$：

$ exp(theta_(c,k)) =
  (pi_(c,k) / pi_(c,#raw("carrier")))
  / (pi_("DL",k) / pi_("DL",#raw("carrier"))). $ <eq:ratio>

例如 $theta_(c,#raw("weather"))$ 為正，表示此公司相對 DL，把同一總量
配置在天氣而非 `carrier` 的比值較高；它本身不表示實際有多少分鐘被人從
`carrier` 改寫到天氣。`weather` 相對 NAS 的公司對比由
$theta_(c,#raw("weather"))-theta_(c,#raw("NAS"))$ 表示，兩係數的協方差
必須放進此對比的標準誤。這個欄位間的比較不能辨認實際責任。
DL 同樣只是模型基準，$theta_("DL",k)=0$ 不是「DL 沒有偏離」
的證據；所有公司的優先次序應由參照不變的公司間對比或可比公司效果的等權平均決定。

#figure(
  table(
    columns: (1.2fr, 1.6fr),
    stroke: hline,
    align: left,
    row-gutter: 0.45em,
    [係數形態], [條件化後的申報結構解讀],
    [`weather` 為正、NAS 與 `late-aircraft` 相對偏低], [相對配置集中於天氣；須比較 `weather`／NAS 與 `weather`／`late-aircraft`],
    [`weather`、NAS 與 `late-aircraft` 都為正], [相對 DL，這三欄對 `carrier` 的份額比都較高；須查各欄原始紀錄],
    [`weather` 與 NAS 一正一負], [這兩個申報欄位的相對份額有差異],
  ),
  caption: [Interpretation of weather contrasts across causes. The entries describe estimands, not estimated results.],
) <tbl:patterns>

#figure(
  image("../outputs/figures/methods_cause_contrasts.png", width: 100%),
  caption: [Exploratory origin-month carrier contrasts and 95% intervals, relative to DL. These are not the final audit list.],
) <fig:methods-contrasts>

@fig:methods-contrasts 把兩條邊界並排：同一公司的天氣對 `carrier` 對比，
未必與天氣對 NAS 的對比同向。圖中零線代表與 DL 的對比為零，不表示 DL
正確；正式排序仍採 #ref(<ch:deliver>) 的可比公司等權平均與驗證規則。

== 為何是 pseudo-likelihood，以及會在哪裡失效

真實航班的分鐘數並非相互獨立的 Poisson 計數：總量已被申報制度固定，分配
又可能由系統規則產生整段全記或多欄同時為零。這裡使用 Poisson 的估計方程式
而非相信其完整分布。對任意非負分鐘數，可最大化省去與參數無關項的
pseudo log-likelihood
$ sum_(u,k) [m_(u k) log mu_(u k) - mu_(u k)]. $
它不要求 $m_(u k)$ 是整數；即使資料記為整數分鐘，結果量的仍是時間長度，
而非獨立事件的次數。保留零值是此準則的性質，單有零質量點並不足以支持
Poisson 的分布假設。令 $bold(z)_(u k)$ 為 @eq:composition 的迴歸設計列。
當 @eq:poisson 的*條件平均數*設定正確時，對迴歸係數的 score 為
$sum_(u,k) bold(z)_(u k)(m_(u k)-mu_(u k))$，其期望為零；
分布的變異數即使不是 Poisson 形式，係數仍可由 pseudo-likelihood 估計
@gourieroux1984pml。這個論點不能替條件平均數設錯、聚合遺漏變異，或高維
固定效應偏誤提供保證，也不能使用 Poisson 的 model-based 標準誤。

單元只以氣象平均控制公司之間的天氣差異；如果真實氣象效果是非線性的，
$exp(bold(psi)^top overline(bold(x))_u)$ 通常不等於單元內
$exp(bold(psi)^top bold(x)_i)$ 的平均。航班聚合可能因此留下組成差異。
應比較單元內氣象分散程度與公司係數，並在 #ref(<ch:sim>) 以含逐班非線性
氣象效果的生成機制檢查偏誤。這是選擇可計算的單元模型所付出的近似成本。

若某家公司在可比的層中從不記入某成因，對應係數可能趨向無窮，屬
separation 而非強烈且精確的效果 @correia2019existence。估計時須檢查
各公司 × 成因的正值與分離狀態，不能把不收斂的係數排上清單。高維固定
效應隨單元數增加，也可能引入有限樣本偏誤；穩健標準誤只處理變異數，
不會消掉係數偏誤 @weidner2019bias。

探索性估計按起點月份單元聚類，容許同一單元的四欄結果相關；起點與
目的地月份之間共享的航管或天氣衝擊，則仍可能使區間過窄。分鐘數的
右偏分布也可能讓少數大型單元主導估計。正式分析將檢查這些單元的影響，
並以 #ref(<ch:sim>) 評估共同衝擊下的區間涵蓋率，再選擇聚類或重抽方式。

= 輔助診斷：天氣欄的啟用與零值 <ch:within>

== 從研究問題到估計量

令 $i$ 為航班，$s(i)$ 為起點機場 × 年月層，$c(i)$ 為執飛公司，
$Y_i = bb(1)(#raw("reported_weather_delay_min") > 0)$。同一場天氣可以造成
不同長度的延誤；這個結果只問申報表*有沒有啟用天氣欄*，不把分鐘數的大小混入。
所比較的母體是達申報門檻的已執飛航班。
此模型補充主分析：它區分「天氣欄啟用率較低」與「啟用率相近但申報分鐘數較少」，
不單獨決定公司的查核名次，也不能代替非 `carrier` 份額。

#figure(
  image("../outputs/figures/methods_y1_rate.png", width: 72%),
  caption: [Unadjusted positive-weather share by carrier with more than 5,000 flights. This is not an audit list.],
) <fig:methods-y1>

天氣欄是否啟用是二元結果。@fig:methods-y1 顯示各公司未調整的啟用率
有所不同，但各公司營運的機場與遭遇的天氣也不同，不能直接將此差異
歸於申報行為。下式在控制機場月份與氣象條件後估計公司間的差異。

估計式為

$ bb(E)[Y_i | s(i), c(i), bold(x)_i] = alpha_(s(i)) + gamma_(c(i))
  + bold(x)_i^top bold(beta), quad gamma_"DL" = 0. $ <eq:within>

若條件平均數確為 @eq:within 的加法形式，$gamma_c$ 就是公司 $c$ 相對 DL、
在同一層和同樣 $bold(x)$ 下記入天氣的機率差；乘以 100 後以百分點表示。兩家非基準公司
的差為 $gamma_c - gamma_d$。若真實機率對天氣為非線性、公司效果又隨機型
變動，估計係數只是樣本中按航班配置加權的*線性投影*，不能無條件稱為所有
航班上的平均因果效果。#tc(<chen2023lpm>) 討論了線性機率模型與非線性機率模型
之平均偏效果的關係，這裡保留條件平均數形式與此限制。

DL 只是辨識模型所需的參照水準，不代表它的申報正確。換一家公司作基準會改變
各 $gamma_c$ 的表示，卻不改變任意兩家公司 $gamma_c-gamma_d$ 的差。
清單不能因 DL 的係數設定為零就視它為已通過查核；要比較全部公司，須報
參照不變的公司間對比或把係數重新中心化到可比公司效果的等權平均。

== 為何固定在同一機場與月份比較

各公司營運的機場不同，跑道、除冰能力、航管慣例與季節性天氣都可能影響
成因申報。@eq:within 讓每個起點機場月份各有截距 $alpha_s$，避免以
多雪與少雪機場之間的差異估計公司效果。目的地月份環境仍可能影響結果，
正式診斷須另加目的地 × 年月效果，檢查公司效果是否改變；兩端交叉固定效應
不能只靠一次起點層去均值消去，須另作交替吸收。定義 $D_(i c)=bb(1)(c(i)=c)$，對任何變數
$a_i$ 令 $tilde(a)_i=a_i-overline(a)_(s(i))$。從 @eq:within 扣除同層平均，得到

$ tilde(Y)_i = sum_(c != "DL") tilde(D)_(i c) gamma_c
  + tilde(bold(x))_i^top bold(beta) + tilde(epsilon)_i. $ <eq:demean>

$alpha_s$ 正好消失，對去均值後的資料做最小平方即可估 $gamma_c$。
#tc(<mundlak1978pooling>) 的 within estimator 提供這個固定效應消去方式。
單一公司層內所有 $tilde(D)_(i c)$ 都為零，無法識別公司對比；兩家公司若從不
出現在同一或連通的層，彼此也缺少可比路徑。層數很多不等於每家公司的效果
都精確，真正有用的是層內的競飛與氣象重疊。

層內 $Y_i$ 全為零或全為一時，該層仍可能影響線性機率模型的斜率估計：
$tilde(Y)_i=0$ 仍會進入最小平方的估計條件。因此不能僅因結果全同便
刪除整層。共用估計程序目前會排除此類層，正式估計前須修正篩選條件並
重新估計。這與單一公司層無法識別公司間對比，是不同的問題。

== 後續檢驗：Logistic GLMM 與組內相關 <sec:glmm>

若調整機場月份與氣象條件後，同機尾號或同機場日期的航班仍有明顯的
天氣欄啟用相關，將以廣義線性混合模型（GLMM）作敏感度分析。
目標是明確建模群組異質性與組內相關，檢查公司啟用率差異及其不確定性
是否穩定。#tc(<brooks2017glmmtmb>) 說明 GLMM 如何透過隨機效果處理
重複觀測相關；本題擬先使用 Logistic GLMM，保留公司與兩端機場月份固定效果。

令 $Y_i$ 仍為天氣欄是否記入正分鐘，$s(i)$、$r(i)$ 分別為起點與目的地
機場 × 年月，$h(i)$ 為機尾號，$g(i)$ 為起點機場 × 日期。
候選模型為

$ Y_i | bold(x)_i,u,v tilde.op op("Bernoulli")(p_i), quad
  op("logit")(p_i)=alpha_(s(i))+kappa_(r(i))+gamma_(c(i))
  +bold(x)_i^top bold(beta)+u_(h(i))+v_(g(i)), $ <eq:glmm>

其中 $u_h tilde.op N(0,sigma_u^2)$、$v_g tilde.op N(0,sigma_v^2)$ 為候選隨機
截距。先檢查各群的重複觀測與剩餘相關，再比較加入機尾號、機場日期及兩者
並用的設定。公司效果維持固定，避免同時引入公司收縮而混合不同的比較目的。
隨機效果與公司、航網及氣象共變數間的關係須另作檢查；均值為零、分布形式及
群組間獨立等假設，不能僅由模型可以收斂而視為成立。

*資料與輸出。* 使用與 @eq:within 相同的有效申報航班，一班一列，加入機尾號
及機場日期識別碼。@eq:glmm 的 $gamma_c-gamma_d$ 是給定隨機效果後的
對數優勢比，不能直接與線性機率模型的機率差比較。比較時對隨機效果積分，
再於共同可比的航班條件分布上平均各公司的預測機率，報告標準化公司機率差及
其區間。另與具有相同固定效果的普通 logistic 模型對照，以區分連結函數改變
和加入隨機效果的影響；在 #ref(<ch:sim>) 檢查偏誤、區間涵蓋與估計穩定性。

*與過度離散的關係。* Poisson 混合模型可透過發生率異質性產生超過 Poisson
基準的變異，但逐班 Bernoulli 結果的變異數由其機率決定；本節主要處理航班間
相關，而非對單筆二元結果另估離散參數。分鐘主模型使用的 PPML 本來就不要求
變異數等於平均數 @santossilva2006log，故不能僅憑分鐘數變異較大，就改用
Poisson GLMM。本節是待評估的啟用率診斷方案，不改變二類分鐘主排序的定義。

== 天氣、航班特徵與資訊時點 <sec:weather-sets>

$bold(x)_i$ 分別包含起點與目的地的能見度、雲幕、風速、降水與飛行規則類別，
另有距離、班表壅塞、前一航段及實際到達延誤長度。起點和目的地不能只留一端：
兩地天氣可能不同，申報人卻只填一組成因分鐘數。天氣使用三種資訊設定：

#figure(
  table(
    columns: (auto, 1.3fr, 1.4fr),
    stroke: hline,
    align: left,
    row-gutter: 0.45em,
    [設定], [氣象欄位], [係數所回答的比較],
    [起飛前], [`wx_`], [給定事前可知的天氣],
    [事件時刻], [`oracle_origin_dep_*`、`oracle_dest_scharr_*`], [給定事後填報時已發生的天氣],
    [兩組並用], [以上兩組], [再控制事前至事件時刻的天氣變化],
  ),
  caption: [Three weather information sets for the same flights. The audit sample is defined using `wx_` throughout.],
) <tbl:weather>

本題研究的是事後填報，不是事前預測，所以事件時刻的 `oracle_` 可作回溯共變量。
`oracle_dest_arr_*` 的時點卻隨本班實際到達而變，會把延誤長度機械帶入氣象
欄位，故以前綴排除。`reported_*` 是被稽核的結果，不能拿來解釋本班延誤。
雪深、陣風與前一航段的結構性缺失依其適用條件處理，不當作隨機缺失插補。

實際到達延誤長度在填表時已知，控制它可比較同樣延誤長度的申報；但它本身
受公司營運與天氣影響。納入後 $gamma_c$ 是*給定延誤長度*的申報對比，
不是公司對所有航班的總效果。申報門檻更使研究母體依到達延誤而定：
如果公司影響延誤是否達門檻，條件化可能使原本不同的航班組成進入比較。
這兩點須與未控制延誤長度、及以全部已執飛航班為分母的描述並列，
但描述後者時，須將未達門檻者視為無申報義務，而非申報零分鐘。

延誤長度作為 $bold(x)_i$ 的一項，而不作為分層的依據。它不能取代機場月份：
同樣長度的延誤，在暴雪中的機場與晴天的機場有不同的真實成因，只依長度
比較會把各公司航點與天氣的差異算進 $gamma_c$。也不在機場月份之內再依長度
分層：層切得愈細，只有一家公司的層愈多，而這類層對公司對比沒有資訊。
以連續共變量控制的代價是函數形式：短延誤與長延誤的成因分配若不同，線性項
未必足以描述。因此另將延誤長度分成數段，於各段內重估，檢查公司的排序
是否改變。

== 係數的不確定性與可檢查條件

二元結果本身異質變異，且同層航班共享機場狀況、同機尾號航班共享機隊與輪轉。
故不用同方差的 OLS 標準誤；按層 $S$ 與機尾號 $T$ 雙向聚類，變異數由

$ hat(V) = hat(V)_S + hat(V)_T - hat(V)_(S ∩ T) $ <eq:twoway>

組成 @cameron2011multiway。交集項避免同一對相關被加兩次。這個估計仍仰賴
足夠多的有效層與機尾號；小公司的比較可能只由少數競飛層支撐。

分析時應檢查三件事：各公司在多少層與 DL 或其他公司競飛；公司之間的天氣
分布是否重疊；三種資訊設定下 $gamma_c$ 的方向與大小是否穩定。
若 fit 的值在觀測支持內大量超出零到一，線性平均函數不適合把係數讀作固定
機率差，應縮小解讀範圍或改用非線性平均函數。這些檢查處理的是模型近似與
重疊問題；即使全部通過，@eq:mixture 的未觀測真實成因仍無法由資料驗證。

= 清單的排序、證據與界限 <ch:deliver>

主清單以 @eq:binary-poisson 的二類責任邊界為核心。二類公司效果先相對所有
可比公司的*等權平均*重新中心化，使包括 DL 在內的每家公司都有相對該平均的對比；
這只改變參照點，不改變任意兩家公司之間的差。全部可比公司按中心化效果的
絕對值由大到小排序，並分欄標示偏高或偏低。每列附效果大小、區間、
可比層數、航班等權的份額檢查，以及 `late-aircraft` 改分組後的方向。
三種氣象資訊設定下的二類公司效果並列，檢查排序對資訊時點是否敏感。

用完整共變異矩陣計算公司對比的標準誤，對全部可比公司的雙側檢定作
Benjamini–Hochberg 校正 @benjamini1995fdr。通過者在*證據欄*標記，
不通過者仍保留在排序清單。四欄模型另列天氣、NAS 與 `late-aircraft`
相對 `carrier` 的係數，以及天氣相對 NAS 的分類對比；天氣欄啟用率
用於解讀零值。這些診斷分欄呈現，不壓成單一分數，也不改變二類排序。

點排序之外，依公司效果的聯合不確定性報告可能的名次範圍；比較兩家公司
的絕對偏離程度時須納入係數共變異數，不能只看各自區間是否重疊。
名次範圍重疊表示優先次序尚不明確，不直接宣稱兩家公司效果相同。
BH 標記的 false discovery rate 與名次範圍的涵蓋率均由 #ref(<ch:sim>)
在設定過的情境下檢查；若未達預定標準，正式使用前須調整並重新驗證。

未觀測的機隊耐受度、航路天氣與航管指令仍可能解釋公司差異；模擬只測試設定過的
競爭解釋。氣象基準不是逐班真實成因，清單供監理優先查核，不構成違規認定。
取消的航班沒有到達延誤，不進入稽核樣本。樣本期間的成因分類適用當時制度，
不能直接延伸到後續修訂的分類 @dot2026causes。

= 驗證：以已知真值檢查清單 <ch:sim>

== 目的、基底與未知的真值

真實資料沒有逐班延誤的客觀成因，無法直接檢驗實證清單是否正確。
模擬研究先設定逐班成因真值，再對指定公司植入已知的申報偏移，
依正式分析程序重新產出排序與證據標記。
驗證單位是公司，與交付物一致；流程見 @fig:sim-flow。

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let border = rgb("#4b6277")
    let arrow = (stroke: (paint: border, thickness: 0.9pt), mark: (end: ">"))
    let boxes = (
      (0.0, rgb("#e7f1fb"), [保留觀測航班], [公司、氣象、航線], [與到達延誤]),
      (2.95, rgb("#f1f4f8"), [生成潛在成因], [五欄非負分鐘], [總量固定]),
      (5.90, rgb("#fff0d5"), [植入申報偏移], [指定公司改報], [維持總分鐘數]),
      (8.85, rgb("#e7f1fb"), [重估分析模型], [相同樣本], [相同分析規則]),
      (11.80, rgb("#e9f4e9"), [評估查核清單], [比較排序標記], [與植入答案]),
    )
    for (x, fill, title, detail1, detail2) in boxes {
      rect((x, 0.2), (x + 2.6, 2.1), fill: fill, stroke: border)
      content((x + 1.3, 1.62), text(size: 9pt, weight: "bold")[#title])
      content((x + 1.3, 1.02), text(size: 7.5pt)[#detail1])
      content((x + 1.3, 0.70), text(size: 7.5pt)[#detail2])
    }
    for x in (2.6, 5.55, 8.50, 11.45) {
      line((x, 1.15), (x + 0.35, 1.15), ..arrow)
    }
  }),
  caption: [Plasmode simulation 從觀測航班到公司層級查核清單的生成與評估流程。],
) <fig:sim-flow>
#tc(<brakenhoff2018profiling>) 的醫療機構 profiling 研究也以真實個體特徵
作為模擬基底，按機構評估漏列與誤列；本題借其*評估層級與情境設計*，
不借用二元結果模型。依 #tc(<morris2017simulation>) 的 ADEMP，分別檢查
零偏移時的排序與 BH 誤標、已知改報的標記力，以及真實受天氣影響不同時的誤標。

plasmode simulation 固定真實航班的層、公司、氣象、機尾號、前一航段與到達延誤，
只替換申報成因 @schreck2023plasmode。十五分鐘以下仍無申報，生成後再取與
正式分析相同的稽核樣本；公司航網、班次不均與稀疏層因此保留。

由於資料不含客觀成因，生成機制先設定一組*潛在真實成因分鐘數* $T_(i,k)$，
$k$ 取五個申報成因。逐班保留可分配總量 $L_i$，使

$ T_(i,k) >= 0, quad sum_k T_(i,k)=L_i, quad
  H_i = bb(1)(T_(i,#raw("weather")) > 0). $ <eq:true-causes>

$H_i$ 表示本班是否真的有天氣造成延誤；$T_(i,k)$ 與 $H_i$ 都不是從
`reported_*` 成因欄反推的真值。其分布由事先指定的氣象反應情境生成，
涵蓋機隊容忍度與航路天氣未被測站捕捉的可能性。因為真實的 $T_(i,k)$
無法校準，模擬只能說「在這些設定下能否分辨」，不能推算實證清單的真實誤列率。

== 資料如何生成，與分析模型如何區隔

先令 $bold(w)_i$ 為事件時刻氣象。潛在天氣成因的機率可設為

$ op("logit") P(H_i=1 | bold(w)_i,s(i),r(i),t(i))
  = a_(s(i)) + a_(r(i)) + f(bold(w)_i) + b_(t(i)), $ <eq:latent>

$s(i)$ 與 $r(i)$ 分別是起點、目的地機場 × 年月，$t(i)$ 是機尾號；$f$
含兩端的能見度、雲幕、風、降水與交互作用。先依 $H_i$
與 $L_i$ 生成滿足 @eq:true-causes 的真實五欄分鐘數；再以同一套*申報規則*
決定是否在填報時啟用天氣欄：

$ op("logit") P(Y_i=1 | H_i,bold(w)_i,c(i))
  = kappa_(H_i) + g(bold(w)_i) + v_(d(i)) + v_(t(i))
    + delta_(c(i))^Y. $ <eq:reporting>

$d(i)$ 是機場 × 日期；$v_d$ 與 $v_t$ 使同日與同機尾號的申報相關。
$delta_c^Y=0$ 表示在相同真實成因與天氣下，公司沒有額外的申報傾向。
資料生成使用 logit 與氣象交互作用，天氣欄啟用分析則使用線性機率模型
與較簡單的氣象項；
目的地月份效果使模擬也能檢查只控制起點時的遺漏偏差。
藉此檢驗分析模型設定不完全時的表現。

先依 @eq:reporting 抽出是否啟用天氣欄，再在固定 $L_i$ 下分配申報分鐘：
若抽到不啟用，將天氣真值分鐘轉至其他欄；若抽到啟用而天氣真值為零，
從有正分鐘數的其他欄移入天氣欄。植入公司效果時也遵守此順序，
使天氣啟用狀態與最後的申報分鐘一致；只改動分鐘配置的情境須固定已抽出的
$Y_i$，啟用時保留天氣欄正分鐘數，不啟用時不得移入天氣欄。
令 $R_(i,k)$ 為模擬後可供分析的五欄*申報*，它由真值加上改報量形成：

$ R_(i,k)=T_(i,k)+Delta_(i,k), quad
  sum_k Delta_(i,k)=0, quad R_(i,k)>=0. $ <eq:misreport>

共同的填報雜訊可讓 $R_i$ 偏離 $T_i$，但所有公司使用同一規則；再對指定公司
額外植入已知的改報偏移。分別模擬 `carrier` 移往 `weather`、NAS 或
`late-aircraft`，以及 NAS 移往 `weather`、`weather` 移往 NAS，
並加入隨延誤長度整段全記天氣；每次移轉量
不得超過移轉當下來源欄可用的分鐘數。若 $Y_i=0$，$R_(i,#raw("weather"))=0$；
若 $Y_i=1$，天氣欄有正分鐘數。五欄申報總量維持 $L_i$，並保留多零、
整段全記與右尾形態，避免只用平滑的 multinomial 模型替分析模型出題。

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: hline,
    align: left,
    row-gutter: 0.45em,
    [情境], [資料如何生成], [檢查什麼],
    [零偏移], [各公司共用 @eq:reporting 的規則與申報雜訊，沒有公司特有偏移], [航網與天氣不同時的名次與 BH 誤標],
    [植入偏移], [只對指定公司改動五欄分鐘數的移轉規則或天氣啟用率], [二類主清單對責任邊界偏移的排序與標記力，以及輔助模型對偏移形式的辨識能力],
    [機隊差異], [保持申報規則相同，只提高指定公司在 @eq:latent 的真實天氣成因機率], [識別假設違反時的名次與誤標],
  ),
  caption: [Three base simulation scenarios.],
) <tbl:scenarios>

== 植入範圍與受影響公司的組成

植入設計的三個維度見 @tbl:sim-factors。受植入公司集合事先指定，並保留其
真實航班配置。#tc(<brakenhoff2018profiling>) 支持同時改變異常者比例與
機構規模；#tc(<steinbuss2020outlier>) 支持把局部異常與全域異常分開評估。

#figure(
  table(
    columns: (1fr, 1.7fr, 1.7fr),
    stroke: hline,
    align: left,
    row-gutter: 0.5em,
    [Factor], [Settings], [What it tests],
    [Planted carriers], [Few or many], [Peer-reference contamination],
    [Carrier flight volume], [Across sizes; concentrated in small or large carriers], [Misses driven by unequal volume],
    [Injection scope], [All eligible flights; selected airports, months, or weather states], [Local signals masked by an overall average],
  ),
  caption: [Simulation factors crossed with the base scenarios.],
) <tbl:sim-factors>

條件式植入只改動指定情境內的航班，仍按 @eq:misreport 在五欄間轉移分鐘數，
維持非負、總量及零值關係。Steinbuss 與 Böhm 的生成法需要已標記為正常的資料；
本題沒有這種標籤，因此只借用其異常型態的區分，不照搬生成分布。

條件式植入包含兩種情形：一種仍會改變公司整體的平均申報配置；
另一種只影響特定條件下的航班，使公司整體平均接近不變。
主模型估計*平均公司效應*，未納入公司 × 機場或公司 × 天氣的
交互作用。後一種情境若未排在前段且未獲 BH 標記，表示主清單難以凸顯這類局部異常，
不能把它解讀成平均效應估計有偏。故局部情境的名次與標記率單獨報，
並列受影響條件內的申報差異與全體航班平均差異；不把它混入全域改報的
偵測力摘要。此項檢查用於界定清單能辨認的異常範圍。

== 真值、重複分析與表現指標

植入參數 $delta_c^Y$ 在 logit 尺度上；天氣欄啟用模型的 $gamma_c$ 卻是
線性投影的機率差，不能把前者直接當成後者的「真值」。二類與四欄配置模型
也可能因聚合與固定效應而有不同的目標值。每個情境先由生成機制計算
理論條件平均：對天氣啟用模型的設計矩陣作同一個投影，得 $gamma_c^*$；
對兩個配置模型，分別將理論平均分鐘數代入各自的 Poisson score 求得
對應目標。這是分析程序在該情境下應估到的值；若目的是判定「申報規則
是否被改動」，另以植入的 $delta_c$ 作真正的公司異常標籤。

每次重複都重新生成 $T_i$ 與五欄申報、重新套樣本規則、重新擬合二類主模型
與成因診斷模型，並依 #ref(<ch:deliver>) 的規則形成完整排序與 BH 證據標記。
模擬和真實資料須用同一規則；看到模擬結果再挑門檻會高估表現。
對 `carrier` 移往非 `carrier` 與非 `carrier` 欄彼此互換，分別報告名次與標記力。
後一種改報不改變二類份額，
二類排序理應無法辨認；若要查天氣欄的分類習慣，須另看四欄診斷。
另依受影響公司數、班次規模與偏移範圍分組，比較相同偏移幅度在小公司
是否較難檢出，以及多家公司同向偏移時，未植入公司是否被標記為反向異常。
局部偏移能否反映在清單前段，也須單獨檢查。

以被植入公司為陽性標籤，逐次記錄 BH 正確標記、誤標與漏標。
標記的偵測力是植入者獲標記的比例；每次獲標記者中未被植入者的比例是
查核誤標比例，沒有標記時定為零。統計意義的 false discovery proportion 則須以
生成機制下真正滿足中心化公司效果為零者作虛無集合，跨重複平均為 false discovery rate。
由於重新中心化會使未植入公司的效果也可能非零，查核誤標比例與統計 FDR 須分開報告。
另報植入者的名次分布、無偏移公司的標記率與方向判對率。這些指標分別評估
完整排序與證據標記，比單一 accuracy 更適合少數公司被植入的情境。真實資料仍沒有這些
標籤，不能把模擬指標寫成實際航空公司的判錯率。

#figure(
  table(
    columns: (1.3fr, 1.6fr),
    stroke: hline,
    align: left,
    row-gutter: 0.45em,
    [表現指標], [實際回答的問題],
    [零偏移下的 BH 誤標率與 false discovery rate], [申報規則相同時的標記情形],
    [偏移情境的標記力、名次與方向判定], [不同偏移形式與公司規模下的標記與排序表現],
    [受影響公司數與局部情境的分層結果], [可比公司等權平均受污染或偏移只在部分航班發生時，清單覆蓋到哪裡],
    [$hat(gamma)_c$、$hat(theta)_(c,k)$ 的偏誤], [線性近似、聚合或固定效應是否扭曲係數],
    [係數區間與名次範圍的涵蓋率], [標準誤與排序不確定性是否對同日、同尾號與右尾資料有效],
    [機隊差異情境的名次與 BH 誤標率], [真實天氣成因不同時的排序與標記情形],
  ),
  caption: [Performance measures for estimates, intervals, and the audit list.],
) <tbl:performance>

若重複次數為 $B$、某項表現指標是每次為零或一的事件，其 Monte Carlo
標準誤為 $sqrt(hat(p)(1-hat(p))/B)$。通過標準應在執行前依名目水準與
Monte Carlo 標準誤訂定；所需 $B$ 由可接受的 Monte Carlo 誤差與單次
高維 Poisson 擬合時間共同決定。若區間涵蓋不足，要分辨是係數偏誤、
單元聚類遺漏同層相關，還是重尾單元主導估計，而不是只換一個名目信賴水準。

plasmode 保留了觀測到的公司與航班配置，不能模擬「同一班若改由另一公司
執飛，是否仍會達十五分鐘門檻」的反事實選入過程。它可量測已觀測樣本結構
下的名次與標記錯誤，不能解決門檻造成的全部選擇問題。#tc(<karim2026plasmode>) 的
提醒亦適用：在設定過的機制下通過，只表示未被這些情境否證。

#bibliography("refs.yml", style: "apa", title: [參考文獻])
