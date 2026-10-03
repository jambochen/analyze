#set document(title: "美國航班天氣延誤申報的歸因異常稽核：方法與分析計畫")
#set page(paper: "a4", margin: (x: 2.6cm, y: 2.4cm), numbering: "1")
#set text(font: ("Noto Sans CJK TC",), lang: "zh", region: "tw", size: 11pt)
#set par(justify: true, leading: 0.85em, first-line-indent: 0em)
#show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans CJK TC"), size: 9.5pt)
#set heading(numbering: "1.1")
#show heading: it => block(above: 1.4em, below: 0.7em)[#it]
#show heading.where(level: 1): it => block(above: 1.8em, below: 0.9em)[#it]
#set figure(gap: 0.8em)
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
#show regex("[\p{Han}，。、；：「」（）][ \n]+[\p{Han}，。、；：「」（）]"): it => it.text.replace(regex("[ \n]+"), "")

#let rows(name) = csv("../outputs/tables/" + name + ".csv").slice(1)
#let pick(tbl, key, col: 1) = float(tbl.find(r => r.at(0) == key).at(col))
#let pct(x, d: 0) = str(calc.round(x * 100, digits: d)) + "%"
#let thou(x) = {
  let c = str(int(x)).clusters()
  c.enumerate().map(((i, d)) => if i > 0 and calc.rem(c.len() - i, 3) == 0 { "," + d } else { d }).join()
}
#let tc(..keys) = keys.pos().map(k => cite(k, form: "prose")).join([、])
#let hline = (x, y) => if y == 0 { (bottom: 0.7pt) }

#let dims = rows("sample_dims")
#let cells = rows("cell_carriers")
#let crossing = rows("cluster_crossing")

#align(center)[
  #text(size: 17pt, weight: "bold")[美國航班天氣延誤申報的歸因異常稽核]

  #v(0.3em)
  #text(size: 12pt)[方法與分析計畫]

  #v(0.6em)
  #text(size: 10.5pt)[統計研究所　學期分析報告]
]

#v(1.2em)

#block(fill: rgb("#f7fafc"), inset: 1em, radius: 3pt, width: 100%)[
  #text(size: 10pt)[
    *摘要。* 本研究以機場實測氣象為基準，比較各航空公司在同一機場、同一月份、
    同樣天氣條件下的延誤成因申報，產出供監理單位使用的查核優先清單。資料中沒有
    任何欄位記錄延誤的真實成因，清單的每一列都無從對照答案，結論是否可信取決於
    方法本身。本報告因此只處理方法：前半逐一說明所用的模型與推論方法，各自的
    出處、假設，以及假設在本資料上如何檢查；後半是分析計畫，以設定真值的模擬
    研究為主幹，另含六個方向，分別處理清單的不確定性、未觀測混淆、由模型導出
    的限制條件、氣象判定以外的航班、判定偏離的參照分布，以及機場層級的清單。本報告不呈現估計結果。
  ]
]

= 問題與資料 <ch:recap>

美國運輸統計局要求到達延誤達門檻的航班申報延誤成因，把延誤分鐘數拆入數個
成因欄位。欄位由航空公司自行填報，監理機關不例行查核；交通部監察長辦公室
認定其不可信，但未指出偏誤方向。本研究的問題是：以實測氣象為基準，哪些
航空公司的天氣歸因系統性地不同於其他公司，差異的方向與大小能否量化並排序。
偏高與偏低皆列入。

公開揭露的指標會改變航空公司的行為，已有先例。準點率排名以到達延誤十五分鐘
為界，#tc(<forbes2015disclosure>) 發現各公司對門檻附近航班的反應不一，且與
公司內部的制度有關。成因申報同樣影響準點績效的認定，各公司的申報行為有差異
並不意外，問題在於如何量化。

稽核樣本限於有申報義務且起點或目的地遭遇惡劣天氣的航班，共
#thou(pick(dims, "航班數")) 班，涵蓋 #thou(pick(dims, "執飛航空公司")) 家執飛公司、
#thou(pick(dims, "起點機場")) 座起點機場與 #thou(pick(dims, "機尾號")) 個機尾號。
比較在 stratum（層）內進行，一層是一座機場的一個年月，共
#thou(pick(dims, "機場 × 年月格子")) 層。樣本的界定與缺失的處理已於提案交代，
此處不重述。

資料沒有真實成因，目標因此分兩層，前者是後者可信的前提：

- *方法驗證。* 在設定真值的模擬資料上，確認方法能以可接受的錯誤率找回植入的
  偏移，且不把無偏移的公司列入清單（#ref(<ch:sim>)）。
- *實證應用。* 同一套方法用於真實資料，產出監理優先清單。

以下記號全篇通用。

#figure(
  table(
    columns: (auto, 1fr),
    stroke: hline,
    align: (left, left),
    row-gutter: 0.4em,
    [記號], [意義],
    [$i$], [航班],
    [$s$], [層，機場 × 年月],
    [$c$], [執飛公司；reference level 為 DL],
    [$k$], [成因欄位：`carrier`、`weather`、`NAS`、`late-aircraft`；reference level 為 `carrier`],
    [$u$], [單元，層 × 執飛公司],
    [$bold(x)_i$], [航班層的共變量：氣象欄位（#ref(<sec:basis>)）與排定時已知的欄位],
    [$m_(u k)$], [單元 $u$ 內記入成因 $k$ 的延誤分鐘數合計],
  ),
  caption: [記號。],
) <tbl:notation>

= 方法 <ch:methods>

本章每一節依序寫模型、出處、假設，以及假設在本資料上的成立情形或檢查方式。
各節的摘要見@tbl:methods。

== 估計對象 <sec:estimand>

申報是兩個決策的結果：延誤是否記入天氣欄，以及延誤分鐘數在各成因欄位之間
如何分配。兩者合為一個 two-part model，binary part 是「天氣欄是否為正」，
conditional part 是成因之間的分配。被估計的量在兩部分都是公司因子的主效應：
給定層與氣象共變量，公司 $c$ 相對於 reference level 的差異。

公司直接作為因子進入模型，其係數即所求，標準誤隨估計一併得到。另一種做法是
先以氣象建立預測模型、再比較各公司的殘差，但殘差沒有現成的抽樣分布，清單的
每一列便沒有區間。

主效應要讀成申報行為的差異，須假設同一層、同樣實測氣象下，各公司實際遭遇的
天氣延誤比例相同。這個假設不可由資料檢驗，機隊與航線特性都可能使它不成立。
本報告以兩種方式處理：#ref(<sec:confound>)量化須有多強的未觀測混淆才能改變
結論，#ref(<ch:sim>)則直接模擬假設被違反的情境，量測誤列的比例。

== 氣象基準的資訊集 <sec:basis>

*兩組氣象。* 資料集為每一班航班提供兩組氣象欄位。`wx_` 是排定起飛時刻之前
觀測到的氣象；`oracle_` 是航班自身的起飛與到達時刻所在小時的氣象。

*兩組都可用。* 發布者把 `oracle_` 與其餘欄位隔開，是因為資料集原本的用途是
預測延誤：起飛前不可能知道事件時刻的天氣，拿它預測即構成洩漏。本研究不預測
延誤，被分析的是事後填寫的成因申報。申報人填表時已知道航班實際遭遇的天氣，
`oracle_` 屬於他的資訊集；發布者也明言這組欄位供回溯分析使用。只以 `wx_` 為
共變量，等於用少於申報人的資訊評斷他的申報，起飛後天氣才惡化的航班會被誤判
為偏離。

*並列而不擇一。* 兩組氣象回答不同的問題，公司主效應在三種設定下分別估計，
並列報告。

#figure(
  table(
    columns: (auto, 1fr),
    stroke: hline,
    align: (left, left),
    row-gutter: 0.45em,
    [共變量], [公司主效應的意義],
    [`wx_`], [給定起飛前可知的天氣，各公司的申報差異],
    [`oracle_`], [給定航班實際遭遇的天氣，各公司的申報差異],
    [兩組並用], [給定起飛前的天氣與其後的變化，各公司的申報差異],
  ),
  caption: [氣象共變量的三種設定。],
) <tbl:basis>

三種設定下結論相同的公司，其偏離與資訊集的選擇無關。結論不同的公司，偏離
來自起飛後的天氣變化，須個別解讀。兩組之間的差另有用途，見
#ref(<sec:restrict>)。

*假設與檢查。*

- _觀測時點不受結果影響。_ 目的地的 `oracle_` 有兩個版本。實際到達時刻的
  版本，延誤越久取到的觀測越晚，共變量成了結果的函數，不可用。本研究取排定
  到達時刻的版本，其時點由班表決定。起點取起飛時刻的觀測，那是申報人確知的
  天氣。
- _兩組設定落在同一批航班上。_ 惡劣天氣的判定維持以 `wx_` 為準，樣本不隨
  共變量的設定改變，三組係數才可比較。
- _缺失與申報無關。_ `oracle_` 的缺失來自測站未觀測，不插補。缺觀測的航班
  在歸因率與延誤長度上是否不同於其餘航班，須先檢查。

== 層固定效應與 within estimator <sec:within>

*模型。* binary part 的結果變數 $Y_i$ 表示航班 $i$ 的天氣欄是否為正。

$ bb(E)[Y_i | s, c, bold(x)_i] = alpha_(s(i)) + gamma_(c(i)) + bold(x)_i^top bold(beta),
  quad gamma_"DL" = 0 $ <eq:within>

這是 binomial GLM with identity link，$gamma_c$ 即公司 $c$ 相對於 DL 的機率差。
$alpha_s$ 為層的固定效應，吸收跑道配置、除冰能力、當地航管慣例與該月的天氣
型態。估計採 within transformation：各變數減去層內平均後以最小平方法估計，
結果與放入全部層的指示變數相同，但不必建立 #thou(pick(dims, "機場 × 年月格子"))
欄的設計矩陣。

*出處。* 層的固定效應以條件化消去，是 #tc(<mundlak1978pooling>) 與
#tc(<chamberlain1980covariance>) 以來的標準處理。identity link 而非 logit link
的依據是 #tc(<chen2023lpm>)：該文給出線性模型的係數能近似非線性指標模型的
平均偏效果的條件。

*為何不用 logit link。* 理由有二。其一，logit link 之下層的固定效應無法以
減去層內平均消去；把層的指示變數直接放入則有 incidental parameter 問題，改用
條件概似又須對每一層計算組合數量級的分母，層內班次達上千時不可行。其二，
監理單位要的是機率差，identity link 的係數直接就是。

*假設與檢查。*

- _條件平均數為線性。_ identity link 不保證配適值落在 $[0, 1]$。此處不使用
  個別航班的配適機率，只使用係數 $gamma_c$，它估計的是平均機率差。線性設定
  錯誤時 $gamma_c$ 偏離多少，由模擬研究量測：生成模型刻意取非線性的形式
  （#ref(<ch:sim>)），within estimator 的偏誤與涵蓋率即是此假設的檢查。
- _層內須有比較。_ 只含一家公司、或結果全同的層，對 $gamma_c$ 沒有資訊，
  估計前即排除。含兩家以上公司的層佔 #pct(pick(cells, "兩家以上格子佔比"))，
  涵蓋 #pct(pick(cells, "兩家以上格子的航班佔比")) 的航班；每層公司數的中位數
  為 #str(int(pick(cells, "每格公司數中位數")))，多數層的比較只發生在少數幾家
  公司之間，小公司的係數因此雜訊較大。
- _設計矩陣不退化。_ 層內恆定的變數經 within transformation 後全為零；完全
  共線的欄位則使標準誤無定義。兩種退化都不一定報錯，估計後須檢查標準誤為
  有限正數。

== 成因分配的 multinomial logit <sec:simplex>

*資料結構。* 各成因欄位的分鐘數加總等於該航班的到達延誤，此限制在樣本內
嚴格成立，申報因此是 compositional data，落在 simplex 上。逐欄各自建模會違反
這個限制：某一欄多記的分鐘必然是其他欄少記的。`security` 欄幾乎不填報，剔除後
為四個成因。

*模型。* 分配以 multinomial logit 描述。令單元 $u$ 的線性預測量為

$ zeta_(u k) = lambda_(s(u), k) + theta_(c(u), k) + overline(bold(w))_u^top bold(psi)_k,
  quad theta_(c, #raw("carrier")) = 0, quad theta_("DL", k) = 0, $ <eq:lp>

則延誤記入成因 $k$ 的比例為

$ pi_(u k) = exp(zeta_(u k)) / (sum_j exp(zeta_(u j))). $ <eq:mnl>

$lambda_(s, k)$ 是層 × 成因的固定效應，吸收該機場該月的基準成因分配。
$overline(bold(w))_u$ 是單元內的氣象平均；氣象在單元內不隨成因變動，必須與
成因交互才進得了模型，$bold(psi)_k$ 讀作「天氣越惡劣，延誤越多記入成因 $k$」。
$theta_(c, k)$ 是公司 × 成因的係數，以唯一可歸責的 `carrier` 欄為 reference
level，讀作公司 $c$ 相對於 DL、相對於承認自身責任，多把延誤記入成因 $k$ 的
對數比。任意兩個成因之間的比較是 contrast，例如
$theta_(c, #raw("weather")) - theta_(c, #raw("NAS"))$，不必另立結果變數。

*估計。* @eq:mnl 是非線性模型，層 × 成因的固定效應數以萬計，無法直接估計。
改用 multinomial 與 Poisson log-linear model 的等價關係：

$ bb(E)[m_(u k)] = exp(eta_u + zeta_(u k)). $ <eq:poisson>

$eta_u$ 為單元的固定效應，吸收該公司在該層的申報總量；給定單元總量後，
@eq:poisson 的條件分布即 @eq:mnl，兩者對 $theta$ 的最大概似估計相同。等價關係
出自 #tc(<baker1994multinomial>)，#tc(<lee2017poisson>) 把它推廣到含群集結構的
情形。兩組高維固定效應以交替投影消去，演算法見 #tc(<guimaraes2010hdfe>) 與
#tc(<correia2019ppmlhdfe>)。

*為何不用對數比轉換。* 對數比轉換是 compositional data 的標準處理，但它要求
每一個分量為正。本資料的零是結構性的：某成因不成立即不填報，並非微量被捨去。
#tc(<alzeley2026zeros>) 指出結構性零不可插補，須以條件分布處理。若改為刪去含
零的單元，被刪的集中在班次少的單元，也就是小公司，而小公司正是清單要涵蓋的
對象。計數模型中零是合法的觀測值，不需插補也不需刪樣本。
#tc(<koslovsky2023zidm>) 的零膨脹 Dirichlet-multinomial 是另一條可行路線，
它以模型分離結構性零與抽樣零；本資料的零由申報制度決定，不需模型推斷，故
未採用。

*假設與檢查。*

- _Poisson 在此是 pseudo-likelihood。_ 分鐘數不是獨立事件的計數，Poisson 的
  變異數假設不成立。估計量的一致性只要求條件平均數 @eq:poisson 設定正確，
  不要求分布正確；變異數則不取 Poisson 的名目值，改用 cluster-robust 估計
  （#ref(<sec:cluster>)）。
- _incidental parameter。_ #tc(<weidner2019bias>) 證明多組固定效應的 Poisson
  估計量雖一致，卻帶有漸近偏誤，信賴區間的涵蓋率可能不足。該文處理三組固定
  效應，本模型為單元與層 × 成因兩組，結構同類而不相同，結論不能直接套用，
  須以模擬研究量測實際的涵蓋率（#ref(<ch:sim>)）。涵蓋率若不足，
  #tc(<fernandezval2013panel>) 對雙向固定效應的非線性模型給出解析與 jackknife
  兩種偏誤校正，Poisson 在其涵蓋範圍內。
- _重尾。_ 延誤分鐘數強右偏。#tc(<egger2026heavy>) 指出結果變數重尾時，
  Poisson pseudo-likelihood 的常規區間失效，並提出 bootstrap 修正。本模型的
  區間是否受影響列為#ref(<sec:rank>)的工作。
- _separation。_ 某公司在所有層都不曾記入某成因時，對應的 $theta_(c,k)$ 的
  最大概似估計不存在。#tc(<correia2019existence>) 證明被 separation 的觀測對
  其餘參數的估計不帶資訊，可安全移除。移除的若全是零，會不對稱地剔除某一類
  單元；檢查方式是改以班次門檻整批排除小單元，零與非零一併移除，比較係數
  是否隨門檻改變。

== 歸因強度的質量點 <sec:twopart>

已記入天氣的航班，其天氣分鐘數佔到達延誤的比例 $R_i in (0, 1]$ 在 $1$ 處有
質量點：一旦認定天氣，整段延誤全數記入天氣。beta 迴歸在區間端點無定義，
不能用。

*模型。* 質量點是可觀測的，不需要潛在變數來判斷哪些觀測屬於它，故拆成兩個
結果變數分別以 @eq:within 估計：$bb(1)(R_i = 1)$，以及給定 $R_i < 1$ 時的
$R_i$。兩者的公司主效應分開報告，不合成單一係數。

*出處。* #tc(<ospina2011inflated>) 的 zero-or-one inflated beta regression 把
比例型結果寫成端點的質量與區間內連續分布的混合，本研究採同樣的分解。差別在
區間內的部分：該文以 beta 分布建模，本研究只設定條件平均數並以 within
estimator 估計，原因同#ref(<sec:within>)，beta 的概似無法消去上萬個層的固定
效應。比例型結果只設定條件平均數而不設定分布的做法，見
#tc(<papke1996fractional>)。

*假設與檢查。* $R_i$ 只定義在已記入天氣的航班上，是以結果變數篩選的子樣本；
各公司記入天氣的機率不同時，此處的比較落在組成不同的子樣本上。這一節的係數
因此只輔助解讀，清單的數字取自#ref(<sec:within>)與#ref(<sec:simplex>)。

== 有限混合模型 <sec:mixture>

*問題。* 「整段延誤記入天氣」是同一個觀測值，卻可能出自不同的產生機制，
例如申報系統的預設值，或延誤確實全由天氣造成。機制不可觀測，但它們對共變量
的反應不同：若延誤全由天氣造成，延誤越長，其間混入其他成因的機會越大，整段
記入天氣的機率應隨延誤長度遞減。這是一個 monotonicity restriction。

*模型。* 令 $Z_i = bb(1)(R_i = 1)$，$bold(z)_i$ 為延誤長度與氣象嚴重度，
$Lambda$ 為 logistic 函數。

$ P(Z_i = 1) = sum_(k=1)^K w_(c(i), k) thin Lambda(a_k + bold(z)_i^top bold(b)_k),
  quad a_1 < dots.c < a_K $ <eq:mixture>

每個 mixture component 有自己的截距與斜率；$w_(c, k)$ 為公司 $c$ 的觀測屬於
component $k$ 的比例，由 softmax 轉換給出，其未轉換的參數設階層常態先驗，
班次少的公司向整體平均收縮。所關心的量是各 component 的延誤長度係數的正負，
以及各公司在各 component 上的權重。

*出處。* 貝氏有限混合模型的設定、成分數與標籤問題依
#tc(<grun2024mixture>)。成分的編號不可識別，各鏈之間標籤互換會使收斂診斷
失效 @stephens2000label，此處以截距遞增的限制固定編號。抽樣用
No-U-Turn Sampler @hoffman2011nuts。

*採貝氏估計的理由。* 清單需要的是「某公司有多少比例屬於某 component」這類
參數的函數，其不確定性可由後驗樣本直接計算；班次少的公司由階層先驗收縮，
不必另做一次。

*假設與檢查。*

- _可識別性。_ 二元結果的混合只靠共變量的變異識別，屬弱可識別。
  #tc(<betancourt2017hmc>) 說明 divergence 是後驗幾何困難的徵兆，可作為弱
  可識別的判讀依據。
- _抽樣有效。_ 以 #tc(<vehtari2019rhat>) 的 rank-normalized $hat(R)$ 與有效
  樣本數判斷。成分數 $K$ 的比較以診斷通過為先決條件：
  #tc(<vehtari2015loo>) 的 leave-one-out 交叉驗證假設後驗樣本有效，診斷未
  通過的 $K$ 不進入比較。
- _能否找回已知的機制。_ 模擬研究中有一種偏移型態正是「依規則整段記入天氣，
  且機率隨延誤長度遞增」（#ref(<ch:sim>)），用以量測此模型的檢定力。

== 變異數估計 <sec:cluster>

*方法。* @eq:within 的標準誤採 two-way cluster-robust variance estimator，
同時對層與機尾號 cluster：

$ hat(V) = hat(V)_S + hat(V)_T - hat(V)_(S ∩ T), $ <eq:twoway>

三項分別是對層、對機尾號、對兩者交集 cluster 的 sandwich 估計量。出處為
#tc(<cameron2011multiway>)。

*為何是這兩個維度。* #tc(<abadie2017cluster>) 主張 cluster 的層級由抽樣與
指派機制決定，而非由「殘差是否相關」的事後檢定決定。同一層的航班共用當日的
機場狀況；同一架飛機的航班共用機型與當日的輪轉。兩個維度並非巢狀：一個
機尾號所跨層數的中位數為 #crossing.at(0).at(1)，一層內班次的中位數僅
#crossing.at(0).at(2)，對層 cluster 吸收不到機尾號維度的相關。

*假設與檢查。* cluster-robust 推論是 cluster 數趨於無窮下的漸近結果。
#tc(<mackinnon2022guide>) 指出 cluster 數少或 cluster 大小懸殊時，sandwich
估計量會低估變異。本資料的總 cluster 數很大，但單一公司的係數只由該公司出現
的層與飛機決定，小公司的有效 cluster 數遠小於總數，大小也極不均勻。
#tc(<mackinnon2024twoway>) 的 two-way jackknife 是針對此情形的修正，列為
#ref(<sec:rank>)的工作。@eq:poisson 的標準誤目前對單元 cluster，同一節一併
檢查。

== shrinkage 與 multiple comparisons <sec:shrink>

*shrinkage。* 各公司的班次相差超過一個數量級，班次少的公司係數雜訊大，直接
排序會讓雜訊把它們推到清單兩端。設 $hat(gamma)_c tilde.op N(gamma_c, s_c^2)$、
$gamma_c tilde.op N(mu, tau^2)$，empirical Bayes 估計量為

$ tilde(gamma)_c = hat(mu) + (1 - B_c)(hat(gamma)_c - hat(mu)),
  quad B_c = s_c^2 / (s_c^2 + hat(tau)^2), $ <eq:eb>

$hat(mu)$ 取精確度加權平均，$hat(tau)^2$ 以動差法估計，即觀測到的公司間變異
扣除平均抽樣變異，負值截為零。出處為 #tc(<efron1973stein>)。$hat(tau)^2$ 為零
時所有公司收縮到同一點，排序隨之消失，此步驟因此同時檢查公司間的差異是否
超出抽樣雜訊。

*multiple comparisons。* 清單是排序，但「哪幾家顯著偏離」用到了門檻，數十個
係數同時檢定須校正。採 #tc(<benjamini1995fdr>) 的程序控制 false discovery
rate，跨所有結果變數的係數一起校正。

*假設與檢查。*

- _可交換性。_ @eq:eb 假設各公司的 $gamma_c$ 來自同一個常態分布。主線與區域
  航空的營運型態不同，是否宜共用一個先驗須檢查；替代方案是
  #tc(<gu2020invidious>) 的非參數 empirical Bayes，不設定先驗的形狀。
- _估計量之間獨立。_ @eq:eb 與 Benjamini–Hochberg 的原始證明都假設各
  $hat(gamma)_c$ 獨立。本模型的係數共用同一個 reference level，也共用層，
  並不獨立。相依對 false discovery rate 的實際影響，由模擬研究的 $H_0$ 情境
  直接量測。

== 摘要

#figure(
  text(size: 9.5pt, table(
    columns: (auto, auto, 1fr, 1fr),
    stroke: hline,
    align: (left, left, left, left),
    row-gutter: 0.35em,
    [方法], [用於], [主要假設], [檢查方式],
    [within estimator \ #ref(<sec:within>)], [binary part 的公司主效應], [條件平均數為線性], [模擬：非線性生成模型下的偏誤與涵蓋率],
    [Poisson log-linear \ #ref(<sec:simplex>)], [成因分配的公司 × 成因係數], [條件平均數正確；固定效應不造成偏誤；尾部不破壞區間], [模擬的涵蓋率；bootstrap 區間；排除小單元],
    [兩組氣象並列 \ #ref(<sec:basis>)], [氣象共變量], [觀測時點不受結果影響], [目的地取排定到達時刻；三種設定並列],
    [質量點分解 \ #ref(<sec:twopart>)], [歸因強度], [子樣本可比], [不檢查，降為輔助解讀],
    [有限混合模型 \ #ref(<sec:mixture>)], [整段記入天氣的產生機制], [成分可識別；抽樣有效], [divergence、$hat(R)$、有效樣本數；模擬的檢定力],
    [two-way cluster-robust \ #ref(<sec:cluster>)], [標準誤], [cluster 數夠多且大小均勻], [two-way jackknife],
    [empirical Bayes \ #ref(<sec:shrink>)], [排序前的收縮], [可交換；估計量獨立], [非參數 empirical Bayes 對照],
    [Benjamini–Hochberg \ #ref(<sec:shrink>)], [顯著家數], [檢定統計量獨立], [模擬：$H_0$ 下的 false discovery rate],
  )),
  caption: [各方法的主要假設與檢查方式。],
) <tbl:methods>

表中半數的檢查落在模擬研究。這些假設多半沒有只靠真實資料就能回答的
檢定，要知道區間是否有效，須在真值已知處量測。

= 方法驗證：模擬研究 <ch:sim>

模擬研究依 #tc(<morris2017simulation>) 的 ADEMP 框架設計與報告，依序交代
目的、資料生成機制、估計對象、方法與表現指標。

== 目的

回答三個問題：沒有任何公司偏移時，方法會把幾家列入清單；植入已知大小的偏移
後，偏移多大才偵測得到；識別假設被違反時，無偏移的公司有多少被誤列。

== 資料生成機制 <sec:dgm>

採 plasmode simulation @schreck2023plasmode：層、氣象、執飛公司、
到達延誤與前一航段全部保留真實資料，只把申報成因換成模擬值。本資料的不平衡、
稀疏的層與共變量之間的相關，正是方法最可能失效之處，純參數模擬會把這些特徵
抹平。

#figure(
  table(
    columns: (auto, 1fr),
    stroke: hline,
    align: (left, left),
    row-gutter: 0.45em,
    [情境], [生成方式],
    [$H_0$], [以「成因分配只依層與氣象、與公司無關」的模型配適真實資料，由配適的模型抽出申報成因。所有公司的真值為零],
    [植入偏移], [在 $H_0$ 資料上，對指定公司植入大小為 $delta$ 的偏移。型態四種：`carrier` 移往 `weather`；`NAS` 與 `late-aircraft` 移往 `weather`；由 `carrier` 均勻移出至其餘三欄；依規則整段記入 `weather`，且機率隨延誤長度遞增],
    [識別假設被違反], [不植入申報偏移，但令指定公司實際的天氣延誤比例較高，對應機隊特性造成的差異],
  ),
  caption: [三類情境。偏移型態取自真實資料中觀察到的申報形態，$delta$ 與被植入公司的班次規模皆設為多個水準。],
) <tbl:scenarios>

*生成模型須比分析模型有彈性。* #tc(<shaw2025plasmode>) 指出，生成模型與分析
模型相同時，模擬量到的只是模型對自己的配適，方法的表現會被高估。$H_0$ 的
生成模型因此加入分析模型沒有的項，例如氣象共變量之間的交互作用與非線性的
連結函數。#ref(<sec:within>)的線性假設即由此檢查。

*生成模型用事件時刻的氣象。* 成因分配由 `oracle_` 氣象決定，這對應申報人
實際擁有的資訊。分析模型則依@tbl:basis 的三種設定各估一次。只用 `wx_` 的
設定少了生成機制用到的資訊，它在 $H_0$ 下的誤列率就是資訊集不對齊的代價，
其餘兩種設定是對照。

== 估計對象

@eq:within 的 $gamma_c$、@eq:lp 的 $theta_(c, k)$ 及其 contrast，以及由這些
係數決定的各公司的 rank。真值由生成機制給定：$H_0$ 下全為零，植入偏移的
情境下由 $delta$ 決定。

== 方法

#ref(<ch:methods>)的全部方法，加上#ref(<sec:rank>)的 confidence set for
ranks 與#ref(<sec:restrict>)的 test of restrictions。每一次重複都執行與真實
資料分析完全相同的程式，包含層的篩選、收縮與校正。

== 表現指標

#figure(
  table(
    columns: (auto, 1fr),
    stroke: hline,
    align: (left, left),
    row-gutter: 0.45em,
    [指標], [回答的問題],
    [$H_0$ 下的 type I error 與 false discovery rate], [無偏移時是否誤列],
    [各 $delta$ 下的檢定力], [偏移多大才偵測得到，與公司班次的關係],
    [估計量的偏誤與信賴區間的涵蓋率], [高維固定效應與重尾結果下，區間是否仍有效],
    [rank 的還原率與 confidence set for ranks 的涵蓋率], [被植入的公司能否排到清單前段],
    [識別假設被違反時的誤列率], [未觀測的差異會把無偏移者推上清單多少],
    [理論參照與 empirical null 的 false discovery rate], [參照分布由資料估計時，誤列是否減少],
    [三種氣象設定各自的誤列率], [只用起飛前氣象時，資訊集不對齊的代價],
    [test of restrictions 的檢定力], [monotonicity 與交互作用檢定能否偵測對應型態的偏移],
  ),
  caption: [表現指標。各指標另報 Monte Carlo 標準誤。],
) <tbl:performance>

== 結論的界限

模擬只能否證方法，不能證實方法 @karim2026plasmode。通過的只是所設定
的偏移型態與生成機制；未設定的偏移型態是否偵測得到，模擬回答不了。真實資料
的清單若與模擬中「識別假設被違反」的情境形態相似，也無法由模擬區分兩者。
這兩點會在最終報告中明說。

== 計算成本

每一次重複都要重新配適含高維固定效應的 Poisson 模型，這是本研究第一個同一
運算大量重複的環節。重複次數由 Monte Carlo 標準誤的目標精確度反推，是否只
抽取部分的層，待實測單次配適的時間後決定。

= 分析計畫 <ch:plan>

以下各節各自處理清單的一個弱點。

== 清單的不確定性 <sec:rank>

*問題。* 交付物是排序，現有的區間卻是係數的區間。監理單位問的是「排在最前面
的那一家有多確定是最前面」，係數的區間不能直接回答：兩家的係數區間不重疊，
不代表整張清單的順序都確定。機構排名的這個問題由
#tc(<goldstein1996league>) 提出。

*rank 的推論。* 對 rank 本身建立信賴集合。#tc(<almohamad2017ranks>) 給出整張
清單同時成立的集合，#tc(<chetverikov2024csranks>) 另提供單一公司的邊際集合與
實作。兩者的輸入是係數估計與其共變異矩陣，可直接接在現有估計之後。
成因分配的 rank 可改用 #tc(<bazylik2024multinomial>) 針對 multinomial 資料的
版本；#tc(<chandra2025ranks>) 提供有限樣本下有效的集合，作為漸近版本的對照。

*清單該列幾家。* rank 的信賴集合若很寬，完整排序即無意義，交付物應改為分級。
#tc(<gu2020invidious>) 把排序與選擇寫成 compound decision，在控制誤選率之下
決定選出哪些單位；#tc(<kline2023reportcard>) 以同一框架為雇主評級，在評級的
資訊量與排序錯誤之間取捨。後者的處境與本研究相同：對一批機構的行為差異給出
可公開的分級。

*區間本身是否有效。* rank 的推論以係數的區間為輸入，係數的區間若失準，其後
全部失準。待檢查的有三項，前兩項的理由見#ref(<sec:simplex>)，第三項見
#ref(<sec:cluster>)：incidental parameter 造成的偏誤、重尾結果下的區間，以及
小公司的有效 cluster 數。做法是以 cluster bootstrap 與 two-way jackknife
重算區間，與 sandwich 區間並列；三者的涵蓋率在模擬研究中量測。

== 未觀測混淆 <sec:confound>

*問題。* #ref(<sec:estimand>)的識別假設可能因機隊特性而不成立：小型支線機對
低雲與強風的容忍度較低，確實較常被天氣拖延。此時公司主效應反映的是機隊差異。

*敏感度分析。* #tc(<cinelli2020sensitivity>) 以 partial $R^2$ 表示未觀測變數
與處理、與結果的關聯強度，回答「要多強的未觀測混淆才能使係數歸零或變號」，
並以已觀測共變量的強度作為比較基準。本研究的處理變數是公司，在層內才有變異，
#tc(<strezhnev2026ovb>) 指出處理在群集層變動時原公式須修正；
#tc(<wainstein2025wls>) 處理加權估計的情形。計畫對清單前段的每一家公司報告
使其係數歸零所需的混淆強度，並以氣象共變量的強度為基準。

*機尾號固定效應。* 機型隨機尾號固定，加入機尾號的固定效應可消去機隊特性。
代價在識別：此時公司主效應只能由曾為多家公司執飛的飛機識別。這類飛機若過少，
或只連通部分公司，其餘公司的主效應即不可識別。計畫先計算公司之間由共同
機尾號構成的連通結構，確認哪些公司之間可以比較，再決定此模型能回答的範圍。
不連通的部分只能由敏感度分析處理。

== 由模型導出的限制條件 <sec:restrict>

*問題。* 因子水準間的比較只能說某公司高於另一公司，而偏高可以有正當的解釋。
test of restrictions 不依賴公司之間的比較：由申報的機械關係導出一個應成立的
限制，檢驗各公司的資料是否違反。

- _monotonicity restriction。_ 延誤越長，整段記入天氣的機率應遞減
  （#ref(<sec:mixture>)）。迴歸函數是否單調可用 #tc(<chetverikov2012monotonicity>)
  的無母數檢定，不必先設定函數形式。
- _共變量 × 公司的交互作用檢定。_ 同一架飛機的前一航段晚到，會消耗班表留下
  的周轉時間，本班的延誤便有明確的機械成因，應記入 `late-aircraft`。以排定
  周轉時間減去前一航段的到達延誤為剩餘緩衝，兩個輸入都在本班起飛前確定，
  不受本班的申報影響。檢定的是剩餘緩衝對天氣歸因的影響是否因公司而異：
  剩餘緩衝小時反而更常記入天氣的公司，違反此限制。
- _天氣變化 × 公司的交互作用檢定。_ 以排定到達時刻的 `oracle_` 飛行規則類別
  減去起飛前的 `wx_` 類別，為航班起飛後的天氣變化（#ref(<sec:basis>)）。
  天氣惡化時更常記入天氣是應有的反應。檢定的是此反應是否因公司而異：天氣
  未惡化時仍提高天氣歸因、或惡化時毫無反應的公司，違反此限制。比較發生在
  同一家公司的不同航班之間，不受各公司航網差異的影響。
- _對實際天氣的 monotonicity restriction。_ 記入天氣的機率應隨事件時刻的
  天氣嚴重度遞增。此限制對 `wx_` 不成立也無妨，申報人依據的本來就不是起飛前
  的天氣；對 `oracle_` 則應成立。逐公司檢定，不遞增的區段即須解釋。
  飛行規則類別是四級的有序變數，有序解釋變數加上單調限制時的檢定與信賴區域見
  #tc(<espinosa2021ordinal>)。

*待處理。* 剩餘緩衝在資料中恆為正（緩衝若耗盡，同一架飛機無法執飛本班），
沒有自然的斷點，只能取低於第 $q$ 分位數者為一組。$q$ 是選定的值，須報告結論
隨 $q$ 的變化。各項檢定的檢定力在模擬研究中量測（@tbl:performance）。

== 氣象判定以外的航班 <sec:negctl>

*問題。* 稽核樣本以起飛前的 `wx_` 判定惡劣天氣。以事件時刻的 `oracle_` 再
判定一次，有申報義務的航班分成四類。

#figure(
  table(
    columns: (auto, auto, auto),
    stroke: hline,
    align: (left, left, left),
    row-gutter: 0.45em,
    [], [事件時刻惡劣], [事件時刻良好],
    [起飛前惡劣], [稽核樣本], [稽核樣本],
    [起飛前良好], [未納入], [未納入：negative control],
  ),
  caption: [依兩組氣象判定的四類航班。惡劣指起點或目的地為 IFR 或 LIFR。],
) <tbl:fourfold>

下列的兩類不在稽核樣本內，各有用途。主樣本維持不變，使結論能與提案及既有
輸出對照；下列兩類各自另做一個分析。

*起飛前良好、事件時刻惡劣。* 這些航班確實遭遇惡劣天氣，只因起飛前尚未轉壞
而被排除。以同一組模型在這批航班上另行估計公司主效應。兩批航班的結論若
一致，清單即不依賴樣本以哪一組氣象判定；若不一致，差異本身說明哪些公司的
申報對「起飛後才出現的天氣」反應不同。

*兩組氣象皆良好：negative control。* #tc(<lipsitch2010negative>) 的想法是
找一個預期沒有效應的情境，若該處仍觀察到關聯，即顯示有未測量的因素在作用。
其假設與設計準則見 #tc(<shi2020negative>) 的綜述。
兩端機場在起飛前與事件時刻都不是惡劣天氣的航班，記入天氣沒有機場觀測支持。
各公司在這類航班上記入天氣的機率有兩個用途：

- _方向性的證據。_ 主樣本的公司主效應只說明某公司高於其他公司。這裡的參照
  點是「沒有觀測到的天氣」，比例偏高的公司須說明天氣從何而來。
- _檢查主效應的來源。_ 某公司在 negative control 上的主效應若與主樣本同樣
  大小，它在主樣本的偏高便不是對天氣的反應較強，而是一種與天氣無關的記帳
  習慣。兩者的監理意義不同。

*假設與檢查。*

- _機場天氣良好不等於沒有天氣。_ 航路上的對流天氣不在測站觀測內，前段航班
  因天氣晚到也可能被記入天氣。此處的比例因此不會是零，有訊號的是公司之間的
  差異。層改取起點 × 目的地 × 年月，同一層的航班飛同一條航路，航路天氣在
  公司之間相同。層變細之後含兩家以上公司的層會減少，須先確認可比較的範圍。
- _樣本來源。_ 這兩類航班有申報義務，只是未通過惡劣天氣的判定，可由同一份
  已清理的資料取得，處理方式與稽核樣本相同。

*延伸。* 機場逐時氣象可算出航班排定時間窗內機場處於惡劣天氣的時數，把
「惡劣與否」換成連續的曝露量。時間窗須以排定的起飛與到達時刻界定；以實際
到達時刻界定，延誤越久時間窗越長，曝露量即成為結果的函數。

== 判定偏離的參照分布 <sec:null>

*問題。* 檢定的虛無假設是「公司主效應恰為零」。樣本達百萬班時標準誤極小，
任何非零的差異都會顯著。而即使沒有任何公司的申報有問題，氣象共變量未能完全
調整的差異也會使主效應不為零。以理論上的零為參照，顯著與否便失去鑑別力。
醫療機構的評比有同樣的處境 @normand2007profiling：
#tc(<xia2019profiling>) 指出，慣用的固定效應與隨機效應做法都會不成比例地
標記大機構，原因是參照分布沒有納入風險調整不完全所造成的機構間變異。

*方法。* 該文的做法是 empirical null @efron2007size：參照分布由資料
估計，不取理論值。設公司 $c$ 的估計量在「申報無異常」之下服從

$ hat(gamma)_c tilde.op N(mu_0, tau_0^2 + s_c^2), $ <eq:null>

$s_c^2$ 是抽樣變異，$tau_0^2$ 是與申報異常無關的公司間變異。各公司的 $s_c$
相差懸殊，參照分布因此隨公司而異，這一點同 #tc(<hartman2022null>) 的個別化
empirical null。以 @eq:null 算出的 $p$ 值再進入#ref(<sec:shrink>)的
Benjamini–Hochberg 程序。

*$mu_0$ 與 $tau_0^2$ 從哪裡估計。* empirical null 通常由估計值分布的中央部分
估計，前提是多數單位屬於虛無。本研究只有 #thou(pick(dims, "執飛航空公司")) 家
公司，中央部分無從界定。改由#ref(<sec:negctl>)的 negative control 估計：在
兩組氣象皆良好的航班上以同一組模型估計公司主效應，其分布的位置與離散即
$mu_0$ 與 $tau_0^2$。這是 #tc(<schuemie2014calibration>) 的 empirical
calibration 的做法，該文以預期無效應處的估計值分布校正主分析的 $p$ 值。

*與 shrinkage 的關係。* @eq:eb 的 $tau^2$ 是公司間的全部變異。此處把它拆成
兩部分：negative control 上也出現的 $tau_0^2$，以及只在惡劣天氣下出現的
其餘部分。清單要找的是後者。

*呈現。* 清單另附一張 funnel plot @spiegelhalter2005funnel：以估計量的
精確度為橫軸，畫出 @eq:null 的界限。班次少的公司界限寬，是否越界一望即知，
不必把所有公司排成一列。

*假設與檢查。*

- _negative control 與主樣本的偏誤來源相同。_ #tc(<hwang2021calibration>)
  檢視了 empirical calibration 在不同偏誤情境下的表現。本研究中此假設只部分
  成立：機隊對惡劣天氣的容忍度不同，這項差異只在惡劣天氣下起作用，天氣良好
  的航班量不到它。$tau_0^2$ 因此只涵蓋與天氣無關的公司間變異，機隊造成的
  部分仍由#ref(<sec:confound>)處理。
- _公司數少。_ 估計 $tau_0^2$ 的資料點只有公司數那麼多，估計值本身不確定，須把它的抽樣誤差
  傳入 @eq:null。機場層級的清單單位數多得多，empirical null 在該處較能發揮。
- _與天氣無關的偏高仍有監理意義。_ 以 @eq:null 判定，會放過在任何天氣下
  都偏好記入天氣的公司。這類公司由#ref(<sec:negctl>)的 negative control
  本身標出，兩個判定並列報告。
- _實際的錯誤率。_ 理論參照與 empirical null 兩種判定在 $H_0$ 情境與識別
  假設被違反的情境下的 false discovery rate，由模擬研究比較
  （#ref(<ch:sim>)）。

== 機場層級的清單 <sec:airport>

*問題。* 清單目前以公司為單位。監理單位派員查核時，「某公司在哪幾座機場」
比「某公司」更可執行。

*方法上的困難。* 公司 × 機場的單元班次稀少，許多單元內結果全同，出現
complete separation 或 quasi-complete separation，最大概似估計不存在。可行的
處理有兩條。其一是 #tc(<firth1993bias>) 的懲罰概似，估計量在 separation 下
仍有限；相應的 cluster-robust 推論見 #tc(<mackinnon2024logit>)。其二是階層
模型，令公司 × 機場的效應向公司的主效應收縮。公司與機場並非巢狀，須設為交叉
分類的隨機效應；此結構在大樣本下的估計見 #tc(<ghosh2020backfitting>)，貝氏抽樣見
#tc(<papaspiliopoulos2021crossed>)。此層的抽樣誤差相對於真實變異大得多，收縮預期會實質改變排序。

*順序。* 此方向的工作量最大，並引出新的識別問題：一座機場若只有少數公司，
公司 × 機場的效應與機場本身的效應難以分開。它排在其餘方向之後，且以公司
層級的方法通過模擬驗證為前提。

= 界限 <ch:limits>

- *清單不是違規認定。* 方法量到的是申報行為在公司之間的差異。由差異推到
  申報不實，須有#ref(<sec:estimand>)的識別假設，而該假設不可檢驗。認定需要
  航空公司的內部運行紀錄。
- *模擬的結論限於所設定的情境。* 見#ref(<ch:sim>)。
- *氣象基準仍少於申報人所知。* 兩組氣象都是機場測站的逐時觀測。申報人另有
  機組回報、航管指令與簽派紀錄，這些不在公開資料內。
- *取消的航班不在樣本內。* 取消班沒有到達延誤，不進入稽核樣本；取消本身也是
  一次成因申報，其與延誤申報的關係另行檢視。

#bibliography("refs.yml", style: "apa", title: [參考文獻])
