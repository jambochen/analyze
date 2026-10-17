#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#let ink = rgb("#22262b")
#let slate = rgb("#33414a")
#let mute = rgb("#6b7178")
#let sq-dark = rgb("#9c7a5b")
#let sq-light = rgb("#e6dccd")
#let rule = rgb("#d5d8db")

// 藍、橘、青留給統計圖的資料分組，版面不用，避免聽眾把版面色讀成資料類別。
// 進度條畫成棋盤的一列：每頁一格，走過的頁深淺交替。封面也算一頁，與 PDF 頁次一致。
#let rank-progress = context {
  let cur = utils.slide-counter.get().first() + 1
  let total = utils.last-slide-counter.final().first() + 1
  stack(dir: ltr, ..range(1, total + 1).map(i => box(
    width: 100% / total,
    height: 5pt,
    fill: if i > cur { rgb("#eef0f1") } else if calc.odd(i) { sq-dark } else { sq-light },
  )))
}

#let deck(info, body) = {
  show: metropolis-theme.with(
    aspect-ratio: "16-9",
    align: horizon,
    header: self => text(font: "Noto Serif CJK TC", weight: 600, utils.display-current-heading(depth: self.slide-level)),
    footer-progress: false,
    footer-right: context text(size: 0.7em, fill: mute, {
      let n = utils.slide-counter.get().first()
      let last = utils.last-slide-counter.final().first()
      if n > last [附錄] else [#(n + 1) / #(last + 1)]
    }),
    config-common(new-section-slide-fn: none),
    config-page(background: place(bottom, rank-progress)),
    config-colors(
      primary: sq-dark,
      primary-light: sq-light,
      secondary: slate,
      neutral-lightest: white,
      neutral-dark: slate,
      neutral-darkest: ink,
    ),
    info,
  )
  set text(font: "Noto Sans CJK TC", size: 20pt, weight: 350, lang: "zh", region: "tw")
  set par(leading: 0.9em)
  show heading: set text(font: "Noto Serif CJK TC", weight: 600)
  show raw: set text(font: "Noto Sans Mono CJK TC", size: 1.2em, fill: slate)
  show regex("[\p{Han}，。、；：「」（）][ \n]+[\p{Han}，。、；：「」（）]"): it => it.text.replace(regex("[ \n]+"), "")
  set table(stroke: none, inset: (x: 0.6em, y: 0.45em), align: left)
  show table.cell.where(y: 0): set text(weight: 500, fill: slate)
  set figure(numbering: none, gap: 0.5em)
  show figure.caption: set text(size: 0.7em, fill: mute)
  show figure: set block(spacing: 1.2em)
  body
}

// 色塊只用於定義性的整段：研究問題、核心定義。
#let define(body) = block(
  width: 100%,
  inset: (x: 1em, y: 0.8em),
  fill: sq-light.lighten(55%),
  stroke: (left: 3pt + sq-dark),
  body,
)

// 封面標題與頁標題同用襯線字
#let title-font(body) = text(font: "Noto Serif CJK TC", weight: 600, body)

#let caption(body) = text(size: 0.8em, fill: mute, body)
