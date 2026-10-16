// Lichess Glicko-2 逐局重放。計算逐行對應 scripts/glicko.py 的參考實作，
// 兩者都移植自 lila 的 org.goochjs.glicko2（RatingCalculator.java @ bfad573339）。
// 不得做任何改變浮點運算順序的改寫（例如合併乘法、用 mul_add），否則與參考實作
// 的逐位元一致性會失效。

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

const MULTIPLIER: f64 = 173.7178;
const DEFAULT_RATING: f64 = 1500.0;
const DEFAULT_DEVIATION: f64 = 350.0;
const DEFAULT_VOLATILITY: f64 = 0.06;
const CONVERGENCE_TOLERANCE: f64 = 0.000001;
const ITERATION_MAX: i32 = 3000;

fn g(phi: f64) -> f64 {
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    1.0 / (1.0 + 3.0 * phi * phi / pi2).sqrt()
}

fn f(x: f64, delta: f64, phi: f64, v: f64, a: f64, tau: f64) -> f64 {
    let ex = x.exp();
    let d = phi * phi + v + ex;
    (ex * (delta * delta - phi * phi - v - ex) / (2.0 * d * d)) - ((x - a) / (tau * tau))
}

fn new_rating(mu: f64, phi: f64, sigma: f64, mu_o: f64, phi_o: f64, s: f64, tau: f64) -> Option<[f64; 3]> {
    let go = g(phi_o);
    let e = 1.0 / (1.0 + (-1.0 * go * (mu - mu_o)).exp());
    let v = 1.0 / (go * go * e * (1.0 - e));
    let outcome = go * (s - e);
    let delta = v * outcome;
    let a = (sigma * sigma).ln();
    let mut big_a = a;
    let mut big_b;
    if delta * delta > phi * phi + v {
        big_b = (delta * delta - phi * phi - v).ln();
    } else {
        let mut k = 1.0;
        big_b = a - k * tau.abs();
        while f(big_b, delta, phi, v, a, tau) < 0.0 {
            k += 1.0;
            big_b = a - k * tau.abs();
        }
    }
    let mut fa = f(big_a, delta, phi, v, a, tau);
    let mut fb = f(big_b, delta, phi, v, a, tau);
    let mut it = 0;
    while (big_b - big_a).abs() > CONVERGENCE_TOLERANCE && it < ITERATION_MAX {
        it += 1;
        let c = big_a + ((big_a - big_b) * fa) / (fb - fa);
        let fc = f(c, delta, phi, v, a, tau);
        if fc * fb < 0.0 {
            big_a = big_b;
            fa = fb;
        } else {
            fa = fa / 2.0;
        }
        big_b = c;
        fb = fc;
    }
    if it == ITERATION_MAX {
        return None;
    }
    let new_sigma = (big_a / 2.0).exp();
    let phi_star = (phi * phi + new_sigma * new_sigma).sqrt();
    let new_phi = 1.0 / ((1.0 / (phi_star * phi_star)) + (1.0 / v)).sqrt();
    Some([mu + new_phi * new_phi * outcome, new_phi, new_sigma])
}

fn update(w: [f64; 3], b: [f64; 3], s: f64, tau: f64, floor: f64) -> ([f64; 3], [f64; 3]) {
    let rw = floor.max(w[0]);
    let rb = floor.max(b[0]);
    let mw = (rw - DEFAULT_RATING) / MULTIPLIER;
    let pw = w[1] / MULTIPLIER;
    let mb = (rb - DEFAULT_RATING) / MULTIPLIER;
    let pb = b[1] / MULTIPLIER;
    match (
        new_rating(mw, pw, w[2], mb, pb, s, tau),
        new_rating(mb, pb, b[2], mw, pw, 1.0 - s, tau),
    ) {
        (Some(nw), Some(nb)) => (
            [nw[0] * MULTIPLIER + DEFAULT_RATING, nw[1] * MULTIPLIER, nw[2]],
            [nb[0] * MULTIPLIER + DEFAULT_RATING, nb[1] * MULTIPLIER, nb[2]],
        ),
        _ => ([rw, w[1], w[2]], [rb, b[1], b[2]]),
    }
}

#[derive(Clone, Copy)]
struct Config {
    tau: f64,
    floor: f64,
    // 以表頭校正：每個 (帳號, 速度類別) 維護真實等級分可能落在的區間 [lo, hi)，
    // 每局與表頭隱含的區間取交集，空集合時以表頭重設。關閉時即從 1500 起的純重放
    anchor: bool,
    // 表頭整數的解讀：真為 Double.toInt 截斷，假為四捨五入
    truncate: bool,
    // 資料內首局表頭不是 1500 的帳號，在資料起點前已有對局（左截斷），RD 未知，以此值起算
    rd_trunc: f64,
}

#[derive(Clone, Copy)]
struct State {
    lo: f64,
    hi: f64,
    rd: f64,
    vol: f64,
    updated: bool,
}

/// 單方在對局前與表頭比對並校正，回傳旗標：
/// 位元 0 校正前的點估計與表頭相符、1 校正前的區間與表頭相容、2 不相容而重設、3 左截斷起算。
fn observe(s: &mut State, h: i32, c: &Config) -> u8 {
    let mid = (s.lo + s.hi) / 2.0;
    let point = if c.truncate { mid as i32 == h } else { mid.round() as i32 == h };
    if !c.anchor || h == i32::MIN {
        return point as u8 | (point as u8) << 1;
    }
    let (hl, hh) = if c.truncate { (h as f64, h as f64 + 1.0) } else { (h as f64 - 0.5, h as f64 + 0.5) };
    if !s.updated && h != 1500 {
        *s = State { lo: hl, hi: hh, rd: c.rd_trunc, vol: DEFAULT_VOLATILITY, updated: false };
        return point as u8 | 1 << 3;
    }
    let (nlo, nhi) = (s.lo.max(hl), s.hi.min(hh));
    let consistent = nlo < nhi || (s.lo == s.hi && hl <= s.lo && s.lo < hh);
    if consistent {
        if s.lo != s.hi {
            s.lo = nlo;
            s.hi = nhi;
        }
        point as u8 | 1 << 1
    } else {
        s.lo = hl;
        s.hi = hh;
        point as u8 | 1 << 2
    }
}

fn advance(s: &mut State, new: [f64; 3], mid: f64, floor: f64) {
    // 區間退化為一點（純重放或尚未校正）時直接採用結果，與參考實作逐位元一致；
    // 整個區間都在下限之下時，計算輸入都是下限，結果與區間內位置無關
    if s.lo == s.hi || s.hi <= floor {
        s.lo = new[0];
        s.hi = new[0];
    } else {
        let shift = new[0] - mid;
        s.lo += shift;
        s.hi += shift;
    }
    s.rd = new[1];
    s.vol = new[2];
    s.updated = true;
}

#[allow(clippy::too_many_arguments)]
fn run(
    white: &[u32],
    black: &[u32],
    perf: &[u8],
    score: &[f64],
    rated: &[bool],
    wa: &[i32],
    ba: &[i32],
    n_perf: usize,
    c: Config,
    mut sink: impl FnMut(usize, [f64; 3], [f64; 3], u8, u8),
) {
    let n_players = white.iter().chain(black.iter()).copied().max().map_or(0, |m| m as usize + 1);
    let init = State { lo: DEFAULT_RATING, hi: DEFAULT_RATING, rd: DEFAULT_DEVIATION, vol: DEFAULT_VOLATILITY, updated: false };
    let mut state = vec![init; n_players * n_perf];
    for i in 0..white.len() {
        let iw = white[i] as usize * n_perf + perf[i] as usize;
        let ib = black[i] as usize * n_perf + perf[i] as usize;
        let fw = observe(&mut state[iw], wa[i], &c);
        let fb = observe(&mut state[ib], ba[i], &c);
        let (sw, sb) = (state[iw], state[ib]);
        let pw = [(sw.lo + sw.hi) / 2.0, sw.rd, sw.vol];
        let pb = [(sb.lo + sb.hi) / 2.0, sb.rd, sb.vol];
        sink(i, pw, pb, fw, fb);
        if rated[i] {
            let (nw, nb) = update(pw, pb, score[i], c.tau, c.floor);
            advance(&mut state[iw], nw, pw[0], c.floor);
            advance(&mut state[ib], nb, pb[0], c.floor);
        }
    }
}

type Out<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
);

/// 依序重放，回傳每局白方與黑方對局前（校正後）的 rating、RD、volatility 與 observe 的旗標。
/// perf 為 0..n_perf 的速度類別代碼；rated 為假的對局不更新但仍記錄對局前狀態。
/// white_anchor、black_anchor 為可用來校正的表頭等級分，不可用時為 i32::MIN。
#[pyfunction]
#[pyo3(signature = (white, black, perf, score, rated, white_anchor, black_anchor, n_perf, tau=0.75, floor=800.0, anchor=true, truncate=true, rd_trunc=350.0))]
#[allow(clippy::too_many_arguments)]
fn replay<'py>(
    py: Python<'py>,
    white: PyReadonlyArray1<'py, u32>,
    black: PyReadonlyArray1<'py, u32>,
    perf: PyReadonlyArray1<'py, u8>,
    score: PyReadonlyArray1<'py, f64>,
    rated: PyReadonlyArray1<'py, bool>,
    white_anchor: PyReadonlyArray1<'py, i32>,
    black_anchor: PyReadonlyArray1<'py, i32>,
    n_perf: usize,
    tau: f64,
    floor: f64,
    anchor: bool,
    truncate: bool,
    rd_trunc: f64,
) -> PyResult<Out<'py>> {
    let (white, black, perf, score, rated, wa, ba) = (
        white.as_slice()?, black.as_slice()?, perf.as_slice()?, score.as_slice()?, rated.as_slice()?,
        white_anchor.as_slice()?, black_anchor.as_slice()?,
    );
    let c = Config { tau, floor, anchor, truncate, rd_trunc };
    let n = white.len();
    let (out, fws, fbs) = py.detach(|| {
        let mut out: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(n));
        let (mut fws, mut fbs) = (Vec::with_capacity(n), Vec::with_capacity(n));
        run(white, black, perf, score, rated, wa, ba, n_perf, c, |_, sw, sb, fw, fb| {
            for k in 0..3 {
                out[k].push(sw[k]);
                out[k + 3].push(sb[k]);
            }
            fws.push(fw);
            fbs.push(fb);
        });
        (out, fws, fbs)
    });
    let [a, b, c_, d, e, f] = out;
    Ok((
        PyArray1::from_vec(py, a),
        PyArray1::from_vec(py, b),
        PyArray1::from_vec(py, c_),
        PyArray1::from_vec(py, d),
        PyArray1::from_vec(py, e),
        PyArray1::from_vec(py, f),
        PyArray1::from_vec(py, fws),
        PyArray1::from_vec(py, fbs),
    ))
}

/// 以 rayon 平行跑多組規則與參數，每組只回傳 observe 的旗標（白方低四位元、黑方高四位元），
/// 避免同時持有多份完整的浮點輸出。configs 的每項為
/// (perfs 的索引, rateds 的索引, tau, floor, anchor, truncate, rd_trunc)。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn replay_match<'py>(
    py: Python<'py>,
    white: PyReadonlyArray1<'py, u32>,
    black: PyReadonlyArray1<'py, u32>,
    score: PyReadonlyArray1<'py, f64>,
    white_anchor: PyReadonlyArray1<'py, i32>,
    black_anchor: PyReadonlyArray1<'py, i32>,
    perfs: Vec<PyReadonlyArray1<'py, u8>>,
    rateds: Vec<PyReadonlyArray1<'py, bool>>,
    configs: Vec<(usize, usize, f64, f64, bool, bool, f64)>,
    n_perf: usize,
) -> PyResult<Vec<Bound<'py, PyArray1<u8>>>> {
    let (white, black, score, wa, ba) =
        (white.as_slice()?, black.as_slice()?, score.as_slice()?, white_anchor.as_slice()?, black_anchor.as_slice()?);
    let perfs: Vec<&[u8]> = perfs.iter().map(|a| a.as_slice()).collect::<Result<_, _>>()?;
    let rateds: Vec<&[bool]> = rateds.iter().map(|a| a.as_slice()).collect::<Result<_, _>>()?;
    let n = white.len();
    let res: Vec<Vec<u8>> = py.detach(|| {
        configs
            .par_iter()
            .map(|&(ip, ir, tau, floor, anchor, truncate, rd_trunc)| {
                let c = Config { tau, floor, anchor, truncate, rd_trunc };
                let mut out = vec![0u8; n];
                run(white, black, perfs[ip], score, rateds[ir], wa, ba, n_perf, c, |i, _, _, fw, fb| {
                    out[i] = fw | fb << 4;
                });
                out
            })
            .collect()
    });
    Ok(res.into_iter().map(|v| PyArray1::from_vec(py, v)).collect())
}

/// 只用表頭、不經重放的對局史檢查，用來把不相符歸因到資料本身的斷點。
/// 每個 (帳號, 速度類別) 維持「表頭預期的下一局等級分」：更新的局為本局加 RatingDiff，
/// 未更新的局為本局本身。回傳每局雙方的旗標與此前在該類別的局數。
/// 旗標位元：0 本局與預期相符（首個可用表頭以 1500 為預期）、1 首局、2 首個可用表頭為 1500、
/// 3 至本局為止曾有任何不符、4 同一鏈的上一局未更新。
/// 表頭不可用（缺值或重算期間未更新而留著舊值）以 i32::MIN 表示：不檢查、不改變預期。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn header_chain<'py>(
    py: Python<'py>,
    white: PyReadonlyArray1<'py, u32>,
    black: PyReadonlyArray1<'py, u32>,
    perf: PyReadonlyArray1<'py, u8>,
    white_elo: PyReadonlyArray1<'py, i32>,
    black_elo: PyReadonlyArray1<'py, i32>,
    white_diff: PyReadonlyArray1<'py, i32>,
    black_diff: PyReadonlyArray1<'py, i32>,
    rated: PyReadonlyArray1<'py, bool>,
    n_perf: usize,
) -> PyResult<(
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let (white, black, perf, we, be, wd, bd, rated) = (
        white.as_slice()?, black.as_slice()?, perf.as_slice()?, white_elo.as_slice()?,
        black_elo.as_slice()?, white_diff.as_slice()?, black_diff.as_slice()?, rated.as_slice()?,
    );
    let n = white.len();
    let (wf, bf, wn, bn) = py.detach(|| {
        let n_players = white.iter().chain(black.iter()).copied().max().map_or(0, |m| m as usize + 1);
        let mut st = vec![Chain { expect: 1500, nb: 0, seen: false, origin: false, broken: false, prev_unrated: false }; n_players * n_perf];
        let (mut wf, mut bf, mut wn, mut bn) = (vec![0u8; n], vec![0u8; n], vec![0u32; n], vec![0u32; n]);
        for i in 0..n {
            for (id, elo, diff, flags, nb) in [
                (white[i], we[i], wd[i], &mut wf, &mut wn),
                (black[i], be[i], bd[i], &mut bf, &mut bn),
            ] {
                let s = &mut st[id as usize * n_perf + perf[i] as usize];
                let first = s.nb == 0;
                let mut ok = false;
                if elo != i32::MIN {
                    if !s.seen {
                        s.origin = elo == 1500;
                        s.seen = true;
                    }
                    ok = elo == s.expect;
                    s.broken |= !ok;
                }
                flags[i] = ok as u8 | (first as u8) << 1 | (s.origin as u8) << 2 | (s.broken as u8) << 3 | (s.prev_unrated as u8) << 4;
                nb[i] = s.nb;
                s.nb += 1;
                if elo != i32::MIN {
                    s.expect = if rated[i] { elo + diff } else { elo };
                } else if rated[i] {
                    s.expect = i32::MIN;
                }
                s.prev_unrated = !rated[i];
            }
        }
        (wf, bf, wn, bn)
    });
    Ok((PyArray1::from_vec(py, wf), PyArray1::from_vec(py, bf), PyArray1::from_vec(py, wn), PyArray1::from_vec(py, bn)))
}

#[derive(Clone, Copy)]
struct Chain {
    expect: i32,
    nb: u32,
    seen: bool,
    origin: bool,
    broken: bool,
    prev_unrated: bool,
}

/// 各局獨立做一次更新：給定雙方對局前的 rating、RD、volatility 與白方得分，回傳雙方局後的 rating。
/// 用於以表頭等級分配上重放的 RD 檢驗表頭的 RatingDiff，各局互不依賴，故平行計算。
#[pyfunction]
#[pyo3(signature = (wr, wd, wv, br, bd, bv, score, tau=0.75, floor=800.0))]
#[allow(clippy::too_many_arguments)]
fn step<'py>(
    py: Python<'py>,
    wr: PyReadonlyArray1<'py, f64>,
    wd: PyReadonlyArray1<'py, f64>,
    wv: PyReadonlyArray1<'py, f64>,
    br: PyReadonlyArray1<'py, f64>,
    bd: PyReadonlyArray1<'py, f64>,
    bv: PyReadonlyArray1<'py, f64>,
    score: PyReadonlyArray1<'py, f64>,
    tau: f64,
    floor: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (wr, wd, wv, br, bd, bv, score) = (
        wr.as_slice()?, wd.as_slice()?, wv.as_slice()?, br.as_slice()?, bd.as_slice()?, bv.as_slice()?, score.as_slice()?,
    );
    let (a, b): (Vec<f64>, Vec<f64>) = py.detach(|| {
        (0..wr.len())
            .into_par_iter()
            .map(|i| {
                let (nw, nb) = update([wr[i], wd[i], wv[i]], [br[i], bd[i], bv[i]], score[i], tau, floor);
                (nw[0], nb[0])
            })
            .unzip()
    });
    Ok((PyArray1::from_vec(py, a), PyArray1::from_vec(py, b)))
}

#[pymodule]
fn glicko_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(replay, m)?)?;
    m.add_function(wrap_pyfunction!(replay_match, m)?)?;
    m.add_function(wrap_pyfunction!(header_chain, m)?)?;
    m.add_function(wrap_pyfunction!(step, m)?)?;
    Ok(())
}
