// scripts/continuation_selection.py 界限與敏感度分析的加權平均。
// trim_means：玩家 block bootstrap 的加權平均與修剪平均。numpy 寫法要展開「重抽 × 列」的權重矩陣，
// 每次重抽都要掃過數個與矩陣同大小的暫存陣列，單執行緒且受記憶體頻寬限制；這裡每次重抽只持有一列權重，各次重抽以 rayon 平行。
// tilt_means：有參數敏感度分析的傾斜平均，各 gamma 的求根以 rayon 平行。

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

// 保留權重的下端 share 比例；numpy 版為 take = clip(target - (cumsum - w), 0, w)。
// 累計權重超過 target 之後 take 全為 0，可提前結束
fn trim<I: Iterator<Item = usize>>(idx: I, y: &[f64], wb: &[f64], target: f64) -> f64 {
    let mut prev = 0.0;
    let mut acc = 0.0;
    for j in idx {
        acc += (target - prev).max(0.0).min(wb[j]) * y[j];
        prev += wb[j];
        if prev >= target {
            break;
        }
    }
    acc / target
}

/// 回傳 (mean, low, high)：mean 形狀 (B+1,)，low、high 形狀 (B+1, K)。
/// 第 0 列的權重為 w，第 b 列為 counts[b-1, player] * w；counts 為 (B, 玩家數) 的重抽次數。
/// low、high 為保留下端、上端 shares[b, k] 比例權重的修剪平均，y 須已依升冪排序；K 為 0 時只算 mean，y 不必排序。
#[pyfunction]
fn trim_means<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    player: PyReadonlyArray1<'py, i64>,
    w: PyReadonlyArray1<'py, f64>,
    counts: PyReadonlyArray2<'py, f64>,
    shares: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (np_, k) = (counts.shape()[1], shares.shape()[1]);
    let rows = counts.shape()[0] + 1;
    assert_eq!(shares.shape()[0], rows);
    let (y, player, w, counts, shares) = (y.as_slice()?, player.as_slice()?, w.as_slice()?, counts.as_slice()?, shares.as_slice()?);
    let n = y.len();
    let res: Vec<(f64, Vec<f64>, Vec<f64>)> = py.detach(|| {
        (0..rows)
            .into_par_iter()
            .map(|b| {
                let wb: Vec<f64> = if b == 0 {
                    w.to_vec()
                } else {
                    let c = &counts[(b - 1) * np_..b * np_];
                    (0..n).map(|j| c[player[j] as usize] * w[j]).collect()
                };
                let tot: f64 = wb.iter().sum();
                let wy: f64 = wb.iter().zip(y).map(|(a, b)| a * b).sum();
                let s = &shares[b * k..(b + 1) * k];
                let lo = s.iter().map(|&sh| trim(0..n, y, &wb, sh * tot)).collect();
                let hi = s.iter().map(|&sh| trim((0..n).rev(), y, &wb, sh * tot)).collect();
                (wy / tot, lo, hi)
            })
            .collect()
    });
    let mean: Vec<f64> = res.iter().map(|r| r.0).collect();
    let lo: Vec<f64> = res.iter().flat_map(|r| r.1.iter().copied()).collect();
    let hi: Vec<f64> = res.iter().flat_map(|r| r.2.iter().copied()).collect();
    Ok((PyArray1::from_vec(py, mean), PyArray1::from_vec(py, lo).reshape([rows, k])?, PyArray1::from_vec(py, hi).reshape([rows, k])?))
}

// scipy.optimize.brentq 的移植（scipy/optimize/Zeros/brentq.c），預設容差相同，求根結果與 scipy 只差浮點誤差
fn brentq<F: Fn(f64) -> f64>(f: F, xa: f64, xb: f64) -> f64 {
    let (xtol, rtol) = (2e-12, 4.0 * f64::EPSILON);
    let (mut xpre, mut xcur) = (xa, xb);
    let (mut xblk, mut fblk, mut spre, mut scur) = (0.0, 0.0, 0.0, 0.0);
    let (mut fpre, mut fcur) = (f(xpre), f(xcur));
    if fpre == 0.0 {
        return xpre;
    }
    if fcur == 0.0 {
        return xcur;
    }
    assert!(fpre.is_sign_negative() != fcur.is_sign_negative(), "f(a) 與 f(b) 同號");
    for _ in 0..100 {
        if fpre != 0.0 && fcur != 0.0 && fpre.is_sign_negative() != fcur.is_sign_negative() {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return xcur;
        }
        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        xcur += if scur.abs() > delta { scur } else if sbis > 0.0 { delta } else { -delta };
        fcur = f(xcur);
    }
    xcur
}

/// 選擇上的有參數模型：續下者屬於「兩組都會續下」的機率為 expit(alpha + gamma y)，加權平均等於 share。
/// 對每個 gamma 求 alpha，回傳以該機率加權的 y 平均。各 gamma 以 rayon 平行；
/// numpy 版逐一 gamma 呼叫 brentq，每次求值只對一個 gamma 算 exp，無法向量化也無法平行
#[pyfunction]
fn tilt_means<'py>(py: Python<'py>, y: PyReadonlyArray1<'py, f64>, w: PyReadonlyArray1<'py, f64>, share: f64, gammas: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (y, w, gammas) = (y.as_slice()?, w.as_slice()?, gammas.as_slice()?);
    let wsum: f64 = w.iter().sum();
    let out: Vec<f64> = py.detach(|| {
        gammas
            .par_iter()
            .map(|&g| {
                if g == 0.0 || share == 1.0 {
                    return w.iter().zip(y).map(|(a, b)| a * b).sum::<f64>() / wsum;
                }
                let f = |a: f64| w.iter().zip(y).map(|(wj, yj)| wj / (1.0 + (-(a + g * yj)).exp())).sum::<f64>() / wsum - share;
                let a = brentq(f, -60.0 - g.abs(), 60.0 + g.abs());
                let (mut num, mut den) = (0.0, 0.0);
                for (wj, yj) in w.iter().zip(y) {
                    let pi = wj / (1.0 + (-(a + g * yj)).exp());
                    num += pi * yj;
                    den += pi;
                }
                num / den
            })
            .collect()
    });
    Ok(PyArray1::from_vec(py, out))
}

#[pymodule]
fn weighted_trim_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trim_means, m)?)?;
    m.add_function(wrap_pyfunction!(tilt_means, m)?)?;
    Ok(())
}
