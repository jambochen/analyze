// scripts/likelihood.py 的熱點。
// vgh、score：logit_ll（以得分為反應、線性預測式 X theta 的 fractional logit）的負對數概似、梯度、Hessian 與逐列分數。
// jax 版在 CPU 上逐塊依序執行，XLA 對這類逐元素運算與加總只平行到兩三個核；這裡各列以 rayon 分塊平行，一次掃過資料同時累加三者。
// group_outer、dyad_meat：sandwich 共變異的 meat。numpy 版以 np.unique 排序再逐欄 bincount，全為單執行緒。

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

const BLOCK: usize = 4096;

// log sigmoid(z)，兩側都不溢位
fn log_sigmoid(z: f64) -> f64 {
    if z >= 0.0 { -(-z).exp().ln_1p() } else { z - z.exp().ln_1p() }
}

/// 回傳 (負對數概似, 對 free 的梯度, 對 free 的 Hessian)。x 為 (n, p) 的 C-order 矩陣，theta 為長 p 的完整參數
#[pyfunction]
fn vgh<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    theta: PyReadonlyArray1<'py, f64>,
    free: PyReadonlyArray1<'py, i64>,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let p = x.shape()[1];
    let (x, y, theta) = (x.as_slice()?, y.as_slice()?, theta.as_slice()?);
    let free: Vec<usize> = free.as_slice()?.iter().map(|&j| j as usize).collect();
    let k = free.len();
    let (v, g, h) = py.detach(|| {
        y.par_chunks(BLOCK)
            .enumerate()
            .map(|(c, yb)| {
                let (mut v, mut g, mut h) = (0.0, vec![0.0; k], vec![0.0; k * k]);
                let mut xf = vec![0.0; k];
                for (r, &yi) in yb.iter().enumerate() {
                    let row = &x[(c * BLOCK + r) * p..(c * BLOCK + r + 1) * p];
                    let z: f64 = row.iter().zip(theta).map(|(a, b)| a * b).sum();
                    let lp = log_sigmoid(z);
                    v -= yi * lp + (1.0 - yi) * (lp - z);
                    let pr = lp.exp();
                    let (res, wt) = (yi - pr, pr * (1.0 - pr));
                    for (a, &j) in free.iter().enumerate() {
                        xf[a] = row[j];
                        g[a] -= res * xf[a];
                    }
                    for a in 0..k {
                        let wa = wt * xf[a];
                        for b in a..k {
                            h[a * k + b] += wa * xf[b];
                        }
                    }
                }
                (v, g, h)
            })
            .reduce(
                || (0.0, vec![0.0; k], vec![0.0; k * k]),
                |(v1, mut g1, mut h1), (v2, g2, h2)| {
                    g1.iter_mut().zip(&g2).for_each(|(a, b)| *a += b);
                    h1.iter_mut().zip(&h2).for_each(|(a, b)| *a += b);
                    (v1 + v2, g1, h1)
                },
            )
    });
    let mut h = h;
    for a in 0..k {
        for b in 0..a {
            h[a * k + b] = h[b * k + a];
        }
    }
    Ok((v, PyArray1::from_vec(py, g), PyArray1::from_vec(py, h).reshape([k, k])?))
}

/// 逐列的對數概似對 free 的分數，(n, len(free))
#[pyfunction]
fn score<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    theta: PyReadonlyArray1<'py, f64>,
    free: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = x.shape()[1];
    let (x, y, theta) = (x.as_slice()?, y.as_slice()?, theta.as_slice()?);
    let free: Vec<usize> = free.as_slice()?.iter().map(|&j| j as usize).collect();
    let (n, k) = (y.len(), free.len());
    let mut out = vec![0.0; n * k];
    py.detach(|| {
        out.par_chunks_mut(k.max(1)).enumerate().for_each(|(i, o)| {
            let row = &x[i * p..(i + 1) * p];
            let z: f64 = row.iter().zip(theta).map(|(a, b)| a * b).sum();
            let res = y[i] - 1.0 / (1.0 + (-z).exp());
            for (a, &j) in free.iter().enumerate() {
                o[a] = res * row[j];
            }
        })
    });
    Ok(PyArray1::from_vec(py, out).reshape([n, k])?)
}

// 依 key 分組，回傳各組分數和的外積之和 sum_g s_g s_g'（k x k）。key 以平行排序分組，各組再分塊平行累加。
// key 的長度可為列數的整數倍，第 i 個鍵對應第 i % n 列：一列同時屬於兩組（一局的兩位玩家）時把鍵接在一起。
// 不以組的代碼開累加陣列：代碼是全庫的玩家代碼，遠大於樣本內的組數，每條執行緒一份會配置、清零與相加大量記憶體
fn group_outer_impl(s: &[f64], k: usize, key: &[u64]) -> Vec<f64> {
    let n = s.len() / k.max(1);
    let mut idx: Vec<(u64, u32)> = key.iter().enumerate().map(|(i, &g)| (g, i as u32)).collect();
    idx.par_sort_unstable();
    let starts: Vec<usize> = (0..idx.len()).filter(|&i| i == 0 || idx[i].0 != idx[i - 1].0).chain([idx.len()]).collect();
    starts
        .par_windows(2)
        .fold(
            || (vec![0.0; k * k], vec![0.0; k]),
            |(mut acc, mut sg), wd| {
                sg.iter_mut().for_each(|v| *v = 0.0);
                for &(_, i) in &idx[wd[0]..wd[1]] {
                    let r = i as usize % n;
                    let row = &s[r * k..(r + 1) * k];
                    sg.iter_mut().zip(row).for_each(|(a, b)| *a += b);
                }
                for a in 0..k {
                    for b in 0..k {
                        acc[a * k + b] += sg[a] * sg[b];
                    }
                }
                (acc, sg)
            },
        )
        .map(|(acc, _)| acc)
        .reduce(|| vec![0.0; k * k], |mut a, b| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
            a
        })
}

/// 依 key 分組的 meat：sum_g s_g s_g'。s 為 (n, k)
#[pyfunction]
fn group_outer<'py>(py: Python<'py>, s: PyReadonlyArray2<'py, f64>, key: PyReadonlyArray1<'py, i64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let k = s.shape()[1];
    let (s, key) = (s.as_slice()?, key.as_slice()?);
    let key: Vec<u64> = key.iter().map(|&g| g as u64).collect();
    let m = py.detach(|| group_outer_impl(s, k, &key));
    Ok(PyArray1::from_vec(py, m).reshape([k, k])?)
}

/// dyadic 的 meat：任兩局只要有一位共同玩家，分數的乘積就計入。sum_p S_p S_p' 對同一對手組合的對局
/// 重複計了兩次，減去以無序對手組合為群的一次。s 為 (n, k)，w、b 為兩位玩家的代碼
#[pyfunction]
fn dyad_meat<'py>(py: Python<'py>, s: PyReadonlyArray2<'py, f64>, w: PyReadonlyArray1<'py, i64>, b: PyReadonlyArray1<'py, i64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let k = s.shape()[1];
    let (s, w, b) = (s.as_slice()?, w.as_slice()?, b.as_slice()?);
    let m = py.detach(|| {
        let players: Vec<u64> = w.iter().chain(b).map(|&p| p as u64).collect();
        let np_ = players.iter().copied().max().unwrap_or(0) + 1;
        let pair: Vec<u64> = w.iter().zip(b).map(|(&x, &y)| x.min(y) as u64 * np_ + x.max(y) as u64).collect();
        let (mut m, q) = rayon::join(|| group_outer_impl(s, k, &players), || group_outer_impl(s, k, &pair));
        m.iter_mut().zip(&q).for_each(|(a, b)| *a -= b);
        m
    });
    Ok(PyArray1::from_vec(py, m).reshape([k, k])?)
}

#[pymodule]
fn likelihood_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vgh, m)?)?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(group_outer, m)?)?;
    m.add_function(wrap_pyfunction!(dyad_meat, m)?)?;
    Ok(())
}
