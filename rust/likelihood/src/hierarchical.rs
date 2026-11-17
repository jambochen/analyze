use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}

fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 { -(-x).exp().ln_1p() } else { x - x.exp().ln_1p() }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn hierarchical_logp<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
    x: PyReadonlyArray2<'py, f64>,
    z_width: PyReadonlyArray2<'py, f64>,
    a: PyReadonlyArray1<'py, i64>,
    b: PyReadonlyArray1<'py, i64>,
    ha: PyReadonlyArray1<'py, f64>,
    hb: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, i64>,
    mean_width: PyReadonlyArray1<'py, f64>,
    players: usize,
    slope: bool,
    blocks: usize,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let (n, p, q) = (x.shape()[0], x.shape()[1], z_width.shape()[1]);
    let (theta, x, zw) = (theta.as_slice()?, x.as_slice()?, z_width.as_slice()?);
    let (a, b, ha, hb, y, mw) = (a.as_slice()?, b.as_slice()?, ha.as_slice()?, hb.as_slice()?, y.as_slice()?, mean_width.as_slice()?);
    let k = if slope { 2 } else { 1 };
    let z_start = p + q + k;
    assert_eq!(theta.len(), z_start + players * k + usize::from(slope));
    assert_eq!(zw.len(), n * q);
    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);
    assert_eq!(ha.len(), n);
    assert_eq!(hb.len(), n);
    assert_eq!(y.len(), n);
    assert_eq!(mw.len(), q);
    assert!(blocks > 0 && players > 0);
    let (value, gradient) = py.detach(|| {
        let sd: Vec<f64> = theta[p + q..z_start].iter().map(|v| v.exp()).collect();
        let zs = &theta[z_start..z_start + players * k];
        let t = if slope { theta[theta.len() - 1] } else { 0.0 };
        let (pr, pc) = (sigmoid(t), sigmoid(-t));
        let (rho, corr_scale) = (pr - pc, 2.0 * (pr * pc).sqrt());
        let mut u = vec![0.0; players * k];
        for i in 0..players {
            u[i * k] = sd[0] * zs[i * k];
            if slope { u[i * k + 1] = sd[1] * (rho * zs[i * k] + corr_scale * zs[i * k + 1]); }
        }
        // 固定分塊及合併順序，使加總不隨 rayon 的排程改變。
        let block_size = n.div_ceil(blocks).max(1);
        let chunks: Vec<_> = (0..n.div_ceil(block_size)).into_par_iter().map(|chunk| {
            let mut val = 0.0;
            let mut g = vec![0.0; p + q + players * k];
            for r in chunk * block_size..((chunk + 1) * block_size).min(n) {
                let (ai, bi) = (a[r] as usize, b[r] as usize);
                let xr = &x[r * p..(r + 1) * p];
                let zr = &zw[r * q..(r + 1) * q];
                let mut eta = xr.iter().zip(&theta[..p]).map(|(v, c)| v * c).sum::<f64>() + u[ai * k] - u[bi * k];
                if slope { eta += u[ai * k + 1] * ha[r] - u[bi * k + 1] * hb[r]; }
                let width = zr.iter().zip(&theta[p..p + q]).map(|(v, c)| v * c).sum::<f64>().exp();
                let (upper, lower) = (eta + width, eta - width);
                let (lp, de, dw) = match y[r] {
                    0 => { let d = -sigmoid(upper); (log_sigmoid(-upper), d, width * d) },
                    2 => { let d = sigmoid(-lower); (log_sigmoid(lower), d, -width * d) },
                    1 => {
                        let (su, sl) = (sigmoid(-upper), sigmoid(lower));
                        (log_sigmoid(upper) + log_sigmoid(-lower) + (-(-2.0 * width).exp_m1()).ln(),
                         su - sl, width * (su + sl) + 2.0 * width / (2.0 * width).exp_m1())
                    },
                    _ => panic!("ordinal outcome outside 0..=2"),
                };
                val += lp;
                for j in 0..p { g[j] += de * xr[j]; }
                for j in 0..q { g[p + j] += dw * zr[j]; }
                g[p + q + ai * k] += de;
                g[p + q + bi * k] -= de;
                if slope {
                    g[p + q + ai * k + 1] += de * ha[r];
                    g[p + q + bi * k + 1] -= de * hb[r];
                }
            }
            (val, g)
        }).collect();
        let mut val = 0.0;
        let mut g = vec![0.0; p + q + players * k];
        for (v, gs) in chunks { val += v; for (out, d) in g.iter_mut().zip(gs) { *out += d; } }
        let mut grad = vec![0.0; theta.len()];
        let normal_const = -0.5 * (2.0 * std::f64::consts::PI).ln();
        for j in 0..p {
            val += normal_const - 2.5_f64.ln() - 0.5 * (theta[j] / 2.5).powi(2);
            grad[j] = g[j] - theta[j] / 6.25;
        }
        for j in 0..q {
            let d = theta[p + j] - mw[j];
            val += normal_const - 0.5 * d * d;
            grad[p + j] = g[p + j] - d;
        }
        for j in 0..k {
            // log(sd) 參數化必須包含 HalfNormal 的 Jacobian。
            val += 2.0_f64.ln() + normal_const - 0.5 * sd[j] * sd[j] + theta[p + q + j];
            grad[p + q + j] = 1.0 - sd[j] * sd[j];
        }
        for i in 0..players {
            let g0 = g[p + q + i * k];
            grad[p + q] += g0 * u[i * k];
            grad[z_start + i * k] = g0 * sd[0];
            if slope {
                let g1 = g[p + q + i * k + 1];
                grad[p + q + 1] += g1 * u[i * k + 1];
                grad[z_start + i * k] += g1 * sd[1] * rho;
                grad[z_start + i * k + 1] = g1 * sd[1] * corr_scale;
                grad[theta.len() - 1] += g1 * sd[1] * (2.0 * pr * pc * zs[i * k] - 0.5 * corr_scale * rho * zs[i * k + 1]);
            }
        }
        for (j, z) in zs.iter().enumerate() {
            val += normal_const - 0.5 * z * z;
            grad[z_start + j] -= z;
        }
        if slope {
            // Beta(2,2) 加上 logit Jacobian，對應原模型的 LKJ(2)。
            val += 6.0_f64.ln() + 2.0 * (log_sigmoid(t) + log_sigmoid(-t));
            grad[theta.len() - 1] -= 2.0 * rho;
        }
        (val, grad)
    });
    Ok((value, PyArray1::from_vec(py, gradient)))
}
