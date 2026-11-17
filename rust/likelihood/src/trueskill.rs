// 各序列只能依事件順序更新；獨立候選交給 Rayon，三類機率不使用套件的 expected_score。
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

fn cdf(x: f64) -> f64 { 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2) }

fn replay(a: &[i64], b: &[i64], y: &[i64], events: &[i64], weight: &[f64], cfg: &[f64], save: bool) -> (Vec<f64>, Vec<f64>) {
    let n = a.len();
    let players = (*a.iter().chain(b).max().unwrap() + 1) as usize;
    let mut state = vec![(0.0_f64, cfg[0]*cfg[0]); players];
    let eps = cfg[3];
    let mut loss = 0.0;
    let mut count = 0.0;
    let mut clipped = 0.0;
    let mut out = if save { vec![0.0; n*3] } else { Vec::new() };
    for &ev in events {
        let i = (ev.abs()-1) as usize;
        let (ai, bi) = (a[i] as usize, b[i] as usize);
        if ev > 0 {
            if weight[i] == 0.0 && !save { continue; }
            let (sa, sb) = (state[ai], state[bi]);
            let c = (sa.1+sb.1+2.0*cfg[1].powi(2)+2.0).sqrt();
            let d = sa.0-sb.0;
            // draw interval 一律放在左半側相減，避免兩個接近一的 CDF 相減。
            let mut p = [cdf((-eps-d)/c), cdf((eps-d.abs())/c)-cdf((-eps-d.abs())/c), cdf((d-eps)/c)];
            for v in &mut p {
                assert!(v.is_finite() && *v >= 0.0);
                if *v < 1e-300 { clipped += 1.0; *v = 1e-300; }
            }
            let total: f64 = p.iter().sum();
            for v in &mut p { *v /= total; }
            loss -= weight[i]*p[y[i] as usize].ln();
            count += weight[i];
            if save { out[i*3..i*3+3].copy_from_slice(&p); }
        } else {
            let (ma, va) = state[ai];
            let (mb, vb) = state[bi];
            let (va, vb) = (va+cfg[1]*cfg[1], vb+cfg[1]*cfg[1]);
            let c2 = va+vb+2.0;
            let c = c2.sqrt();
            let diff = ma-mb;
            // 一對一 truncated Normal moments；libm erfc 避免低精度近似在連續更新中累積。
            let pdf = |x: f64| (-0.5*x*x).exp()/(2.0*std::f64::consts::PI).sqrt();
            let (v,w) = if y[i] == 1 {
                let lo = (-eps-diff.abs())/c;
                let hi = (eps-diff.abs())/c;
                let z = cdf(hi)-cdf(lo);
                assert!(z > 0.0);
                let v = (pdf(lo)-pdf(hi))/z;
                let w = v*v+(hi*pdf(hi)-lo*pdf(lo))/z;
                (if diff < 0.0 { -v } else { v }, w)
            } else {
                let sign = if y[i] == 2 { 1.0 } else { -1.0 };
                let z = (sign*diff-eps)/c;
                let probability = cdf(z);
                assert!(probability > 0.0);
                let v = pdf(z)/probability;
                (sign*v, v*(v+z))
            };
            let sa = (ma+va/c*v, va*(1.0-va/c2*w));
            let sb = (mb-vb/c*v, vb*(1.0-vb/c2*w));
            assert!(sa.0.is_finite() && sb.0.is_finite() && sa.1.is_finite() && sb.1.is_finite());
            assert!(sa.1 > 0.0 && sb.1 > 0.0);
            state[ai] = sa;
            state[bi] = sb;
        }
    }
    (vec![loss/count, count, clipped], out)
}

#[pyfunction]
pub fn trueskill_replay<'py>(py: Python<'py>, a: PyReadonlyArray1<'py, i64>, b: PyReadonlyArray1<'py, i64>,
    y: PyReadonlyArray1<'py, i64>, events: PyReadonlyArray1<'py, i64>, weight: PyReadonlyArray1<'py, f64>,
    configs: PyReadonlyArray2<'py, f64>, save: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (a,b,y,events,weight) = (a.as_slice()?,b.as_slice()?,y.as_slice()?,events.as_slice()?,weight.as_slice()?);
    let configs = configs.as_slice()?;
    assert_eq!(configs.len()%4, 0);
    assert!(!save || configs.len() == 4);
    let results: Vec<_> = py.detach(|| configs.par_chunks(4).map(|c| replay(a,b,y,events,weight,c,save)).collect());
    let stats = results.iter().flat_map(|r| r.0.iter().copied()).collect();
    let predictions: Vec<_> = results.into_iter().flat_map(|r| r.1).collect();
    let rows = predictions.len()/3;
    Ok((PyArray1::from_vec(py, stats).reshape([configs.len()/4,3])?,
        PyArray1::from_vec(py, predictions).reshape([rows,3])?))
}
