//! Fitting one cluster: smart initialisation, null and full MAP fits, likelihood ratio test
//! and per-intron effect sizes. Mirrors `dirichlet_multinomial_anova_mc` in
//! `leafcutter/R/dm_glm_multi_conc.R` and `leaf_cutter_effect_sizes` in
//! `leafcutter/R/differential_splicing.R`.

use crate::design::{ClusterData, Design};
use crate::dm::DmModel;
use crate::lbfgs::{self, LbfgsParams, Status};
use crate::special::chisq_sf;
use serde::{Deserialize, Serialize};

/// Model / optimiser settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FitParams {
    /// Gamma shape of the prior on each concentration parameter.
    pub conc_shape: f64,
    /// Gamma rate of the prior on each concentration parameter.
    pub conc_rate: f64,
    /// Ridge term protecting the method-of-moments initialiser against colinear covariates.
    pub smart_init_regularizer: f64,
    /// Initial concentration for every intron.
    pub init_conc: f64,
    /// If the LRT p-value is below this, the null is refitted from the full solution.
    pub refit_null_below_p: f64,
    #[serde(skip)]
    pub lbfgs: LbfgsParams,
    /// Cap on L-BFGS iterations (overrides `lbfgs.max_iter`), mirrors R's per-cluster timeout.
    pub max_iter: usize,
    /// Extra starting points for the full model beyond the warm start from the null fit
    /// (0 = exactly R's behaviour). The Dirichlet-multinomial posterior can have several
    /// modes that differ in the concentrations; the best of all starts is kept.
    pub full_extra_starts: usize,
    /// Only run the extra starts when a full-fit concentration moved by more than this factor
    /// from the null fit (a sign of a mode switch). `1.0` disables the gate (always restart).
    pub restart_conc_ratio: f64,
}

impl Default for FitParams {
    fn default() -> Self {
        FitParams {
            conc_shape: 1.0001,
            conc_rate: 1e-4,
            smart_init_regularizer: 0.001,
            init_conc: 10.0,
            refit_null_below_p: 0.001,
            lbfgs: LbfgsParams::default(),
            max_iter: 2000,
            full_extra_starts: 1,
            restart_conc_ratio: 1.0,
        }
    }
}

/// A fitted Dirichlet-multinomial GLM.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fit {
    pub p: usize,
    pub k: usize,
    /// Row-major `P x K`, rows sum to zero.
    pub beta: Vec<f64>,
    pub conc: Vec<f64>,
    /// Log posterior at the optimum (the quantity `rstan::optimizing` reports as `value`).
    pub value: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub converged: bool,
}

impl Fit {
    pub fn beta_row(&self, q: usize) -> &[f64] {
        &self.beta[q * self.k..(q + 1) * self.k]
    }
}

/// Solve the small symmetric positive definite system `A x = B` (A is `p x p`, B is `p x k`)
/// by Gaussian elimination with partial pivoting. Returns row-major `p x k`.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, p: usize, k: usize) -> Vec<f64> {
    for col in 0..p {
        // pivot
        let mut piv = col;
        for r in col + 1..p {
            if a[r * p + col].abs() > a[piv * p + col].abs() {
                piv = r;
            }
        }
        if piv != col {
            for c in 0..p {
                a.swap(col * p + c, piv * p + c);
            }
            for c in 0..k {
                b.swap(col * k + c, piv * k + c);
            }
        }
        let d = a[col * p + col];
        if d == 0.0 {
            continue;
        }
        for r in col + 1..p {
            let f = a[r * p + col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..p {
                a[r * p + c] -= f * a[col * p + c];
            }
            for c in 0..k {
                b[r * k + c] -= f * b[col * k + c];
            }
        }
    }
    // back substitution
    let mut x = vec![0.0; p * k];
    for r in (0..p).rev() {
        for c in 0..k {
            let mut v = b[r * k + c];
            for j in r + 1..p {
                v -= a[r * p + j] * x[j * k + c];
            }
            let d = a[r * p + r];
            x[r * k + c] = if d != 0.0 { v / d } else { 0.0 };
        }
    }
    x
}

/// Method-of-moments initialisation of `beta` (the "smart" init of `dm_glm_multi_conc.R`):
/// regress `log((y+1) / rowSums(y+1))` on the design with a small ridge penalty and centre
/// each row. Returns row-major `P x K`. Stan's `beta_scale`/`beta_raw` split of the same
/// matrix is a pure reparameterisation, so the centred coefficients are the initial `beta`.
pub fn smart_init(
    counts: &[u32],
    n: usize,
    k: usize,
    design: &Design,
    regularizer: f64,
) -> Vec<f64> {
    let p = design.p;
    // X'X + reg I  and  X' y_norm
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p * k];
    let mut ynorm = vec![0.0; k];
    for i in 0..n {
        let y = &counts[i * k..(i + 1) * k];
        let tot: f64 = y.iter().map(|&v| v as f64 + 1.0).sum();
        for j in 0..k {
            ynorm[j] = ((y[j] as f64 + 1.0) / tot).ln();
        }
        let x = design.row(i);
        for q in 0..p {
            for r in 0..p {
                xtx[q * p + r] += x[q] * x[r];
            }
            for j in 0..k {
                xty[q * k + j] += x[q] * ynorm[j];
            }
        }
    }
    for q in 0..p {
        xtx[q * p + q] += regularizer;
    }
    let mut beta = solve(xtx, xty, p, k);
    center_rows(&mut beta, p, k);
    beta
}

/// The same initialisation computed from the collapsed representation: `X'X` and
/// `X' y_norm` are sums over samples that only depend on the design cell and the integer
/// counts, so they can be accumulated from the histograms (samples with zero total contribute
/// `-ln K` to every intron). Identical to [`smart_init`] up to floating point summation order.
pub fn smart_init_collapsed(data: &ClusterData, regularizer: f64) -> Vec<f64> {
    let (p, k) = (data.p, data.k);
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p * k];
    let lnk = (k as f64).ln();
    let mut s = vec![0.0; k];
    for cell in &data.cells {
        let n_c = cell.n_pos + cell.n_zero;
        if n_c == 0.0 {
            continue;
        }
        // Σ_n y_norm_nj = Σ_n ln(y_nj + 1) - Σ_n ln(tot_n + K)
        let mut log_tot = 0.0;
        for &(v, m) in &cell.total_hist {
            log_tot += m * ((v as f64) + k as f64).ln();
        }
        log_tot += cell.n_zero * lnk;
        for (sj, hist) in s.iter_mut().zip(&cell.intron_hist) {
            let mut acc = 0.0;
            for &(v, m) in hist {
                acc += m * ((v as f64) + 1.0).ln();
            }
            *sj = acc - log_tot;
        }
        let x = &cell.x;
        for q in 0..p {
            for r in 0..p {
                xtx[q * p + r] += n_c * x[q] * x[r];
            }
            for j in 0..k {
                xty[q * k + j] += x[q] * s[j];
            }
        }
    }
    for q in 0..p {
        xtx[q * p + q] += regularizer;
    }
    let mut beta = solve(xtx, xty, p, k);
    center_rows(&mut beta, p, k);
    beta
}

/// Subtract each row's mean (rows of `beta` are only identified up to a constant).
pub fn center_rows(beta: &mut [f64], p: usize, k: usize) {
    for q in 0..p {
        let row = &mut beta[q * k..(q + 1) * k];
        let m = row.iter().sum::<f64>() / k as f64;
        for v in row.iter_mut() {
            *v -= m;
        }
    }
}

/// Maximise the log posterior from the given starting point.
pub fn fit_model(data: &ClusterData, beta0: &[f64], conc0: &[f64], params: &FitParams) -> Fit {
    let (p, k) = (data.p, data.k);
    assert_eq!(beta0.len(), p * k);
    assert_eq!(conc0.len(), k);
    let model = DmModel::new(data, params.conc_shape, params.conc_rate);
    let mut theta = Vec::with_capacity(model.n_params());
    theta.extend_from_slice(beta0);
    theta.extend(conc0.iter().map(|c| c.max(1e-300).ln()));
    let mut lb = params.lbfgs.clone();
    lb.max_iter = params.max_iter;
    let mut grad_buf = vec![0.0; theta.len()];
    let res = lbfgs::minimize(
        |x, g| {
            let ll = model.log_posterior(x, &mut grad_buf);
            for (gi, gb) in g.iter_mut().zip(&grad_buf) {
                *gi = -gb;
            }
            -ll
        },
        &mut theta,
        &lb,
    );
    let (beta, log_conc) = theta.split_at(p * k);
    let mut beta = beta.to_vec();
    center_rows(&mut beta, p, k);
    Fit {
        p,
        k,
        beta,
        conc: log_conc.iter().map(|v| v.exp()).collect(),
        value: -res.f,
        iterations: res.iterations,
        evaluations: res.evaluations,
        converged: matches!(res.status, Status::Converged | Status::LineSearchFailed),
    }
}

/// Output of the likelihood ratio test for one cluster.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LrtResult {
    pub loglr: f64,
    pub df: usize,
    pub p: f64,
    pub fit_null: Fit,
    pub fit_full: Fit,
    pub refit_null: bool,
}

/// Fit the null and full models and compute the likelihood ratio test.
///
/// * `counts` – sample-major `N x K` counts (already filtered).
/// * `x_full` – full design, `N x P_full`.
/// * `null_cols` – which columns of `x_full` make up the null design (e.g. everything but
///   the group column).
/// * `cached_null` – an existing null fit for the same samples/design (e.g. from a cohort
///   cache); it must have been fitted on `x_full.select_columns(null_cols)`.
pub fn lrt(
    counts: &[u32],
    n: usize,
    k: usize,
    x_full: &Design,
    null_cols: &[usize],
    params: &FitParams,
    cached_null: Option<Fit>,
) -> LrtResult {
    let data_full = ClusterData::build(counts, n, k, x_full);
    let data_null = data_full.select_columns(null_cols);
    let p_null = null_cols.len();

    let mut fit_null = match cached_null {
        Some(f) if f.p == p_null && f.k == k => f,
        _ => {
            let beta0 = smart_init_collapsed(&data_null, params.smart_init_regularizer);
            fit_model(&data_null, &beta0, &vec![params.init_conc; k], params)
        }
    };

    // warm start the full model from the null solution: null rows copied, other rows zero
    let p_full = x_full.p;
    let mut beta0 = vec![0.0; p_full * k];
    for (i, &c) in null_cols.iter().enumerate() {
        beta0[c * k..(c + 1) * k].copy_from_slice(fit_null.beta_row(i));
    }
    let mut fit_full = fit_model(&data_full, &beta0, &fit_null.conc, params);
    if params.full_extra_starts > 0 {
        let moved = fit_full
            .conc
            .iter()
            .zip(&fit_null.conc)
            .map(|(a, b)| (a / b).ln().abs())
            .fold(0.0, f64::max);
        if params.restart_conc_ratio <= 1.0 || moved > params.restart_conc_ratio.ln() {
            // start 1: method-of-moments init of the full design, fresh concentrations
            let beta_mm = smart_init_collapsed(&data_full, params.smart_init_regularizer);
            let alt = fit_model(&data_full, &beta_mm, &vec![params.init_conc; k], params);
            if alt.value > fit_full.value {
                fit_full = alt;
            }
            if params.full_extra_starts > 1 {
                // start 2: warm start with high concentrations (the near-multinomial mode)
                let alt = fit_model(&data_full, &beta0, &vec![100.0; k], params);
                if alt.value > fit_full.value {
                    fit_full = alt;
                }
            }
        }
    }

    let df = (p_full - p_null) * (k - 1);
    let mut loglr = fit_full.value - fit_null.value;
    let mut refit = false;
    if chisq_sf(2.0 * loglr, df as f64) < params.refit_null_below_p {
        let mut beta_n = vec![0.0; p_null * k];
        for (i, &c) in null_cols.iter().enumerate() {
            beta_n[i * k..(i + 1) * k].copy_from_slice(fit_full.beta_row(c));
        }
        let refit_null = fit_model(&data_null, &beta_n, &fit_full.conc, params);
        if refit_null.value > fit_null.value {
            refit = true;
            fit_null = refit_null;
            loglr = fit_full.value - fit_null.value;
        }
    }
    let p = chisq_sf(2.0 * loglr, df as f64);
    LrtResult {
        loglr,
        df,
        p,
        fit_null,
        fit_full,
        refit_null: refit,
    }
}

/// Per-intron effect sizes (`leaf_cutter_effect_sizes`): the log effect size is the group
/// row of `beta`; baseline / perturbed PSI are the concentration-weighted softmax of the
/// intercept row without / with the group effect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectSize {
    pub logef: f64,
    pub baseline: f64,
    pub perturbed: f64,
    pub deltapsi: f64,
}

pub fn effect_sizes(fit: &Fit, intercept_row: usize, group_row: usize) -> Vec<EffectSize> {
    let k = fit.k;
    let b0 = fit.beta_row(intercept_row);
    let b1 = fit.beta_row(group_row);
    let to_psi = |g: &[f64]| -> Vec<f64> {
        let m = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut w: Vec<f64> = (0..k).map(|j| (g[j] - m).exp() * fit.conc[j]).collect();
        let s: f64 = w.iter().sum();
        for v in w.iter_mut() {
            *v /= s;
        }
        w
    };
    let base = to_psi(b0);
    let pert = to_psi(&(0..k).map(|j| b0[j] + b1[j]).collect::<Vec<_>>());
    (0..k)
        .map(|j| EffectSize {
            logef: b1[j],
            baseline: base[j],
            perturbed: pert[j],
            deltapsi: pert[j] - base[j],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_small_system() {
        // A = [[4,1],[1,3]], B = [[1,2],[3,4]] -> x = A^-1 B
        let x = solve(vec![4.0, 1.0, 1.0, 3.0], vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        // A x: row0 = 4*x0 + x1
        let r00 = 4.0 * x[0] + x[2];
        let r01 = 4.0 * x[1] + x[3];
        let r10 = x[0] + 3.0 * x[2];
        let r11 = x[1] + 3.0 * x[3];
        assert!((r00 - 1.0).abs() < 1e-12 && (r01 - 2.0).abs() < 1e-12);
        assert!((r10 - 3.0).abs() < 1e-12 && (r11 - 4.0).abs() < 1e-12);
    }

    #[test]
    fn collapsed_smart_init_matches_dense() {
        let n = 12;
        let k = 3;
        let counts: Vec<u32> = vec![
            5, 3, 0, 0, 0, 0, 9, 1, 2, 4, 4, 4, 0, 7, 1, 30, 0, 2, 1, 1, 1, 0, 0, 0, 8, 2, 2, 5, 3,
            0, 5, 3, 0, 6, 6, 1,
        ];
        let group: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let cov: Vec<f64> = (0..n).map(|i| (i % 3) as f64 - 1.0).collect();
        let x = Design::from_columns(n, &[&vec![1.0; n], &group, &cov]);
        let dense = smart_init(&counts, n, k, &x, 0.001);
        let data = ClusterData::build(&counts, n, k, &x);
        let coll = smart_init_collapsed(&data, 0.001);
        for (a, b) in dense.iter().zip(&coll) {
            assert!((a - b).abs() < 1e-10, "{dense:?} vs {coll:?}");
        }
    }

    #[test]
    fn lrt_no_signal_has_small_loglr() {
        // same proportions in both groups -> loglr ~ 0, p ~ 1
        let n = 40;
        let k = 3;
        let mut counts = Vec::new();
        for i in 0..n {
            let base = [20u32, 10, 5];
            for b in base.iter().take(k) {
                counts.push(b + (i % 3) as u32);
            }
        }
        let group: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
        let x = Design::from_columns(n, &[&vec![1.0; n], &group]);
        let res = lrt(&counts, n, k, &x, &[0], &FitParams::default(), None);
        assert_eq!(res.df, 2);
        assert!(res.loglr.abs() < 1e-3, "loglr={}", res.loglr);
        assert!(res.p > 0.99);
        assert!(res.fit_null.converged && res.fit_full.converged);
    }

    #[test]
    fn lrt_detects_strong_signal() {
        let n = 40;
        let k = 3;
        let mut counts = Vec::new();
        for i in 0..n {
            let base = if i < n / 2 {
                [30u32, 5, 5]
            } else {
                [5u32, 30, 5]
            };
            for b in base.iter().take(k) {
                counts.push(b + (i % 2) as u32);
            }
        }
        let group: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
        let x = Design::from_columns(n, &[&vec![1.0; n], &group]);
        let res = lrt(&counts, n, k, &x, &[0], &FitParams::default(), None);
        assert!(res.loglr > 50.0, "loglr={}", res.loglr);
        assert!(res.p < 1e-20);
        let es = effect_sizes(&res.fit_full, 0, 1);
        assert!(es[0].deltapsi < -0.4 && es[1].deltapsi > 0.4, "{es:?}");
        let s: f64 = es.iter().map(|e| e.deltapsi).sum();
        assert!(s.abs() < 1e-9);
    }
}
