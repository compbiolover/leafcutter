//! Fitting one cluster: smart initialisation, null and full MAP fits, likelihood ratio test
//! and per-intron effect sizes. Mirrors `dirichlet_multinomial_anova_mc` in
//! `leafcutter/R/dm_glm_multi_conc.R` and `leaf_cutter_effect_sizes` in
//! `leafcutter/R/differential_splicing.R`.

use crate::design::{ClusterData, Design};
use crate::dm::{ConcParam, DmModel, MultinomialModel};
use crate::lbfgs::{self, LbfgsParams, Status};
use crate::special::chisq_sf;
use serde::{Deserialize, Serialize};

/// How `beta` is initialised before the L-BFGS fit (leafcutter-ds `--init`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InitStrategy {
    /// Bayesian ridge regression per junction on `log((y+1)/rowsum(y+1))`
    /// (scikit-learn's `BayesianRidge`, leafcutter-ds default).
    Brr,
    /// Ridge regression with a 0.001 penalty (the R package's "smart" init).
    Rr,
    /// Multinomial logistic regression fitted from zero.
    Mult,
    /// All zeros.
    Zero,
}

impl std::str::FromStr for InitStrategy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "brr" => Ok(InitStrategy::Brr),
            "rr" => Ok(InitStrategy::Rr),
            "mult" => Ok(InitStrategy::Mult),
            "0" | "zero" => Ok(InitStrategy::Zero),
            _ => Err(format!("unknown init strategy '{s}' (brr, rr, mult, 0)")),
        }
    }
}

/// Which package's fitting procedure to reproduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Procedure {
    /// leafcutter-ds (Python/Pyro): null fit from the init; full fit from the null solution
    /// and from a fresh init, keep the better; every fit starts with `conc = init_conc`;
    /// refit the null from the full solution if p < 0.001.
    Python,
    /// The R/Stan package: full fit warm-started from the null (including the
    /// concentrations); optional extra starts; refit rule as above.
    R,
}

/// Model / optimiser settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FitParams {
    /// Gamma shape of the prior on each concentration parameter.
    pub conc_shape: f64,
    /// Gamma rate of the prior on each concentration parameter.
    pub conc_rate: f64,
    pub init: InitStrategy,
    pub procedure: Procedure,
    /// Upper bound of the concentrations (`conc = conc_max * sigmoid(u)`); `inf` selects the
    /// unbounded log parameterisation of the R model.
    pub conc_max: f64,
    /// Pseudocount added to the Dirichlet parameters.
    pub eps: f64,
    /// Ridge term of the `rr` initialiser.
    pub smart_init_regularizer: f64,
    /// Initial concentration for every intron.
    pub init_conc: f64,
    /// If the LRT p-value is below this, the null is refitted from the full solution.
    pub refit_null_below_p: f64,
    #[serde(skip)]
    pub lbfgs: LbfgsParams,
    /// Cap on L-BFGS iterations (overrides `lbfgs.max_iter`).
    pub max_iter: usize,
    /// R procedure only: extra starting points for the full model beyond the warm start.
    pub full_extra_starts: usize,
    /// R procedure only: only run the extra starts when a concentration moved by more than
    /// this factor between the null and full fits (`1.0` = always).
    pub restart_conc_ratio: f64,
}

impl Default for FitParams {
    fn default() -> Self {
        FitParams::python()
    }
}

impl FitParams {
    /// leafcutter-ds defaults (`--init brr`, torch L-BFGS settings, `conc_max = 3000`,
    /// `eps = 1e-8`).
    pub fn python() -> Self {
        FitParams {
            conc_shape: 1.0001,
            conc_rate: 1e-4,
            init: InitStrategy::Brr,
            procedure: Procedure::Python,
            conc_max: 3000.0,
            eps: 1e-8,
            smart_init_regularizer: 0.001,
            init_conc: 10.0,
            refit_null_below_p: 0.001,
            lbfgs: LbfgsParams::torch(),
            max_iter: 500,
            full_extra_starts: 1,
            restart_conc_ratio: 1.0,
        }
    }

    /// The R package's procedure with the tighter stopping rules of this crate.
    pub fn r_like() -> Self {
        FitParams {
            init: InitStrategy::Rr,
            procedure: Procedure::R,
            conc_max: f64::INFINITY,
            eps: 0.0,
            lbfgs: LbfgsParams::default(),
            max_iter: 2000,
            ..FitParams::python()
        }
    }

    /// Exactly `rstan::optimizing`'s stopping rules and a single start.
    pub fn r_exact() -> Self {
        FitParams {
            lbfgs: LbfgsParams::stan(),
            full_extra_starts: 0,
            ..FitParams::r_like()
        }
    }

    pub fn conc_param(&self) -> ConcParam {
        if self.conc_max.is_finite() {
            ConcParam::Sigmoid { max: self.conc_max }
        } else {
            ConcParam::Log
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
    let cp = params.conc_param();
    let model = DmModel::with_options(data, params.conc_shape, params.conc_rate, params.eps, cp);
    let mut theta = Vec::with_capacity(model.n_params());
    theta.extend_from_slice(beta0);
    theta.extend(conc0.iter().map(|&c| cp.unconstrained(c)));
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
    let (beta, conc_u) = theta.split_at(p * k);
    let mut beta = beta.to_vec();
    center_rows(&mut beta, p, k);
    Fit {
        p,
        k,
        beta,
        conc: conc_u.iter().map(|&u| cp.conc(u)).collect(),
        value: -res.f,
        iterations: res.iterations,
        evaluations: res.evaluations,
        converged: matches!(res.status, Status::Converged | Status::LineSearchFailed),
    }
}

/// Fit the multinomial logistic regression (no concentrations) from `beta0`; used by the
/// `mult` initialisation.
pub fn fit_multinomial(data: &ClusterData, beta0: &[f64], params: &FitParams) -> Vec<f64> {
    let model = MultinomialModel { data };
    let mut theta = beta0.to_vec();
    let mut lb = params.lbfgs.clone();
    lb.max_iter = params.max_iter;
    let mut grad_buf = vec![0.0; theta.len()];
    let _ = lbfgs::minimize(
        |x, g| {
            let ll = model.log_likelihood(x, &mut grad_buf);
            for (gi, gb) in g.iter_mut().zip(&grad_buf) {
                *gi = -gb;
            }
            -ll
        },
        &mut theta,
        &lb,
    );
    center_rows(&mut theta, data.p, data.k);
    theta
}

/// Symmetric eigendecomposition by cyclic Jacobi rotations (for the tiny `P x P` systems of
/// the initialisers). Returns `(eigenvalues, eigenvectors as columns, row-major p x p)`.
fn jacobi_eigen(a_in: &[f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = vec![0.0; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..p {
            for j in 0..p {
                if i != j {
                    off += a[i * p + j] * a[i * p + j];
                }
            }
        }
        if off < 1e-30 {
            break;
        }
        for pi in 0..p {
            for qi in pi + 1..p {
                let apq = a[pi * p + qi];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[pi * p + pi];
                let aqq = a[qi * p + qi];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for kk in 0..p {
                    let akp = a[kk * p + pi];
                    let akq = a[kk * p + qi];
                    a[kk * p + pi] = c * akp - s * akq;
                    a[kk * p + qi] = s * akp + c * akq;
                }
                for kk in 0..p {
                    let apk = a[pi * p + kk];
                    let aqk = a[qi * p + kk];
                    a[pi * p + kk] = c * apk - s * aqk;
                    a[qi * p + kk] = s * apk + c * aqk;
                }
                for kk in 0..p {
                    let vkp = v[kk * p + pi];
                    let vkq = v[kk * p + qi];
                    v[kk * p + pi] = c * vkp - s * vkq;
                    v[kk * p + qi] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..p).map(|i| a[i * p + i]).collect(), v)
}

/// Sufficient statistics of one Bayesian ridge regression: `n`, column means of `X`, mean of
/// `y`, and the raw (uncentred) `X'X`, `X'y`, `y'y`.
struct RidgeStats {
    n: f64,
    xm: Vec<f64>,
    ym: f64,
    xtx: Vec<f64>,
    xty: Vec<f64>,
    yty: f64,
}

/// scikit-learn's `BayesianRidge` (defaults: `fit_intercept=True`, `max_iter=300`,
/// `tol=1e-3`, `alpha_1 = alpha_2 = lambda_1 = lambda_2 = 1e-6`) solved from sufficient
/// statistics, so the 300-iteration loop costs `O(P^2)` per iteration whatever `N` is.
/// Returns the coefficients (the intercept is discarded, as leafcutter-ds discards
/// `reg.intercept_`).
fn bayesian_ridge_stats(st: &RidgeStats, p: usize) -> Vec<f64> {
    let nf = st.n;
    // centred moments: Xc'Xc = X'X - n x̄x̄', Xc'yc = X'y - n x̄ ȳ, yc'yc = y'y - n ȳ²
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for q in 0..p {
        xty[q] = st.xty[q] - nf * st.xm[q] * st.ym;
        for s in 0..p {
            xtx[q * p + s] = st.xtx[q * p + s] - nf * st.xm[q] * st.xm[s];
        }
    }
    let yty = (st.yty - nf * st.ym * st.ym).max(0.0);
    let (eig, vecs) = jacobi_eigen(&xtx, p);
    let eig: Vec<f64> = eig.iter().map(|e| e.max(0.0)).collect();
    let proj: Vec<f64> = (0..p)
        .map(|kk| (0..p).map(|q| vecs[q * p + kk] * xty[q]).sum())
        .collect();
    let coef_for = |alpha: f64, lambda: f64| -> Vec<f64> {
        let mut c = vec![0.0; p];
        for kk in 0..p {
            let w = proj[kk] / (eig[kk] + lambda / alpha);
            for q in 0..p {
                c[q] += vecs[q * p + kk] * w;
            }
        }
        c
    };
    let var_y = yty / nf;
    let mut alpha = 1.0 / (var_y + f64::EPSILON);
    let mut lambda = 1.0;
    let (a1, a2, l1, l2) = (1e-6, 1e-6, 1e-6, 1e-6);
    let mut coef_old: Vec<f64> = vec![0.0; p];
    for it in 0..300 {
        let coef = coef_for(alpha, lambda);
        // ||yc - Xc coef||² = yc'yc - 2 coef'Xc'yc + coef'Xc'Xc coef
        let mut quad = 0.0;
        let mut lin = 0.0;
        for q in 0..p {
            lin += coef[q] * xty[q];
            for s in 0..p {
                quad += coef[q] * xtx[q * p + s] * coef[s];
            }
        }
        let rmse = (yty - 2.0 * lin + quad).max(0.0);
        let gamma: f64 = (0..p)
            .map(|kk| alpha * eig[kk] / (lambda + alpha * eig[kk]))
            .sum();
        lambda = (gamma + 2.0 * l1) / (coef.iter().map(|c| c * c).sum::<f64>() + 2.0 * l2);
        alpha = (nf - gamma + 2.0 * a1) / (rmse + 2.0 * a2);
        if it != 0
            && coef
                .iter()
                .zip(&coef_old)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                < 1e-3
        {
            break;
        }
        coef_old = coef;
    }
    coef_for(alpha, lambda)
}

/// scikit-learn's `BayesianRidge` on a dense design `x` (`n x p`, sample-major) and response
/// `y`; see [`bayesian_ridge_stats`].
pub fn bayesian_ridge(x: &[f64], n: usize, p: usize, y: &[f64]) -> Vec<f64> {
    let mut st = RidgeStats {
        n: n as f64,
        xm: vec![0.0; p],
        ym: 0.0,
        xtx: vec![0.0; p * p],
        xty: vec![0.0; p],
        yty: 0.0,
    };
    for i in 0..n {
        let r = &x[i * p..(i + 1) * p];
        st.ym += y[i];
        st.yty += y[i] * y[i];
        for q in 0..p {
            st.xm[q] += r[q];
            st.xty[q] += r[q] * y[i];
            for s in 0..p {
                st.xtx[q * p + s] += r[q] * r[s];
            }
        }
    }
    st.ym /= n as f64;
    for m in st.xm.iter_mut() {
        *m /= n as f64;
    }
    bayesian_ridge_stats(&st, p)
}

/// leafcutter-ds's `brr` initialisation: Bayesian ridge regression of every junction's
/// `log((y+1)/rowsum(y+1))` on the design, rows centred. One pass over the samples collects
/// the sufficient statistics of all `K` regressions.
pub fn brr_init(counts: &[u32], n: usize, k: usize, design: &Design) -> Vec<f64> {
    let p = design.p;
    let nf = n as f64;
    let mut xm = vec![0.0; p];
    let mut xtx = vec![0.0; p * p];
    let mut ym = vec![0.0; k];
    let mut yty = vec![0.0; k];
    let mut xty = vec![0.0; p * k];
    let mut ynorm = vec![0.0; k];
    for i in 0..n {
        let y = &counts[i * k..(i + 1) * k];
        let tot: f64 = y.iter().map(|&v| v as f64 + 1.0).sum();
        let ltot = tot.ln();
        for j in 0..k {
            ynorm[j] = (y[j] as f64 + 1.0).ln() - ltot;
            ym[j] += ynorm[j];
            yty[j] += ynorm[j] * ynorm[j];
        }
        let x = design.row(i);
        for q in 0..p {
            xm[q] += x[q];
            for s in 0..p {
                xtx[q * p + s] += x[q] * x[s];
            }
            for j in 0..k {
                xty[q * k + j] += x[q] * ynorm[j];
            }
        }
    }
    for m in xm.iter_mut() {
        *m /= nf;
    }
    let mut beta = vec![0.0; p * k];
    for j in 0..k {
        let st = RidgeStats {
            n: nf,
            xm: xm.clone(),
            ym: ym[j] / nf,
            xtx: xtx.clone(),
            xty: (0..p).map(|q| xty[q * k + j]).collect(),
            yty: yty[j],
        };
        let coef = bayesian_ridge_stats(&st, p);
        for q in 0..p {
            beta[q * k + j] = coef[q];
        }
    }
    center_rows(&mut beta, p, k);
    beta
}

/// Initial `beta` for a design according to the chosen strategy.
pub fn init_beta(
    strategy: InitStrategy,
    counts: &[u32],
    n: usize,
    k: usize,
    design: &Design,
    data: &ClusterData,
    params: &FitParams,
) -> Vec<f64> {
    match strategy {
        InitStrategy::Brr => brr_init(counts, n, k, design),
        InitStrategy::Rr => smart_init_collapsed(data, params.smart_init_regularizer),
        InitStrategy::Zero => vec![0.0; design.p * k],
        InitStrategy::Mult => fit_multinomial(data, &vec![0.0; design.p * k], params),
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
    /// Whether the full fit from the fresh initialisation beat the one from the null solution.
    pub smart_init_improved: bool,
}

/// Fit the null and full models and compute the likelihood ratio test.
///
/// * `counts` – sample-major `N x K` counts (already filtered).
/// * `x_full` – full design, `N x P_full`.
/// * `null_cols` – which columns of `x_full` make up the null design (leafcutter-ds and this
///   crate put the intercept and confounders first, so this is `0..P_null`).
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
    let x_null = x_full.select_columns(null_cols);
    let data_full = ClusterData::build(counts, n, k, x_full);
    let data_null = data_full.select_columns(null_cols);
    let p_null = null_cols.len();
    let p_full = x_full.p;
    let init_conc = vec![params.init_conc; k];

    let mut fit_null = match cached_null {
        Some(f) if f.p == p_null && f.k == k => f,
        _ => {
            let beta0 = init_beta(params.init, counts, n, k, &x_null, &data_null, params);
            fit_model(&data_null, &beta0, &init_conc, params)
        }
    };

    // full model from the null solution: null rows copied, group rows zero
    let mut beta_from_null = vec![0.0; p_full * k];
    for (i, &c) in null_cols.iter().enumerate() {
        beta_from_null[c * k..(c + 1) * k].copy_from_slice(fit_null.beta_row(i));
    }
    let mut smart_init_improved = false;
    let mut fit_full = match params.procedure {
        Procedure::Python => {
            let a = fit_model(&data_full, &beta_from_null, &init_conc, params);
            let beta_smart = init_beta(params.init, counts, n, k, x_full, &data_full, params);
            let b = fit_model(&data_full, &beta_smart, &init_conc, params);
            if b.value > a.value {
                smart_init_improved = true;
                b
            } else {
                a
            }
        }
        Procedure::R => {
            let mut best = fit_model(&data_full, &beta_from_null, &fit_null.conc, params);
            if params.full_extra_starts > 0 {
                let moved = best
                    .conc
                    .iter()
                    .zip(&fit_null.conc)
                    .map(|(a, b)| (a / b).ln().abs())
                    .fold(0.0, f64::max);
                if params.restart_conc_ratio <= 1.0 || moved > params.restart_conc_ratio.ln() {
                    let beta_mm = smart_init_collapsed(&data_full, params.smart_init_regularizer);
                    let alt = fit_model(&data_full, &beta_mm, &init_conc, params);
                    if alt.value > best.value {
                        smart_init_improved = true;
                        best = alt;
                    }
                    if params.full_extra_starts > 1 {
                        let alt = fit_model(&data_full, &beta_from_null, &vec![100.0; k], params);
                        if alt.value > best.value {
                            best = alt;
                        }
                    }
                }
            }
            best
        }
    };
    let _ = &mut fit_full;

    let df = (p_full - p_null) * (k - 1);
    let mut loglr = fit_full.value - fit_null.value;
    let mut refit = false;
    if chisq_sf(2.0 * loglr, df as f64) < params.refit_null_below_p {
        let mut beta_n = vec![0.0; p_null * k];
        for (i, &c) in null_cols.iter().enumerate() {
            beta_n[i * k..(i + 1) * k].copy_from_slice(fit_full.beta_row(c));
        }
        let conc0 = match params.procedure {
            Procedure::Python => init_conc.clone(),
            Procedure::R => fit_full.conc.clone(),
        };
        let refit_null = fit_model(&data_null, &beta_n, &conc0, params);
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
        smart_init_improved,
    }
}

/// Per-intron effect sizes for one non-baseline design column (`leaf_cutter_effect_sizes` /
/// leafcutter-ds `task`): the log effect size is that row of `beta`; `psi` is the
/// concentration-weighted softmax of the intercept row plus that row; `deltapsi` is the
/// difference to the baseline PSI (intercept row only).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectSize {
    pub logef: f64,
    pub psi: f64,
    pub deltapsi: f64,
}

/// Concentration-weighted softmax of a logit vector: `normalize(softmax(g) * conc)`.
pub fn psi_from_logits(g: &[f64], conc: &[f64]) -> Vec<f64> {
    let m = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut w: Vec<f64> = g.iter().zip(conc).map(|(v, c)| (v - m).exp() * c).collect();
    let s: f64 = w.iter().sum();
    for v in w.iter_mut() {
        *v /= s;
    }
    w
}

/// Baseline PSI (intercept row) and, per group row, the effect sizes of every intron.
pub fn effect_sizes(
    fit: &Fit,
    intercept_row: usize,
    group_rows: &[usize],
) -> (Vec<f64>, Vec<Vec<EffectSize>>) {
    let k = fit.k;
    let b0 = fit.beta_row(intercept_row);
    let base = psi_from_logits(b0, &fit.conc);
    let per_group = group_rows
        .iter()
        .map(|&r| {
            let b1 = fit.beta_row(r);
            let pert = psi_from_logits(
                &(0..k).map(|j| b0[j] + b1[j]).collect::<Vec<_>>(),
                &fit.conc,
            );
            (0..k)
                .map(|j| EffectSize {
                    logef: b1[j],
                    psi: pert[j],
                    deltapsi: pert[j] - base[j],
                })
                .collect()
        })
        .collect();
    (base, per_group)
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
    fn bayesian_ridge_matches_sklearn() {
        // sklearn.linear_model.BayesianRidge().fit(X, y).coef_ with
        // X = [[1,0,0.3],[1,0,-1.2],[1,0,0.5],[1,0,0.5],[1,1,2.0],[1,1,-0.7],[1,1,0.3],[1,1,0.1]],
        // y = [0.2,-0.4,0.1,0.9,1.3,0.5,0.8,1.1]
        let x = vec![
            1.0, 0.0, 0.3, 1.0, 0.0, -1.2, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 1.0, 2.0, 1.0, 1.0,
            -0.7, 1.0, 1.0, 0.3, 1.0, 1.0, 0.1,
        ];
        let y = vec![0.2, -0.4, 0.1, 0.9, 1.3, 0.5, 0.8, 1.1];
        let c = bayesian_ridge(&x, 8, 3, &y);
        assert!(
            c[0].abs() < 1e-12,
            "intercept column must get a zero coefficient: {c:?}"
        );
        let expect = [0.0, 0.48554743, 0.33689104];
        for i in 1..3 {
            assert!((c[i] - expect[i]).abs() < 1e-6, "{c:?} vs {expect:?}");
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
        for params in [FitParams::python(), FitParams::r_like()] {
            let res = lrt(&counts, n, k, &x, &[0], &params, None);
            assert_eq!(res.df, 2);
            assert!(res.loglr.abs() < 1e-3, "loglr={}", res.loglr);
            assert!(res.p > 0.99);
            assert!(res.fit_null.converged && res.fit_full.converged);
        }
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
        for params in [FitParams::python(), FitParams::r_like()] {
            let res = lrt(&counts, n, k, &x, &[0], &params, None);
            assert!(res.loglr > 50.0, "loglr={}", res.loglr);
            assert!(res.p < 1e-20);
            let (_, es) = effect_sizes(&res.fit_full, 0, &[1]);
            assert!(
                es[0][0].deltapsi < -0.4 && es[0][1].deltapsi > 0.4,
                "{es:?}"
            );
            let s: f64 = es[0].iter().map(|e| e.deltapsi).sum();
            assert!(s.abs() < 1e-9);
        }
    }
}
