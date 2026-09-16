//! Dirichlet-multinomial GLM: log posterior and analytic gradient.
//!
//! Parameters are packed as `theta = [beta (P x K, row-major, beta[p*K + k]), log_conc (K)]`.
//! Each row of `beta` is a covariate's effect on the K introns; the softmax is invariant to
//! adding a constant to a row, so rows are kept sum-to-zero (K-1 effective degrees of
//! freedom, exactly like Stan's `simplex * scale` parameterisation in `dm_glm_multi_conc.stan`).
//! The gradient of a sum-to-zero row is itself sum-to-zero, so L-BFGS never leaves that
//! subspace.
//!
//! Per sample `n` with proportions `s = softmax(beta' x_n)`, `a = conc ⊙ s`, `A = Σ a`,
//! `Y = Σ y_n`:
//! `ℓ_n = lgamma(A) + Σ_k lgamma(a_k + y_k) - lgamma(A + Y) - Σ_k lgamma(a_k)`
//! plus a `Gamma(shape, rate)` prior on every `conc_k` (constants dropped, as Stan's `~`).
//! This is the objective `rstan::optimizing` maximises (MAP without Jacobian).

use crate::design::ClusterData;
use crate::special::{digamma, lgamma};

/// Walks `lgamma(a + v)` and `digamma(a + v)` over increasing integer shifts `v`, using the
/// recurrences `lgamma(x + 1) = lgamma(x) + ln x` and `digamma(x + 1) = digamma(x) + 1/x`
/// for small gaps (a log and a division per step instead of two special-function calls) and
/// direct evaluation for large jumps. Histograms are stored sorted, so most steps are small.
struct ShiftWalker {
    a: f64,
    v: u32,
    lg: f64,
    dg: f64,
}

impl ShiftWalker {
    const MAX_STEP: u32 = 24;

    #[inline]
    fn new(a: f64) -> Self {
        ShiftWalker {
            a,
            v: 0,
            lg: lgamma(a),
            dg: digamma(a),
        }
    }

    /// `(lgamma(a + v), digamma(a + v))` for `v >= self.v`.
    #[inline]
    fn at(&mut self, v: u32) -> (f64, f64) {
        debug_assert!(v >= self.v);
        let gap = v - self.v;
        if gap > Self::MAX_STEP {
            let x = self.a + v as f64;
            self.lg = lgamma(x);
            self.dg = digamma(x);
        } else {
            let mut x = self.a + self.v as f64;
            for _ in 0..gap {
                self.lg += x.ln();
                self.dg += 1.0 / x;
                x += 1.0;
            }
        }
        self.v = v;
        (self.lg, self.dg)
    }
}

/// The model for one cluster.
pub struct DmModel<'a> {
    pub data: &'a ClusterData,
    pub conc_shape: f64,
    pub conc_rate: f64,
}

impl<'a> DmModel<'a> {
    pub fn new(data: &'a ClusterData, conc_shape: f64, conc_rate: f64) -> Self {
        DmModel {
            data,
            conc_shape,
            conc_rate,
        }
    }

    #[inline]
    pub fn n_params(&self) -> usize {
        self.data.p * self.data.k + self.data.k
    }

    /// Log posterior (up to a constant) and its gradient. Returns `-inf` for infeasible points
    /// (a concentration underflowing to zero).
    pub fn log_posterior(&self, theta: &[f64], grad: &mut [f64]) -> f64 {
        let k = self.data.k;
        let p = self.data.p;
        debug_assert_eq!(theta.len(), p * k + k);
        let (beta, log_conc) = theta.split_at(p * k);
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        let (grad_beta, grad_lc) = grad.split_at_mut(p * k);

        let mut conc = vec![0.0; k];
        for j in 0..k {
            conc[j] = log_conc[j].exp();
            if !conc[j].is_finite() || conc[j] <= 0.0 {
                return f64::NEG_INFINITY;
            }
        }

        let mut eta = vec![0.0; k];
        let mut s = vec![0.0; k];
        let mut a = vec![0.0; k];
        let mut big_g = vec![0.0; k];
        let mut ll = 0.0;

        for cell in &self.data.cells {
            if cell.n_pos == 0.0 {
                continue;
            }
            // eta = beta' x
            let x = &cell.x;
            let mut emax = f64::NEG_INFINITY;
            for j in 0..k {
                let mut e = 0.0;
                for q in 0..p {
                    e += beta[q * k + j] * x[q];
                }
                eta[j] = e;
                if e > emax {
                    emax = e;
                }
            }
            let mut ssum = 0.0;
            for j in 0..k {
                s[j] = (eta[j] - emax).exp();
                ssum += s[j];
            }
            let mut big_a = 0.0;
            for j in 0..k {
                s[j] /= ssum;
                a[j] = conc[j] * s[j];
                big_a += a[j];
            }
            if big_a <= 0.0 || !big_a.is_finite() {
                return f64::NEG_INFINITY;
            }

            // total terms: n_pos * lgamma(A) - Σ M lgamma(A + V)
            ll += cell.n_pos * lgamma(big_a);
            let mut g_a = cell.n_pos * digamma(big_a);
            let mut walker = ShiftWalker::new(big_a);
            for &(v, m) in &cell.total_hist {
                let (lg, dg) = walker.at(v);
                ll -= m * lg;
                g_a -= m * dg;
            }

            // per-intron terms: Σ m (lgamma(a_k + v) - lgamma(a_k))
            for j in 0..k {
                let h = &cell.intron_hist[j];
                let mut gj = g_a;
                if !h.is_empty() {
                    let aj = a[j];
                    if aj <= 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    let mut walker = ShiftWalker::new(aj);
                    let (lga, dga) = (walker.lg, walker.dg);
                    for &(v, m) in h {
                        let (lg, dg) = walker.at(v);
                        ll += m * (lg - lga);
                        gj += m * (dg - dga);
                    }
                }
                big_g[j] = gj;
            }

            // chain rule through a = conc ⊙ softmax(eta)
            let mut t = 0.0;
            for j in 0..k {
                t += big_g[j] * a[j];
            }
            for j in 0..k {
                let deta = big_g[j] * a[j] - s[j] * t;
                for q in 0..p {
                    grad_beta[q * k + j] += deta * x[q];
                }
                grad_lc[j] += big_g[j] * a[j];
            }
        }

        // Gamma(shape, rate) prior on conc, on the log scale (no Jacobian, as rstan::optimizing)
        let sm1 = self.conc_shape - 1.0;
        for j in 0..k {
            ll += sm1 * log_conc[j] - self.conc_rate * conc[j];
            grad_lc[j] += sm1 - self.conc_rate * conc[j];
        }
        ll
    }

    /// Dense reference implementation (one term per sample) used in tests.
    #[cfg(test)]
    pub fn log_posterior_dense(
        &self,
        theta: &[f64],
        counts: &[u32],
        design: &crate::design::Design,
    ) -> f64 {
        let k = self.data.k;
        let p = self.data.p;
        let (beta, log_conc) = theta.split_at(p * k);
        let conc: Vec<f64> = log_conc.iter().map(|v| v.exp()).collect();
        let mut ll = 0.0;
        for n in 0..design.n {
            let x = design.row(n);
            let y = &counts[n * k..(n + 1) * k];
            let eta: Vec<f64> = (0..k)
                .map(|j| (0..p).map(|q| beta[q * k + j] * x[q]).sum())
                .collect();
            let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = eta.iter().map(|e| (e - m).exp()).collect();
            let z: f64 = ex.iter().sum();
            let a: Vec<f64> = (0..k).map(|j| conc[j] * ex[j] / z).collect();
            let big_a: f64 = a.iter().sum();
            let big_y: f64 = y.iter().map(|&v| v as f64).sum();
            ll += lgamma(big_a) - lgamma(big_a + big_y);
            for j in 0..k {
                ll += lgamma(a[j] + y[j] as f64) - lgamma(a[j]);
            }
        }
        for j in 0..k {
            ll += (self.conc_shape - 1.0) * log_conc[j] - self.conc_rate * conc[j];
        }
        ll
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::Design;

    fn toy() -> (Vec<u32>, usize, usize, Design) {
        // 8 samples, 3 introns, design [1, group, continuous covariate]
        let counts = vec![
            10, 5, 2, //
            8, 6, 0, //
            0, 0, 0, //
            30, 1, 9, //
            2, 20, 1, //
            0, 25, 3, //
            4, 4, 4, //
            1, 0, 12,
        ];
        let d = Design::from_columns(
            8,
            &[
                &[1.0; 8],
                &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                &[0.3, -1.2, 0.5, 0.5, 2.0, -0.7, 0.3, 0.1],
            ],
        );
        (counts, 8, 3, d)
    }

    #[test]
    fn collapsed_matches_dense() {
        let (counts, n, k, d) = toy();
        let cd = ClusterData::build(&counts, n, k, &d);
        let model = DmModel::new(&cd, 1.0001, 1e-4);
        let theta: Vec<f64> = vec![
            0.3, -0.1, -0.2, 0.5, -0.5, 0.0, 0.1, 0.2, -0.3, 2.0, 1.5, 2.5,
        ];
        let mut g = vec![0.0; theta.len()];
        let ll = model.log_posterior(&theta, &mut g);
        let ll_dense = model.log_posterior_dense(&theta, &counts, &d);
        assert!((ll - ll_dense).abs() < 1e-9, "{ll} vs {ll_dense}");
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let (counts, n, k, d) = toy();
        let cd = ClusterData::build(&counts, n, k, &d);
        let model = DmModel::new(&cd, 1.0001, 1e-4);
        let theta: Vec<f64> = vec![
            0.3, -0.1, -0.2, 0.5, -0.5, 0.0, 0.1, 0.2, -0.3, 2.0, 1.5, 2.5,
        ];
        let mut g = vec![0.0; theta.len()];
        let _ = model.log_posterior(&theta, &mut g);
        let mut scratch = vec![0.0; theta.len()];
        for i in 0..theta.len() {
            let h = 1e-6;
            let mut tp = theta.clone();
            tp[i] += h;
            let mut tm = theta.clone();
            tm[i] -= h;
            let fd = (model.log_posterior(&tp, &mut scratch)
                - model.log_posterior(&tm, &mut scratch))
                / (2.0 * h);
            assert!(
                (fd - g[i]).abs() < 1e-6 * (1.0 + fd.abs()),
                "param {i}: fd={fd} analytic={}",
                g[i]
            );
        }
        // beta rows' gradients are sum-to-zero
        for q in 0..cd.p {
            let s: f64 = g[q * k..(q + 1) * k].iter().sum();
            assert!(s.abs() < 1e-9);
        }
    }
}
