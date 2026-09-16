//! A compact limited-memory BFGS minimiser with a strong-Wolfe line search
//! (Nocedal & Wright, algorithms 3.5/3.6 with cubic interpolation).
//!
//! Convergence criteria mirror the defaults of Stan's L-BFGS (which `rstan::optimizing`
//! uses): absolute / relative change in objective, gradient norm, relative gradient
//! (`g' H g / |f|`) and parameter change.

/// Optimiser settings.
///
/// Stan's defaults are `history = 5`, `tol_rel_obj = 1e4`, `tol_rel_grad = 1e7` (both scaled
/// by machine epsilon). Those stop noticeably early in the flat concentration directions of
/// this model (fits end up ~1e-2 below the optimum in log posterior), so the defaults here
/// are tighter: `history = 10`, `tol_rel_obj = 1e2`, `tol_rel_grad = 1e4`, which brings the
/// fits within ~1e-5 of the optimum at about twice the cost. Use [`LbfgsParams::stan`] for
/// Stan's values.
#[derive(Clone, Debug)]
pub struct LbfgsParams {
    /// Number of correction pairs kept.
    pub history: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Stop when |f_k - f_{k-1}| < tol_obj.
    pub tol_obj: f64,
    /// Stop when |f_k - f_{k-1}| / max(|f_k|, |f_{k-1}|, eps) < tol_rel_obj * eps.
    pub tol_rel_obj: f64,
    /// Stop when ||g|| < tol_grad.
    pub tol_grad: f64,
    /// Stop when g' H g / max(|f|, eps) < tol_rel_grad * eps.
    pub tol_rel_grad: f64,
    /// Stop when ||x_k - x_{k-1}|| < tol_param.
    pub tol_param: f64,
    /// Maximum number of function evaluations inside one line search.
    pub max_line_search: usize,
    /// Print one line per iteration to stderr.
    pub trace: bool,
}

impl Default for LbfgsParams {
    fn default() -> Self {
        LbfgsParams {
            history: 10,
            max_iter: 2000,
            tol_obj: 1e-12,
            tol_rel_obj: 1e2,
            tol_grad: 1e-8,
            tol_rel_grad: 1e4,
            tol_param: 1e-8,
            max_line_search: 40,
            trace: false,
        }
    }
}

impl LbfgsParams {
    /// The stopping rules `rstan::optimizing` uses by default.
    pub fn stan() -> Self {
        LbfgsParams {
            history: 5,
            tol_rel_obj: 1e4,
            tol_rel_grad: 1e7,
            ..Default::default()
        }
    }
}

/// Why the optimiser stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Converged on one of the tolerance criteria.
    Converged,
    /// Line search could not find an acceptable step (typically already at the optimum
    /// up to floating point precision).
    LineSearchFailed,
    /// Hit `max_iter`.
    MaxIter,
    /// Objective was not finite at the starting point.
    NonFiniteStart,
}

/// Result of a minimisation. `x` is updated in place by [`minimize`].
#[derive(Clone, Debug)]
pub struct LbfgsResult {
    pub f: f64,
    pub grad_norm: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub status: Status,
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Minimise `f` starting from `x`. `f(x, grad)` must fill `grad` and return the objective.
/// Returns non-finite values from `f` are treated as infeasible points by the line search.
pub fn minimize<F>(mut f: F, x: &mut [f64], params: &LbfgsParams) -> LbfgsResult
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x.len();
    let eps = f64::EPSILON;
    let mut g = vec![0.0; n];
    let mut evals = 1usize;
    let mut fx = f(x, &mut g);
    if !fx.is_finite() {
        return LbfgsResult {
            f: fx,
            grad_norm: f64::NAN,
            iterations: 0,
            evaluations: evals,
            status: Status::NonFiniteStart,
        };
    }

    let m = params.history.max(1);
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(m);
    let mut alpha = vec![0.0; m];

    let mut d = vec![0.0; n];
    let mut x_new = vec![0.0; n];
    let mut g_new = vec![0.0; n];
    let mut status = Status::MaxIter;
    let mut iter = 0usize;

    while iter < params.max_iter {
        // --- search direction: d = -H g via two-loop recursion
        d.copy_from_slice(&g);
        let k = s_hist.len();
        for i in (0..k).rev() {
            alpha[i] = rho_hist[i] * dot(&s_hist[i], &d);
            for j in 0..n {
                d[j] -= alpha[i] * y_hist[i][j];
            }
        }
        if k > 0 {
            let sy = dot(&s_hist[k - 1], &y_hist[k - 1]);
            let yy = dot(&y_hist[k - 1], &y_hist[k - 1]);
            let gamma = if yy > 0.0 { sy / yy } else { 1.0 };
            for v in d.iter_mut() {
                *v *= gamma;
            }
        }
        for i in 0..k {
            let beta = rho_hist[i] * dot(&y_hist[i], &d);
            for j in 0..n {
                d[j] += (alpha[i] - beta) * s_hist[i][j];
            }
        }
        for v in d.iter_mut() {
            *v = -*v;
        }
        let mut dg = dot(&d, &g);
        if dg >= 0.0 {
            // not a descent direction: reset memory and use steepest descent
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            for j in 0..n {
                d[j] = -g[j];
            }
            dg = -dot(&g, &g);
            if dg >= 0.0 {
                status = Status::Converged;
                break;
            }
        }

        // relative gradient criterion: g' H g / |f|  (d = -H g)
        let gnorm = dot(&g, &g).sqrt();
        if gnorm < params.tol_grad {
            status = Status::Converged;
            break;
        }
        if (-dg) / fx.abs().max(eps) < params.tol_rel_grad * eps {
            status = Status::Converged;
            break;
        }

        // --- line search
        let step0 = if iter == 0 {
            (1.0 / gnorm).min(1.0)
        } else {
            1.0
        };
        let ls = line_search(
            &mut f,
            x,
            fx,
            &g,
            &d,
            dg,
            step0,
            &mut x_new,
            &mut g_new,
            params.max_line_search,
        );
        evals += ls.evals;
        if params.trace {
            eprintln!(
                "iter {iter:4} f={fx:.10} |g|={gnorm:.3e} gHg/|f|={:.3e} ls={:?} evals={}",
                (-dg) / fx.abs().max(eps),
                ls.result,
                ls.evals
            );
        }
        let Some((step, f_new)) = ls.result else {
            status = Status::LineSearchFailed;
            break;
        };

        // --- update history
        let mut s = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut snorm2 = 0.0;
        for j in 0..n {
            s[j] = step * d[j];
            y[j] = g_new[j] - g[j];
            snorm2 += s[j] * s[j];
        }
        let sy = dot(&s, &y);
        if sy > 1e-300 {
            if s_hist.len() == m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s);
            y_hist.push(y);
        }

        let df = (f_new - fx).abs();
        let f_prev = fx;
        x.copy_from_slice(&x_new);
        g.copy_from_slice(&g_new);
        fx = f_new;
        iter += 1;

        if df < params.tol_obj {
            status = Status::Converged;
            break;
        }
        if df / fx.abs().max(f_prev.abs()).max(eps) < params.tol_rel_obj * eps {
            status = Status::Converged;
            break;
        }
        if snorm2.sqrt() < params.tol_param {
            status = Status::Converged;
            break;
        }
    }

    LbfgsResult {
        f: fx,
        grad_norm: dot(&g, &g).sqrt(),
        iterations: iter,
        evaluations: evals,
        status,
    }
}

struct LineSearchOutcome {
    /// (step, f_new); x_new and g_new are filled by the caller's buffers.
    result: Option<(f64, f64)>,
    evals: usize,
}

const C1: f64 = 1e-4;
const C2: f64 = 0.9;

/// Strong-Wolfe line search. On success `x_new`/`g_new` hold the accepted point.
#[allow(clippy::too_many_arguments)]
fn line_search<F>(
    f: &mut F,
    x: &[f64],
    f0: f64,
    _g0: &[f64],
    d: &[f64],
    dg0: f64,
    step0: f64,
    x_new: &mut [f64],
    g_new: &mut [f64],
    max_evals: usize,
) -> LineSearchOutcome
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x.len();
    let mut evals = 0usize;
    let mut eval_at =
        |a: f64, x_new: &mut [f64], g_new: &mut [f64], evals: &mut usize| -> (f64, f64) {
            for j in 0..n {
                x_new[j] = x[j] + a * d[j];
            }
            *evals += 1;
            let fa = f(x_new, g_new);
            let dga = if fa.is_finite() {
                dot(g_new, d)
            } else {
                f64::NAN
            };
            (fa, dga)
        };

    // Bracketing phase.
    let mut a_prev = 0.0;
    let mut f_prev = f0;
    let mut dg_prev = dg0;
    let mut a = step0;
    let max_step = 1e10;
    let mut bracket: Option<(f64, f64, f64, f64, f64, f64)> = None; // (lo, f_lo, dg_lo, hi, f_hi, dg_hi)
    while evals < max_evals {
        let (fa, dga) = eval_at(a, x_new, g_new, &mut evals);
        if !fa.is_finite() || fa > f0 + C1 * a * dg0 || (a_prev > 0.0 && fa >= f_prev) {
            bracket = Some((a_prev, f_prev, dg_prev, a, fa, dga));
            break;
        }
        if dga.abs() <= -C2 * dg0 {
            return LineSearchOutcome {
                result: Some((a, fa)),
                evals,
            };
        }
        if dga >= 0.0 {
            bracket = Some((a, fa, dga, a_prev, f_prev, dg_prev));
            break;
        }
        a_prev = a;
        f_prev = fa;
        dg_prev = dga;
        a *= 2.0;
        if a > max_step {
            break;
        }
    }
    let Some((mut lo, mut f_lo, mut dg_lo, mut hi, mut f_hi, mut dg_hi)) = bracket else {
        return LineSearchOutcome {
            result: None,
            evals,
        };
    };

    // Zoom phase.
    while evals < max_evals {
        // cubic interpolation between lo and hi, safeguarded
        let mut a_j = cubic_min(lo, f_lo, dg_lo, hi, f_hi, dg_hi);
        let (lo_b, hi_b) = if lo < hi { (lo, hi) } else { (hi, lo) };
        let width = hi_b - lo_b;
        if !a_j.is_finite() || a_j <= lo_b + 0.05 * width || a_j >= hi_b - 0.05 * width {
            a_j = 0.5 * (lo + hi);
        }
        if width < 1e-16 * lo_b.abs().max(1.0) {
            break;
        }
        let (fj, dgj) = eval_at(a_j, x_new, g_new, &mut evals);
        if !fj.is_finite() || fj > f0 + C1 * a_j * dg0 || fj >= f_lo {
            hi = a_j;
            f_hi = fj;
            dg_hi = dgj;
        } else {
            if dgj.abs() <= -C2 * dg0 {
                return LineSearchOutcome {
                    result: Some((a_j, fj)),
                    evals,
                };
            }
            if dgj * (hi - lo) >= 0.0 {
                hi = lo;
                f_hi = f_lo;
                dg_hi = dg_lo;
            }
            lo = a_j;
            f_lo = fj;
            dg_lo = dgj;
        }
    }
    // Fall back to the best point with sufficient decrease found so far, if any.
    if lo > 0.0 && f_lo.is_finite() && f_lo < f0 {
        let (fa, _) = eval_at(lo, x_new, g_new, &mut evals);
        if fa.is_finite() && fa < f0 {
            return LineSearchOutcome {
                result: Some((lo, fa)),
                evals,
            };
        }
    }
    LineSearchOutcome {
        result: None,
        evals,
    }
}

/// Minimiser of the cubic interpolating (a, fa, ga) and (b, fb, gb).
fn cubic_min(a: f64, fa: f64, ga: f64, b: f64, fb: f64, gb: f64) -> f64 {
    if !fa.is_finite() || !fb.is_finite() || !ga.is_finite() || !gb.is_finite() {
        return f64::NAN;
    }
    let d1 = ga + gb - 3.0 * (fa - fb) / (a - b);
    let disc = d1 * d1 - ga * gb;
    if disc < 0.0 {
        return f64::NAN;
    }
    let d2 = disc.sqrt() * (b - a).signum();
    b - (b - a) * (gb + d2 - d1) / (gb - ga + 2.0 * d2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rosenbrock() {
        let mut x = vec![-1.2, 1.0];
        let res = minimize(
            |x, g| {
                let (a, b) = (x[0], x[1]);
                g[0] = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
                g[1] = 200.0 * (b - a * a);
                (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
            },
            &mut x,
            &LbfgsParams::default(),
        );
        assert!(res.f < 1e-12, "{res:?}");
        assert!((x[0] - 1.0).abs() < 1e-5 && (x[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn quadratic_with_infeasible_region() {
        // f = sum (x_i - 3)^2 but undefined for x_0 > 4 (mimics log-domain blow-ups)
        let mut x = vec![0.0; 5];
        let res = minimize(
            |x, g| {
                if x[0] > 4.0 {
                    return f64::NAN;
                }
                let mut f = 0.0;
                for i in 0..5 {
                    g[i] = 2.0 * (x[i] - 3.0);
                    f += (x[i] - 3.0).powi(2);
                }
                f
            },
            &mut x,
            &LbfgsParams::default(),
        );
        assert!(res.f < 1e-14, "{res:?}");
    }
}
