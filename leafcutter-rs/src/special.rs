//! Special functions: log-gamma, digamma and the chi-square survival function.

/// Natural log of the gamma function.
#[inline]
pub fn lgamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// Digamma function psi(x) = d/dx ln Gamma(x), for x > 0.
///
/// Uses the recurrence psi(x) = psi(x+1) - 1/x to shift the argument to >= 10 and then the
/// asymptotic expansion. Accuracy ~1e-14 for x > 0.
pub fn digamma(mut x: f64) -> f64 {
    debug_assert!(x > 0.0, "digamma requires x > 0, got {x}");
    let mut acc = 0.0;
    while x < 10.0 {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // ln x - 1/(2x) - sum B_{2n} / (2n x^{2n})
    let series = inv2
        * (1.0 / 12.0
            - inv2
                * (1.0 / 120.0
                    - inv2
                        * (1.0 / 252.0
                            - inv2
                                * (1.0 / 240.0
                                    - inv2
                                        * (1.0 / 132.0
                                            - inv2 * (691.0 / 32760.0 - inv2 / 12.0))))));
    acc + x.ln() - 0.5 * inv - series
}

/// Regularized upper incomplete gamma function Q(a, x) = Gamma(a, x) / Gamma(a).
///
/// Series for x < a + 1, continued fraction otherwise (Numerical Recipes `gammq`).
pub fn gammq(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if !x.is_finite() {
        return 0.0;
    }
    if x < a + 1.0 {
        1.0 - gser(a, x)
    } else {
        gcf(a, x)
    }
}

fn gser(a: f64, x: f64) -> f64 {
    let gln = lgamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..10_000 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-16 {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

fn gcf(a: f64, x: f64) -> f64 {
    let gln = lgamma(a);
    let fpmin = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut i = 1.0;
    loop {
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 || i > 10_000.0 {
            break;
        }
        i += 1.0;
    }
    (-x + a * x.ln() - gln).exp() * h
}

/// Upper tail probability of a chi-square distribution: P(X > x) with `df` degrees of freedom.
/// Matches R's `pchisq(x, df, lower.tail = FALSE)`.
pub fn chisq_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gammq(0.5 * df, 0.5 * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digamma_known_values() {
        let euler = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + euler).abs() < 1e-13);
        assert!((digamma(0.5) - (-euler - 2.0 * 2f64.ln())).abs() < 1e-13);
        assert!((digamma(10.0) - 2.251_752_589_066_721).abs() < 1e-12);
        // recurrence check
        for &x in &[0.1, 0.7, 2.3, 17.5, 1234.5] {
            assert!((digamma(x + 1.0) - digamma(x) - 1.0 / x).abs() < 1e-12);
        }
    }

    #[test]
    fn digamma_matches_lgamma_derivative() {
        for &x in &[0.05f64, 0.3, 1.0, 3.7, 25.0, 400.0] {
            let h = 1e-4 * x;
            let fd = (lgamma(x + h) - lgamma(x - h)) / (2.0 * h);
            assert!(
                (fd - digamma(x)).abs() < 1e-6 * (1.0 + digamma(x).abs()),
                "x={x} fd={fd} psi={}",
                digamma(x)
            );
        }
    }

    #[test]
    fn chisq_sf_matches_r() {
        // scipy.stats.chi2.sf(x, df) (identical to R's pchisq(lower.tail=FALSE))
        let cases = [
            (1.0, 1.0, 0.31731050786291115),
            (5.0, 3.0, 0.1717971442967335),
            (20.0, 4.0, 0.0004993992273873336),
            (100.0, 7.0, 1.0787979671702833e-18),
            (1000.0, 9.0, 1.7240681189224405e-209),
        ];
        for (x, df, expect) in cases {
            let got = chisq_sf(x, df);
            assert!(
                ((got - expect) / expect).abs() < 1e-9,
                "x={x} df={df} got={got} expect={expect}"
            );
        }
        assert_eq!(chisq_sf(-1.0, 3.0), 1.0);
        assert_eq!(chisq_sf(0.0, 3.0), 1.0);
    }
}
