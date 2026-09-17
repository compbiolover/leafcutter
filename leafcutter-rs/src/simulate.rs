//! Synthetic Dirichlet-multinomial splicing data for tests and benchmarks (no `rand`
//! dependency; a small xorshift generator is enough here).

use crate::ds::Cluster;

/// xorshift64* pseudo random generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in (0, 1).
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Gamma(shape, 1) via Marsaglia–Tsang.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            return self.gamma(shape + 1.0) * self.uniform().powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.uniform();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
    pub fn dirichlet(&mut self, alpha: &[f64]) -> Vec<f64> {
        let mut v: Vec<f64> = alpha.iter().map(|&a| self.gamma(a)).collect();
        let s: f64 = v.iter().sum();
        for x in v.iter_mut() {
            *x /= s;
        }
        v
    }
    pub fn poisson(&mut self, mean: f64) -> u32 {
        if mean <= 0.0 {
            return 0;
        }
        if mean < 30.0 {
            let l = (-mean).exp();
            let mut k = 0u32;
            let mut p = 1.0;
            loop {
                p *= self.uniform();
                if p <= l {
                    return k;
                }
                k += 1;
            }
        }
        // normal approximation for large means
        let v = mean + mean.sqrt() * self.normal();
        v.round().max(0.0) as u32
    }
    pub fn multinomial(&mut self, n: u32, probs: &[f64]) -> Vec<u32> {
        let mut out = vec![0u32; probs.len()];
        let mut remaining = n;
        let mut rem_p = 1.0;
        for j in 0..probs.len() {
            if remaining == 0 {
                break;
            }
            if j == probs.len() - 1 {
                out[j] = remaining;
                break;
            }
            let p = (probs[j] / rem_p).clamp(0.0, 1.0);
            let x = self.binomial(remaining, p);
            out[j] = x;
            remaining -= x;
            rem_p -= probs[j];
            if rem_p <= 0.0 {
                break;
            }
        }
        out
    }
    pub fn binomial(&mut self, n: u32, p: f64) -> u32 {
        if p <= 0.0 {
            return 0;
        }
        if p >= 1.0 {
            return n;
        }
        if n < 64 {
            (0..n).filter(|_| self.uniform() < p).count() as u32
        } else {
            let m = n as f64 * p;
            let v = m + (m * (1.0 - p)).sqrt() * self.normal();
            v.round().clamp(0.0, n as f64) as u32
        }
    }
}

/// Simulation settings.
#[derive(Clone, Debug)]
pub struct SimParams {
    pub n_samples: usize,
    pub n_clusters: usize,
    pub seed: u64,
    /// Fraction of clusters with a real group effect.
    pub frac_differential: f64,
    /// Mean per-sample read depth per cluster (log-normal around this).
    pub mean_depth: f64,
    /// Fraction of samples with zero reads in a cluster.
    pub dropout: f64,
    pub max_introns: usize,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            n_samples: 100,
            n_clusters: 1000,
            seed: 1,
            frac_differential: 0.1,
            mean_depth: 40.0,
            dropout: 0.05,
            max_introns: 6,
        }
    }
}

/// Simulated dataset: group labels (0/1) and clusters with LeafCutter-style names.
pub struct SimData {
    pub samples: Vec<String>,
    pub group: Vec<u8>,
    pub clusters: Vec<Cluster>,
    /// Which clusters carry a true effect.
    pub differential: Vec<bool>,
}

pub fn simulate(params: &SimParams) -> SimData {
    let mut rng = Rng::new(params.seed);
    let n = params.n_samples;
    let samples: Vec<String> = (0..n).map(|i| format!("sample{}", i + 1)).collect();
    let group: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
    let mut clusters = Vec::with_capacity(params.n_clusters);
    let mut differential = Vec::with_capacity(params.n_clusters);
    let depth_sd = 0.6;
    let mut pos = 10_000u64;
    for c in 0..params.n_clusters {
        let k = 2 + (rng.next_u64() % (params.max_introns as u64 - 1)) as usize;
        let base = rng.dirichlet(&vec![1.0; k]);
        let conc = (2.5 + 0.8 * rng.normal()).exp(); // ~ lognormal, median 12
        let diff = rng.uniform() < params.frac_differential;
        let mut alt = base.clone();
        if diff {
            let logit: Vec<f64> = base.iter().map(|p| p.ln() + 1.2 * rng.normal()).collect();
            let m = logit.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = logit.iter().map(|v| (v - m).exp()).collect();
            let s: f64 = e.iter().sum();
            alt = e.iter().map(|v| v / s).collect();
        }
        let chrom = format!("chr{}", 1 + c % 22);
        let start = pos;
        let introns: Vec<String> = (0..k)
            .map(|j| {
                format!(
                    "{chrom}:{}:{}:clu_{}_NA",
                    start + 100 * j as u64,
                    start + 5000 + 300 * j as u64,
                    c + 1
                )
            })
            .collect();
        pos += 20_000;
        let mut counts = vec![0u32; n * k];
        for i in 0..n {
            if rng.uniform() < params.dropout {
                continue;
            }
            let depth = (params.mean_depth.ln() + depth_sd * rng.normal()).exp();
            let total = rng.poisson(depth);
            let mean = if group[i] == 1 { &alt } else { &base };
            let alpha: Vec<f64> = mean.iter().map(|p| (p * conc).max(1e-3)).collect();
            let probs = rng.dirichlet(&alpha);
            let y = rng.multinomial(total, &probs);
            counts[i * k..(i + 1) * k].copy_from_slice(&y);
        }
        clusters.push(Cluster {
            name: format!("{chrom}:clu_{}_NA", c + 1),
            annotations: Vec::new(),
            introns,
            n,
            counts,
        });
        differential.push(diff);
    }
    SimData {
        samples,
        group,
        clusters,
        differential,
    }
}
