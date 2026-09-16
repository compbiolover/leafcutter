//! Differential splicing driver: per-cluster filtering (identical to
//! `differential_splicing` in `leafcutter/R/differential_splicing.R`), the parallel loop over
//! clusters, and multiple-testing correction.

use crate::design::Design;
use crate::fit::{effect_sizes, lrt, Fit, FitParams};
use crate::nullcache::NullCache;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Filtering thresholds (defaults of `differential_splicing`) plus the fit settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DsParams {
    /// Don't test clusters with more introns than this.
    pub max_cluster_size: usize,
    /// Ignore introns used (>= 1 read) in fewer than this many samples.
    pub min_samples_per_intron: usize,
    /// Require this many samples per group with at least `min_coverage` reads.
    pub min_samples_per_group: usize,
    /// Read threshold for the `min_samples_per_group` rule.
    pub min_coverage: u32,
    pub fit: FitParams,
}

impl Default for DsParams {
    fn default() -> Self {
        DsParams {
            max_cluster_size: 10,
            min_samples_per_intron: 5,
            min_samples_per_group: 4,
            min_coverage: 20,
            fit: FitParams::default(),
        }
    }
}

/// One intron cluster's counts for the samples in the request, sample-major (`counts[n*k + j]`).
#[derive(Clone, Debug)]
pub struct Cluster {
    pub name: String,
    pub introns: Vec<String>,
    pub n: usize,
    pub counts: Vec<u32>,
}

impl Cluster {
    #[inline]
    pub fn k(&self) -> usize {
        self.introns.len()
    }
}

/// Per intron output row of the effect size table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntronResult {
    pub intron: String,
    pub logef: f64,
    pub baseline: f64,
    pub perturbed: f64,
    pub deltapsi: f64,
}

/// Per cluster result. `status` is `"Success"` or the reason the cluster was not tested,
/// using the same strings as the R package.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterResult {
    pub cluster: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loglr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub df: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p: Option<f64>,
    #[serde(rename = "p.adjust", skip_serializing_if = "Option::is_none")]
    pub p_adjust: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_storey: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub introns: Vec<IntronResult>,
    /// Number of samples that entered the fit and number of design cells of the full model.
    #[serde(default)]
    pub n_samples: usize,
    #[serde(default)]
    pub refit_null: bool,
    #[serde(default)]
    pub evaluations: usize,
    /// Log posterior at the null / full optimum (`value` of `rstan::optimizing`).
    #[serde(default)]
    pub value_null: f64,
    #[serde(default)]
    pub value_full: f64,
    /// Whether both fits met a convergence criterion.
    #[serde(default)]
    pub converged: bool,
}

impl ClusterResult {
    fn skipped(name: &str, why: &str) -> Self {
        ClusterResult {
            cluster: name.to_string(),
            status: why.to_string(),
            loglr: None,
            df: None,
            p: None,
            p_adjust: None,
            q_storey: None,
            introns: Vec::new(),
            n_samples: 0,
            refit_null: false,
            evaluations: 0,
            value_null: f64::NAN,
            value_full: f64::NAN,
            converged: false,
        }
    }
    pub fn is_success(&self) -> bool {
        self.status == "Success"
    }
}

/// Prepared inputs to a cluster fit after filtering: the per-sample counts and design.
#[derive(Clone, Debug)]
pub struct PreparedCluster {
    pub introns: Vec<String>,
    pub n: usize,
    pub k: usize,
    pub counts: Vec<u32>,
    pub x_full: Design,
    /// Columns of `x_full` forming the null design.
    pub null_cols: Vec<usize>,
    /// Indices (into the request's sample list) of the samples kept.
    pub samples_used: Vec<usize>,
}

/// Apply the R package's filters. `x` is the 0/1 group vector over the request samples and
/// `confounders` an optional `N x C` matrix. Returns the prepared cluster or a skip reason.
pub fn prepare_cluster(cluster: &Cluster, x: &[f64], confounders: Option<&Design>, params: &DsParams) -> Result<PreparedCluster, String> {
    let k = cluster.k();
    let n = cluster.n;
    assert_eq!(x.len(), n);
    if k > params.max_cluster_size {
        return Err("Too many introns in cluster".into());
    }
    if k <= 1 {
        return Err("<=1 junction in cluster".into());
    }
    let totals: Vec<u32> = (0..n).map(|i| cluster.counts[i * k..(i + 1) * k].iter().sum()).collect();
    let samples_used: Vec<usize> = (0..n).filter(|&i| totals[i] > 0).collect();
    if samples_used.len() <= 1 {
        return Err("<=1 sample with coverage>0".into());
    }
    let covered: Vec<bool> = samples_used.iter().map(|&i| totals[i] >= params.min_coverage).collect();
    if covered.iter().filter(|&&c| c).count() <= 1 {
        return Err("<=1 sample with coverage>min_coverage".into());
    }
    let x_subset: Vec<f64> = samples_used.iter().map(|&i| x[i]).collect();
    let introns_to_use: Vec<usize> = (0..k)
        .filter(|&j| samples_used.iter().filter(|&&i| cluster.counts[i * k + j] > 0).count() >= params.min_samples_per_intron)
        .collect();
    if introns_to_use.len() < 2 {
        return Err("<2 introns used in >=min_samples_per_intron samples".into());
    }
    // at least two groups with >= min_samples_per_group well-covered samples
    let mut groups: Vec<(f64, usize)> = Vec::new();
    for (idx, &xi) in x_subset.iter().enumerate() {
        if !covered[idx] {
            continue;
        }
        match groups.iter_mut().find(|(g, _)| *g == xi) {
            Some((_, c)) => *c += 1,
            None => groups.push((xi, 1)),
        }
    }
    if groups.iter().filter(|(_, c)| *c >= params.min_samples_per_group).count() < 2 {
        return Err("Not enough valid samples".into());
    }
    let k2 = introns_to_use.len();
    let mut counts = Vec::with_capacity(samples_used.len() * k2);
    for &i in &samples_used {
        for &j in &introns_to_use {
            counts.push(cluster.counts[i * k + j]);
        }
    }
    let n2 = samples_used.len();
    let ones = vec![1.0; n2];
    let mut cols: Vec<Vec<f64>> = vec![ones, x_subset];
    if let Some(ch) = confounders {
        let sub = ch.select_rows(&samples_used);
        for c in 0..sub.p {
            if sub.column_sd(c) > 0.0 {
                cols.push(sub.column(c));
            }
        }
    }
    let col_refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
    let x_full = Design::from_columns(n2, &col_refs);
    let null_cols: Vec<usize> = (0..x_full.p).filter(|&c| c != 1).collect();
    Ok(PreparedCluster {
        introns: introns_to_use.iter().map(|&j| cluster.introns[j].clone()).collect(),
        n: n2,
        k: k2,
        counts,
        x_full,
        null_cols,
        samples_used,
    })
}

/// Test a single cluster: filter, fit, LRT, effect sizes.
pub fn test_cluster(cluster: &Cluster, x: &[f64], confounders: Option<&Design>, params: &DsParams, null_cache: Option<&NullCache>) -> ClusterResult {
    let prep = match prepare_cluster(cluster, x, confounders, params) {
        Ok(p) => p,
        Err(why) => return ClusterResult::skipped(&cluster.name, &why),
    };
    let cached: Option<Fit> = null_cache.and_then(|c| c.get(&cluster.name, &prep));
    let from_cache = cached.is_some();
    let res = lrt(&prep.counts, prep.n, prep.k, &prep.x_full, &prep.null_cols, &params.fit, cached.clone());
    if cached.is_none() {
        if let Some(c) = null_cache {
            c.put(&cluster.name, &prep, &res.fit_null);
        }
    }
    if !res.loglr.is_finite() || !res.fit_full.value.is_finite() || !res.fit_null.value.is_finite() {
        return ClusterResult::skipped(&cluster.name, "Error: non-finite fit");
    }
    let es = effect_sizes(&res.fit_full, 0, 1);
    ClusterResult {
        cluster: cluster.name.clone(),
        status: "Success".into(),
        loglr: Some(res.loglr),
        df: Some(res.df),
        p: Some(res.p),
        p_adjust: None,
        q_storey: None,
        introns: prep
            .introns
            .iter()
            .zip(es)
            .map(|(name, e)| IntronResult { intron: name.clone(), logef: e.logef, baseline: e.baseline, perturbed: e.perturbed, deltapsi: e.deltapsi })
            .collect(),
        n_samples: prep.n,
        refit_null: res.refit_null,
        evaluations: res.fit_full.evaluations + if from_cache { 0 } else { res.fit_null.evaluations },
        value_null: res.fit_null.value,
        value_full: res.fit_full.value,
        converged: res.fit_null.converged && res.fit_full.converged,
    }
}

/// Benjamini–Hochberg adjusted p-values (`p.adjust(method = "fdr")`); `None` entries are
/// ignored and stay `None`.
pub fn bh_adjust(p: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut idx: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_some()).collect();
    let m = idx.len();
    let mut out = vec![None; p.len()];
    if m == 0 {
        return out;
    }
    idx.sort_by(|&a, &b| p[b].unwrap().partial_cmp(&p[a].unwrap()).unwrap_or(std::cmp::Ordering::Equal));
    // descending p: q_(i) = min(1, cummin(m/i * p_(i)))
    let mut running = 1.0f64;
    for (pos, &i) in idx.iter().enumerate() {
        let rank = (m - pos) as f64;
        let v = (m as f64 / rank * p[i].unwrap().min(1.0)).min(running);
        running = v;
        out[i] = Some(v.min(1.0));
    }
    out
}

/// Storey q-values with a fixed `lambda` (pi0 = #{p > lambda} / ((1 - lambda) m), then the
/// BH-style step-up with `pi0` in place of 1). `lambda = 0.5` is a robust default.
pub fn storey_qvalues(p: &[Option<f64>], lambda: f64) -> Vec<Option<f64>> {
    let m = p.iter().filter(|v| v.is_some()).count();
    if m == 0 {
        return vec![None; p.len()];
    }
    let above = p.iter().filter(|v| matches!(v, Some(x) if *x > lambda)).count();
    let pi0 = (above as f64 / ((1.0 - lambda) * m as f64)).clamp(0.0, 1.0);
    let pi0 = if pi0 == 0.0 { 1.0 / m as f64 } else { pi0 };
    bh_adjust(p).into_iter().map(|q| q.map(|v| (v * pi0).min(1.0))).collect()
}

/// Run the test over all clusters in parallel (rayon) and fill in the adjusted p-values.
pub fn run(clusters: &[Cluster], x: &[f64], confounders: Option<&Design>, params: &DsParams, null_cache: Option<&NullCache>) -> Vec<ClusterResult> {
    let mut results: Vec<ClusterResult> = clusters
        .par_iter()
        .map(|c| test_cluster(c, x, confounders, params, null_cache))
        .collect();
    finalize(&mut results);
    results
}

/// Compute BH and Storey adjusted p-values over a set of results.
pub fn finalize(results: &mut [ClusterResult]) {
    let p: Vec<Option<f64>> = results.iter().map(|r| r.p).collect();
    let bh = bh_adjust(&p);
    let st = storey_qvalues(&p, 0.5);
    for (r, (b, s)) in results.iter_mut().zip(bh.into_iter().zip(st)) {
        r.p_adjust = b;
        r.q_storey = s;
    }
}

/// Count of each status string, for the "Differential splicing summary" the R tool prints.
pub fn status_summary(results: &[ClusterResult]) -> Vec<(String, usize)> {
    let mut v: Vec<(String, usize)> = Vec::new();
    for r in results {
        match v.iter_mut().find(|(s, _)| *s == r.status) {
            Some((_, c)) => *c += 1,
            None => v.push((r.status.clone(), 1)),
        }
    }
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bh_matches_r() {
        // R: p.adjust(c(0.01, 0.04, 0.03, 0.5, NA, 0.2), method="fdr")
        let p = [Some(0.01), Some(0.04), Some(0.03), Some(0.5), None, Some(0.2)];
        let q = bh_adjust(&p);
        let expect = [0.05, 0.06666666666666667, 0.06666666666666667, 0.5, f64::NAN, 0.25];
        for i in 0..p.len() {
            match (q[i], expect[i]) {
                (None, e) => assert!(e.is_nan()),
                (Some(v), e) => assert!((v - e).abs() < 1e-12, "{i}: {v} vs {e}"),
            }
        }
    }

    #[test]
    fn filters_replicate_r_reasons() {
        let params = DsParams::default();
        let n = 12;
        let x: Vec<f64> = (0..n).map(|i| if i < 6 { 0.0 } else { 1.0 }).collect();
        // single junction
        let c = Cluster { name: "a".into(), introns: vec!["i1".into()], n, counts: vec![5; n] };
        assert_eq!(prepare_cluster(&c, &x, None, &params).unwrap_err(), "<=1 junction in cluster");
        // too many introns
        let c = Cluster { name: "a".into(), introns: (0..11).map(|i| format!("i{i}")).collect(), n, counts: vec![5; n * 11] };
        assert_eq!(prepare_cluster(&c, &x, None, &params).unwrap_err(), "Too many introns in cluster");
        // low coverage everywhere
        let c = Cluster { name: "a".into(), introns: vec!["i1".into(), "i2".into()], n, counts: vec![1; n * 2] };
        assert_eq!(prepare_cluster(&c, &x, None, &params).unwrap_err(), "<=1 sample with coverage>min_coverage");
        // second intron rarely used
        let mut counts = vec![0u32; n * 2];
        for i in 0..n {
            counts[i * 2] = 30;
            if i < 3 {
                counts[i * 2 + 1] = 4;
            }
        }
        let c = Cluster { name: "a".into(), introns: vec!["i1".into(), "i2".into()], n, counts };
        assert_eq!(prepare_cluster(&c, &x, None, &params).unwrap_err(), "<2 introns used in >=min_samples_per_intron samples");
        // one group under-covered
        let mut counts = vec![0u32; n * 2];
        for i in 0..n {
            counts[i * 2] = if i < 6 { 30 } else { 5 };
            counts[i * 2 + 1] = 5;
        }
        let c = Cluster { name: "a".into(), introns: vec!["i1".into(), "i2".into()], n, counts };
        assert_eq!(prepare_cluster(&c, &x, None, &params).unwrap_err(), "Not enough valid samples");
        // success, with a constant confounder column dropped
        let counts: Vec<u32> = (0..n).flat_map(|i| [30 + i as u32, 10]).collect();
        let c = Cluster { name: "a".into(), introns: vec!["i1".into(), "i2".into()], n, counts };
        let conf = Design::from_columns(n, &[&vec![1.0; n], &(0..n).map(|i| i as f64).collect::<Vec<_>>()]);
        let prep = prepare_cluster(&c, &x, Some(&conf), &params).unwrap();
        assert_eq!(prep.x_full.p, 3);
        assert_eq!(prep.null_cols, vec![0, 2]);
        let r = test_cluster(&c, &x, Some(&conf), &params, None);
        assert!(r.is_success(), "{}", r.status);
        assert_eq!(r.df, Some(1));
    }
}
