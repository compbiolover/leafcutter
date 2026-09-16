//! Differential splicing driver: per-cluster filtering (identical to `task` in
//! `leafcutter/differential_splicing/differential_splicing.py`), the parallel loop over
//! clusters, and multiple-testing correction.

use crate::design::Design;
use crate::fit::{effect_sizes, lrt, EffectSize, Fit, FitParams};
use crate::io::Phenotype;
use crate::nullcache::NullCache;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Filtering thresholds (the `leafcutter-ds` command line defaults) plus the fit settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DsParams {
    /// Don't test clusters with more introns than this.
    pub max_cluster_size: usize,
    /// Ignore introns used (>= 1 read) in fewer than this many samples.
    pub min_samples_per_intron: usize,
    /// Categorical phenotype: require this many samples per group with at least
    /// `min_coverage` reads, in at least two groups.
    pub min_samples_per_group: usize,
    /// Read threshold for the rule above.
    pub min_coverage: u32,
    /// Continuous phenotype: require this many distinct values among covered samples.
    pub min_unique_vals: usize,
    pub fit: FitParams,
}

impl Default for DsParams {
    fn default() -> Self {
        DsParams {
            max_cluster_size: usize::MAX,
            min_samples_per_intron: 5,
            min_samples_per_group: 3,
            min_coverage: 20,
            min_unique_vals: 10,
            fit: FitParams::default(),
        }
    }
}

/// One intron cluster's counts for the samples in the request, sample-major (`counts[n*k + j]`).
#[derive(Clone, Debug)]
pub struct Cluster {
    /// `chr:clu`
    pub name: String,
    /// Full intron names (including a leafcutter2 annotation field when present).
    pub introns: Vec<String>,
    /// Sorted unique leafcutter2 annotations of the introns (empty for 4-field names).
    pub annotations: Vec<String>,
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
    /// PSI in the baseline group (intercept row).
    pub psi_baseline: f64,
    /// Per non-baseline group present in the cluster: log effect size, PSI and ΔPSI.
    pub effects: BTreeMap<String, EffectSize>,
}

/// Per cluster result. `status` is `"Success"` or the reason the cluster was not tested,
/// using the same strings as leafcutter-ds and the R package.
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub genes: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub introns: Vec<IntronResult>,
    /// Number of samples that entered the fit.
    #[serde(default)]
    pub n_samples: usize,
    #[serde(default)]
    pub refit_null: bool,
    #[serde(default)]
    pub smart_init_improved: bool,
    #[serde(default)]
    pub evaluations: usize,
    /// Log posterior at the null / full optimum.
    #[serde(default)]
    pub value_null: f64,
    #[serde(default)]
    pub value_full: f64,
    /// Whether both fits met a convergence criterion.
    #[serde(default)]
    pub converged: bool,
}

impl ClusterResult {
    fn skipped(c: &Cluster, why: &str) -> Self {
        ClusterResult {
            cluster: c.name.clone(),
            status: why.to_string(),
            loglr: None,
            df: None,
            p: None,
            p_adjust: None,
            q_storey: None,
            genes: None,
            annotations: c.annotations.clone(),
            introns: Vec::new(),
            n_samples: 0,
            refit_null: false,
            smart_init_improved: false,
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
    /// `[intercept, confounders..., group columns...]`
    pub x_full: Design,
    /// Columns of `x_full` forming the null design (`0..P_null`).
    pub null_cols: Vec<usize>,
    /// Names of the group columns (`P_null..P_full`), in design order.
    pub group_names: Vec<String>,
    /// Indices (into the request's sample list) of the samples kept.
    pub samples_used: Vec<usize>,
}

/// Apply leafcutter-ds's filters. Returns the prepared cluster or a skip reason.
pub fn prepare_cluster(
    cluster: &Cluster,
    pheno: &Phenotype,
    confounders: Option<&Design>,
    params: &DsParams,
) -> Result<PreparedCluster, String> {
    let k = cluster.k();
    let n = cluster.n;
    assert_eq!(pheno.len(), n);
    if k > params.max_cluster_size {
        return Err("Too many introns in cluster".into());
    }
    if k <= 1 {
        return Err("<=1 junction in cluster".into());
    }
    // one pass: per-sample totals and per-intron number of samples with a read
    let mut totals: Vec<u32> = Vec::with_capacity(n);
    let mut used_by = vec![0usize; k];
    for i in 0..n {
        let y = &cluster.counts[i * k..(i + 1) * k];
        let mut tot = 0u32;
        for j in 0..k {
            if y[j] > 0 {
                used_by[j] += 1;
                tot += y[j];
            }
        }
        totals.push(tot);
    }
    let samples_used: Vec<usize> = (0..n).filter(|&i| totals[i] > 0).collect();
    if samples_used.len() <= 1 {
        return Err("<=1 sample with coverage>0".into());
    }
    let covered: Vec<bool> = samples_used
        .iter()
        .map(|&i| totals[i] >= params.min_coverage)
        .collect();
    if covered.iter().filter(|&&c| c).count() <= 1 {
        return Err("<=1 sample with coverage>min_coverage".into());
    }
    let introns_to_use: Vec<usize> = (0..k)
        .filter(|&j| used_by[j] >= params.min_samples_per_intron)
        .collect();
    if introns_to_use.len() < 2 {
        return Err("<2 introns used in >=min_samples_per_intron samples".into());
    }
    let n2 = samples_used.len();

    // phenotype columns (the only part depending on x)
    let mut group_cols: Vec<(String, Vec<f64>)> = Vec::new();
    match pheno {
        Phenotype::Categorical { levels, codes } => {
            let mut per_level = vec![0usize; levels.len()];
            for (idx, &i) in samples_used.iter().enumerate() {
                if covered[idx] {
                    per_level[codes[i]] += 1;
                }
            }
            if per_level
                .iter()
                .filter(|&&c| c >= params.min_samples_per_group)
                .count()
                < 2
            {
                return Err("Not enough valid samples".into());
            }
            for (lvl, name) in levels.iter().enumerate().skip(1) {
                let col: Vec<f64> = samples_used
                    .iter()
                    .map(|&i| if codes[i] == lvl { 1.0 } else { 0.0 })
                    .collect();
                if col.iter().any(|&v| v != 0.0) {
                    group_cols.push((name.clone(), col));
                }
            }
        }
        Phenotype::Continuous { values, .. } => {
            let mut uniq: Vec<f64> = samples_used
                .iter()
                .enumerate()
                .filter(|(idx, _)| covered[*idx])
                .map(|(_, &i)| values[i])
                .collect();
            uniq.sort_by(|a, b| a.partial_cmp(b).unwrap());
            uniq.dedup();
            if uniq.len() < params.min_unique_vals {
                return Err("Not enough valid samples".into());
            }
            group_cols.push((
                "x".to_string(),
                samples_used.iter().map(|&i| values[i]).collect(),
            ));
        }
    }

    let k2 = introns_to_use.len();
    let mut counts = Vec::with_capacity(n2 * k2);
    for &i in &samples_used {
        for &j in &introns_to_use {
            counts.push(cluster.counts[i * k + j]);
        }
    }
    let mut cols: Vec<Vec<f64>> = vec![vec![1.0; n2]];
    if let Some(ch) = confounders {
        let sub = ch.select_rows(&samples_used);
        for c in 0..sub.p {
            if sub.column_sd(c) > 0.0 {
                cols.push(sub.column(c));
            }
        }
    }
    let p_null = cols.len();
    let group_names: Vec<String> = group_cols.iter().map(|(n, _)| n.clone()).collect();
    for (_, c) in group_cols {
        cols.push(c);
    }
    let col_refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
    let x_full = Design::from_columns(n2, &col_refs);
    Ok(PreparedCluster {
        introns: introns_to_use
            .iter()
            .map(|&j| cluster.introns[j].clone())
            .collect(),
        n: n2,
        k: k2,
        counts,
        x_full,
        null_cols: (0..p_null).collect(),
        group_names,
        samples_used,
    })
}

/// Test a single cluster: filter, fit, LRT, effect sizes.
pub fn test_cluster(
    cluster: &Cluster,
    pheno: &Phenotype,
    confounders: Option<&Design>,
    params: &DsParams,
    null_cache: Option<&NullCache>,
) -> ClusterResult {
    let prep = match prepare_cluster(cluster, pheno, confounders, params) {
        Ok(p) => p,
        Err(why) => return ClusterResult::skipped(cluster, &why),
    };
    let cached: Option<Fit> = null_cache.and_then(|c| c.get(&cluster.name, &prep));
    let from_cache = cached.is_some();
    let res = lrt(
        &prep.counts,
        prep.n,
        prep.k,
        &prep.x_full,
        &prep.null_cols,
        &params.fit,
        cached.clone(),
    );
    if !from_cache {
        if let Some(c) = null_cache {
            c.put(&cluster.name, &prep, &res.fit_null);
        }
    }
    if !res.loglr.is_finite() || !res.fit_full.value.is_finite() || !res.fit_null.value.is_finite()
    {
        return ClusterResult::skipped(cluster, "Error: non-finite fit");
    }
    let p_null = prep.null_cols.len();
    let group_rows: Vec<usize> = (p_null..prep.x_full.p).collect();
    let (base, per_group) = effect_sizes(&res.fit_full, 0, &group_rows);
    let introns = prep
        .introns
        .iter()
        .enumerate()
        .map(|(j, name)| IntronResult {
            intron: name.clone(),
            psi_baseline: base[j],
            effects: prep
                .group_names
                .iter()
                .zip(&per_group)
                .map(|(g, es)| (g.clone(), es[j].clone()))
                .collect(),
        })
        .collect();
    ClusterResult {
        cluster: cluster.name.clone(),
        status: "Success".into(),
        loglr: Some(res.loglr),
        df: Some(res.df),
        p: Some(res.p),
        p_adjust: None,
        q_storey: None,
        genes: None,
        annotations: cluster.annotations.clone(),
        introns,
        n_samples: prep.n,
        refit_null: res.refit_null,
        smart_init_improved: res.smart_init_improved,
        evaluations: res.fit_full.evaluations
            + if from_cache {
                0
            } else {
                res.fit_null.evaluations
            },
        value_null: res.fit_null.value,
        value_full: res.fit_full.value,
        converged: res.fit_null.converged && res.fit_full.converged,
    }
}

/// Benjamini–Hochberg adjusted p-values (`p.adjust(method = "fdr")` /
/// `scipy.stats.false_discovery_control`); `None` entries are ignored and stay `None`.
pub fn bh_adjust(p: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut idx: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_some()).collect();
    let m = idx.len();
    let mut out = vec![None; p.len()];
    if m == 0 {
        return out;
    }
    idx.sort_by(|&a, &b| {
        p[b].unwrap()
            .partial_cmp(&p[a].unwrap())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut running = 1.0f64;
    for (pos, &i) in idx.iter().enumerate() {
        let rank = (m - pos) as f64;
        let v = (m as f64 / rank * p[i].unwrap().min(1.0)).min(running);
        running = v;
        out[i] = Some(v.min(1.0));
    }
    out
}

/// Storey q-values with a fixed `lambda`.
pub fn storey_qvalues(p: &[Option<f64>], lambda: f64) -> Vec<Option<f64>> {
    let m = p.iter().filter(|v| v.is_some()).count();
    if m == 0 {
        return vec![None; p.len()];
    }
    let above = p
        .iter()
        .filter(|v| matches!(v, Some(x) if *x > lambda))
        .count();
    let pi0 = (above as f64 / ((1.0 - lambda) * m as f64)).clamp(0.0, 1.0);
    let pi0 = if pi0 == 0.0 { 1.0 / m as f64 } else { pi0 };
    bh_adjust(p)
        .into_iter()
        .map(|q| q.map(|v| (v * pi0).min(1.0)))
        .collect()
}

/// Run the test over all clusters in parallel (rayon) and fill in the adjusted p-values.
pub fn run(
    clusters: &[Cluster],
    pheno: &Phenotype,
    confounders: Option<&Design>,
    params: &DsParams,
    null_cache: Option<&NullCache>,
) -> Vec<ClusterResult> {
    let mut results: Vec<ClusterResult> = clusters
        .par_iter()
        .map(|c| test_cluster(c, pheno, confounders, params, null_cache))
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

/// Count of each status string.
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

/// Convenience: a two-level categorical phenotype from a 0/1 vector (0 = baseline).
pub fn binary_phenotype(x: &[u8], baseline: &str, other: &str) -> Phenotype {
    Phenotype::Categorical {
        levels: vec![baseline.to_string(), other.to_string()],
        codes: x.iter().map(|&v| v as usize).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bh_matches_r() {
        let p = [
            Some(0.01),
            Some(0.04),
            Some(0.03),
            Some(0.5),
            None,
            Some(0.2),
        ];
        let q = bh_adjust(&p);
        let expect = [
            0.05,
            0.06666666666666667,
            0.06666666666666667,
            0.5,
            f64::NAN,
            0.25,
        ];
        for i in 0..p.len() {
            match (q[i], expect[i]) {
                (None, e) => assert!(e.is_nan()),
                (Some(v), e) => assert!((v - e).abs() < 1e-12, "{i}: {v} vs {e}"),
            }
        }
    }

    fn two_intron(n: usize, counts: Vec<u32>) -> Cluster {
        Cluster {
            name: "a".into(),
            introns: vec!["i1".into(), "i2".into()],
            annotations: vec![],
            n,
            counts,
        }
    }

    #[test]
    fn filters_replicate_python_reasons() {
        let params = DsParams {
            min_samples_per_group: 4,
            ..Default::default()
        };
        let n = 12;
        let x: Vec<u8> = (0..n).map(|i| if i < 6 { 0 } else { 1 }).collect();
        let pheno = binary_phenotype(&x, "ctrl", "case");
        let c = Cluster {
            name: "a".into(),
            introns: vec!["i1".into()],
            annotations: vec![],
            n,
            counts: vec![5; n],
        };
        assert_eq!(
            prepare_cluster(&c, &pheno, None, &params).unwrap_err(),
            "<=1 junction in cluster"
        );
        let big = Cluster {
            name: "a".into(),
            introns: (0..11).map(|i| format!("i{i}")).collect(),
            annotations: vec![],
            n,
            counts: vec![5; n * 11],
        };
        let p10 = DsParams {
            max_cluster_size: 10,
            ..params.clone()
        };
        assert_eq!(
            prepare_cluster(&big, &pheno, None, &p10).unwrap_err(),
            "Too many introns in cluster"
        );
        assert_eq!(
            prepare_cluster(&two_intron(n, vec![1; n * 2]), &pheno, None, &params).unwrap_err(),
            "<=1 sample with coverage>min_coverage"
        );
        let mut counts = vec![0u32; n * 2];
        for i in 0..n {
            counts[i * 2] = 30;
            if i < 3 {
                counts[i * 2 + 1] = 4;
            }
        }
        assert_eq!(
            prepare_cluster(&two_intron(n, counts), &pheno, None, &params).unwrap_err(),
            "<2 introns used in >=min_samples_per_intron samples"
        );
        let mut counts = vec![0u32; n * 2];
        for i in 0..n {
            counts[i * 2] = if i < 6 { 30 } else { 5 };
            counts[i * 2 + 1] = 5;
        }
        assert_eq!(
            prepare_cluster(&two_intron(n, counts), &pheno, None, &params).unwrap_err(),
            "Not enough valid samples"
        );
        // success, with a constant confounder column dropped; design order [1, conf, group]
        let counts: Vec<u32> = (0..n).flat_map(|i| [30 + i as u32, 10]).collect();
        let c = two_intron(n, counts);
        let conf = Design::from_columns(
            n,
            &[&vec![1.0; n], &(0..n).map(|i| i as f64).collect::<Vec<_>>()],
        );
        let prep = prepare_cluster(&c, &pheno, Some(&conf), &params).unwrap();
        assert_eq!(prep.x_full.p, 3);
        assert_eq!(prep.null_cols, vec![0, 1]);
        assert_eq!(prep.group_names, vec!["case".to_string()]);
        assert_eq!(
            prep.x_full.column(2),
            x.iter().map(|&v| v as f64).collect::<Vec<_>>()
        );
        let r = test_cluster(&c, &pheno, Some(&conf), &params, None);
        assert!(r.is_success(), "{}", r.status);
        assert_eq!(r.df, Some(1));
        assert!(r.introns[0].effects.contains_key("case"));
    }

    #[test]
    fn three_groups_and_continuous() {
        let n = 30;
        let codes: Vec<usize> = (0..n).map(|i| i % 3).collect();
        let pheno = Phenotype::Categorical {
            levels: vec!["a".into(), "b".into(), "c".into()],
            codes,
        };
        let counts: Vec<u32> = (0..n)
            .flat_map(|i| [30 + (i % 3) as u32 * 10, 10 + (i % 2) as u32])
            .collect();
        let c = two_intron(n, counts);
        let params = DsParams::default();
        let prep = prepare_cluster(&c, &pheno, None, &params).unwrap();
        assert_eq!(prep.group_names, vec!["b".to_string(), "c".to_string()]);
        let r = test_cluster(&c, &pheno, None, &params, None);
        assert!(r.is_success());
        assert_eq!(r.df, Some(2));
        // continuous
        let values: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let pheno = Phenotype::Continuous {
            values,
            scale_factor: 1.0,
        };
        let prep = prepare_cluster(&c, &pheno, None, &params).unwrap();
        assert_eq!(prep.group_names, vec!["x".to_string()]);
        let r = test_cluster(&c, &pheno, None, &params, None);
        assert!(r.is_success());
        assert_eq!(r.df, Some(1));
        // too few unique values
        let values: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let pheno = Phenotype::Continuous {
            values,
            scale_factor: 1.0,
        };
        assert_eq!(
            prepare_cluster(&c, &pheno, None, &params).unwrap_err(),
            "Not enough valid samples"
        );
    }
}
