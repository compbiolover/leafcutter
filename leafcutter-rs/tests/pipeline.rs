//! End-to-end checks on simulated data: statuses, power, calibration, store round trip,
//! null-fit cache reuse and JSON serialisation of results.

use leafcutter_rs::ds::{self, binary_phenotype, DsParams};
use leafcutter_rs::nullcache::NullCache;
use leafcutter_rs::simulate::{simulate, SimParams};
use leafcutter_rs::store::{write_store, Store};

fn sim(n: usize, m: usize, seed: u64) -> leafcutter_rs::simulate::SimData {
    simulate(&SimParams {
        n_samples: n,
        n_clusters: m,
        seed,
        ..Default::default()
    })
}

#[test]
fn recovers_simulated_effects_and_is_calibrated() {
    let data = sim(60, 400, 7);
    let x = binary_phenotype(&data.group, "control", "case");
    let params = DsParams::default();
    let results = ds::run(&data.clusters, &x, None, &params, None);
    assert_eq!(results.len(), 400);
    let tested: Vec<_> = results.iter().filter(|r| r.is_success()).collect();
    assert!(tested.len() > 380, "only {} clusters tested", tested.len());
    let truth: std::collections::HashMap<&str, bool> = data
        .clusters
        .iter()
        .zip(&data.differential)
        .map(|(c, &d)| (c.name.as_str(), d))
        .collect();
    let sig: Vec<_> = tested
        .iter()
        .filter(|r| r.p_adjust.unwrap() < 0.05)
        .collect();
    let tp = sig.iter().filter(|r| truth[r.cluster.as_str()]).count();
    let n_true = tested.iter().filter(|r| truth[r.cluster.as_str()]).count();
    assert!(
        tp as f64 >= 0.6 * n_true as f64,
        "power too low: {tp}/{n_true}"
    );
    assert!(
        (sig.len() - tp) as f64 <= 0.15 * sig.len() as f64 + 2.0,
        "too many false discoveries: {}/{}",
        sig.len() - tp,
        sig.len()
    );
    // null p-values roughly uniform: at most ~10% below 0.05
    let null_p: Vec<f64> = tested
        .iter()
        .filter(|r| !truth[r.cluster.as_str()])
        .map(|r| r.p.unwrap())
        .collect();
    let frac = null_p.iter().filter(|&&p| p < 0.05).count() as f64 / null_p.len() as f64;
    assert!(frac < 0.10, "null p < 0.05 fraction {frac}");
    for r in &tested {
        assert!(r.converged, "{} did not converge", r.cluster);
        assert!(
            r.loglr.unwrap() > -1e-6,
            "{} negative loglr {}",
            r.cluster,
            r.loglr.unwrap()
        );
        let s: f64 = r.introns.iter().map(|i| i.effects["case"].deltapsi).sum();
        assert!(s.abs() < 1e-8);
    }
}

#[test]
fn store_roundtrip_matches_in_memory_and_supports_subsets() {
    let data = sim(40, 50, 3);
    let dir = std::env::temp_dir().join(format!("lcrs_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sim.lcs");
    write_store(&data.samples, &data.clusters, &path).unwrap();
    let store = Store::open(&path).unwrap();
    let params = DsParams::default();
    let x = binary_phenotype(&data.group, "control", "case");
    let all: Vec<usize> = (0..40).collect();
    for i in 0..store.n_clusters() {
        let c = store.cluster(i, &all);
        assert_eq!(c.counts, data.clusters[i].counts);
        let a = ds::test_cluster(&c, &x, None, &params, None);
        let b = ds::test_cluster(&data.clusters[i], &x, None, &params, None);
        assert_eq!(a.status, b.status);
        assert_eq!(a.loglr, b.loglr);
    }
    // a subset in shuffled order
    let idx = [
        5usize, 3, 30, 12, 1, 22, 9, 15, 18, 7, 8, 33, 34, 35, 36, 37,
    ];
    let c = store.cluster(0, &idx);
    assert_eq!(c.n, idx.len());
    for (r, &s) in idx.iter().enumerate() {
        let k = c.k();
        assert_eq!(
            &c.counts[r * k..(r + 1) * k],
            &data.clusters[0].counts[s * k..(s + 1) * k]
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn null_cache_is_reused_and_gives_identical_results() {
    let data = sim(50, 60, 11);
    let x = binary_phenotype(&data.group, "control", "case");
    let params = DsParams::default();
    let cache = NullCache::in_memory(42);
    let first = ds::run(&data.clusters, &x, None, &params, Some(&cache));
    let n_cached = cache.len();
    assert!(n_cached > 0);
    let second = ds::run(&data.clusters, &x, None, &params, Some(&cache));
    assert_eq!(cache.len(), n_cached);
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(a.status, b.status);
        // The cache holds the best null fit seen. For clusters whose null was not refitted from
        // the full solution the cached fit is bit-identical to the one used in the first run,
        // so the warm-started full fit and the test statistic are reproduced exactly.
        if let (Some(la), Some(lb)) = (a.loglr, b.loglr) {
            if !a.refit_null {
                assert!((la - lb).abs() < 1e-9, "{}: {la} vs {lb}", a.cluster);
            }
        }
    }
    // runs that both use the cache are deterministic
    let again = ds::run(&data.clusters, &x, None, &params, Some(&cache));
    for (a, b) in second.iter().zip(&again) {
        assert_eq!(a.loglr, b.loglr, "{}", a.cluster);
    }
    // a different grouping reuses the same null fits
    let g2: Vec<u8> = (0..50).map(|i| if i % 4 < 2 { 0 } else { 1 }).collect();
    let x2 = binary_phenotype(&g2, "control", "case");
    let third = ds::run(&data.clusters, &x2, None, &params, Some(&cache));
    assert_eq!(cache.len(), n_cached);
    assert!(third.iter().any(|r| r.is_success()));
    // the evaluations counter only counts the full fit when the null came from the cache
    let e1: usize = first.iter().map(|r| r.evaluations).sum();
    let e3: usize = third.iter().map(|r| r.evaluations).sum();
    assert!(e3 < e1);
}

#[test]
fn results_serialise_to_json() {
    let data = sim(30, 5, 5);
    let x = binary_phenotype(&data.group, "control", "case");
    let results = ds::run(&data.clusters, &x, None, &DsParams::default(), None);
    let s = serde_json::to_string(&results).unwrap();
    let back: Vec<ds::ClusterResult> = serde_json::from_str(&s).unwrap();
    assert_eq!(back.len(), 5);
    assert!(s.contains("\"p.adjust\""));
}
