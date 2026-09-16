//! Time the per-cluster phases on a store: `cargo run --release --example profile_setup -- store groups [n_clusters]`
use leafcutter_rs::design::ClusterData;
use leafcutter_rs::ds::{prepare_cluster, test_cluster, DsParams};
use leafcutter_rs::fit::{fit_model, smart_init_collapsed};
use leafcutter_rs::io;
use leafcutter_rs::store::Store;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let store = Store::open(Path::new(&args[1])).unwrap();
    let meta = io::read_groups(Path::new(&args[2])).unwrap();
    let enc = io::encode_design(&meta.groups, &meta.confounders).unwrap();
    let cols = io::sample_indices(&store.header.samples, &meta.samples).unwrap();
    let m: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(200);
    let mut params = DsParams::default();
    params.fit.full_extra_starts = 0;
    let (mut t_gather, mut t_prep, mut t_build, mut t_init, mut t_fit1, mut t_full) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut evals = 0usize;
    for i in 0..m.min(store.n_clusters()) {
        let t = Instant::now();
        let c = store.cluster(i, &cols);
        t_gather += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let prep = match prepare_cluster(&c, &enc.x, None, &params) {
            Ok(p) => p,
            Err(_) => continue,
        };
        t_prep += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let df = ClusterData::build(&prep.counts, prep.n, prep.k, &prep.x_full);
        let dn = df.select_columns(&prep.null_cols);
        t_build += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let b0 = smart_init_collapsed(&dn, params.fit.smart_init_regularizer);
        t_init += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut p1 = params.fit.clone();
        p1.max_iter = 1;
        let f = fit_model(&dn, &b0, &vec![10.0; prep.k], &p1);
        evals += f.evaluations;
        t_fit1 += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let r = test_cluster(&c, &enc.x, None, &params, None);
        t_full += t.elapsed().as_secs_f64();
        let _ = r;
    }
    let ms = |x: f64| x * 1000.0 / m as f64;
    println!(
        "per cluster (ms): gather {:.3} prepare {:.3} build full+null {:.3} smart_init {:.3} null fit 1 iter {:.3} ({:.1} evals) | full test_cluster {:.3}",
        ms(t_gather), ms(t_prep), ms(t_build), ms(t_init), ms(t_fit1), evals as f64 / m as f64, ms(t_full)
    );
}
