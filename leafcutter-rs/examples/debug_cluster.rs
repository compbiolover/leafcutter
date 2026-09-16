//! Trace the null/full fits of one cluster: `cargo run --release --example debug_cluster -- counts groups cluster_name`
use leafcutter_rs::design::ClusterData;
use leafcutter_rs::ds::{prepare_cluster, DsParams};
use leafcutter_rs::fit::{fit_model, smart_init};
use leafcutter_rs::io;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let table = io::read_counts(Path::new(&args[1])).unwrap();
    let meta = io::read_groups(Path::new(&args[2])).unwrap();
    let enc = io::encode_design(&meta.groups, &meta.confounders).unwrap();
    let cols = io::sample_indices(&table.samples, &meta.samples).unwrap();
    let clusters = io::clusters_from_table(&table, &cols).unwrap();
    let c = clusters.iter().find(|c| c.name == args[3]).expect("cluster not found");
    let mut params = DsParams::default();
    params.fit.lbfgs.trace = true;
    let prep = prepare_cluster(c, &enc.x, enc.confounders.as_ref(), &params).unwrap();
    eprintln!("cluster {} n={} k={} p={}", c.name, prep.n, prep.k, prep.x_full.p);
    for i in 0..prep.n {
        eprintln!("  x={:?} y={:?}", prep.x_full.row(i), &prep.counts[i * prep.k..(i + 1) * prep.k]);
    }
    let x_null = prep.x_full.select_columns(&prep.null_cols);
    let dn = ClusterData::build(&prep.counts, prep.n, prep.k, &x_null);
    let df = ClusterData::build(&prep.counts, prep.n, prep.k, &prep.x_full);
    let b0 = smart_init(&prep.counts, prep.n, prep.k, &x_null, params.fit.smart_init_regularizer);
    eprintln!("--- null fit");
    let fnull = fit_model(&dn, &b0, &vec![10.0; prep.k], &params.fit);
    eprintln!("null value {:.8} conc {:?} beta {:?} conv {}", fnull.value, fnull.conc, fnull.beta, fnull.converged);
    eprintln!("--- null restart");
    let fnull2 = fit_model(&dn, &fnull.beta, &fnull.conc, &params.fit);
    eprintln!("null restart value {:.8} (gain {:.3e})", fnull2.value, fnull2.value - fnull.value);
    let mut bf = vec![0.0; prep.x_full.p * prep.k];
    for (i, &col) in prep.null_cols.iter().enumerate() {
        bf[col * prep.k..(col + 1) * prep.k].copy_from_slice(fnull.beta_row(i));
    }
    eprintln!("--- full fit");
    let ffull = fit_model(&df, &bf, &fnull.conc, &params.fit);
    eprintln!("full value {:.8} conc {:?} beta {:?} conv {}", ffull.value, ffull.conc, ffull.beta, ffull.converged);
    eprintln!("--- full restart");
    let ffull2 = fit_model(&df, &ffull.beta, &ffull.conc, &params.fit);
    eprintln!("full restart value {:.8} (gain {:.3e})", ffull2.value, ffull2.value - ffull.value);
    let ffull3 = fit_model(&df, &ffull2.beta, &ffull2.conc, &params.fit);
    eprintln!("full restart2 value {:.8} (gain {:.3e})", ffull3.value, ffull3.value - ffull2.value);
    // optional: start the full fit from an external solution {"beta":[[..],[..]],"conc":[..]}
    if let Some(path) = args.get(4) {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let beta: Vec<f64> = v["beta"].as_array().unwrap().iter().flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap())).collect();
        let conc: Vec<f64> = v["conc"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let model = leafcutter_rs::dm::DmModel::new(&df, params.fit.conc_shape, params.fit.conc_rate);
        let mut theta = beta.clone();
        theta.extend(conc.iter().map(|c| c.ln()));
        let mut g = vec![0.0; theta.len()];
        let ll = model.log_posterior(&theta, &mut g);
        eprintln!("--- external solution: rust objective {:.8} |g|={:.3e}", ll, g.iter().map(|x| x * x).sum::<f64>().sqrt());
        let f = fit_model(&df, &beta, &conc, &params.fit);
        eprintln!("full fit from external value {:.8} conc {:?} beta {:?}", f.value, f.conc, f.beta);
    }
}
