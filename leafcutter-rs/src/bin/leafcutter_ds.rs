//! Command line / child-process front end for the LeafCutter differential splicing port.
//!
//! * `run`         – the `leafcutter-ds` workflow on a counts file or a count store.
//! * `json`        – one request on stdin, one response on stdout (for use as a Node child
//!   process, e.g. from ProteinPaint).
//! * `build-store` – convert a counts file into a memory-mapped cluster-major store.
//! * `simulate`    – write a synthetic dataset for testing and benchmarking.

use clap::{Args, Parser, Subcommand};
use leafcutter_rs::design::Design;
use leafcutter_rs::ds::{self, Cluster, ClusterResult, DsParams};
use leafcutter_rs::fit::{FitParams, InitStrategy};
use leafcutter_rs::genes::{map_clusters_to_genes, read_exons};
use leafcutter_rs::io::{self, Phenotype};
use leafcutter_rs::nullcache::{fingerprint, NullCache};
use leafcutter_rs::simulate::{simulate, SimParams};
use leafcutter_rs::store::{build_store, Store};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "leafcutter_ds",
    version,
    about = "LeafCutter differential splicing (Rust port of leafcutter-ds)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run differential splicing on a counts file (or store) and a groups file.
    Run(RunArgs),
    /// Read a JSON request from stdin and write a JSON response to stdout.
    Json {
        /// Number of threads (overridden by the request's "threads").
        #[arg(short = 'p', long, default_value_t = 0)]
        threads: usize,
    },
    /// Build a memory-mapped cluster-major count store from a counts file.
    BuildStore {
        counts_file: PathBuf,
        store: PathBuf,
    },
    /// Simulate a Dirichlet-multinomial dataset (writes <prefix>_perind_numers.counts.gz and <prefix>_groups.txt).
    Simulate {
        #[arg(short = 'o', long, default_value = "sim")]
        output_prefix: String,
        #[arg(short = 'n', long, default_value_t = 100)]
        samples: usize,
        #[arg(short = 'm', long, default_value_t = 1000)]
        clusters: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 0.1)]
        frac_differential: f64,
        #[arg(long, default_value_t = 40.0)]
        mean_depth: f64,
        /// Also write a count store next to the counts file.
        #[arg(long)]
        store: bool,
    },
}

#[derive(Args, Clone)]
struct FilterArgs {
    /// Only for categorical phenotypes: the group the others are compared with.
    #[arg(short = '0', long, default_value = "Control")]
    baseline_group: String,
    /// Don't test clusters with more introns than this.
    #[arg(short = 's', long, default_value_t = usize::MAX)]
    max_cluster_size: usize,
    /// Ignore introns used (>= 1 read) in fewer than this many samples.
    #[arg(short = 'i', long, default_value_t = 5)]
    min_samples_per_intron: usize,
    /// Categorical phenotype: require this many samples per group with at least min_coverage reads.
    #[arg(short = 'g', long, default_value_t = 3)]
    min_samples_per_group: usize,
    /// Read threshold for min_samples_per_group.
    #[arg(short = 'c', long, default_value_t = 20)]
    min_coverage: u32,
    /// Continuous phenotype: require this many distinct values among covered samples.
    #[arg(short = 'u', long, default_value_t = 10)]
    min_unique_vals: usize,
    /// Initialisation: brr (Bayesian ridge regression), rr (ridge regression), mult
    /// (multinomial logistic regression) or 0.
    #[arg(long, default_value = "brr")]
    init: String,
    /// Reproduce the R package's procedure with rstan's stopping rules and a single start.
    #[arg(long)]
    like_r: bool,
    /// The R package's procedure with this crate's tighter stopping rules.
    #[arg(long)]
    r_procedure: bool,
    /// Maximum L-BFGS iterations per fit (leafcutter-ds: 500).
    #[arg(long)]
    max_iter: Option<usize>,
}

impl FilterArgs {
    fn to_params(&self) -> Result<DsParams, String> {
        let mut fit = if self.like_r {
            FitParams::r_exact()
        } else if self.r_procedure {
            FitParams::r_like()
        } else {
            FitParams::python()
        };
        if !self.like_r && !self.r_procedure {
            fit.init = self.init.parse::<InitStrategy>()?;
        }
        if let Some(m) = self.max_iter {
            fit.max_iter = m;
        }
        Ok(DsParams {
            max_cluster_size: self.max_cluster_size,
            min_samples_per_intron: self.min_samples_per_intron,
            min_samples_per_group: self.min_samples_per_group,
            min_coverage: self.min_coverage,
            min_unique_vals: self.min_unique_vals,
            fit,
        })
    }
}

#[derive(Args)]
struct RunArgs {
    /// Counts file (perind_numers.counts[.gz] / junction_counts.gz) or a count store built with build-store.
    counts: PathBuf,
    /// Groups file: sample, group (labels or numeric), [confounders...]; no header.
    groups: PathBuf,
    #[arg(short = 'o', long, default_value = "leafcutter_ds")]
    output_prefix: String,
    /// Number of threads (0 = all cores).
    #[arg(short = 'p', long, default_value_t = 0)]
    threads: usize,
    /// Exon table (chr, start, end, strand, gene_name) used only to label clusters with genes.
    #[arg(short = 'e', long)]
    exon_file: Option<PathBuf>,
    #[command(flatten)]
    filters: FilterArgs,
    /// Path of a null-fit cache file to read/update.
    #[arg(long)]
    null_cache: Option<PathBuf>,
    /// Also write <prefix>_results.json with the full results.
    #[arg(long)]
    json: bool,
}

/// JSON request for `leafcutter_ds json`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    /// Counts file (perind_numers.counts[.gz]).
    #[serde(default)]
    counts_file: Option<PathBuf>,
    /// Count store built with `build-store`. One of counts_file / store is required.
    #[serde(default)]
    store: Option<PathBuf>,
    /// Groups file, alternative to samples/groups/confounders.
    #[serde(default)]
    groups_file: Option<PathBuf>,
    /// Sample names (subset of the cohort), parallel to `groups`.
    #[serde(default)]
    samples: Vec<String>,
    /// Group label per sample (any number of labels; all numeric = continuous phenotype).
    #[serde(default)]
    groups: Vec<String>,
    /// Confounder columns, each parallel to `samples`; numeric columns are standardised,
    /// others treated as categorical.
    #[serde(default)]
    confounders: Vec<Vec<String>>,
    /// Baseline group for categorical phenotypes.
    #[serde(default = "default_baseline")]
    baseline_group: String,
    /// Exon table for gene labels.
    #[serde(default)]
    exon_file: Option<PathBuf>,
    #[serde(default)]
    threads: Option<usize>,
    #[serde(default)]
    params: DsParams,
    /// Use the R package's procedure and rstan's stopping rules.
    #[serde(default)]
    like_r: bool,
    #[serde(default)]
    null_cache: Option<PathBuf>,
    /// Drop per-intron effect sizes from the response.
    #[serde(default)]
    omit_introns: bool,
    /// Only report clusters with status "Success".
    #[serde(default)]
    only_success: bool,
}

fn default_baseline() -> String {
    "Control".into()
}

#[derive(Serialize)]
struct Response {
    /// Baseline group (categorical) or "mean" (continuous).
    baseline: String,
    /// Non-baseline groups, in the order of the effect size columns.
    groups: Vec<String>,
    continuous: bool,
    n_samples: usize,
    n_clusters: usize,
    confounder_columns: Vec<String>,
    dropped_samples: Vec<String>,
    summary: BTreeMap<String, usize>,
    timing_ms: BTreeMap<String, u128>,
    clusters: Vec<ClusterResult>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn init_threads(threads: usize) {
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }
}

/// Where the counts come from.
enum Source {
    Table(io::CountsTable),
    Store(Store),
}

impl Source {
    fn open(path: &Path) -> Result<Source, String> {
        let mut magic = [0u8; 4];
        let is_store = std::fs::File::open(path)
            .and_then(|mut f| f.read_exact(&mut magic))
            .map(|_| &magic == b"LCCS")
            .unwrap_or(false);
        if is_store {
            Store::open(path).map(Source::Store)
        } else {
            io::read_counts(path).map(Source::Table)
        }
    }
    fn samples(&self) -> &[String] {
        match self {
            Source::Table(t) => &t.samples,
            Source::Store(s) => &s.header.samples,
        }
    }
}

struct Analysis {
    results: Vec<ClusterResult>,
    baseline: String,
    groups: Vec<String>,
    continuous: bool,
    confounder_names: Vec<String>,
    dropped_samples: Vec<String>,
    n_samples: usize,
    timing: BTreeMap<String, u128>,
    with_annotations: bool,
}

fn analyse(
    source: &Source,
    meta: &io::Meta,
    baseline: &str,
    params: &DsParams,
    exon_file: Option<&Path>,
    null_cache_path: Option<&Path>,
) -> Result<Analysis, String> {
    let mut timing = BTreeMap::new();
    let t0 = Instant::now();
    let enc = io::encode_design(meta, baseline)?;
    if !enc.dropped_samples.is_empty() {
        eprintln!(
            "Samples removed due to missing values in covariates: {}",
            enc.dropped_samples.join(",")
        );
    }
    if enc.baseline_missing {
        if let Phenotype::Categorical { levels, .. } = &enc.phenotype {
            eprintln!("Warning: baseline group '{baseline}' is not among the group labels; using '{}' as baseline", levels[0]);
        }
    }
    let samples: Vec<String> = enc
        .sample_rows
        .iter()
        .map(|&i| meta.samples[i].clone())
        .collect();
    let cols = io::sample_indices(source.samples(), &samples)?;
    let n = cols.len();
    if let Phenotype::Categorical { levels, codes } = &enc.phenotype {
        let mut per_level = vec![0usize; levels.len()];
        for &c in codes {
            per_level[c] += 1;
        }
        let ok1 = per_level
            .iter()
            .filter(|&&c| c >= params.min_samples_per_intron)
            .count();
        let ok2 = per_level
            .iter()
            .filter(|&&c| c >= params.min_samples_per_group)
            .count();
        if ok1 < 2 || ok2 < 2 {
            return Err("There are no groups with enough samples to test. You can reduce min_samples_per_intron (-i) or min_samples_per_group (-g), but note that the calibration of leafcutter p-values has only been checked down to n=4 samples per group.".into());
        }
    }
    let cache = null_cache_path.map(|p| {
        let key = fingerprint(&(
            &samples,
            &meta.confounders,
            params.max_cluster_size,
            params.min_samples_per_intron,
            params.min_coverage,
            params.fit.conc_shape.to_bits(),
            params.fit.conc_rate.to_bits(),
            params.fit.eps.to_bits(),
            params.fit.conc_max.to_bits(),
        ));
        NullCache::open(p, key)
    });
    let confounders: Option<&Design> = enc.confounders.as_ref();
    let exons = match exon_file {
        Some(p) => Some(read_exons(p)?),
        None => None,
    };
    let t1 = Instant::now();
    let (mut results, with_annotations) = match source {
        Source::Table(table) => {
            let clusters = io::clusters_from_table(table, &cols)?;
            timing.insert("gather_ms".into(), t1.elapsed().as_millis());
            let t2 = Instant::now();
            let mut r = ds::run(
                &clusters,
                &enc.phenotype,
                confounders,
                params,
                cache.as_ref(),
            );
            timing.insert("fit_ms".into(), t2.elapsed().as_millis());
            if let Some(ex) = &exons {
                let map = map_clusters_to_genes(&clusters, ex);
                for res in r.iter_mut() {
                    res.genes = map.get(&res.cluster).cloned();
                }
            }
            (r, clusters.iter().any(|c| !c.annotations.is_empty()))
        }
        Source::Store(store) => {
            use rayon::prelude::*;
            let t2 = Instant::now();
            let mut r: Vec<ClusterResult> = (0..store.n_clusters())
                .into_par_iter()
                .map(|i| {
                    let c: Cluster = store.cluster(i, &cols);
                    ds::test_cluster(&c, &enc.phenotype, confounders, params, cache.as_ref())
                })
                .collect();
            ds::finalize(&mut r);
            timing.insert("fit_ms".into(), t2.elapsed().as_millis());
            if let Some(ex) = &exons {
                // gene labels need only the intron names, which the header carries
                let shells: Vec<Cluster> = store
                    .header
                    .clusters
                    .iter()
                    .map(|m| Cluster {
                        name: m.name.clone(),
                        introns: m.introns.clone(),
                        annotations: vec![],
                        n: 0,
                        counts: vec![],
                    })
                    .collect();
                let map = map_clusters_to_genes(&shells, ex);
                for res in r.iter_mut() {
                    res.genes = map.get(&res.cluster).cloned();
                }
            }
            let ann = store
                .header
                .clusters
                .iter()
                .any(|m| m.introns.iter().any(|s| s.matches(':').count() >= 4));
            (r, ann)
        }
    };
    let _ = &mut results;
    if let Some(c) = &cache {
        c.save()
            .map_err(|e| format!("cannot save null cache: {e}"))?;
    }
    timing.insert("total_ms".into(), t0.elapsed().as_millis());
    let (baseline_label, continuous) = match &enc.phenotype {
        Phenotype::Categorical { levels, .. } => (levels[0].clone(), false),
        Phenotype::Continuous { .. } => (baseline.to_string(), true),
    };
    Ok(Analysis {
        results,
        baseline: baseline_label,
        groups: enc.phenotype.group_names(),
        continuous,
        confounder_names: enc.confounder_names,
        dropped_samples: enc.dropped_samples,
        n_samples: n,
        timing,
        with_annotations,
    })
}

fn summary(results: &[ClusterResult]) -> BTreeMap<String, usize> {
    ds::status_summary(results).into_iter().collect()
}

fn cmd_run(a: RunArgs) -> Result<(), String> {
    init_threads(a.threads);
    let params = a.filters.to_params()?;
    eprintln!("Loading counts from {}", a.counts.display());
    let source = Source::open(&a.counts)?;
    eprintln!("Loading metadata from {}", a.groups.display());
    let meta = io::read_groups(&a.groups)?;
    eprintln!(
        "Running differential splicing analysis on {} threads...",
        rayon::current_num_threads()
    );
    let an = analyse(
        &source,
        &meta,
        &a.filters.baseline_group,
        &params,
        a.exon_file.as_deref(),
        a.null_cache.as_deref(),
    )?;
    eprintln!("Differential splicing summary:");
    for (s, c) in ds::status_summary(&an.results) {
        eprintln!("  {c:>8}  {s}");
    }
    eprintln!("Timing: {:?}", an.timing);
    let sig = PathBuf::from(format!("{}_cluster_significance.txt", a.output_prefix));
    let eff = PathBuf::from(format!("{}_effect_sizes.txt", a.output_prefix));
    io::write_cluster_table(&sig, &an.results, an.with_annotations).map_err(|e| e.to_string())?;
    io::write_effect_sizes(&eff, &an.results, &an.groups, &an.baseline)
        .map_err(|e| e.to_string())?;
    if a.json {
        let resp = Response {
            baseline: an.baseline.clone(),
            groups: an.groups.clone(),
            continuous: an.continuous,
            n_samples: an.n_samples,
            n_clusters: an.results.len(),
            confounder_columns: an.confounder_names.clone(),
            dropped_samples: an.dropped_samples.clone(),
            summary: summary(&an.results),
            timing_ms: an.timing.clone(),
            clusters: an.results,
        };
        let path = format!("{}_results.json", a.output_prefix);
        std::fs::write(&path, serde_json::to_vec(&resp).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    }
    eprintln!("Wrote {} and {}", sig.display(), eff.display());
    Ok(())
}

fn cmd_json(default_threads: usize) -> Result<(), String> {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let mut req: Request =
        serde_json::from_str(&input).map_err(|e| format!("invalid request: {e}"))?;
    init_threads(req.threads.unwrap_or(default_threads));
    if req.like_r {
        req.params.fit = FitParams::r_exact();
    }
    let path = req
        .counts_file
        .as_ref()
        .or(req.store.as_ref())
        .ok_or("request needs counts_file or store")?;
    let source = Source::open(path)?;
    let meta = match &req.groups_file {
        Some(g) => io::read_groups(g)?,
        None => {
            if req.samples.is_empty() || req.samples.len() != req.groups.len() {
                return Err("samples and groups must be non-empty and of equal length".into());
            }
            for (i, c) in req.confounders.iter().enumerate() {
                if c.len() != req.samples.len() {
                    return Err(format!(
                        "confounder column {i} has {} entries, expected {}",
                        c.len(),
                        req.samples.len()
                    ));
                }
            }
            io::Meta {
                samples: req.samples.clone(),
                groups: req.groups.clone(),
                confounders: req.confounders.clone(),
            }
        }
    };
    let mut an = analyse(
        &source,
        &meta,
        &req.baseline_group,
        &req.params,
        req.exon_file.as_deref(),
        req.null_cache.as_deref(),
    )?;
    let summary = summary(&an.results);
    let n_clusters = an.results.len();
    if req.only_success {
        an.results.retain(|r| r.is_success());
    }
    if req.omit_introns {
        for r in an.results.iter_mut() {
            r.introns.clear();
        }
    }
    let resp = Response {
        baseline: an.baseline,
        groups: an.groups,
        continuous: an.continuous,
        n_samples: an.n_samples,
        n_clusters,
        confounder_columns: an.confounder_names,
        dropped_samples: an.dropped_samples,
        summary,
        timing_ms: an.timing,
        clusters: an.results,
    };
    let out = std::io::stdout();
    let mut lock = out.lock();
    serde_json::to_writer(&mut lock, &resp).map_err(|e| e.to_string())?;
    lock.write_all(b"\n").map_err(|e| e.to_string())?;
    Ok(())
}

fn cmd_simulate(
    prefix: &str,
    n: usize,
    m: usize,
    seed: u64,
    frac: f64,
    depth: f64,
    store: bool,
) -> Result<(), String> {
    let sim = simulate(&SimParams {
        n_samples: n,
        n_clusters: m,
        seed,
        frac_differential: frac,
        mean_depth: depth,
        ..Default::default()
    });
    let counts = PathBuf::from(format!("{prefix}_perind_numers.counts.gz"));
    io::write_counts(&counts, &sim.samples, &sim.clusters).map_err(|e| e.to_string())?;
    let groups = format!("{prefix}_groups.txt");
    let mut g = String::new();
    for (s, gr) in sim.samples.iter().zip(&sim.group) {
        g.push_str(&format!(
            "{s}\t{}\n",
            if *gr == 0 { "control" } else { "case" }
        ));
    }
    std::fs::write(&groups, g).map_err(|e| e.to_string())?;
    let truth = format!("{prefix}_truth.txt");
    let mut t = String::from("cluster\tdifferential\n");
    for (c, d) in sim.clusters.iter().zip(&sim.differential) {
        t.push_str(&format!("{}\t{}\n", c.name, *d as u8));
    }
    std::fs::write(&truth, t).map_err(|e| e.to_string())?;
    if store {
        let path = PathBuf::from(format!("{prefix}.lcs"));
        leafcutter_rs::store::write_store(&sim.samples, &sim.clusters, &path)
            .map_err(|e| e.to_string())?;
        eprintln!("Wrote {}", path.display());
    }
    eprintln!("Wrote {} , {groups} and {truth}", counts.display());
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let res = match cli.cmd {
        Cmd::Run(a) => cmd_run(a),
        Cmd::Json { threads } => {
            let r = cmd_json(threads);
            if let Err(e) = &r {
                println!(
                    "{}",
                    serde_json::to_string(&ErrorResponse { error: e.clone() }).unwrap()
                );
            }
            r
        }
        Cmd::BuildStore { counts_file, store } => {
            let t = Instant::now();
            io::read_counts(&counts_file)
                .and_then(|table| build_store(&table, &store))
                .map(|_| {
                    eprintln!(
                        "Wrote {} in {:.1}s",
                        store.display(),
                        t.elapsed().as_secs_f64()
                    );
                })
        }
        Cmd::Simulate {
            output_prefix,
            samples,
            clusters,
            seed,
            frac_differential,
            mean_depth,
            store,
        } => cmd_simulate(
            &output_prefix,
            samples,
            clusters,
            seed,
            frac_differential,
            mean_depth,
            store,
        ),
    };
    if let Err(e) = res {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
