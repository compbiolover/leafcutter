# leafcutter-rs — leafcutter-ds in Rust

A Rust port of [leafcutter-ds](https://github.com/leafcutter2/leafcutter-ds), the Python/Pyro
re-implementation of LeafCutter's differential splicing test that is compatible with
[leafcutter2](https://github.com/leafcutter2/leafcutter2) annotated clusters. It is built for
interactive use: genome-wide cluster p-values, q-values and per-intron ΔPSI for cohorts of
hundreds to tens of thousands of samples in seconds, with a small memory footprint, callable
from a Node server (e.g. a ProteinPaint portal) as a child process.

It also reproduces the original R package's procedure (`--like-r`), see below.

## What it computes

The statistical procedure of `leafcutter/differential_splicing/{differential_splicing,dm_glm,optim}.py`:

* the same per-cluster filters and skip reasons (`max_cluster_size`, `min_samples_per_intron`,
  `min_samples_per_group`, `min_coverage`, `min_unique_vals`);
* a categorical phenotype with any number of groups (`--baseline_group`, one design column
  per non-baseline group, empty groups dropped per cluster) or a continuous phenotype
  (standardised, one column); numeric confounders standardised, categorical confounders
  one-hot encoded with the first sorted level dropped, samples with missing confounders
  removed;
* a Dirichlet-multinomial GLM with one concentration per junction, `Gamma(1.0001, 1e-4)`
  concentration prior, improper flat prior on the coefficients, `conc` bounded by 3000 through
  a sigmoid and a `1e-8` pseudocount on the Dirichlet parameters, fitted by MAP with L-BFGS
  (`max_iter = 500`, `history = 100`, `tolerance_grad = 1e-7`, `tolerance_change = 1e-9`,
  strong-Wolfe line search: the settings leafcutter-ds passes to `torch.optim.LBFGS`);
* the initialisation strategies `brr` (a port of scikit-learn's `BayesianRidge`, the
  default), `rr`, `mult` and `0`;
* the fitting procedure of `dirichlet_multinomial_anova`: null fit from the init; full fit
  from the null solution and from a fresh init, keeping the better; every fit starts with
  `conc = 10`; likelihood ratio statistic against χ² with `(K−1)(P_full − P_null)` degrees of
  freedom; null refitted from the full solution when p < 0.001;
* Benjamini–Hochberg adjusted p-values (`scipy.stats.false_discovery_control`), plus Storey
  q-values in the JSON output;
* per-intron effect sizes: `logef_<group>`, `psi_<baseline>`, `psi_<group>`, `deltapsi_<group>`;
* gene labels from an exon table (`--exon_file`, same start/end matching rule as
  `map_clusters_to_genes`) and, for leafcutter2 input with `chr:start:end:clu:annotation`
  intron names, the `annotations` column.

Not ported: `leafcutter-bayes` (the spike-and-slab per-junction model), the plotting scripts
and the clustering step (use `leafcutter-cluster` / leafcutter2 for those).

### Differences from the Python implementation

* Arithmetic is `f64`; leafcutter-ds runs in `float32`, where torch's `tolerance_change = 1e-9`
  cannot be resolved at the scale of the objective (~1e3–1e5), so its fits stop early. The
  port converges further and reaches a higher posterior in most clusters (see Validation).
* The log posterior omits the Gamma prior's normalising constant (`K (shape·ln rate − lgamma
  shape)`), which Pyro includes; it cancels in the likelihood ratio.
* The R package's procedure is available with `--like-r` (rstan's stopping rules, ridge
  init, warm-started concentrations, single start) or `--r-procedure` (the same procedure
  with tighter stopping rules and an extra start).

## Performance design

* **Sufficient-statistic collapse.** Samples that share a design row share the fitted
  proportions, and the per-sample likelihood depends on the counts only through integers, so
  counts are histogrammed per (design cell, intron) and per (cell, row total). One objective
  evaluation costs `Σ_cells (K + distinct values)` `lgamma`/`digamma` pairs instead of `N × K`;
  with categorical designs the cost stops growing with `N`. A continuous phenotype or
  confounder makes every sample its own cell, which costs roughly 10× at N = 100 and grows
  linearly with N (5.8 s vs 0.6 s for the 2 000-cluster benchmark below, 4 threads).
* `lgamma`/`digamma` over consecutive integer shifts advance by a log and a division.
* The Bayesian-ridge initialiser works from sufficient statistics collected in one pass, so
  its 300-iteration loop costs `O(P²)` per junction regardless of `N`.
* **No dense matrices.** Counts are kept per cluster as `u32`; a memory-mapped cluster-major
  store (`build-store`) lets a request stream one cluster per thread and gather only the
  requested samples (rows are sample-major inside a cluster block, `u16` or `u32` per cluster).
* **rayon** over clusters; `-p` / `"threads"` selects the thread count.
* The null fit does not depend on the phenotype, so it can be cached per cohort
  (`--null-cache` / `null_cache`) and reused across requests that regroup the same samples.

Measured on 2 000 simulated clusters (`leafcutter_ds simulate`, 2–6 introns per cluster,
mean depth 40, two groups, no confounders) on a 4-core VM, fitting time only:

| samples | 1 thread | 4 threads | per cluster per core | `--like-r`, 4 threads |
|--------:|---------:|----------:|---------------------:|----------------------:|
|     100 |   2.4 s  |   0.60 s  |   1.2 ms             |   0.23 s              |
|     400 |   3.5 s  |   0.93 s  |   1.7 ms             |   0.22 s              |
|   1 000 |   4.4 s  |   1.1 s   |   2.2 ms             |   0.21 s              |
|  10 000 |   8.1 s  |   2.2 s   |   4.0 ms             |   0.39 s              |

Scaled to 30 000 clusters on 8 threads: about 5 s at N = 100 and 15 s at N = 10 000. On the
same machine and data, `leafcutter-ds -p 4` takes 99 s (N = 100) and 92 s (N = 400), and
19 s for the 289-cluster Geuvadis example (0.35 s here). A request for a 100-sample subset of
a 10 000-sample store costs the same as a 100-sample dataset.

Memory: only one cluster per thread is live. Reading a `.counts.gz` file keeps the whole table
in memory (`u32`, ~4 bytes × introns × samples), so for large cohorts build the store once and
run on it; the store is memory-mapped and its file-backed pages are reclaimable.

## Building

```
cd leafcutter-rs
cargo build --release          # binary: target/release/leafcutter_ds
cargo test --release           # unit + integration tests (simulated data)
```

## Command line

```
# same inputs and options as leafcutter-ds
leafcutter_ds run counts_perind_numers.counts.gz groups.txt -0 Control -o out -p 8 \
    [-e exons.txt.gz] [-s INF -i 5 -g 3 -c 20 -u 10] [--init brr] [--json]
#   -> out_cluster_significance.txt, out_effect_sizes.txt [, out_results.json]

# one-off conversion for large cohorts, then run on the store
leafcutter_ds build-store counts_perind_numers.counts.gz cohort.lcs
leafcutter_ds run cohort.lcs groups.txt -0 Control -o out --null-cache cohort_null.json

# synthetic data for tests / benchmarks (groups "control" / "case")
leafcutter_ds simulate -n 1000 -m 20000 -o sim --store
```

Defaults match the `leafcutter-ds` command line (`-s inf -i 5 -g 3 -c 20 -u 10 --init brr`,
baseline `Control`; a baseline that is not among the labels falls back to the first sorted
label with a warning). `--like-r` / `--r-procedure` select the R package's procedure.

## Child-process JSON protocol (Node)

`leafcutter_ds json` reads one request object from stdin and writes one response object to
stdout (errors are written as `{"error": "..."}` with a non-zero exit code):

```jsonc
{
  "store": "/data/cohort.lcs",              // or "counts_file": "...counts.gz"
  "samples": ["s1", "s2", "..."],           // any subset of the cohort, any order
  "groups":  ["ctrl", "caseA", "caseB"],    // labels (any number) or numbers (continuous)
  "baseline_group": "ctrl",
  "confounders": [["b1","b2","..."], ["31","45","..."]],   // optional; numeric -> standardised, else one-hot
  "exon_file": "/data/exons.txt.gz",        // optional, gene labels
  "threads": 8,
  "params": { "max_cluster_size": 10, "min_samples_per_intron": 5, "min_samples_per_group": 3,
              "min_coverage": 20, "min_unique_vals": 10,
              "fit": { "init": "brr", "max_iter": 500 } },     // all optional
  "like_r": false,
  "null_cache": "/data/cohort_null.json",   // optional
  "only_success": true, "omit_introns": false
}
```

Response:

```jsonc
{
  "baseline": "ctrl", "groups": ["caseA", "caseB"], "continuous": false,
  "n_samples": 512, "n_clusters": 31240,
  "confounder_columns": ["conf1=b2", "conf1=b3", "conf2"], "dropped_samples": [],
  "summary": { "Success": 28870, "Too many introns in cluster": 310, "...": 0 },
  "timing_ms": { "fit_ms": 4180, "total_ms": 4302 },
  "clusters": [
    { "cluster": "chr1:clu_12_NA", "status": "Success", "loglr": 8.31, "df": 4,
      "p": 2.5e-4, "p.adjust": 0.012, "q_storey": 0.009,
      "genes": "GENE1,GENE2", "annotations": ["PR", "UP"],
      "introns": [ { "intron": "chr1:1000:2000:clu_12_NA:PR", "psi_baseline": 0.31,
                     "effects": { "caseA": { "logef": 0.8, "psi": 0.52, "deltapsi": 0.21 },
                                  "caseB": { "logef": 0.1, "psi": 0.33, "deltapsi": 0.02 } } } ],
      "n_samples": 498, "refit_null": false, "smart_init_improved": true, "evaluations": 161,
      "value_null": -9123.4, "value_full": -9115.1, "converged": true }
  ]
}
```

Suggested deployment: build the store once per cohort, keep the binary next to the Node
server, spawn it per request with the sample subset and labels, and reuse `null_cache` for
requests that regroup the same cohort.

## Validation

### Against leafcutter-ds (Python)

`scripts/compare_tables.py` compares the two output tables of two runs;
`scripts/py_reference_fits.py` runs the Python model directly and dumps its per-cluster null
and full log posteriors (which its CLI does not write). On the two examples shipped with
leafcutter-ds and on simulated data, with identical options:

| dataset | clusters | skip reasons | genes / annotations | significant at p < 1e-3 | p < 0.05 disagreements | BH < 0.05 disagreements | `leafcutter-ds -p 4` | `leafcutter_ds -p 4` |
|--|--:|:-:|:-:|:-:|:-:|:-:|--:|--:|
| Geuvadis sample (58 samples, sex, population confounder) | 289 | identical | identical | identical (1) | 4 of 254 | 0 | 19 s | 0.35 s |
| leafcutter2 chr10 example (12 samples, annotated introns) | 951 | identical | identical | identical (125) | 1 of 419 | 0 | – | 0.13 s |
| simulated, N = 100 | 2 000 | identical | identical | identical (167) | 9 of 1 998 | 2 | 99 s | 0.59 s |
| simulated, N = 400 | 2 000 | identical | identical | 1 differs (192/193) | 40 of 2 000 | 2 | 92 s | 0.87 s |

The disagreements are clusters where the Python fits stopped early: after removing Pyro's
Gamma normalising constant, the Rust null and full log posteriors are higher than Python's
in 212 and 203 of the 254 tested Geuvadis clusters (by more than 0.1 in 89 and 70) and lower
by more than 1e-3 in only 11 and 20, never by more than 0.1. Effect sizes: PSI and ΔPSI agree
to a median of ~1e-4 (max 4e-3 on the leafcutter2 example); `logef` differs by more than
0.01 for introns whose usage is absent in one group, where the coefficient drifts towards
−∞ and both implementations stop at an arbitrary point.

```
leafcutter-ds -0 male -e exons.txt.gz -o py counts.gz groups.txt -p 4
leafcutter_ds run counts.gz groups.txt -0 male -e exons.txt.gz -o rs -p 4
python3 scripts/compare_tables.py py rs
```

### Against the R package

`scripts/run_r_reference.R` runs the R package's test (sources `leafcutter/R`, compiles the
Stan model with rstan) and `scripts/r_to_reference_json.py` + `scripts/compare_reference.py`
compare it with `leafcutter_ds run --json`. With `--r-procedure`, on 2 000 simulated clusters
at N = 100 / 400 / 1 000: identical skip reasons and identical significant sets at p < 1e-6;
11 / 14 / 27 disagreements at p < 0.05, every one a cluster where the Rust full fit reached a
higher posterior than rstan's (`rstan::optimizing`'s default tolerances also stop early in
the flat concentration directions). R takes 23 / 45 / 78 s on 4 threads for these, the port
0.9 / 1.0 / 1.2 s. To reproduce with a conda-forge R (two quirks: recent stanc no longer
accepts the package's old array syntax, which the script rewrites on the fly, and oneTBB
needs `-DTBB_INTERFACE_NEW` in `~/.R/Makevars` `CXX17FLAGS`):

```
micromamba create -n r -c conda-forge r-base r-rstan r-bh r-rcppeigen r-rcppparallel gxx_linux-64 make \
    r-foreach r-domc r-dplyr r-r.utils r-optparse r-magrittr
micromamba run -n r Rscript scripts/run_r_reference.R sim_perind_numers.counts.gz sim_groups.txt r_out 0 4
python3 scripts/r_to_reference_json.py r_out r.json
leafcutter_ds run sim_perind_numers.counts.gz sim_groups.txt -0 control -o rust --json --r-procedure
python3 scripts/compare_reference.py rust_results.json r.json
```

`scripts/reference_ds.py` is a third, independent numpy/autograd/scipy implementation of the
R/Stan model used during development.

Unit tests check the analytic gradient (both concentration parameterisations, the
pseudocount and the multinomial model) against finite differences, the collapsed objective
against the dense per-sample formula, `digamma` against the derivative of `lgamma`, the χ²
survival function against scipy, BH against `p.adjust`, the Bayesian ridge against
scikit-learn, the phenotype/confounder encoding, and every filter reason; integration tests
cover power and calibration on simulated data, the store round trip and the null cache.

## Layout

```
src/special.rs   lgamma / digamma / chi-square survival function
src/lbfgs.rs     L-BFGS with strong-Wolfe line search (torch and Stan stopping rules)
src/design.rs    design matrices and the collapsed cluster representation
src/dm.rs        Dirichlet-multinomial GLM objective + gradient, multinomial model
src/fit.rs       initialisers (brr / rr / mult / 0), null/full fits, LRT, effect sizes
src/ds.rs        filters, parallel driver, BH / Storey
src/io.rs        counts / groups files, phenotype encoding, result tables
src/genes.rs     exon table and cluster-to-gene labelling
src/store.rs     memory-mapped cluster-major count store
src/nullcache.rs per-cohort cache of null fits
src/simulate.rs  synthetic data
src/bin/leafcutter_ds.rs   CLI + JSON protocol
examples/                  debug_cluster (trace one cluster), profile_setup (per-phase timing)
scripts/                   compare_tables.py, py_reference_fits.py, run_r_reference.R,
                           r_to_reference_json.py, reference_ds.py, compare_reference.py
```
