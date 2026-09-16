# leafcutter-rs — LeafCutter differential splicing in Rust

A port of LeafCutter's differential splicing test (`leafcutter_ds.R`) to Rust, built for
interactive use: genome-wide cluster p-values, q-values and per-intron ΔPSI for a cohort
of hundreds to tens of thousands of samples in seconds, with a small memory footprint, so it
can be called from a Node server (e.g. a ProteinPaint portal) as a child process.

## What it computes

Exactly the statistical procedure of the R package (`leafcutter/R/differential_splicing.R`,
`dm_glm_multi_conc.R`, `inst/stan/dm_glm_multi_conc.stan`, non-robust model):

* per intron cluster, the same filters (`max_cluster_size`, `min_samples_per_intron`,
  `min_samples_per_group`, `min_coverage`) with the same skip reasons;
* a Dirichlet-multinomial GLM fitted by maximum a posteriori estimation (L-BFGS, analytic
  gradient) under the null design (intercept + confounders) and the full design (+ group),
  with the Gamma(1.0001, 1e-4) prior on the per-intron concentrations;
* the likelihood ratio statistic `2 (ℓ_full − ℓ_null)` against χ² with `(K−1)(P_full − P_null)`
  degrees of freedom, the R package's "smart" method-of-moments initialisation, warm start of
  the full fit from the null fit, and the "refit the null from the full solution if p < 0.001"
  rule;
* Benjamini–Hochberg adjusted p-values (`p.adjust(method = "fdr")`) and, in JSON output,
  Storey q-values (λ = 0.5);
* per-intron effect sizes as `leaf_cutter_effect_sizes`: log effect size, baseline PSI,
  perturbed PSI and ΔPSI.

Stan's `simplex × scale` encoding of the coefficient rows is replaced by free sum-to-zero rows
and log-concentrations; the objective and the test are invariant to that reparameterisation.
Two things go beyond the R code, both optional:

* `--full-extra-starts N` (default 1): the posterior can have several modes that differ in
  the concentrations, and a single L-BFGS run (R's behaviour, `N = 0`) lands in whichever mode
  its trajectory finds. Extra starting points for the full model keep the best mode.
* the null fit of a cluster does not depend on the group labels, so it can be cached per
  cohort (`--null-cache`, or `null_cache` in a JSON request) and reused across requests that
  regroup the same samples.

The `robust` outlier model is not ported.

## Performance design

* **Sufficient-statistic collapse.** Samples that share a design row share the fitted
  proportions, and the per-sample likelihood depends on the counts only through integers, so
  counts are histogrammed per (design cell, intron) and per (cell, row total). One objective
  evaluation costs `Σ_cells (K + distinct values)` `lgamma`/`digamma` pairs instead of `N × K`;
  with categorical designs the cost stops growing with `N`. Continuous covariates simply make
  every sample its own cell (the dense computation).
* **No dense matrices.** Counts are kept per cluster as `u32`; a memory-mapped cluster-major
  store (`build-store`) lets a request stream one cluster per thread and gather only the
  requested samples (rows are sample-major inside a cluster block, `u16` or `u32` per cluster).
* **rayon** over clusters; `-p` / `"threads"` selects the thread count.

Measured on 2 000 simulated clusters (4 cores of a small VM, including I/O):

| samples | 1 thread | 4 threads | per cluster per core |
|--------:|---------:|----------:|---------------------:|
|     100 |   1.3 s  |   0.34 s  |  0.65 ms             |
|     400 |   1.6 s  |   0.44 s  |  0.8 ms              |
|   1 000 |   1.9 s  |   0.50 s  |  0.9 ms              |
|  10 000 |   6.1 s  |   1.6 s   |  3.0 ms              |

(`--full-extra-starts 0`; the default extra start roughly doubles the fitting time.) Scaled to
30 000 clusters on 8 threads this is about 3 s at N = 100 and 12 s at N = 10 000 before the
extra start, well inside the 60 s budget. Peak RSS stays in the tens of MB because only one
cluster per thread is live.

## Building

```
cd leafcutter-rs
cargo build --release          # binary: target/release/leafcutter_ds
cargo test                     # unit + integration tests (simulated data)
```

## Command line

```
# same inputs as scripts/leafcutter_ds.R
leafcutter_ds run counts_perind_numers.counts.gz groups.txt -o out -p 8 [--json]
#   -> out_cluster_significance.txt, out_effect_sizes.txt [, out_results.json]

# one-off conversion for large cohorts, then run on the store
leafcutter_ds build-store counts_perind_numers.counts.gz cohort.lcs
leafcutter_ds run cohort.lcs groups.txt -o out --null-cache cohort_null.json

# synthetic data for tests / benchmarks
leafcutter_ds simulate -n 1000 -m 20000 -o sim --store
```

Filter options mirror the R function defaults (`-s 10 -i 5 -g 4 -c 20`); note that the R
*script* uses `-s Inf -g 3`.

## Child-process JSON protocol (Node)

`leafcutter_ds json` reads one request object from stdin and writes one response object to
stdout (errors are written as `{"error": "..."}` with a non-zero exit code):

```jsonc
{
  "store": "/data/cohort.lcs",           // or "counts_file": "...counts.gz"
  "samples": ["s1", "s2", "..."],        // any subset of the cohort, any order
  "groups":  ["case", "ctrl", "..."],    // exactly two labels; first seen = 0 (numeric: sorted)
  "confounders": [["b1","b2","..."], ["31","45","..."]],   // optional; numeric -> standardised, else one-hot
  "threads": 8,
  "params": { "max_cluster_size": 10, "min_samples_per_intron": 5,
              "min_samples_per_group": 4, "min_coverage": 20,
              "fit": { "full_extra_starts": 1 } },        // all optional
  "null_cache": "/data/cohort_null.json", // optional
  "only_success": true, "omit_introns": false
}
```

Response:

```jsonc
{
  "group_names": ["ctrl", "case"], "n_samples": 512, "n_clusters": 31240,
  "confounder_columns": ["V3b2", "V4"],
  "summary": { "Success": 28870, "Too many introns in cluster": 310, "...": 0 },
  "timing_ms": { "fit_ms": 4180, "total_ms": 4302 },
  "clusters": [
    { "cluster": "chr1:clu_12_NA", "status": "Success", "loglr": 8.31, "df": 2,
      "p": 2.5e-4, "p.adjust": 0.012, "q_storey": 0.009,
      "introns": [ { "intron": "chr1:1000:2000:clu_12_NA", "logef": 0.8,
                     "baseline": 0.31, "perturbed": 0.52, "deltapsi": 0.21 } ],
      "n_samples": 498, "refit_null": false, "evaluations": 61,
      "value_null": -9123.4, "value_full": -9115.1, "converged": true }
  ]
}
```

Suggested deployment: build the store once per cohort (plus gene annotation from the
clustering pipeline), keep the binary next to the Node server, spawn it per request with the
sample subset and labels, and reuse `null_cache` for requests that regroup the same cohort.

## Validation

R/rstan is not needed for the tests, but `scripts/reference_ds.py` is an independent
implementation in numpy/autograd/scipy that uses *Stan's own parameterisation*, the R
package's smart initialisation and refit rule, and a different optimiser. On simulated data
(`leafcutter_ds simulate`) the two agree on every skip reason and on the significant set at
p < 0.05, 0.001 and 1e-6; the log likelihood ratios agree to ~1e-3 except for clusters where
the two optimisers settle in different posterior modes (see `--full-extra-starts`). Run

```
leafcutter_ds run sim_perind_numers.counts.gz sim_groups.txt -o rust --json
python3 scripts/reference_ds.py sim_perind_numers.counts.gz sim_groups.txt --out ref.json --max-clusters 300
python3 scripts/compare_reference.py rust_results.json ref.json
```

Unit tests check the analytic gradient against finite differences, the collapsed objective
against the dense per-sample formula, `digamma` against the derivative of `lgamma`, the
χ² survival function against scipy, BH against `p.adjust`, and every filter reason.

## Layout

```
src/special.rs   lgamma / digamma / chi-square survival function
src/lbfgs.rs     L-BFGS with strong-Wolfe line search (Stan's stopping rules)
src/design.rs    design matrices and the collapsed cluster representation
src/dm.rs        Dirichlet-multinomial GLM objective + gradient
src/fit.rs       smart init, null/full fits, LRT, refit rule, effect sizes
src/ds.rs        filters, parallel driver, BH / Storey
src/io.rs        counts / groups files, result tables
src/store.rs     memory-mapped cluster-major count store
src/nullcache.rs per-cohort cache of null fits
src/simulate.rs  synthetic data
src/bin/leafcutter_ds.rs   CLI + JSON protocol
examples/debug_cluster.rs  trace the fits of one cluster
scripts/reference_ds.py    independent Python reference, compare_reference.py
```
