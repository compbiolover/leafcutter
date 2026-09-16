//! Rust port of LeafCutter's differential splicing test.
//!
//! The statistical core mirrors `leafcutter/R/differential_splicing.R`,
//! `leafcutter/R/dm_glm_multi_conc.R` and `leafcutter/inst/stan/dm_glm_multi_conc.stan`:
//! for every intron cluster a Dirichlet-multinomial GLM is fitted under a null design
//! (intercept + confounders) and a full design (+ group) by maximum a posteriori
//! estimation, and the two fits are compared with a likelihood ratio test.
//!
//! Module overview:
//! * [`special`]  – lgamma / digamma / chi-square survival function.
//! * [`lbfgs`]    – a small L-BFGS minimiser with a strong-Wolfe line search.
//! * [`dm`]       – the Dirichlet-multinomial GLM objective and analytic gradient over
//!                  design cells (sufficient-statistic collapse).
//! * [`fit`]      – smart initialisation, null/full fits, likelihood ratio test, effect sizes.
//! * [`ds`]       – filtering, per-cluster driver, parallel loop, BH / Storey q-values.
//! * [`io`]       – LeafCutter counts / groups files and result tables.
//! * [`store`]    – memory-mapped cluster-major count store for very large cohorts.
//! * [`nullcache`]– on-disk cache of null fits keyed on the cohort/design.
//! * [`simulate`] – synthetic data generator used for tests and benchmarks.

pub mod design;
pub mod dm;
pub mod ds;
pub mod fit;
pub mod io;
pub mod lbfgs;
pub mod nullcache;
pub mod simulate;
pub mod special;
pub mod store;
