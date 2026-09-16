#!/usr/bin/env Rscript
# Run the R package's differential splicing test (the code in leafcutter/R, with the Stan
# model compiled from leafcutter/inst/stan/dm_glm_multi_conc.stan) on a counts + groups file
# and write a table that scripts/r_to_reference_json.py converts for compare_reference.py.
#
# Usage: Rscript run_r_reference.R counts_file groups_file out_prefix [max_clusters] [threads]
#
# Only rstan, foreach, doMC, dplyr and R.utils are needed (the full package pulls in shiny,
# Bioconductor packages etc. that the test itself does not use).
suppressPackageStartupMessages({
  library(rstan); library(foreach); library(doMC); library(dplyr); library(R.utils)
})
args <- commandArgs(trailingOnly = TRUE)
counts_file <- args[1]; groups_file <- args[2]; out_prefix <- args[3]
max_clusters <- if (length(args) >= 4) as.integer(args[4]) else 0L
threads <- if (length(args) >= 5) as.integer(args[5]) else 1L

repo <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "..", ".."))
pkg <- file.path(repo, "leafcutter")

# compile (and cache) the Stan model, then expose it the way the package does
model_rds <- file.path(dirname(out_prefix), "dm_glm_multi_conc.rds")
if (file.exists(model_rds)) {
  sm <- readRDS(model_rds)
} else {
  # newer stanc dropped the old array syntax; rewrite `T x[N];` as `array[N] T x;` (same model)
  code <- readLines(file.path(pkg, "inst", "stan", "dm_glm_multi_conc.stan"))
  code <- gsub("^(\\s*)([A-Za-z_<>=0-9\\[\\]\\.]+(?:\\[[A-Za-z0-9_]+\\])?)\\s+([A-Za-z_]+)\\[([A-Za-z0-9_]+)\\];", "\\1array[\\4] \\2 \\3;", code, perl = TRUE)
  cat(code, sep = "\n")
  sm <- stan_model(model_code = paste(code, collapse = "\n"), model_name = "dm_glm_multi_conc")
  saveRDS(sm, model_rds)
}
stanmodels <- list(dm_glm_multi_conc = sm)
source(file.path(pkg, "R", "utils.R"))
source(file.path(pkg, "R", "dm_glm_multi_conc.R"))
source(file.path(pkg, "R", "differential_splicing.R"))

# --- scripts/leafcutter_ds.R input handling
counts <- read.table(counts_file, header = TRUE, check.names = FALSE)
meta <- read.table(groups_file, header = FALSE, stringsAsFactors = FALSE)
colnames(meta)[1:2] <- c("sample", "group")
counts <- counts[, meta$sample]
group_names <- unique(meta$group)
if (is.numeric(meta$group)) group_names <- sort(group_names)
meta$group <- factor(meta$group, group_names)
numeric_x <- as.numeric(meta$group) - 1
confounders <- NULL
if (ncol(meta) > 2) {
  confounders <- meta[, 3:ncol(meta), drop = FALSE]
  for (i in seq_len(ncol(confounders))) if (is.numeric(confounders[, i])) confounders[, i] <- scale(confounders[, i])
  confounders <- model.matrix(~., data = confounders)
  confounders <- confounders[, 2:ncol(confounders), drop = FALSE]
}
if (max_clusters > 0) {
  introns <- get_intron_meta(rownames(counts))
  cid <- paste(introns$chr, introns$clu, sep = ":")
  keep <- cid %in% head(sort(unique(cid)), max_clusters)
  counts <- counts[keep, ]
}
registerDoMC(threads)
cat("Running differential_splicing on", ncol(counts), "samples,", nrow(counts), "introns\n")
t0 <- Sys.time()
results <- differential_splicing(counts, numeric_x, confounders = confounders, max_cluster_size = 10,
                                 min_samples_per_intron = 5, min_samples_per_group = 4, min_coverage = 20,
                                 timeout = 60, robust = FALSE)
cat("Elapsed:", format(Sys.time() - t0), "\n")

tab <- cluster_results_table(results)
tab$value_null <- sapply(results, function(r) if (is.list(r)) r$fit_null$value else NA)
tab$value_full <- sapply(results, function(r) if (is.list(r)) r$fit_full$value else NA)
tab$refit_null <- sapply(results, function(r) if (is.list(r)) r$refit_null_flag else NA)
tab$ret_null <- sapply(results, function(r) if (is.list(r)) r$fit_null$return_code else NA)
tab$ret_full <- sapply(results, function(r) if (is.list(r)) r$fit_full$return_code else NA)
write.table(tab, paste0(out_prefix, "_cluster_significance.txt"), quote = FALSE, sep = "\t", row.names = FALSE)
es <- leaf_cutter_effect_sizes(results)
colnames(es)[3:4] <- c("baseline", "perturbed")
write.table(es, paste0(out_prefix, "_effect_sizes.txt"), quote = FALSE, sep = "\t", row.names = FALSE)
cat("Wrote", paste0(out_prefix, "_cluster_significance.txt"), "\n")
