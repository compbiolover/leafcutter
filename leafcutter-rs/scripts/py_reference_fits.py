#!/usr/bin/env python3
"""Run the Python leafcutter-ds model directly (same preprocessing as its CLI) and dump the
per-cluster fit values it does not write to its output tables: null/full log posterior,
optimiser exit statuses and whether the fresh-init full fit won.

Usage: py_reference_fits.py counts_file groups_file out_prefix [--baseline B] [-p threads]
Writes <out_prefix>_fits.txt (tab separated).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, scale

from leafcutter.differential_splicing.differential_splicing import differential_splicing

ap = argparse.ArgumentParser()
ap.add_argument("counts")
ap.add_argument("groups")
ap.add_argument("out")
ap.add_argument("--baseline", default="Control")
ap.add_argument("-p", type=int, default=1)
ap.add_argument("--init", default="brr")
ap.add_argument("--max_cluster_size", type=float, default=float("inf"))
ap.add_argument("--min_samples_per_group", type=int, default=3)
a = ap.parse_args()

counts = pd.read_table(a.counts, sep=r"\s+")
if not pd.api.types.is_numeric_dtype(counts.iloc[:, 0]):
    counts = counts.set_index(counts.columns[0])
meta = pd.read_table(a.groups, header=None, sep=r"\s+").rename({0: "sample", 1: "group"}, axis=1)
confounders = None
if len(meta.columns) > 2:
    confounders = meta.iloc[:, 2:]
    confounders.columns = confounders.columns.astype(str)
    tr = []
    for col in confounders.columns:
        if pd.api.types.is_numeric_dtype(confounders[col]):
            tr.append((col, StandardScaler(), [col]))
        else:
            tr.append((col, OneHotEncoder(drop="first", sparse_output=False), [col]))
    confounders = ColumnTransformer(tr).fit_transform(confounders)
    confounders = pd.DataFrame(confounders, index=meta["sample"]).dropna()
    meta = meta[meta["sample"].isin(confounders.index)]
    counts = counts[confounders.index]
else:
    counts = counts[meta["sample"]]
if meta["group"].dtype.kind in "OUS":
    uv = sorted(set(meta["group"]), key=lambda g: g != a.baseline)
    meta["group"] = pd.Categorical(meta["group"], categories=uv, ordered=True)
else:
    meta["group"] = scale(meta["group"])

cluster_table, junc_table, status_df = differential_splicing(
    counts, meta["group"], confounders=confounders, max_cluster_size=a.max_cluster_size, min_samples_per_intron=5,
    min_samples_per_group=a.min_samples_per_group, min_coverage=20, init=a.init, device="cpu", num_cores=a.p,
)
cluster_table["cluster"] = cluster_table.index
cols = ["cluster", "status", "loglr", "df", "p", "p.adjust", "null_ll", "full_ll", "null_exit_status", "full_exit_status", "smart_init_improved"]
cluster_table[cols].to_csv(a.out + "_fits.txt", sep="\t", index=False, na_rep="NA")
print(f"wrote {a.out}_fits.txt ({(cluster_table['status'] == 'Success').sum()} successes)")
