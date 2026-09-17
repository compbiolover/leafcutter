#!/usr/bin/env python3
"""Compare two leafcutter-ds style output pairs (Python vs Rust):
   compare_tables.py <prefix_a> <prefix_b>  (each with _cluster_significance.txt and _effect_sizes.txt)
"""
import sys
import numpy as np
import pandas as pd

a, b = sys.argv[1], sys.argv[2]
ca = pd.read_table(a + "_cluster_significance.txt", index_col=0)
cb = pd.read_table(b + "_cluster_significance.txt", index_col=0)
shared = ca.index.intersection(cb.index)
print(f"clusters: {len(ca)} vs {len(cb)}, shared {len(shared)}")
sa, sb = ca.loc[shared, "status"], cb.loc[shared, "status"]
mism = shared[sa != sb]
print(f"status mismatches: {len(mism)}")
for c in mism[:10]:
    print(f"   {c}: {sa[c]!r} vs {sb[c]!r}")
ok = shared[(sa == "Success") & (sb == "Success")]
la, lb = ca.loc[ok, "loglr"].astype(float), cb.loc[ok, "loglr"].astype(float)
pa, pb = ca.loc[ok, "p"].astype(float), cb.loc[ok, "p"].astype(float)
d = (lb - la)
print(f"loglr B-A over {len(ok)} clusters: median {d.median():+.2e}, |diff|>1e-2 in {(d.abs()>1e-2).sum()}, >0.1 in {(d.abs()>0.1).sum()}, >1 in {(d.abs()>1).sum()}; B higher by >1e-3 in {(d>1e-3).sum()}, lower in {(d<-1e-3).sum()}")
for thr in (0.05, 0.001, 1e-6):
    print(f"  p<{thr}: A {(pa<thr).sum()}, B {(pb<thr).sum()}, disagreements {((pa<thr)!=(pb<thr)).sum()}")
qa, qb = ca.loc[ok, "p.adjust"].astype(float), cb.loc[ok, "p.adjust"].astype(float)
print(f"  p.adjust<0.05: A {(qa<0.05).sum()}, B {(qb<0.05).sum()}, disagreements {((qa<0.05)!=(qb<0.05)).sum()}")
if "genes" in ca.columns and "genes" in cb.columns:
    ga, gb = ca.loc[shared, "genes"].fillna("NA"), cb.loc[shared, "genes"].fillna("NA")
    norm = lambda s: ",".join(sorted(str(s).split(",")))
    print(f"  genes column identical for {(ga.map(norm)==gb.map(norm)).sum()}/{len(shared)} clusters")
if "annotations" in ca.columns and "annotations" in cb.columns:
    print(f"  annotations identical for {(ca.loc[shared,'annotations'].fillna('NA')==cb.loc[shared,'annotations'].fillna('NA')).sum()}/{len(shared)} clusters")
worst = d.abs().sort_values(ascending=False).index[:4]
for c in worst:
    print(f"  worst {c}: loglr A {la[c]:.4f} B {lb[c]:.4f}  p {pa[c]:.2e} / {pb[c]:.2e}")
ea = pd.read_table(a + "_effect_sizes.txt", index_col=0)
eb = pd.read_table(b + "_effect_sizes.txt", index_col=0)
common = ea.index.intersection(eb.index)
print(f"effect size rows: {len(ea)} vs {len(eb)}, shared {len(common)}; columns A {list(ea.columns)} B {list(eb.columns)}")
for col in ea.columns:
    if col in eb.columns:
        x, y = ea.loc[common, col].astype(float), eb.loc[common, col].astype(float)
        m = ~(x.isna() | y.isna())
        dd = (x[m] - y[m]).abs()
        print(f"  {col}: max |diff| {dd.max():.3e}, median {dd.median():.2e}, >0.01 in {(dd>0.01).sum()}/{m.sum()}")
