#!/usr/bin/env python3
"""Convert run_r_reference.R output into the JSON layout of reference_ds.py so that
compare_reference.py can compare the Rust results with R's.

Usage: r_to_reference_json.py out_prefix ref.json
"""
import csv
import json
import sys

prefix, out = sys.argv[1], sys.argv[2]
ref = {}
for r in csv.DictReader(open(prefix + "_cluster_significance.txt"), delimiter="\t"):
    if r["status"] != "Success":
        ref[r["cluster"]] = dict(status=r["status"])
        continue
    ref[r["cluster"]] = dict(
        status="Success", loglr=float(r["loglr"]), df=int(r["df"]), p=float(r["p"]),
        value_null=float(r["value_null"]), value_full=float(r["value_full"]),
        refit_null=r["refit_null"] == "TRUE", nit_null=0, nit_full=0, introns={},
    )
for r in csv.DictReader(open(prefix + "_effect_sizes.txt"), delimiter="\t"):
    p = r["intron"].split(":")
    cid = p[0] + ":" + p[-1]
    if cid in ref and ref[cid]["status"] == "Success":
        ref[cid]["introns"][r["intron"]] = dict(logef=float(r["logef"]), baseline=float(r["baseline"]), perturbed=float(r["perturbed"]), deltapsi=float(r["deltapsi"]))
json.dump(ref, open(out, "w"))
print(f"{len(ref)} clusters, {sum(1 for v in ref.values() if v['status'] == 'Success')} successes")
