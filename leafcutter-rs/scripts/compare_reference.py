#!/usr/bin/env python3
"""Compare `leafcutter_ds run --json` output with `reference_ds.py` output."""
import json
import signal
import sys

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

import numpy as np

rust = {c["cluster"]: c for c in json.load(open(sys.argv[1]))["clusters"]}
ref = json.load(open(sys.argv[2]))
status_mismatch = [c for c in ref if rust[c]["status"] != ref[c]["status"]]
print(f"clusters compared: {len(ref)}; status mismatches: {len(status_mismatch)} {status_mismatch[:5]}")
ok = [c for c in ref if ref[c]["status"] == "Success" and rust[c]["status"] == "Success"]
lr_r = np.array([rust[c]["loglr"] for c in ok])
lr_p = np.array([ref[c]["loglr"] for c in ok])
p_r = np.array([rust[c]["p"] for c in ok])
p_p = np.array([ref[c]["p"] for c in ok])
abs_d = np.abs(lr_r - lr_p)
rel_d = abs_d / np.maximum(np.abs(lr_p), 1.0)
print(f"loglr: max abs diff {abs_d.max():.3e}, max rel diff {rel_d.max():.3e}, median abs diff {np.median(abs_d):.3e}")
print(f"       rust > ref (rust found a better optimum) in {(lr_r - lr_p > 1e-6).sum()} clusters, ref > rust in {(lr_p - lr_r > 1e-6).sum()}")
lp_d = np.abs(np.log10(np.maximum(p_r, 1e-300)) - np.log10(np.maximum(p_p, 1e-300)))
print(f"-log10 p: max abs diff {lp_d.max():.3e}")
for thr in (0.05, 0.001, 1e-6):
    print(f"  significant at p<{thr}: rust {(p_r < thr).sum()}, ref {(p_p < thr).sum()}, disagreements {((p_r < thr) != (p_p < thr)).sum()}")
d_max, l_max = 0.0, 0.0
for c in ok:
    for i in rust[c]["introns"]:
        r = ref[c]["introns"][i["intron"]]
        d_max = max(d_max, abs(i["deltapsi"] - r["deltapsi"]))
        l_max = max(l_max, abs(i["logef"] - r["logef"]))
print(f"effect sizes: max |deltapsi diff| {d_max:.3e}, max |logef diff| {l_max:.3e}")
worst = np.argsort(-abs_d)[:5]
for w in worst:
    print(f"  worst: {ok[w]} rust loglr {lr_r[w]:.6f} ref {lr_p[w]:.6f} p {p_r[w]:.3e} / {p_p[w]:.3e}")
