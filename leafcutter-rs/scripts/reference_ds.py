#!/usr/bin/env python3
"""Independent reference implementation of LeafCutter's differential splicing test.

This re-implements `dirichlet_multinomial_anova_mc` (leafcutter/R/dm_glm_multi_conc.R) and
the filtering of `differential_splicing` (leafcutter/R/differential_splicing.R) in numpy,
using the *Stan* parameterisation of `dm_glm_multi_conc.stan` (per covariate a simplex
`beta_raw` times a scale `beta_scale`, plus per-intron concentrations), the R package's
"smart" method-of-moments initialisation (bugs included, e.g. the row-slot mismatch when
copying null coefficients into the full init), the null-refit rule and the effect size
definitions. Gradients come from autograd and the optimiser is scipy's L-BFGS-B with tight
tolerances, so this is a different parameterisation, a different optimiser and a different
code path from the Rust port; agreement between the two validates the port when R/rstan is
not available.

Usage: reference_ds.py counts_file groups_file --out ref.json [--max-clusters N]
"""
import argparse
import gzip
import json
import sys

import autograd.numpy as np
from autograd import value_and_grad
from autograd.scipy.special import gammaln
from scipy.optimize import minimize
from scipy.stats import chi2

CONC_SHAPE, CONC_RATE = 1.0001, 1e-4


def read_counts(path):
    op = gzip.open if open(path, "rb").read(2) == b"\x1f\x8b" else open
    with op(path, "rt") as f:
        header = f.readline().split()
        rows = []
        for line in f:
            t = line.split()
            if not t:
                continue
            rows.append((t[0], np.array([int(v.split("/")[0]) for v in t[1:]])))
    if len(header) == len(rows[0][1]) + 1:
        header = header[1:]
    return header, rows


def read_groups(path):
    meta = [l.split() for l in open(path) if l.strip()]
    return [m[0] for m in meta], [m[1] for m in meta], [[m[c] for m in meta] for c in range(2, len(meta[0]))]


def encode(groups, confs):
    names = []
    for g in groups:
        if g not in names:
            names.append(g)
    try:
        names = sorted(names, key=float)
    except ValueError:
        pass
    assert len(names) == 2
    x = np.array([1.0 if g == names[1] else 0.0 for g in groups])
    cols = []
    for col in confs:
        try:
            v = np.array([float(c) for c in col])
            sd = v.std(ddof=1)
            cols.append((v - v.mean()) / sd if sd > 0 else np.zeros_like(v))
        except ValueError:
            for lvl in sorted(set(col))[1:]:
                cols.append(np.array([1.0 if c == lvl else 0.0 for c in col]))
    conf = np.column_stack(cols) if cols else None
    return x, names, conf


def stan_target(z, scale, log_conc, x, y):
    """log posterior of dm_glm_multi_conc.stan; beta_raw[p] = softmax(z[p])."""
    K = y.shape[1]
    zc = z - np.max(z, axis=1, keepdims=True)
    beta_raw = np.exp(zc) / np.sum(np.exp(zc), axis=1, keepdims=True)  # P x K simplex rows
    beta = scale[:, None] * (beta_raw - 1.0 / K)  # P x K
    conc = np.exp(log_conc)
    eta = x @ beta  # N x K
    eta = eta - np.max(eta, axis=1, keepdims=True)
    s = np.exp(eta) / np.sum(np.exp(eta), axis=1, keepdims=True)
    a = conc[None, :] * s
    A = np.sum(a, axis=1)
    Y = np.sum(y, axis=1)
    ll = np.sum(gammaln(A) + np.sum(gammaln(a + y), axis=1) - gammaln(A + Y) - np.sum(gammaln(a), axis=1))
    ll = ll + np.sum((CONC_SHAPE - 1.0) * log_conc - CONC_RATE * conc)
    return ll


def optimize(x, y, beta_raw0, scale0, conc0):
    P, K = beta_raw0.shape
    theta0 = np.concatenate([np.log(beta_raw0).ravel(), scale0, np.log(conc0)])

    def unpack(th):
        return th[: P * K].reshape(P, K), th[P * K : P * K + P], th[P * K + P :]

    def negf(th):
        z, sc, lc = unpack(th)
        return -stan_target(z, sc, lc, x, y)

    vg = value_and_grad(negf)
    res = minimize(vg, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=5000, maxfun=20000, ftol=1e-15, gtol=1e-10, maxcor=10))
    z, sc, lc = unpack(res.x)
    zc = z - z.max(axis=1, keepdims=True)
    beta_raw = np.exp(zc) / np.exp(zc).sum(axis=1, keepdims=True)
    return dict(value=-res.fun, beta_raw=beta_raw, beta_scale=sc, conc=np.exp(lc), nit=res.nit)


def smart_init(xNull, y, reg=0.001):
    K = y.shape[1]
    y_norm = np.log((y + 1) / (y + 1).sum(axis=1, keepdims=True))
    beta_mm = np.linalg.solve(xNull.T @ xNull + reg * np.eye(xNull.shape[1]), xNull.T @ y_norm)
    beta_norm = beta_mm - beta_mm.mean(axis=1, keepdims=True)
    scale = []
    for row in beta_norm:
        up = row[np.argmax(np.abs(row))] / (1 - 1.0 / K)
        down = row[np.argmax(-np.sign(up) * row)] / (1.0 / K)
        scale.append(up - down)
    scale = np.array(scale)
    beta_raw = beta_norm / (scale[:, None] + 1e-20) + 1.0 / K
    beta_raw = beta_raw / beta_raw.sum(axis=1, keepdims=True)
    return beta_raw, scale


def sanitize(b):
    b = np.clip(b, 1e-6, 1 - 1e-6)
    return b / b.sum(axis=1, keepdims=True)


def anova(xFull, xNull, y):
    K = y.shape[1]
    beta_raw0, scale0 = smart_init(xNull, y)
    fit_null = optimize(xNull, y, beta_raw0, scale0, np.full(K, 10.0))
    Pn, Pf = xNull.shape[1], xFull.shape[1]
    # R: init$beta_raw[1:ncol(xNull),] = sanitized null rows (row slots 1..Pn of the full design)
    init_raw = np.full((Pf, K), 1e-4)
    init_raw[:Pn] = sanitize(fit_null["beta_raw"])
    init_raw = init_raw / init_raw.sum(axis=1, keepdims=True)
    init_scale = np.ones(Pf)
    init_scale[:Pn] = fit_null["beta_scale"]
    fit_full = optimize(xFull, y, init_raw, init_scale, fit_null["conc"])
    loglr = fit_full["value"] - fit_null["value"]
    df = (Pf - Pn) * (K - 1)
    refit = False
    if chi2.sf(2 * loglr, df) < 0.001:
        rn = optimize(xNull, y, sanitize(fit_full["beta_raw"][:Pn]), fit_full["beta_scale"][:Pn], fit_full["conc"])
        if rn["value"] > fit_null["value"]:
            refit = True
            fit_null = rn
            loglr = fit_full["value"] - fit_null["value"]
    return loglr, df, chi2.sf(2 * loglr, df), fit_full, refit, fit_null


def effect_sizes(fit):
    K = fit["beta_raw"].shape[1]
    beta = fit["beta_scale"][:, None] * (fit["beta_raw"] - 1.0 / K)
    conc = fit["conc"]

    def to_psi(b):
        w = np.exp(b - b.max()) * conc
        return w / w.sum()

    base, pert = to_psi(beta[0]), to_psi(beta[0] + beta[1])
    return beta[1], base, pert, pert - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("counts")
    ap.add_argument("groups")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clusters", type=int, default=0)
    ap.add_argument("--cluster", action="append", default=[], help="only these clusters")
    ap.add_argument("--max-cluster-size", type=int, default=10)
    ap.add_argument("--min-samples-per-intron", type=int, default=5)
    ap.add_argument("--min-samples-per-group", type=int, default=4)
    ap.add_argument("--min-coverage", type=int, default=20)
    a = ap.parse_args()

    samples, rows = read_counts(a.counts)
    ms, groups, confs = read_groups(a.groups)
    col = [samples.index(s) for s in ms]
    x, names, conf = encode(groups, confs)
    clusters = {}
    for name, v in rows:
        p = name.split(":")
        clusters.setdefault(p[0] + ":" + p[-1], []).append((name, v[col]))
    out = {}
    for ci, cid in enumerate(sorted(clusters)):
        if a.max_clusters and ci >= a.max_clusters:
            break
        if a.cluster and cid not in a.cluster:
            continue
        introns = clusters[cid]
        if len(introns) > a.max_cluster_size:
            out[cid] = dict(status="Too many introns in cluster")
            continue
        if len(introns) <= 1:
            out[cid] = dict(status="<=1 junction in cluster")
            continue
        y = np.column_stack([v for _, v in introns]).astype(float)
        tot = y.sum(axis=1)
        use = tot > 0
        if use.sum() <= 1:
            out[cid] = dict(status="<=1 sample with coverage>0")
            continue
        tot = tot[use]
        if (tot >= a.min_coverage).sum() <= 1:
            out[cid] = dict(status="<=1 sample with coverage>min_coverage")
            continue
        xs = x[use]
        y = y[use]
        iu = (y > 0).sum(axis=0) >= a.min_samples_per_intron
        if iu.sum() < 2:
            out[cid] = dict(status="<2 introns used in >=min_samples_per_intron samples")
            continue
        y = y[:, iu]
        names_kept = [n for (n, _), keep in zip(introns, iu) if keep]
        vals, counts = np.unique(xs[tot >= a.min_coverage], return_counts=True)
        if (counts >= a.min_samples_per_group).sum() < 2:
            out[cid] = dict(status="Not enough valid samples")
            continue
        xFull = np.column_stack([np.ones(len(xs)), xs])
        xNull = xFull[:, :1]
        if conf is not None:
            ch = conf[use]
            ch = ch[:, ch.std(axis=0, ddof=1) > 0]
            xFull = np.column_stack([xFull, ch])
            xNull = np.column_stack([xNull, ch])
        loglr, df, p, fit_full, refit, fit_null = anova(xFull, xNull, y)
        logef, base, pert, dpsi = effect_sizes(fit_full)
        out[cid] = dict(
            status="Success", loglr=float(loglr), df=int(df), p=float(p), refit_null=bool(refit),
            value_null=float(fit_null["value"]), value_full=float(fit_full["value"]), nit_null=int(fit_null["nit"]), nit_full=int(fit_full["nit"]),
            beta_full=(fit_full["beta_scale"][:, None] * (fit_full["beta_raw"] - 1.0 / y.shape[1])).tolist(), conc_full=fit_full["conc"].tolist(),
            beta_null=(fit_null["beta_scale"][:, None] * (fit_null["beta_raw"] - 1.0 / y.shape[1])).tolist(), conc_null=fit_null["conc"].tolist(),
            introns={n: dict(logef=float(l), baseline=float(b), perturbed=float(q), deltapsi=float(d)) for n, l, b, q, d in zip(names_kept, logef, base, pert, dpsi)},
        )
        if ci % 50 == 0:
            print(f"{ci} clusters done", file=sys.stderr)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
