"""Seed aggregation and scaffold-cluster bootstrap (T9).

The bootstrap resamples whole test scaffold groups, never single molecules. One set of resamples
is drawn per task and shared by every seed, method and conformal variant, so all comparisons are
paired. Two kinds of interval are reported for conformal metrics:
- ci_lo/ci_hi: test-set sampling only (plan T9); models, calibration sets and conformal
  quantiles are held fixed.
- ci_lo_cal/ci_hi_cal: calibration scaffold groups are resampled too and the conformal
  quantiles refit in every resample (DR-005). Models stay fixed.
Post hoc (DR-006): the Winkler score gets both intervals; the size-matched `valid` control
(source "valid_sizematched") gets the test-only interval. The same holds for the composition
control (DR-007, sources "naive_cal_dropNN").
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from molcalib import config
from molcalib import conformal as cf
from molcalib.conformal import nn_bin_edges
from molcalib.metrics import GAUSS_LEVELS, size_matched_per_molecule, winkler_scores

CI = (2.5, 97.5)


def percentile_ci(boot: np.ndarray) -> tuple[float, float]:
    """95% percentile interval; NaNs (empty resampled bins) ignored. With infinite values
    (too few calibration points) linear interpolation gives NaN, so use the nearest rank."""
    finite = np.isfinite(boot[~np.isnan(boot)]).all()
    lo, hi = np.nanpercentile(boot, CI, method="linear" if finite else "nearest")
    return float(lo), float(hi)


def group_weights(groups, n_boot: int = config.N_BOOT, seed: int = config.BOOT_SEED) -> np.ndarray:
    """(n_boot, n) molecule weights: how often each molecule's scaffold group was drawn.

    Each resample draws as many groups as there are, with replacement.
    """
    codes, _ = pd.factorize(np.asarray(groups))
    n_groups = codes.max() + 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    counts = np.stack([np.bincount(d, minlength=n_groups) for d in draws])
    return counts[:, codes].astype(float)


def wmean(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Weighted mean of per-molecule values x for each weight row; w may be 1-D (one sample)."""
    w = np.atleast_2d(w)
    x = np.asarray(x, dtype=float)
    inf = np.isinf(x)
    if not inf.any():
        return (w @ x) / w.sum(axis=1)
    out = (w @ np.where(inf, 0.0, x)) / w.sum(axis=1)  # 0 * inf would give NaN
    drawn = w @ inf.astype(float) > 0
    out[drawn] = np.inf  # an infinite interval that was resampled makes the mean infinite
    return out


def wspearman(w: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation on the resampled data (molecules repeated by their integer weights)."""
    out = []
    for row in np.atleast_2d(w).astype(int):
        out.append(spearmanr(np.repeat(a, row), np.repeat(b, row))[0])
    return np.array(out)


def run_metrics(test: pd.DataFrame, intervals: pd.DataFrame, w: np.ndarray,
                size_matched: dict | None = None) -> dict:
    """Metric values for one run under weights w: {(source, variant, metric): array(len(w))}.

    Point metrics use source = variant = "". With w = ones((1, n)) this reproduces T8.
    size_matched: output of conformal.size_matched_intervals, reported as "valid_sizematched".
    """
    y, mu, sg = (test[c].to_numpy(float) for c in ("y", "mu", "sigma_total"))
    err = y - mu
    nll = 0.5 * np.log(2 * np.pi * sg**2) + err**2 / (2 * sg**2)
    abs_z = np.abs(err) / sg
    gauss_obs = np.stack([wmean(w, (abs_z <= norm.ppf(0.5 + p / 2)).astype(float))
                          for p in GAUSS_LEVELS])  # (levels, B)
    gap = gauss_obs - GAUSS_LEVELS[:, None]
    out = {
        ("", "", "mae"): wmean(w, np.abs(err)),
        ("", "", "rmse"): np.sqrt(wmean(w, err**2)),
        ("", "", "nll"): wmean(w, nll),
        ("", "", "miscal_area"): np.abs(gap).mean(axis=0),
        ("", "", "miscal_signed"): gap.mean(axis=0),
        ("", "", "spearman_sigma_abserr"): wspearman(w, sg, np.abs(err)),
    }
    # farthest test-distance quartile (bins fixed on the full test set)
    nn = test["nn_dist"].to_numpy(float)
    edges = nn_bin_edges(nn)
    far = (np.searchsorted(edges, nn, side="left") == len(edges)).astype(float)
    pos = {r: i for i, r in enumerate(test.index)}
    for (source, variant), iv in intervals.groupby(["source", "variant"], sort=False):
        idx = np.array([pos[r] for r in iv["row"]])
        lo = np.empty(len(test))
        hi = np.empty(len(test))
        lo[idx], hi[idx] = iv["lo"].to_numpy(), iv["hi"].to_numpy()
        hit = ((y >= lo) & (y <= hi)).astype(float)
        out[(source, variant, "coverage")] = wmean(w, hit)
        out[(source, variant, "mean_width")] = wmean(w, hi - lo)
        out[(source, variant, "coverage_far_bin")] = wmean(w, hit * far) / wmean(w, far)
        out[(source, variant, "winkler")] = wmean(w, winkler_scores(y, lo, hi))
    for variant, d in size_matched_per_molecule(y, size_matched or {}).items():
        for metric, x in d.items():
            out[("valid_sizematched", variant, metric)] = wmean(w, x)
    return out


def conformal_boot_with_cal(pred: pd.DataFrame, w_test: np.ndarray, seed: int,
                            alpha: float = config.ALPHA) -> dict:
    """Conformal metrics when calibration groups are also resampled and quantiles refit.

    Resample b uses test weights w_test[b] and a scaffold-group resample of each calibration
    set. Calibration resamples depend on (BOOT_SEED, split seed, source) only, so the ensemble
    and MC dropout runs of one seed share them (paired comparison).
    Returns {(source, variant, metric): array(n_boot)} for coverage, mean_width, coverage_far_bin
    and winkler.
    """
    test = pred[pred["split"] == "test"]
    y, mu, sg, nn = (test[c].to_numpy(float) for c in ("y", "mu", "sigma_total", "nn_dist"))
    edges = nn_bin_edges(nn)
    far = (np.searchsorted(edges, nn, side="left") == len(edges)).astype(float)
    n_boot = len(w_test)
    out = {}
    for k, source in enumerate(cf.SOURCES):
        cal = pred[pred["split"] == source]
        cy, cmu, csg, cnn = (cal[c].to_numpy(float) for c in ("y", "mu", "sigma_total", "nn_dist"))
        w_cal = group_weights(cal["scaffold_id"], n_boot, [config.BOOT_SEED, seed, k])
        res = {(v, m): np.empty(n_boot) for v in cf.VARIANTS
               for m in ("coverage", "mean_width", "coverage_far_bin", "winkler")}
        for b in range(n_boot):
            i = np.repeat(np.arange(len(cal)), w_cal[b].astype(int))
            fits = {
                "split": cf.fit_split(cy[i], cmu[i], alpha),
                "normalized": cf.fit_normalized(cy[i], cmu[i], csg[i], alpha),
                "mondrian": cf.fit_mondrian(cy[i], cmu[i], cnn[i], alpha),
                "mondrian_normalized": cf.fit_mondrian_normalized(cy[i], cmu[i], csg[i], cnn[i],
                                                                  alpha),
            }
            wb = w_test[b]
            for v, fit in fits.items():
                lo, hi = fit.predict(mu, sg, nn)
                hit = ((y >= lo) & (y <= hi)).astype(float)
                res[(v, "coverage")][b] = wmean(wb, hit)[0]
                res[(v, "mean_width")][b] = wmean(wb, hi - lo)[0]
                res[(v, "coverage_far_bin")][b] = (wmean(wb, hit * far) / wmean(wb, far))[0]
                res[(v, "winkler")][b] = wmean(wb, winkler_scores(y, lo, hi))[0]
        out.update({(source, v, m): arr for (v, m), arr in res.items()})
    return out


def _cal_ci(runs: list, key, runs_b=None, key_b=None) -> dict:
    """ci_lo_cal/ci_hi_cal if every run has calibration-resampled values for key."""
    if not all(len(r) > 2 and key in r[2] for r in runs):
        return {"ci_lo_cal": np.nan, "ci_hi_cal": np.nan}
    boot = np.mean([r[2][key] for r in runs], axis=0)
    if runs_b is not None:
        boot = boot - np.mean([r[2][key_b] for r in runs_b], axis=0)
    lo, hi = percentile_ci(boot)
    return {"ci_lo_cal": lo, "ci_hi_cal": hi}


def summarize(per_run: dict) -> pd.DataFrame:
    """per_run: {(task, method, seed): (full, boot[, boot_with_cal])} -> one row per metric.

    mean/sd are over seeds (full test set); the CI is the percentile interval of the
    seed-averaged metric over the shared bootstrap resamples.
    """
    rows = []
    cells = sorted({(t, m) for t, m, _ in per_run})
    for task, method in cells:
        seeds = sorted(s for t, m, s in per_run if (t, m) == (task, method))
        for key in per_run[(task, method, seeds[0])][0]:
            vals = np.array([per_run[(task, method, s)][0][key][0] for s in seeds])
            boot = np.mean([per_run[(task, method, s)][1][key] for s in seeds], axis=0)
            lo, hi = percentile_ci(boot)
            rows.append({"task": task, "method": method, "source": key[0], "variant": key[1],
                         "metric": key[2], "n_seeds": len(seeds), "mean": vals.mean(),
                         "sd": vals.std(ddof=1), "ci_lo": lo, "ci_hi": hi,
                         **_cal_ci([per_run[(task, method, s)] for s in seeds], key),
                         **{f"seed{s}": v for s, v in zip(seeds, vals)}})
    return pd.DataFrame(rows)


def _diff_row(tags, seeds, a, b, key_a, key_b) -> dict:
    d = np.array([a[s][0][key_a][0] - b[s][0][key_b][0] for s in seeds])
    boot = np.mean([a[s][1][key_a] - b[s][1][key_b] for s in seeds], axis=0)
    lo, hi = percentile_ci(boot)
    return {**tags, "n_seeds": len(seeds), "mean_diff": d.mean(), "sd_diff": d.std(ddof=1),
            "ci_lo": lo, "ci_hi": hi,
            **_cal_ci([a[s] for s in seeds], key_a, [b[s] for s in seeds], key_b),
            "n_seeds_positive": int((d > 0).sum()),
            **{f"seed{s}": v for s, v in zip(seeds, d)}}


def paired(per_run: dict) -> pd.DataFrame:
    """Paired per-seed differences with bootstrap CIs.

    "ensemble-mcd": every planned metric, ensemble minus MC dropout.
    "valid-naive_cal": coverage, width, far-bin coverage and Winkler score (post hoc, DR-006)
    of each conformal variant, `valid` calibration minus `naive_cal` calibration, for each base
    method. The post-hoc controls (size-matched, composition) have no paired comparison.
    """
    rows = []
    for task in sorted({t for t, _, _ in per_run}):
        by = {m: {s: v for (t, mm, s), v in per_run.items() if t == task and mm == m}
              for m in ("ensemble", "mcd")}
        seeds = sorted(set(by["ensemble"]) & set(by["mcd"]))
        for key in by["ensemble"][seeds[0]][0]:
            if key[0] not in ("", *cf.SOURCES) or key[2] == "winkler":  # post hoc, DR-006/007
                continue
            rows.append(_diff_row({"task": task, "comparison": "ensemble-mcd", "method": "",
                                   "source": key[0], "variant": key[1], "metric": key[2]},
                                  seeds, by["ensemble"], by["mcd"], key, key))
        for method in ("ensemble", "mcd"):
            for key in by[method][seeds[0]][0]:
                if key[0] != "valid":
                    continue
                naive = ("naive_cal", key[1], key[2])
                rows.append(_diff_row({"task": task, "comparison": "valid-naive_cal",
                                       "method": method, "source": "", "variant": key[1],
                                       "metric": key[2]},
                                      seeds, by[method], by[method], key, naive))
    return pd.DataFrame(rows)


def paired_by_seed_group(per_run: dict, groups: dict) -> pd.DataFrame:
    """"valid-naive_cal" rows of `paired` restricted to groups of seeds, e.g.
    {"1-5": (1, 2, 3, 4, 5), "1-4": (1, 2, 3, 4), "5": (5,)} (post hoc, DR-007). sd is NaN for a
    one-seed group; the bootstrap CIs use that seed's resamples only."""
    out = []
    for name, seeds in groups.items():
        p = paired({k: v for k, v in per_run.items() if k[2] in seeds})
        p = p[p.comparison == "valid-naive_cal"].copy()
        p.insert(1, "seeds", name)
        out.append(p)
    return pd.concat(out, ignore_index=True)
