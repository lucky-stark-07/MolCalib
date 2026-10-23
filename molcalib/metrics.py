"""Test-set metrics (T8). Pure functions of test labels and already-fitted outputs; nothing is fit here."""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from molcalib import config
from molcalib.conformal import nn_bin_edges

GAUSS_LEVELS = np.round(np.arange(0.1, 1.0, 0.1), 1)  # 0.1 ... 0.9
KEEP_FRACS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
N_BINS = 4


def point_metrics(y, mu, sigma) -> dict:
    """MAE, RMSE, Gaussian NLL (with sigma_total) and Spearman(sigma, |error|)."""
    y, mu, sigma = (np.asarray(a, dtype=float) for a in (y, mu, sigma))
    err = y - mu
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + err**2 / (2 * sigma**2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "nll": float(np.mean(nll)),
        "spearman_sigma_abserr": float(spearmanr(sigma, np.abs(err))[0]),
    }


def gaussian_calibration(y, mu, sigma, levels=GAUSS_LEVELS) -> tuple[np.ndarray, float, float]:
    """Observed coverage of central Gaussian intervals mu +- z_{(1+p)/2} sigma at each level p.

    Returns (observed, miscalibration_area, signed_gap). miscalibration_area is the mean
    |observed - expected| over the levels; signed_gap is the mean (observed - expected), negative
    when intervals are too narrow (overconfident).
    """
    abs_z = np.abs(np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / np.asarray(sigma)
    levels = np.asarray(levels)
    observed = np.array([np.mean(abs_z <= norm.ppf(0.5 + p / 2)) for p in levels])
    gap = observed - levels
    return observed, float(np.mean(np.abs(gap))), float(np.mean(gap))


def selective_rmse(y, mu, sigma, keep_fracs=KEEP_FRACS) -> dict[float, float]:
    """RMSE on the most confident fraction of molecules (smallest sigma first; ties keep order)."""
    y, mu, sigma = (np.asarray(a, dtype=float) for a in (y, mu, sigma))
    order = np.argsort(sigma, kind="stable")
    out = {}
    for f in keep_fracs:
        keep = order[: math.ceil(f * len(y))]
        out[f] = float(math.sqrt(np.mean((y[keep] - mu[keep]) ** 2)))
    return out


def winkler_scores(y, lo, hi, alpha: float = config.ALPHA) -> np.ndarray:
    """Per-molecule Winkler (interval) score, post hoc (DR-006):
    (hi - lo) + (2/alpha)(lo - y)[y < lo] + (2/alpha)(y - hi)[y > hi]. Lower is better.
    An infinite interval scores inf (never a miss, so no inf * 0)."""
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    below = np.where(y < lo, lo - y, 0.0)
    above = np.where(y > hi, y - hi, 0.0)
    return (hi - lo) + (2 / alpha) * (below + above)


def interval_metrics(y, lo, hi) -> dict:
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    width = hi - lo
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "n_infinite": int(np.isinf(width).sum()),
        "winkler": float(np.mean(winkler_scores(y, lo, hi))),
    }


def size_matched_per_molecule(y, size_matched: dict) -> dict:
    """{variant: {metric: array(n_test)}}: coverage hit, width and Winkler score of each test
    molecule, averaged over the size-matched subsamples (DR-006). The mean over molecules is the
    mean over subsamples of the test-set metric."""
    y = np.asarray(y, dtype=float)
    out = {}
    for variant, (lo, hi) in size_matched.items():
        out[variant] = {"coverage": ((y >= lo) & (y <= hi)).mean(axis=0),
                        "mean_width": (hi - lo).mean(axis=0),
                        "winkler": winkler_scores(y, lo, hi).mean(axis=0)}
    return out


def size_matched_metrics(pred: pd.DataFrame, size_matched: dict, keys: dict) -> pd.DataFrame:
    """One row per variant: test metrics averaged over the size-matched subsamples (DR-006)."""
    y = pred.loc[pred["split"] == "test", "y"]
    n_cal = int((pred["split"] == "naive_cal").sum())
    return pd.DataFrame([
        {**keys, "source": "valid_sizematched", "variant": v, "n_cal": n_cal,
         "n_subsamples": len(size_matched[v][0]), **{m: float(np.mean(a)) for m, a in d.items()}}
        for v, d in size_matched_per_molecule(y, size_matched).items()
    ])


def median_abs_error(pred: pd.DataFrame, keys: dict) -> pd.DataFrame:
    """Median |y - mu| on naive_cal, valid and test (post-hoc descriptive check, DR-006)."""
    return pd.DataFrame([
        {**keys, "split": s, "n": int(m.sum()),
         "median_abs_error": float(np.median(np.abs(pred.loc[m, "y"] - pred.loc[m, "mu"])))}
        for s in ("naive_cal", "valid", "test") for m in [pred["split"] == s]
    ])


def coverage_by_nn_bin(y, lo, hi, nn, n_bins=N_BINS) -> list[dict]:
    """Coverage and width per test nearest-neighbour-distance bin (test-set quartiles)."""
    y, lo, hi, nn = (np.asarray(a, dtype=float) for a in (y, lo, hi, nn))
    edges = nn_bin_edges(nn, n_bins)
    bins = np.searchsorted(edges, nn, side="left")
    rows = []
    for b in range(len(edges) + 1):
        m = bins == b
        rows.append({
            "bin": b, "n": int(m.sum()),
            "nn_min": float(nn[m].min()), "nn_max": float(nn[m].max()),
            **{k: v for k, v in interval_metrics(y[m], lo[m], hi[m]).items()
               if k not in ("median_width", "winkler")},
        })
    return rows


def evaluate_run(pred: pd.DataFrame, intervals: pd.DataFrame, keys: dict):
    """Metrics for one (task, seed, method) prediction file and its conformal intervals.

    Returns (point, selective, gaussian, conformal, bins) DataFrames, each prefixed with `keys`.
    """
    test = pred[pred["split"] == "test"]
    y, mu, sg = test["y"], test["mu"], test["sigma_total"]
    observed, area, signed = gaussian_calibration(y, mu, sg)
    point = pd.DataFrame([{
        **keys, "n_test": len(test), **point_metrics(y, mu, sg),
        "miscal_area": area, "miscal_signed": signed,
    }])
    gauss = pd.DataFrame({**keys, "level": GAUSS_LEVELS, "observed": observed})
    sel = pd.DataFrame([{**keys, "keep_frac": f, "rmse": r}
                        for f, r in selective_rmse(y, mu, sg).items()])

    conf, bins = [], []
    for (source, variant), iv in intervals.groupby(["source", "variant"], sort=False):
        t = test.loc[iv["row"].to_numpy()]
        lo, hi = iv["lo"].to_numpy(), iv["hi"].to_numpy()
        tag = {**keys, "source": source, "variant": variant}
        conf.append({**tag, **interval_metrics(t["y"], lo, hi)})
        bins += [{**tag, **r} for r in coverage_by_nn_bin(t["y"], lo, hi, t["nn_dist"])]
    return point, sel, gauss, pd.DataFrame(conf), pd.DataFrame(bins)
