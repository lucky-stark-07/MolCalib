"""Split, normalized and Mondrian conformal prediction (T7).

The fit_* functions take calibration arrays only. Test inputs go to Fitted.predict, which has no
label argument, so test labels cannot reach calibration.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from molcalib import config

SOURCES = ("naive_cal", "valid")
VARIANTS = ("split", "normalized", "mondrian", "mondrian_normalized")
N_BINS = 4


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Empirical quantile at level ceil((n+1)(1-alpha))/n; inf if that exceeds n (too few points)."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(np.sort(scores)[k - 1]) if k <= n else math.inf


def nn_bin_edges(nn_cal: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Bin edges from calibration-set quantiles (quartiles by default).

    Duplicate edges are merged and edges that leave a bin empty are dropped, so every bin has
    at least one calibration point. A value x falls in bin searchsorted(edges, x, side="left").
    """
    edges = np.unique(np.quantile(nn_cal, np.linspace(0, 1, n_bins + 1)[1:-1]))
    while len(edges):
        counts = np.bincount(np.searchsorted(edges, nn_cal, side="left"), minlength=len(edges) + 1)
        empty = np.flatnonzero(counts == 0)
        if not len(empty):
            break
        edges = np.delete(edges, min(empty[0], len(edges) - 1))
    return edges


@dataclass(frozen=True)
class Fitted:
    kind: str
    q: np.ndarray  # one quantile, or one per Mondrian bin
    edges: np.ndarray | None = None

    def predict(self, mu, sigma=None, nn_dist=None) -> tuple[np.ndarray, np.ndarray]:
        """Intervals (lo, hi) for test inputs; sigma is needed for "normalized", nn_dist for
        and "mondrian_normalized"."""
        mu = np.asarray(mu, dtype=float)
        if self.kind == "split":
            half = np.full(mu.shape, self.q[0])
        elif self.kind == "normalized":
            half = self.q[0] * np.asarray(sigma, dtype=float)
        else:
            half = self.q[np.searchsorted(self.edges, np.asarray(nn_dist), side="left")]
            if self.kind == "mondrian_normalized":
                half = half * np.asarray(sigma, dtype=float)
        return mu - half, mu + half


def fit_split(y_cal, mu_cal, alpha: float = config.ALPHA) -> Fitted:
    """Score |y - mu|; interval mu +- q."""
    scores = np.abs(np.asarray(y_cal) - np.asarray(mu_cal))
    return Fitted("split", np.array([conformal_quantile(scores, alpha)]))


def fit_normalized(y_cal, mu_cal, sigma_cal, alpha: float = config.ALPHA) -> Fitted:
    """Score |y - mu| / sigma_total; interval mu +- q * sigma_total."""
    scores = np.abs(np.asarray(y_cal) - np.asarray(mu_cal)) / np.asarray(sigma_cal)
    return Fitted("normalized", np.array([conformal_quantile(scores, alpha)]))


def fit_mondrian(y_cal, mu_cal, nn_cal, alpha: float = config.ALPHA, n_bins: int = N_BINS) -> Fitted:
    """Split conformal with a separate quantile per nearest-neighbour-distance bin (score |y - mu|)."""
    scores = np.abs(np.asarray(y_cal) - np.asarray(mu_cal))
    nn_cal = np.asarray(nn_cal)
    edges = nn_bin_edges(nn_cal, n_bins)
    bins = np.searchsorted(edges, nn_cal, side="left")
    q = np.array([conformal_quantile(scores[bins == b], alpha) for b in range(len(edges) + 1)])
    return Fitted("mondrian", q, edges)


def fit_mondrian_normalized(
    y_cal, mu_cal, sigma_cal, nn_cal, alpha: float = config.ALPHA, n_bins: int = N_BINS
) -> Fitted:
    """Like fit_mondrian but the score is |y - mu| / sigma_total; interval mu +- q_bin * sigma_total."""
    scores = np.abs(np.asarray(y_cal) - np.asarray(mu_cal)) / np.asarray(sigma_cal)
    nn_cal = np.asarray(nn_cal)
    edges = nn_bin_edges(nn_cal, n_bins)
    bins = np.searchsorted(edges, nn_cal, side="left")
    q = np.array([conformal_quantile(scores[bins == b], alpha) for b in range(len(edges) + 1)])
    return Fitted("mondrian_normalized", q, edges)


def intervals_for_run(pred: pd.DataFrame, alpha: float = config.ALPHA) -> pd.DataFrame:
    """All (source, variant) intervals for one prediction file (columns of cache/preds).

    Returns long format: row (index into pred), source, variant, lo, hi, for test rows only.
    Calibration uses only the naive_cal or valid rows.
    """
    test = pred[pred["split"] == "test"]
    out = []
    for source in SOURCES:
        cal = pred[pred["split"] == source]
        fits = {
            "split": fit_split(cal["y"], cal["mu"], alpha),
            "normalized": fit_normalized(cal["y"], cal["mu"], cal["sigma_total"], alpha),
            "mondrian": fit_mondrian(cal["y"], cal["mu"], cal["nn_dist"], alpha),
            "mondrian_normalized": fit_mondrian_normalized(
                cal["y"], cal["mu"], cal["sigma_total"], cal["nn_dist"], alpha
            ),
        }
        for variant, fit in fits.items():
            lo, hi = fit.predict(test["mu"], test["sigma_total"], test["nn_dist"])
            out.append(pd.DataFrame(
                {"row": test.index, "source": source, "variant": variant, "lo": lo, "hi": hi}
            ))
    return pd.concat(out, ignore_index=True)


# Post-hoc size-matched control (DR-006)
SIZE_MATCHED_VARIANTS = ("split", "normalized")


def valid_subsamples(pred: pd.DataFrame, n: int, r: int, seed) -> np.ndarray:
    """(r, n) index labels of `valid` rows; each row is one subsample drawn without replacement."""
    valid = pred.index[pred["split"] == "valid"].to_numpy()
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(valid, n, replace=False) for _ in range(r)])


def size_matched_intervals(pred: pd.DataFrame, r: int = config.SIZE_MATCH_R, seed=None,
                           n: int | None = None, alpha: float = config.ALPHA) -> dict:
    """Split and normalized conformal refit on r subsamples of `valid` with n rows each (default:
    the naive_cal size of this prediction file).

    Returns {variant: (lo, hi)} for test rows, each of shape (r, n_test).
    """
    test = pred[pred["split"] == "test"]
    n = int((pred["split"] == "naive_cal").sum()) if n is None else n
    seed = config.SIZE_MATCH_SEED if seed is None else seed
    los = {v: [] for v in SIZE_MATCHED_VARIANTS}
    his = {v: [] for v in SIZE_MATCHED_VARIANTS}
    for idx in valid_subsamples(pred, n, r, seed):
        cal = pred.loc[idx]
        fits = {
            "split": fit_split(cal["y"], cal["mu"], alpha),
            "normalized": fit_normalized(cal["y"], cal["mu"], cal["sigma_total"], alpha),
        }
        for variant, fit in fits.items():
            lo, hi = fit.predict(test["mu"], test["sigma_total"])
            los[variant].append(lo)
            his[variant].append(hi)
    return {v: (np.stack(los[v]), np.stack(his[v])) for v in SIZE_MATCHED_VARIANTS}
