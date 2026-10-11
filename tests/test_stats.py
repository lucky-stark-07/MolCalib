import numpy as np
import pandas as pd

from molcalib import stats
from molcalib.conformal import intervals_for_run
from molcalib.metrics import evaluate_run


def test_group_weights_resample_whole_groups():
    groups = np.array([5, 5, 5, 9, 9, 2, 7, 7, 7, 7])
    w = stats.group_weights(groups, n_boot=200, seed=3)
    assert w.shape == (200, 10)
    for g in np.unique(groups):  # molecules of one scaffold always share a weight
        assert (w[:, groups == g] == w[:, groups == g][:, :1]).all()
    first = w[:, [0, 3, 5, 6]]  # one molecule per group: counts of each group
    assert (first.sum(axis=1) == 4).all()  # each resample draws n_groups groups
    np.testing.assert_array_equal(w, stats.group_weights(groups, n_boot=200, seed=3))


def test_cluster_bootstrap_is_wider_than_iid_for_correlated_groups():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 10)
    x = np.repeat(rng.normal(size=40), 10) + 0.1 * rng.normal(size=400)  # strong group effect
    cluster = stats.wmean(stats.group_weights(groups, 1000, 1), x)
    iid = stats.wmean(stats.group_weights(np.arange(400), 1000, 1), x)
    assert cluster.std() > 2 * iid.std()


def _pred(seed, shift=0.0):
    rng = np.random.default_rng(seed)
    n = 90
    return pd.DataFrame({
        "split": ["naive_cal"] * 30 + ["valid"] * 30 + ["test"] * 30,
        "scaffold_id": np.r_[np.arange(60), np.repeat(np.arange(100, 110), 3)],
        "y": rng.normal(size=n), "mu": rng.normal(size=n) + shift,
        "sigma_total": rng.uniform(0.5, 1.5, n), "nn_dist": rng.uniform(0, 1, n),
    })


def _per_run(preds):
    per_run, w = {}, None
    for key, pred in preds.items():
        iv = intervals_for_run(pred)
        test = pred[pred["split"] == "test"]
        w = stats.group_weights(test["scaffold_id"], 300, 0) if w is None else w
        per_run[key] = (stats.run_metrics(test, iv, np.ones((1, len(test)))),
                        stats.run_metrics(test, iv, w))
    return per_run


def test_full_sample_matches_t8_metrics():
    pred = _pred(0)
    iv = intervals_for_run(pred)
    test = pred[pred["split"] == "test"]
    full = stats.run_metrics(test, iv, np.ones((1, len(test))))
    point, _, _, conf, _ = evaluate_run(pred, iv, {})
    for m in ("mae", "rmse", "nll", "miscal_area", "spearman_sigma_abserr"):
        assert np.isclose(full[("", "", m)][0], point[m].item())
    for _, r in conf.iterrows():
        assert np.isclose(full[(r.source, r.variant, "coverage")][0], r.coverage)


def test_summary_and_paired():
    preds = {("t", m, s): _pred(s, shift=0.0) for m in ("ensemble", "mcd") for s in (1, 2, 3)}
    per_run = _per_run(preds)
    summ = stats.summarize(per_run)
    fin = summ[np.isfinite(summ["mean"])]
    assert ((fin.ci_lo <= fin["mean"] + 1e-12) & (fin["mean"] <= fin.ci_hi + 1e-12)).all()
    inf = summ[np.isinf(summ["mean"])]  # tiny Mondrian bins -> infinite width, reported as inf
    assert len(inf) and np.isinf(inf.ci_hi).all()
    pair = stats.paired(per_run)
    same = pair[(pair.comparison == "ensemble-mcd") & pair.mean_diff.notna()]  # inf - inf is NaN
    assert np.allclose(same[["mean_diff", "ci_lo", "ci_hi"]].to_numpy(), 0)
    assert set(pair.comparison) == {"ensemble-mcd", "valid-naive_cal"}


def test_wmean_handles_infinite_values():
    x = np.array([1.0, np.inf, 3.0])
    w = np.array([[1, 0, 1], [1, 2, 1]], dtype=float)
    np.testing.assert_array_equal(stats.wmean(w, x), [2.0, np.inf])
