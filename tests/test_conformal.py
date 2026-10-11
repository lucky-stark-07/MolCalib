import inspect
import math

import numpy as np
import pandas as pd

from molcalib import conformal as cf


def _data(rng, n):
    nn = rng.uniform(0, 1, n)
    sigma = 0.3 + 1.5 * nn  # noise grows with distance from the training set
    mu = rng.normal(0, 1, n)
    return mu + sigma * rng.normal(size=n), mu, sigma, nn


def test_quantile_level_and_too_few_points():
    s = np.arange(1.0, 10.0)  # n = 9, ceil(10 * 0.9) = 9 -> the maximum
    assert cf.conformal_quantile(s, 0.1) == 9.0
    assert cf.conformal_quantile(s[:8], 0.1) == math.inf  # n = 8: ceil(9 * 0.9) = 9 > n


def test_coverage_is_nominal_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cov = {"split": [], "normalized": [], "mondrian": [], "mondrian_normalized": []}
    bins = []
    for _ in range(300):
        yc, muc, sgc, nnc = _data(rng, 400)
        yt, mut, sgt, nnt = _data(rng, 1000)
        fits = {
            "split": cf.fit_split(yc, muc),
            "normalized": cf.fit_normalized(yc, muc, sgc),
            "mondrian": cf.fit_mondrian(yc, muc, nnc),
            "mondrian_normalized": cf.fit_mondrian_normalized(yc, muc, sgc, nnc),
        }
        for k, f in fits.items():
            lo, hi = f.predict(mut, sgt, nnt)
            hit = (yt >= lo) & (yt <= hi)
            cov[k].append(hit.mean())
            if k.startswith("mondrian"):
                far = nnt > 0.75
                bins.append(hit[far].mean())
    for k, v in cov.items():
        assert abs(np.mean(v) - 0.90) < 0.02, k
    assert abs(np.mean(bins) - 0.90) < 0.03  # Mondrian also covers in the far bin


def test_split_conformal_undercovers_far_bin_but_mondrian_does_not():
    rng = np.random.default_rng(1)
    far_split, far_mond = [], []
    for _ in range(200):
        yc, muc, _, nnc = _data(rng, 400)
        yt, mut, _, nnt = _data(rng, 1000)
        far = nnt > 0.75
        lo, hi = cf.fit_split(yc, muc).predict(mut)
        far_split.append(((yt >= lo) & (yt <= hi))[far].mean())
        lo, hi = cf.fit_mondrian(yc, muc, nnc).predict(mut, nn_dist=nnt)
        far_mond.append(((yt >= lo) & (yt <= hi))[far].mean())
    assert np.mean(far_split) < 0.85
    assert np.mean(far_mond) > 0.87


def test_bin_edges_with_ties():
    assert len(cf.nn_bin_edges(np.zeros(50))) == 0  # single bin
    nn = np.r_[np.zeros(40), np.linspace(0.1, 1, 60)]
    edges = cf.nn_bin_edges(nn)
    counts = np.bincount(np.searchsorted(edges, nn, side="left"), minlength=len(edges) + 1)
    assert (counts > 0).all()
    f = cf.fit_mondrian(np.random.default_rng(0).normal(size=100), np.zeros(100), nn)
    _, hi = f.predict(np.zeros(3), nn_dist=np.array([0.0, 0.5, 5.0]))  # beyond the edges is fine
    assert np.isfinite(hi).all()


def test_calibration_api_has_no_test_inputs():
    for fit in (cf.fit_split, cf.fit_normalized, cf.fit_mondrian, cf.fit_mondrian_normalized):
        assert not [p for p in inspect.signature(fit).parameters if "test" in p]
    assert not [p for p in inspect.signature(cf.Fitted.predict).parameters if p.startswith("y")]


def test_test_labels_do_not_change_intervals():
    rng = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame({
        "split": ["naive_cal"] * 40 + ["valid"] * 40 + ["test"] * 40,
        "y": rng.normal(size=n), "mu": rng.normal(size=n),
        "sigma_total": rng.uniform(0.5, 1.5, n), "nn_dist": rng.uniform(0, 1, n),
    })
    a = cf.intervals_for_run(df)
    df2 = df.copy()
    df2.loc[df2["split"] == "test", "y"] = 1e6  # mock: garbage test labels
    pd.testing.assert_frame_equal(a, cf.intervals_for_run(df2))
    assert set(a["source"]) == set(cf.SOURCES) and set(a["variant"]) == set(cf.VARIANTS)
    assert len(a) == 40 * 8
