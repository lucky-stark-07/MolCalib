import math

import numpy as np
import pandas as pd
import pytest

from molcalib import metrics as mt


def test_point_metrics_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    mu = np.array([1.0, 2.0, 3.0, 2.0])  # one error of 2
    m = mt.point_metrics(y, mu, np.ones(4))
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["nll"] == pytest.approx(0.5 * math.log(2 * math.pi) + 4 / 2 / 4)
    s = np.array([0.1, 0.2, 0.3, 5.0])
    assert mt.point_metrics(y, mu, s)["spearman_sigma_abserr"] > 0


def test_gaussian_calibration_detects_over_and_good_calibration():
    rng = np.random.default_rng(0)
    n = 200_000
    sigma = rng.uniform(0.5, 2.0, n)
    y = sigma * rng.normal(size=n)
    obs, area, _ = mt.gaussian_calibration(y, np.zeros(n), sigma)
    np.testing.assert_allclose(obs, mt.GAUSS_LEVELS, atol=0.01)
    assert area < 0.01
    _, area2, signed2 = mt.gaussian_calibration(y, np.zeros(n), sigma / 2)  # overconfident
    assert area2 > 0.1 and signed2 < -0.1


def test_selective_rmse():
    err = np.array([0.1, 0.2, 0.3, 4.0])
    r = mt.selective_rmse(err, np.zeros(4), sigma=err, keep_fracs=(1.0, 0.75, 0.5))
    assert r[1.0] == pytest.approx(math.sqrt(np.mean(err**2)))
    assert r[0.75] == pytest.approx(math.sqrt(np.mean(err[:3] ** 2)))
    assert r[1.0] > r[0.75] > r[0.5]


def test_interval_and_bin_coverage():
    y = np.arange(8.0)
    lo, hi = y - 1, y + 1
    hi[0] = y[0] - 0.5  # one miss
    m = mt.interval_metrics(y, lo, hi)
    assert m["coverage"] == pytest.approx(7 / 8) and m["n_infinite"] == 0
    rows = mt.coverage_by_nn_bin(y, lo, hi, nn=np.linspace(0, 1, 8))
    assert sum(r["n"] for r in rows) == 8 and len(rows) == 4
    assert rows[0]["coverage"] == pytest.approx(0.5)


def test_evaluate_run_shapes():
    rng = np.random.default_rng(1)
    n = 60
    pred = pd.DataFrame({
        "split": ["naive_cal"] * 20 + ["valid"] * 20 + ["test"] * 20,
        "y": rng.normal(size=n), "mu": rng.normal(size=n),
        "sigma_total": rng.uniform(0.5, 1.5, n), "nn_dist": rng.uniform(0, 1, n),
    })
    from molcalib.conformal import VARIANTS, intervals_for_run

    point, sel, gauss, conf, bins = mt.evaluate_run(pred, intervals_for_run(pred), {"task": "t"})
    assert len(point) == 1 and point["n_test"].item() == 20
    assert len(sel) == len(mt.KEEP_FRACS) and len(gauss) == len(mt.GAUSS_LEVELS)
    assert len(conf) == 2 * len(VARIANTS)
    assert bins.groupby(["source", "variant"])["n"].sum().eq(20).all()
