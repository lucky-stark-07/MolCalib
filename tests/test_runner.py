from pathlib import Path

import pandas as pd

from molcalib import runner

TASK = "caco2_wang"
SPLITS = Path(__file__).resolve().parents[1] / "splits"


def _run(tmp_path, **kw):
    return runner.run_grid(
        [TASK], [1], SPLITS, tmp_path / "preds", tmp_path / "compute.csv",
        "cpu", epochs=3, n_members=2, mc_t=3, **kw,
    )


def test_grid_order():
    g = runner.grid(["solubility_aqsoldb", TASK], [1, 2])
    assert [m for m, _, _ in g] == ["mcd"] * 4 + ["ensemble"] * 4  # MC dropout first
    assert [t for _, t, _ in g][:4] == [TASK, TASK, "solubility_aqsoldb", "solubility_aqsoldb"]


def test_resume_and_determinism(tmp_path):
    a = _run(tmp_path)
    assert a == {"done": 2, "skipped": 0, "failed": 0}
    first = {p.name: pd.read_parquet(p) for p in (tmp_path / "preds").glob("*_s1_*.parquet")}
    assert {"caco2_wang_s1_mcd.parquet", "caco2_wang_s1_ensemble.parquet"} <= set(first)
    assert not list((tmp_path / "preds").glob("*.tmp"))

    assert _run(tmp_path) == {"done": 0, "skipped": 2, "failed": 0}  # resumable

    log = pd.read_csv(tmp_path / "compute.csv")
    assert list(log["status"]) == ["ok", "ok"] and log["git_commit"].notna().all()

    # same seed, fresh directory -> identical predictions
    other = tmp_path / "again"
    other.mkdir()
    _run(other)
    for name, df in first.items():
        pd.testing.assert_frame_equal(df, pd.read_parquet(other / "preds" / name))


def test_failure_is_logged_and_skipped(tmp_path, monkeypatch):
    real = runner.run_task_seed

    def flaky(df, task, seed, method, *a, **k):
        if method == "mcd":
            raise RuntimeError("boom")
        return real(df, task, seed, method, *a, **k)

    monkeypatch.setattr(runner, "run_task_seed", flaky)
    assert _run(tmp_path) == {"done": 1, "skipped": 0, "failed": 1}
    assert list(pd.read_csv(tmp_path / "compute.csv")["status"]) == ["failed", "ok"]
    assert not (tmp_path / "preds" / "caco2_wang_s1_mcd.parquet").exists()
