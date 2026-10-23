"""Laptop: conformal intervals (T7), test metrics (T8), statistics (T9) and report (T10).

Also the post-hoc checks of DR-006 (size-matched `valid` control, Winkler score, median
absolute error per split). Writes cache/conformal/*.parquet, results/metrics_*.csv,
results/stats_*.csv, results/summary.md and results/fig_*.png. `--report-only` rebuilds the
report from results/.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from molcalib import config
from molcalib.conformal import intervals_for_run, size_matched_intervals
from molcalib.metrics import evaluate_run, median_abs_error, size_matched_metrics
from molcalib.report import write_report
from molcalib.runner import git_commit
from molcalib.stats import (
    conformal_boot_with_cal,
    group_weights,
    paired,
    run_metrics,
    summarize,
)

log = logging.getLogger(__name__)

TABLES = ("point", "selective", "gaussian", "conformal", "nn_bins")
POST_HOC = ("size_matched", "abs_error")  # DR-006


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="cache/preds")
    ap.add_argument("--conformal-out", default="cache/conformal")
    ap.add_argument("--results", default="results")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.report_only:
        log.info("wrote %s", write_report(Path(a.results)))
        return
    conf_dir, res_dir = Path(a.conformal_out), Path(a.results)
    conf_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    parts = {t: [] for t in TABLES + POST_HOC}
    per_run = {}
    for task in config.TASKS:
        weights = None  # one set of scaffold-group resamples per task, shared by all runs
        for seed in config.SEEDS:
            for method in ("ensemble", "mcd"):
                src = Path(a.preds) / f"{task}_s{seed}_{method}.parquet"
                if not src.exists():
                    log.warning("missing: %s", src)
                    continue
                pred = pd.read_parquet(src)
                iv = intervals_for_run(pred, config.ALPHA)
                iv.to_parquet(conf_dir / f"{task}_s{seed}_{method}.parquet", index=False)
                keys = {"task": task, "seed": seed, "method": method}
                for name, df in zip(TABLES, evaluate_run(pred, iv, keys)):
                    parts[name].append(df)
                # same subsamples for ensemble and MC dropout of one seed
                sm = size_matched_intervals(pred, seed=[config.SIZE_MATCH_SEED, seed])
                parts["size_matched"].append(size_matched_metrics(pred, sm, keys))
                parts["abs_error"].append(median_abs_error(pred, keys))
                test = pred[pred["split"] == "test"]
                if weights is None:
                    weights = group_weights(test["scaffold_id"])
                per_run[(task, method, seed)] = (
                    run_metrics(test, iv, np.ones((1, len(test))), sm),
                    run_metrics(test, iv, weights, sm),
                    conformal_boot_with_cal(pred, weights, seed, config.ALPHA),
                )
                log.info("%s seed %d %s: done", task, seed, method)

    tables = {f"metrics_{n}": pd.concat(dfs, ignore_index=True) for n, dfs in parts.items()}
    tables["stats_summary"] = summarize(per_run)
    tables["stats_paired"] = paired(per_run)
    commit = git_commit()
    for name, df in tables.items():
        df.insert(0, "git_commit", commit)
        df.to_csv(res_dir / f"{name}.csv", index=False, float_format="%.6g")
    log.info("wrote %s", ", ".join(f"{n}.csv" for n in tables))
    log.info("wrote %s", write_report(res_dir))


if __name__ == "__main__":
    main()
