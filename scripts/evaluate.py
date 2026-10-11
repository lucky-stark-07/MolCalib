"""Laptop: conformal intervals (T7), test metrics (T8) and statistics (T9).

Writes cache/conformal/*.parquet, results/metrics_*.csv and results/stats_*.csv. The report (T10)
is added to this script later.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from molcalib import config
from molcalib.conformal import intervals_for_run
from molcalib.metrics import evaluate_run
from molcalib.runner import git_commit
from molcalib.stats import group_weights, paired, run_metrics, summarize

log = logging.getLogger(__name__)

TABLES = ("point", "selective", "gaussian", "conformal", "nn_bins")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="cache/preds")
    ap.add_argument("--conformal-out", default="cache/conformal")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conf_dir, res_dir = Path(a.conformal_out), Path(a.results)
    conf_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    parts = {t: [] for t in TABLES}
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
                test = pred[pred["split"] == "test"]
                if weights is None:
                    weights = group_weights(test["scaffold_id"])
                per_run[(task, method, seed)] = (
                    run_metrics(test, iv, np.ones((1, len(test)))),
                    run_metrics(test, iv, weights),
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


if __name__ == "__main__":
    main()
