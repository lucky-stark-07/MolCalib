"""Laptop: conformal intervals (T7) and test metrics (T8) for every prediction file.

Writes cache/conformal/*.parquet and results/metrics_*.csv. Statistics and the report (T9, T10)
are added to this script later.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from molcalib import config
from molcalib.conformal import intervals_for_run
from molcalib.metrics import evaluate_run
from molcalib.runner import git_commit

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
    for task in config.TASKS:
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
                log.info("%s seed %d %s: done", task, seed, method)

    commit = git_commit()
    for name, dfs in parts.items():
        df = pd.concat(dfs, ignore_index=True)
        df.insert(0, "git_commit", commit)
        df.to_csv(res_dir / f"metrics_{name}.csv", index=False, float_format="%.6g")
    log.info("wrote %s", ", ".join(f"metrics_{t}.csv" for t in TABLES))


if __name__ == "__main__":
    main()
