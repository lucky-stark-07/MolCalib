"""Laptop: conformal intervals for every prediction file -> cache/conformal/ (T7).

Metrics, statistics and the report (T8-T10) are added to this script later.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from molcalib import config
from molcalib.conformal import intervals_for_run

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="cache/preds")
    ap.add_argument("--out", default="cache/conformal")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in config.TASKS:
        for seed in config.SEEDS:
            for method in ("ensemble", "mcd"):
                src = Path(a.preds) / f"{task}_s{seed}_{method}.parquet"
                if not src.exists():
                    log.warning("missing: %s", src)
                    continue
                iv = intervals_for_run(pd.read_parquet(src), config.ALPHA)
                iv.to_parquet(out / f"{task}_s{seed}_{method}.parquet", index=False)
                log.info("%s seed %d %s: %d interval rows", task, seed, method, len(iv))


if __name__ == "__main__":
    main()
