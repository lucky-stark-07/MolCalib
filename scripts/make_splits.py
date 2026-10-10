"""Laptop: write splits/{task}_seed{seed}.csv for upload to Kaggle."""

import argparse
import logging
from pathlib import Path

from molcalib.config import SEEDS, TASKS
from molcalib.data import build_split

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--out", default="splits")
    ap.add_argument("--tdc-path", default="cache/tdc")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for task in args.tasks:
        for seed in args.seeds:
            df = build_split(task, seed, args.tdc_path)
            df.to_csv(out / f"{task}_seed{seed}.csv", index=False)
            log.info("%s seed %d: %s", task, seed, df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
