"""Kaggle/laptop: resumable training and prediction over the run grid (see molcalib/runner.py)."""

import argparse
import logging
from pathlib import Path

import torch

from molcalib import config
from molcalib.runner import run_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N")
    ap.add_argument("--smoke", action="store_true",
                    help="caco2 only, seed 1, 3 epochs, 2 members, T=5; writes under cache/smoke/")
    ap.add_argument("--tasks", nargs="+", default=list(config.TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out", default=None, help="prediction dir (default cache/preds)")
    ap.add_argument("--compute-csv", default=None, help="default results/compute.csv")
    ap.add_argument("--log", default=None, help="progress log (default cache/progress.log)")
    args = ap.parse_args()

    epochs, n_members, mc_t = config.EPOCHS, config.N_MEMBERS, config.MC_T
    base = Path("cache/smoke") if args.smoke else None
    if args.smoke:
        args.tasks, args.seeds = ["caco2_wang"], [1]
        epochs, n_members, mc_t = 3, 2, 5
    out = Path(args.out or (base / "preds" if base else "cache/preds"))
    compute = Path(args.compute_csv or (base / "compute.csv" if base else "results/compute.csv"))
    log_path = Path(args.log or (base / "progress.log" if base else "cache/progress.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    run_grid(args.tasks, args.seeds, Path(args.splits), out, compute, args.device,
             epochs, n_members, mc_t)


if __name__ == "__main__":
    main()
