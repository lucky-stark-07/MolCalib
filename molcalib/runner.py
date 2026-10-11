"""Run grid for T5: resumable, errors are logged and skipped, never fatal."""

import csv
import logging
import os
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from molcalib import config
from molcalib.train import run_task_seed

log = logging.getLogger(__name__)

METHODS = ("mcd", "ensemble")  # all MC dropout runs first, then the ensembles
COMPUTE_FIELDS = [
    "timestamp", "git_commit", "task", "seed", "method", "device", "epochs", "n_models",
    "train_s", "wall_s", "status",
]


def git_commit() -> str:
    """Commit hash of the code; MOLCALIB_COMMIT overrides it (no git checkout on Kaggle)."""
    if os.environ.get("MOLCALIB_COMMIT"):
        return os.environ["MOLCALIB_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def grid(tasks, seeds) -> list[tuple[str, str, int]]:
    """(method, task, seed) in run order: MC dropout before ensembles, smallest task first."""
    tasks = [t for t in config.TASKS if t in tasks]  # config.TASKS is ordered by dataset size
    return [(m, t, s) for m in METHODS for t in tasks for s in seeds]


def out_path(out_dir: Path, task: str, seed: int, method: str) -> Path:
    return out_dir / f"{task}_s{seed}_{method}.parquet"


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPUTE_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_grid(
    tasks, seeds, splits_dir: Path, out_dir: Path, compute_csv: Path, device: str,
    epochs: int = config.EPOCHS, n_members: int = config.N_MEMBERS, mc_t: int = config.MC_T,
) -> dict[str, int]:
    commit = git_commit()
    counts = {"done": 0, "skipped": 0, "failed": 0}
    for method, task, seed in grid(tasks, seeds):
        label = f"{task} seed {seed} {method}"
        if out_path(out_dir, task, seed, method).exists():
            log.info("skip (exists): %s", label)
            counts["skipped"] += 1
            continue
        t0 = time.perf_counter()
        row = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": commit, "task": task, "seed": seed, "method": method,
            "device": device, "epochs": epochs,
            "n_models": n_members if method == "ensemble" else 1,
        }
        try:
            df = pd.read_csv(splits_dir / f"{task}_seed{seed}.csv")
            _, secs = run_task_seed(
                df, task, seed, method, out_dir, epochs, n_members, mc_t, device
            )
            row.update(train_s=round(sum(secs), 2), status="ok")
            counts["done"] += 1
            log.info("done: %s (train %.0fs)", label, sum(secs))
        except Exception:  # noqa: BLE001  per-run errors are logged and skipped, never fatal
            log.error("FAILED: %s\n%s", label, traceback.format_exc())
            row.update(train_s="", status="failed")
            counts["failed"] += 1
        row["wall_s"] = round(time.perf_counter() - t0, 2)
        _append_row(compute_csv, row)
    log.info("finished: %s", counts)
    return counts
