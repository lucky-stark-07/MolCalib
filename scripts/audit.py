"""Laptop: audit splits/*.csv -> results/split_audit.json, results/fig_nn_distance.png."""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from molcalib.audit import DIST_SPLITS, audit_split, largest_groups
from molcalib.config import SEEDS, TASKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    report = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runs": {},
    }
    frames = {}
    for task in TASKS:
        for seed in SEEDS:
            df = pd.read_csv(Path(args.splits) / f"{task}_seed{seed}.csv")
            run = audit_split(df)  # raises on any overlap
            run["largest_groups"] = largest_groups(df)  # post hoc, DR-007
            report["runs"][f"{task}_seed{seed}"] = run
            frames[(task, seed)] = df
    (out / "split_audit.json").write_text(json.dumps(report, indent=2))

    # Seed 1 only for the figure; the other seeds are in the json.
    fig, axes = plt.subplots(1, len(TASKS), figsize=(4 * len(TASKS), 3), sharey=False)
    for ax, task in zip(axes, TASKS):
        df = frames[(task, SEEDS[0])]
        for s in DIST_SPLITS:
            ax.hist(df.loc[df["split"] == s, "nn_dist"], bins=30, range=(0, 1),
                    histtype="step", density=True, label=s)
        ax.set_title(f"{task} (seed {SEEDS[0]})", fontsize=9)
        ax.set_xlabel("1 - max Tanimoto to train_fit")
    axes[0].set_ylabel("density")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "fig_nn_distance.png", dpi=150)


if __name__ == "__main__":
    main()
