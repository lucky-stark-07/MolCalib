"""Split audit (T3): scaffold leakage checks and distance-to-train summaries.

Post hoc (DR-007): the largest scaffold groups per split file, and the naive_cal groups removed
by the composition control.
"""

import pandas as pd

from molcalib import config
from molcalib.conformal import composition_source, dominant_test_absent_groups
from molcalib.data import SPLITS, bemis_murcko

# Pairs that must share no scaffold. Any overlap is a hard failure.
DISJOINT_PAIRS = (("train_fit", "valid"), ("train_fit", "test"), ("valid", "test"))
DIST_SPLITS = ("naive_cal", "valid", "test")


class LeakageError(AssertionError):
    pass


def audit_split(df: pd.DataFrame) -> dict:
    """Audit one split file (columns split, scaffold_id, nn_dist). Raises LeakageError on overlap."""
    sc = {s: set(g["scaffold_id"]) for s, g in df.groupby("split")}
    overlaps = {f"{a}|{b}": len(sc[a] & sc[b]) for a, b in DISJOINT_PAIRS}
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise LeakageError(f"scaffold overlap between splits: {bad}")

    cal = df[df["split"] == "naive_cal"]
    shared = cal["scaffold_id"].isin(sc["train_fit"])
    sizes = {s: g.groupby("scaffold_id").size() for s, g in df.groupby("split")}
    return {
        "n": df["split"].value_counts().to_dict(),
        "scaffold_overlap": overlaps,
        "naive_cal_frac_molecules_sharing_scaffold_with_train_fit": float(shared.mean()),
        "n_scaffold_groups": {s: len(v) for s, v in sizes.items()},
        "largest_group": {s: int(v.max()) for s, v in sizes.items()},
        "largest_group_frac_of_valid": float(sizes["valid"].max() / sizes["valid"].sum()),
        "nn_dist": {
            s: df.loc[df["split"] == s, "nn_dist"]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .round(4)
            .to_dict()
            for s in DIST_SPLITS
        },
    }


def group_scaffold(smiles) -> str:
    """The one Bemis-Murcko scaffold shared by all molecules of a scaffold group ("" = acyclic)."""
    scaffolds = {bemis_murcko(s) for s in smiles}
    if len(scaffolds) != 1:
        raise ValueError(f"scaffold group with several scaffolds: {sorted(map(str, scaffolds))}")
    return scaffolds.pop()


def largest_groups(df: pd.DataFrame, k: int = config.N_LARGEST_GROUPS) -> list[dict]:
    """The k largest scaffold groups of a split file (columns smiles, split, scaffold_id), post hoc
    (DR-007): scaffold SMILES, size, molecules and share of each partition, partitions holding it."""
    totals = df["split"].value_counts()
    counts = df.groupby(["scaffold_id", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(SPLITS), fill_value=0)
    n = counts.sum(axis=1)
    top = sorted(n.index, key=lambda g: (-n[g], g))[:k]  # ties broken by scaffold_id
    rows = []
    for g in top:
        c = counts.loc[g]
        rows.append({
            "scaffold_id": int(g),
            "scaffold": group_scaffold(df.loc[df["scaffold_id"] == g, "smiles"]),
            "n": int(n[g]),
            "n_by_split": {s: int(c[s]) for s in SPLITS},
            "share_of_split": {s: float(c[s] / totals[s]) if totals.get(s) else 0.0
                               for s in SPLITS},
            "in_splits": [s for s in SPLITS if c[s]],
        })
    return rows


def composition_rows(pred: pd.DataFrame, keys: dict,
                     thresholds=config.COMPOSITION_THRESHOLDS) -> pd.DataFrame:
    """Per threshold: naive_cal size, molecules removed and remaining, removed scaffolds (DR-007)."""
    naive = pred[pred["split"] == "naive_cal"]
    rows = []
    for t in thresholds:
        groups = dominant_test_absent_groups(pred, t)
        removed = naive[naive["scaffold_id"].isin(groups)]
        rows.append({**keys, "source": composition_source(t), "threshold": t,
                     "n_naive_cal": len(naive), "n_removed": len(removed),
                     "n_remaining": len(naive) - len(removed), "n_groups_removed": len(groups),
                     "removed_scaffolds": " ".join(
                         repr(group_scaffold(removed.loc[removed["scaffold_id"] == g, "smiles"]))
                         for g in groups)})
    return pd.DataFrame(rows)
