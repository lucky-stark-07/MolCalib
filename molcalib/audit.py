"""Split audit (T3): scaffold leakage checks and distance-to-train summaries."""

import pandas as pd

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
