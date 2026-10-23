from pathlib import Path

import pandas as pd
import pytest

from molcalib.audit import (
    LeakageError,
    audit_split,
    composition_rows,
    group_scaffold,
    largest_groups,
)
from molcalib.config import SEEDS, TASKS
from molcalib.data import bemis_murcko

SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"


def _df(rows):
    return pd.DataFrame(rows, columns=["split", "scaffold_id", "nn_dist"])


def test_bemis_murcko_no_chirality_and_invalid():
    assert bemis_murcko("C[C@H](N)c1ccccc1") == bemis_murcko("CC(N)c1ccccc1") == "c1ccccc1"
    assert bemis_murcko("CCO") == ""
    assert bemis_murcko("not a smiles") is None


def test_audit_detects_overlap():
    df = _df([("train_fit", 0, None), ("valid", 0, 0.5), ("test", 1, 0.5), ("naive_cal", 0, 0.5)])
    with pytest.raises(LeakageError):
        audit_split(df)


def test_audit_passes_on_disjoint():
    df = _df([("train_fit", 0, None), ("valid", 1, 0.5), ("test", 2, 0.5), ("naive_cal", 0, 0.1)])
    assert audit_split(df)["naive_cal_frac_molecules_sharing_scaffold_with_train_fit"] == 1.0


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("seed", SEEDS)
def test_real_splits_scaffold_disjoint(task, seed):
    df = pd.read_csv(SPLITS_DIR / f"{task}_seed{seed}.csv")
    audit_split(df)  # raises LeakageError on any overlap


def _mols(rows):
    """rows: (split, scaffold_id, smiles) -> split-file-like frame."""
    return pd.DataFrame(rows, columns=["split", "scaffold_id", "smiles"]).assign(nn_dist=0.5)


def test_largest_groups_reports_scaffold_shares_and_partitions():
    df = _mols([("train_fit", 0, "CCO"), ("train_fit", 0, "CCCC"), ("naive_cal", 0, "CCN"),
                ("train_fit", 1, "Cc1ccccc1"), ("naive_cal", 1, "Oc1ccccc1"),
                ("valid", 2, "C1CCCCC1"), ("test", 3, "c1ccncc1"), ("test", 3, "Cc1ccncc1")])
    top = largest_groups(df, k=2)
    assert [g["scaffold"] for g in top] == ["", "c1ccccc1"]  # acyclic group = ""
    assert top[0]["n"] == 3 and top[0]["in_splits"] == ["train_fit", "naive_cal"]
    assert top[0]["share_of_split"] == {"train_fit": 2 / 3, "naive_cal": 0.5, "valid": 0.0,
                                        "test": 0.0}
    assert largest_groups(df, k=4)[2]["scaffold_id"] == 3  # ties broken by id: 3 (n=2) > 2
    with pytest.raises(ValueError):
        group_scaffold(["CCO", "c1ccccc1"])  # one group must have one scaffold


def test_composition_rows_counts_removed_molecules():
    pred = _mols([("naive_cal", 0, "CCO")] * 6 + [("naive_cal", 1, "c1ccccc1O")] * 3
                 + [("naive_cal", 2, "C1CCC1")] + [("test", 3, "c1ccncc1")] * 2)
    rows = composition_rows(pred, {"task": "t"}, thresholds=(0.05, 0.5)).set_index("threshold")
    assert rows.at[0.05, "n_removed"] == 10 and rows.at[0.05, "n_remaining"] == 0
    assert rows.at[0.5, "n_removed"] == 6 and rows.at[0.5, "n_remaining"] == 4
    assert rows.at[0.5, "removed_scaffolds"] == "''"


@pytest.mark.parametrize("seed", SEEDS)
def test_solubility_dominant_groups_are_acyclic_and_benzene(seed):
    """The finding of DR-007, checked on the committed splits."""
    df = pd.read_csv(SPLITS_DIR / f"solubility_aqsoldb_seed{seed}.csv")
    top = {g["scaffold"]: g for g in largest_groups(df, k=2)}
    assert set(top) == {"", "c1ccccc1"}
    assert top[""]["n"] == 2940 and top["c1ccccc1"]["n"] == 1753
    assert all(g["n_by_split"]["test"] == 0 for g in top.values())
    if seed == 5:  # TDC put the whole acyclic group into valid
        assert top[""]["share_of_split"]["valid"] == 1.0
        assert top[""]["in_splits"] == ["valid"]
    else:
        assert all(g["n_by_split"]["valid"] == 0 for g in top.values())
        both = sum(g["share_of_split"]["naive_cal"] for g in top.values())
        assert 0.6 < both < 0.75  # about two-thirds of naive_cal
