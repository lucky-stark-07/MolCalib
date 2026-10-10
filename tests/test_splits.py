from pathlib import Path

import pandas as pd
import pytest

from molcalib.audit import LeakageError, audit_split
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
