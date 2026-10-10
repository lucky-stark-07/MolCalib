"""TDC loading, scaffolds and split construction (see docs/IMPLEMENTATION_PLAN.md, "Splits")."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

log = logging.getLogger(__name__)

NAIVE_CAL_FRAC = 0.1
SPLITS = ("train_fit", "naive_cal", "valid", "test")


def bemis_murcko(smiles: str) -> str | None:
    """Bemis-Murcko scaffold SMILES without chirality; None if the SMILES does not parse.

    Acyclic molecules give the empty string "", which is treated as one scaffold group.
    """
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Keep columns smiles, y, scaffold; drop unparseable SMILES and log the count."""
    out = pd.DataFrame({"smiles": df["Drug"].to_numpy(), "y": df["Y"].to_numpy()})
    out["scaffold"] = [bemis_murcko(s) for s in out["smiles"]]
    bad = int(out["scaffold"].isna().sum())
    if bad:
        log.warning("%s: dropped %d unparseable SMILES", name, bad)
    return out.dropna(subset=["scaffold"]).reset_index(drop=True)


def load_raw(task: str, seed: int, path: str = "cache/tdc"):
    """Return TDC (train, valid, test) frames for one task and seed."""
    from tdc.benchmark_group import admet_group

    Path(path).mkdir(parents=True, exist_ok=True)  # TDC uses a non-recursive os.mkdir
    group = admet_group(path=path)
    train, valid = group.get_train_valid_split(seed=seed, benchmark=task)
    test = group.get(task)["test"]
    return train, valid, test


def nn_distance(smiles: list[str], ref_smiles: list[str]) -> np.ndarray:
    """1 - max Tanimoto similarity to ref_smiles (Morgan radius 2, 2048 bits)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fps(smis):
        return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smis]

    ref = fps(ref_smiles)
    return np.array([1.0 - max(DataStructs.BulkTanimotoSimilarity(f, ref)) for f in fps(smiles)])


def build_split(task: str, seed: int, path: str = "cache/tdc") -> pd.DataFrame:
    """Columns: smiles, y, split, scaffold_id, nn_dist (NaN for train_fit)."""
    raw = load_raw(task, seed, path)
    train, valid, test = (
        clean(d, f"{task}/{n}") for d, n in zip(raw, ("train", "valid", "test"))
    )
    # naive_cal: seeded molecule-level 10% of train; the rest is train_fit
    rng = np.random.default_rng(seed)
    cal_idx = rng.choice(len(train), size=round(NAIVE_CAL_FRAC * len(train)), replace=False)
    is_cal = np.zeros(len(train), dtype=bool)
    is_cal[cal_idx] = True
    train["split"] = np.where(is_cal, "naive_cal", "train_fit")
    valid["split"] = "valid"
    test["split"] = "test"
    df = pd.concat([train, valid, test], ignore_index=True)
    df["scaffold_id"] = pd.factorize(df["scaffold"])[0]
    fit_smiles = df.loc[df["split"] == "train_fit", "smiles"].tolist()
    df["nn_dist"] = np.nan
    rest = df["split"] != "train_fit"
    df.loc[rest, "nn_dist"] = nn_distance(df.loc[rest, "smiles"].tolist(), fit_smiles)
    return df[["smiles", "y", "split", "scaffold_id", "nn_dist"]]
