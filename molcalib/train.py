"""Training and prediction for deep ensembles and MC dropout (T4).

Training uses train_fit only, with no validation loader and no checkpoint selection: the model
after the last epoch is used.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from lightning import pytorch as pl

from molcalib import config
from molcalib.models import aggregate, build_model, predict_mve, set_mc_dropout

log = logging.getLogger(__name__)

PRED_SPLITS = ("naive_cal", "valid", "test")


def _dataset(smiles, y=None) -> MoleculeDataset:
    ys = [None] * len(smiles) if y is None else y
    pts = [MoleculeDatapoint.from_smi(s, None if v is None else np.array([v])) for s, v in zip(smiles, ys)]
    ds = MoleculeDataset(pts)
    ds.cache = True
    return ds


def train_model(train_df: pd.DataFrame, seed: int, epochs: int, device: str):
    """Train one model on train_fit rows (columns smiles, y). Returns the last-epoch model."""
    pl.seed_everything(seed, workers=True)
    ds = _dataset(train_df["smiles"].tolist(), train_df["y"].tolist())
    scaler = ds.normalize_targets()  # train_fit statistics only
    loader = build_dataloader(ds, batch_size=config.BATCH_SIZE, num_workers=0, seed=seed)
    model = build_model(scaler)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if device.startswith("cuda") else "cpu",
        devices=[int(device.split(":")[1])] if ":" in device else 1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, loader)
    return model


def _eval_loader(smiles):
    return build_dataloader(_dataset(smiles), batch_size=config.BATCH_SIZE, shuffle=False)


def _frame(df: pd.DataFrame, agg: dict[str, np.ndarray]) -> pd.DataFrame:
    out = df[["smiles", "split", "scaffold_id", "y", "nn_dist"]].copy()
    for k, v in agg.items():
        out[k] = v
    return out[["smiles", "split", "scaffold_id", "y", "mu", "sigma_alea", "sigma_epi",
                "sigma_total", "nn_dist"]]


def run_task_seed(
    split_df: pd.DataFrame,
    task: str,
    seed: int,
    method: str,
    out_dir: Path,
    epochs: int = config.EPOCHS,
    n_members: int = config.N_MEMBERS,
    mc_t: int = config.MC_T,
    device: str = "cpu",
) -> Path:
    """Train and predict for one (task, split seed, method); write cache/preds-style parquet.

    method "ensemble": n_members models, dropout off at inference; the raw per-member
    predictions are also written to {task}_s{seed}_ensemble_members.parquet.
    method "mcd": one model, mc_t stochastic passes with only nn.Dropout in train mode.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fit = split_df[split_df["split"] == "train_fit"]
    pred = split_df[split_df["split"].isin(PRED_SPLITS)].reset_index()  # keeps file row index
    loader = _eval_loader(pred["smiles"].tolist())

    if method == "ensemble":
        mus, vars_ = [], []
        for k in range(n_members):
            model = train_model(fit, config.member_seed(seed, k), epochs, device)
            model.eval()
            mu, var = predict_mve(model, loader, device)
            mus.append(mu)
            vars_.append(var)
        members = pd.DataFrame(
            {
                "member": np.repeat(np.arange(n_members), len(pred)),
                "row": np.tile(pred["index"].to_numpy(), n_members),
                "mu": np.concatenate(mus),
                "var": np.concatenate(vars_),
            }
        )
        members.to_parquet(out_dir / f"{task}_s{seed}_ensemble_members.parquet", index=False)
        agg = aggregate(np.stack(mus), np.stack(vars_))
    elif method == "mcd":
        model = train_model(fit, config.mcd_seed(seed), epochs, device)
        set_mc_dropout(model)
        torch.manual_seed(config.mcd_seed(seed))
        passes = [predict_mve(model, loader, device) for _ in range(mc_t)]
        agg = aggregate(np.stack([p[0] for p in passes]), np.stack([p[1] for p in passes]))
    else:
        raise ValueError(f"unknown method: {method}")

    path = out_dir / f"{task}_s{seed}_{method}.parquet"
    _frame(pred, agg).to_parquet(path, index=False)
    return path
