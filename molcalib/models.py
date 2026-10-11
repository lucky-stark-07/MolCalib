"""Chemprop v2 D-MPNN with a mean-variance head, and prediction helpers."""

import numpy as np
import torch
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, MveFFN, UnscaleTransform
from sklearn.preprocessing import StandardScaler
from torch import nn

from molcalib import config


def build_model(scaler: StandardScaler) -> MPNN:
    """D-MPNN, mean aggregation, Gaussian-NLL (MVE) head.

    Chemprop's default optimiser and LR schedule come with MPNN. The UnscaleTransform maps
    mu and variance back to the original target scale in eval mode only.
    """
    return MPNN(
        BondMessagePassing(d_h=config.HIDDEN, depth=config.DEPTH, dropout=config.DROPOUT),
        MeanAggregation(),
        MveFFN(
            n_tasks=1,
            input_dim=config.HIDDEN,
            hidden_dim=config.FFN_HIDDEN,
            n_layers=config.FFN_LAYERS,
            dropout=config.DROPOUT,
            output_transform=UnscaleTransform.from_standard_scaler(scaler),
        ),
        batch_norm=False,
    )


def set_mc_dropout(model: nn.Module) -> None:
    """Eval mode, then only nn.Dropout modules back to train mode.

    The model must not be put in train mode as a whole: the output UnscaleTransform acts only
    in eval mode, and BatchNorm (if present) must stay in eval mode.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_mve(model: MPNN, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """One forward pass over loader in the model's current train/eval state.

    Returns (mu, var) on the original target scale, each of shape (N,).
    """
    model.to(device)
    mus, vars_ = [], []
    for batch in loader:
        bmg, V_d, X_d, *_ = batch
        bmg.to(device)
        out = model(bmg, V_d, X_d)  # (B, n_tasks=1, 2)
        mus.append(out[:, 0, 0].cpu().numpy())
        vars_.append(out[:, 0, 1].cpu().numpy())
    return np.concatenate(mus), np.concatenate(vars_)


def aggregate(mus: np.ndarray, vars_: np.ndarray) -> dict[str, np.ndarray]:
    """Combine K predictions (shape (K, N)): ensemble members or MC dropout passes.

    total variance = mean of predicted variances (aleatoric) + variance of means (epistemic,
    population variance over the K predictions).
    """
    alea = vars_.mean(axis=0)
    epi = mus.var(axis=0)
    return {
        "mu": mus.mean(axis=0),
        "sigma_alea": np.sqrt(alea),
        "sigma_epi": np.sqrt(epi),
        "sigma_total": np.sqrt(alea + epi),
    }
