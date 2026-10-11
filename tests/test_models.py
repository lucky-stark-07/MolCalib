from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molcalib.models import aggregate, predict_mve, set_mc_dropout
from molcalib.train import _eval_loader, train_model

SPLIT = Path(__file__).resolve().parents[1] / "splits" / "caco2_wang_seed1.csv"


def _tiny():
    df = pd.read_csv(SPLIT)
    fit = df[df["split"] == "train_fit"].head(48)
    return fit, _eval_loader(fit["smiles"].tolist())


def test_aggregate_total_variance():
    mus = np.array([[0.0, 1.0], [2.0, 1.0]])
    vars_ = np.array([[1.0, 4.0], [3.0, 4.0]])
    a = aggregate(mus, vars_)
    np.testing.assert_allclose(a["mu"], [1.0, 1.0])
    np.testing.assert_allclose(a["sigma_alea"] ** 2, [2.0, 4.0])
    np.testing.assert_allclose(a["sigma_epi"] ** 2, [1.0, 0.0])  # population variance of means
    np.testing.assert_allclose(a["sigma_total"] ** 2, [3.0, 4.0])


def test_mc_dropout_varies_and_eval_is_repeatable():
    fit, loader = _tiny()
    model = train_model(fit, seed=1, epochs=3, device="cpu")

    model.eval()
    mu1, var1 = predict_mve(model, loader, "cpu")
    mu2, var2 = predict_mve(model, loader, "cpu")
    np.testing.assert_array_equal(mu1, mu2)
    np.testing.assert_array_equal(var1, var2)

    set_mc_dropout(model)
    assert not model.training  # whole model stays in eval mode (output transform unscales)
    assert any(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))
    mu3, _ = predict_mve(model, loader, "cpu")
    mu4, _ = predict_mve(model, loader, "cpu")
    assert not np.allclose(mu3, mu4)
    # same scale as the eval-mode prediction (unscaling still active)
    assert abs(mu3.mean() - mu1.mean()) < 1.0


def test_training_is_deterministic_for_a_seed():
    fit, loader = _tiny()
    preds = []
    for _ in range(2):
        m = train_model(fit, seed=7, epochs=3, device="cpu")
        m.eval()
        preds.append(predict_mve(m, loader, "cpu")[0])
    np.testing.assert_array_equal(preds[0], preds[1])
