# MolCalib — implementation tasks for Claude Code (GNN version)

## Context (read first)

MolCalib evaluates established uncertainty quantification (UQ) methods for molecular property prediction under scaffold shift:

- deep ensembles (Lakshminarayanan et al., 2017)
- MC dropout (Gal & Ghahramani, 2016)
- split conformal, normalized conformal and Mondrian conformal prediction (Vovk et al.)

The base model is a Chemprop D-MPNN (Yang et al., 2019; Chemprop v2: Heid et al., 2023). Prior UQ work to cite: Hirschfeld et al. 2020; Scalia et al. 2020.

**Research question:** On TDC ADMET scaffold splits, does split conformal reach its nominal coverage? Does it depend on where the calibration data comes from? How does UQ quality degrade with distance from the training set?

### Non-negotiables

- No novelty claims anywhere in code, README or comments.
- Test data is never used for training, calibration, checkpoint selection or any model choice.
- Fixed hyperparameters, chosen up front. No tuning. **No best-checkpoint selection on any validation set: always use the last epoch.**
- Deterministic given a seed. Resumable. Per-run errors are logged and skipped, never fatal.

## Compute split

| Where | What |
|---|---|
| Laptop CPU | data, splits, scaffolds, split audit, smoke test, conformal, metrics, stats, report |
| Kaggle T4 (overnight, "Save & Run All" background session, 12 h limit) | all GNN training and prediction |

Splits are computed **once on the laptop** and uploaded to Kaggle as a dataset. Kaggle only trains and predicts on those exact files. This guarantees identical splits and avoids TDC downloads on Kaggle.

## Stack

- Python 3.11, environment managed with **PDM** (`pdm init`, `pdm add`, `pdm install`). Commit `pyproject.toml` and `pdm.lock`.
- Runtime dependencies: `chemprop` v2 (pin the version), PyTorch, RDKit, PyTDC (pin), numpy, pandas, scipy, scikit-learn, matplotlib, pyarrow
- Dev dependency group: pytest (`pdm add -dG test pytest`)
- Run everything through `pdm run` (e.g. `pdm run python scripts/make_splits.py`, `pdm run pytest`)
- For Kaggle, export a plain requirements file: `pdm export -o requirements-kaggle.txt --without-hashes --prod`, and commit it. Kaggle installs it with pip; don't run PDM on Kaggle.
- Before writing model code, check the installed Chemprop v2 API (`MPNN`, `BondMessagePassing`, mean-variance `FFN`, data loaders). Do not rely on memory of the API.

## Repo layout

```
pyproject.toml     # PDM-managed project metadata + dependencies
pdm.lock           # committed lock file
molcalib/
  config.py        # tasks, seeds, alpha, hyperparameters
  data.py          # TDC loading, splits, scaffolds -> splits/*.csv
  audit.py         # split audit
  models.py        # Chemprop MPNN with mean-variance head; MC dropout helper
  train.py         # ensemble + MC dropout training and prediction
  conformal.py     # split, normalized, Mondrian
  metrics.py       # coverage, width, ranking, calibration
  stats.py         # seed aggregation, scaffold-cluster bootstrap
  report.py        # tables + figures -> results/
scripts/
  make_splits.py   # laptop: writes splits/ for upload to Kaggle
  train_all.py     # Kaggle: resumable training over the run grid
  evaluate.py      # laptop: conformal + metrics + stats + report
kaggle/molcalib_train.ipynb  # thin notebook: install, run train_all.py
tests/
splits/            # committed: exact split files
cache/             # gitignored
results/           # committed
```

## Fixed configuration

| Setting | Value |
|---|---|
| Tasks (TDC `admet_group`) | `caco2_wang`, `lipophilicity_astrazeneca`, `solubility_aqsoldb` (regression only) |
| Seeds | 1, 2, 3, 4, 5 |
| Alpha | 0.1 (90% target coverage) |
| Model | Chemprop D-MPNN, hidden 300, depth 3, mean aggregation, dropout 0.1 in message passing and FFN, FFN 1 hidden layer of 300 |
| Head | mean-variance (Gaussian NLL) |
| Training | Chemprop default optimiser and LR schedule, batch 64, **fixed 60 epochs, last epoch kept** |
| Targets | standardise with `train_fit` stats; un-standardise mu and sigma |
| Ensemble | 5 members, different init seeds, same `train_fit` data, dropout off at inference |
| MC dropout | 1 model. At inference, put only `nn.Dropout` modules in train mode (BatchNorm stays in eval). T = 30 passes. |
| Uncertainty | total variance = mean of predicted variances (aleatoric) + variance of means (epistemic) |

Run grid: 3 tasks × 5 seeds × (5 ensemble members + 1 MC model) = 90 trainings.
Expected time on a T4: about 4–6 h. Measure it in the smoke run and update this estimate.

## Splits (per task, per seed)

1. Load `train_val` and the fixed `test` from `admet_group`. Get `train`/`valid` from `group.get_train_valid_split(seed=seed, benchmark=name)`.
2. From `train`, hold out a random **molecule-level** 10% sample as `naive_cal` (seeded). The model trains on the remaining 90%, called `train_fit`.
3. `valid` is the **scaffold-disjoint calibration set**. It is never used for training or checkpointing.
4. Both calibration sources use the same trained models. Only the calibration data differs.

## Tasks (in order)

### T1. Repo setup (laptop)
Create the layout with PDM (`pyproject.toml`, `pdm.lock`), `requirements-kaggle.txt` (exported), `config.py`, `.gitignore` (include `.venv/`, `.pdm-python`, `cache/`), MIT licence, and a `Makefile` whose targets wrap `pdm run`: `test`, `splits`, `audit`, `smoke`, `train`, `evaluate`.
**Done when:** `pdm install` works on a fresh clone and `pdm run pytest` runs.

### T2. Splits (laptop)
`make splits`:

- Build the splits above. Compute the Bemis–Murcko scaffold (no chirality) and a scaffold-group id for every molecule.
- Drop unparseable SMILES and log the count.
- Write `splits/{task}_seed{seed}.csv` with columns `smiles`, `y`, `split` (`train_fit` / `naive_cal` / `valid` / `test`), `scaffold_id`.

### T3. Split audit (laptop, before any training)
For each task and seed:

- Assert zero scaffold overlap between `train_fit` and `valid`, `train_fit` and `test`, and `valid` and `test`. Any overlap is a hard failure.
- Report how much `naive_cal` shares scaffolds with `train_fit` (expected: high).
- Report scaffold-group sizes and the largest single group in `valid`.
- Report the nearest-neighbour Tanimoto distance to `train_fit` (Morgan r=2, 2048 bits) for `naive_cal`, `valid` and `test`. Store `nn_dist` per molecule in the split files.

Output: `results/split_audit.json` and `results/fig_nn_distance.png`.

### T4. Model + training (`models.py`, `train.py`)
- Chemprop MPNN with mean-variance head, as configured.
- Training uses `train_fit` only. Either no validation loader, or a validation loader used only for logging; it must never select checkpoints.
- Prediction writes `cache/preds/{task}_s{seed}_{method}.parquet` with columns `smiles`, `split`, `scaffold_id`, `y`, `mu`, `sigma_alea`, `sigma_epi`, `sigma_total`, `nn_dist`, for the `naive_cal`, `valid` and `test` rows.
- Save each ensemble member's raw predictions too, so the ensemble can be re-aggregated without retraining.

### T5. Runner (`scripts/train_all.py`)
Loops over the run grid.

- **Resumable:** skip a run if its output file already exists.
- **Ordering:** smallest task first. Run all MC dropout models before the ensembles, so that a run cut short still has complete results for something.
- Logs GPU seconds and wall time per training run to `results/compute.csv`.
- Writes progress to a log file after each run.
- Flags:
  - `--device cuda|cpu`
  - `--smoke`: caco2 only, seed 1, 3 epochs, 2 members, T = 5
  - `--tasks`, `--seeds`: to split the grid across two sessions or the two T4s

### T6. Kaggle notebook
A thin notebook that:

1. installs `requirements-kaggle.txt` with pip;
2. reads `splits/` from an uploaded Kaggle dataset;
3. runs `train_all.py --device cuda`;
4. writes `cache/preds/` and `results/compute.csv` to `/kaggle/working`.

If Kaggle exposes two T4s, run two processes with different `--tasks` on `cuda:0` and `cuda:1`.

### T7. Conformal (`conformal.py`, laptop)
Functions take calibration arrays only. Test arrays are passed separately at apply time.

- **Split conformal:** score = |y − mu|; q is the ⌈(n+1)(1−α)⌉/n empirical quantile; interval = mu ± q.
- **Normalized conformal:** score = |y − mu| / sigma_total; interval = mu ± q · sigma_total.
- **Mondrian conformal:** separate quantile per NN-distance bin, with bin edges from calibration-set quartiles.
- Run each on both calibration sources (`naive_cal`, `valid`) and both base methods.

### T8. Metrics (`metrics.py`)
On test, per task / seed / method / conformal variant:

- MAE, RMSE
- Coverage and mean interval width
- Coverage per test NN-distance quartile bin
- Spearman correlation between sigma_total and |error|
- Gaussian NLL
- Miscalibration area: expected vs observed coverage of Gaussian intervals at levels 0.1–0.9
- Selective prediction: RMSE when keeping the most-confident 100%, 90%, …, 50%

### T9. Statistics (`stats.py`)
- Mean ± std across the 5 seeds.
- 95% CIs by **cluster bootstrap over test scaffold groups** (1000 resamples, seeded). Never resample individual molecules.
- Paired per-seed differences for: ensemble vs MC dropout; `naive_cal` vs `valid` calibration.

### T10. Report (`report.py`)
Generate `results/summary.md` automatically from the json and csv files (no hand-typed numbers). Include the compute-cost table: GPU-minutes per method next to its UQ metrics.

Figures:

- coverage by calibration source (with CIs)
- coverage vs NN-distance bin
- selective prediction curves
- NN-distance histogram

### T11. Tests
- Scaffold disjointness on real split files.
- Conformal coverage on synthetic exchangeable data is 0.90 ± 0.02 over many trials.
- Conformal calibration never receives test labels (check signatures and a mock).
- MC dropout gives different outputs across passes; the ensemble with dropout off gives identical outputs across repeated calls.
- Determinism in smoke mode.

### T12. README
Sections:

1. Question
2. Setup
3. Prior work (with citations)
4. Reproduce: `pdm install`, `make splits audit`, then train on Kaggle, then `make evaluate`
5. Results (link to `summary.md`)
6. Limitations:
   - 3 regression tasks only
   - marginal coverage only
   - fixed hyperparameters, untuned
   - construction of TDC `test.csv` not verified from source; checked empirically in T3

## Run order and timeline

| When | Step |
|---|---|
| Tonight, by ~01:00 | T1–T4, T11 tests, `make splits audit`, CPU smoke run of `train_all.py --smoke --device cpu` |
| ~01:00 | Upload `splits/` + code to Kaggle; start the background session. Laptop can sleep. |
| Morning | Download predictions. Build T7–T10 on CPU while training finishes if needed. |
| Afternoon | Report, README; send results for a variance/CI check |
| Evening | Publish |

## Cut order if time runs out

1. Mondrian conformal
2. Miscalibration area and NLL
3. Third task (`solubility_aqsoldb`, the largest)

Never cut the split audit, the leakage tests, or the scaffold-cluster bootstrap.