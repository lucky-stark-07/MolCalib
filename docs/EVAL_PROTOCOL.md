# Evaluation protocol

**Status: written on 2026-10-12, after all results were produced. This is a record of the protocol
as executed, not a pre-registration.** The plan it follows (`docs/IMPLEMENTATION_PLAN.md`) was
fixed before any model was trained; decisions made after seeing results are listed under
"Deviations and post-hoc additions" at the end. Design details: `docs/DESIGN.md`.

## Endpoints

Three regression tasks from the TDC `admet_group` benchmark, fixed before any training:
`caco2_wang`, `lipophilicity_astrazeneca`, `solubility_aqsoldb`. Unparseable SMILES are dropped
and counted (2 test molecules of `solubility_aqsoldb`). Targets are used as provided by TDC.

## Splits

- `test`: the fixed TDC test set, identical for all seeds.
- `train` / `valid`: `group.get_train_valid_split(seed, benchmark)` (TDC scaffold split).
- `naive_cal`: a seeded random 10% of `train` (molecule level); the rest is `train_fit`.
- Scaffolds: Bemis-Murcko, no chirality. Hard requirement, checked before training: zero
  scaffold overlap between `train_fit`, `valid` and `test` (passed for every task and seed).
- Splits are computed once on the laptop and committed (`splits/`).

## Seeds

Split seeds 1, 2, 3, 4, 5. Ensemble member k of split seed s uses seed `100*s + k`; the MC
dropout model uses `100*s + 99`. Bootstrap seed 0. All in `molcalib/config.py`.

## Models

Chemprop v2 D-MPNN (hidden 300, depth 3, mean aggregation, dropout 0.1 in message passing and
FFN, one FFN hidden layer of 300) with a mean-variance (Gaussian NLL) head. Targets standardised
with `train_fit` statistics. Chemprop default optimiser and learning-rate schedule, batch 64,
fixed 60 epochs, last epoch kept. Trained on `train_fit` only. No hyperparameter tuning, no
validation loader, no checkpoint selection.

## UQ methods

- Deep ensemble: 5 members, dropout off at inference.
- MC dropout: 1 model, T = 30 passes with only `nn.Dropout` modules in train mode.
- Both: σ²_total = mean of predicted variances + variance of predicted means.
- Conformal (α = 0.1), each calibrated on `naive_cal` and on `valid`, for both base methods:
  split (|y − μ|), normalized (|y − μ| / σ_total), Mondrian (|y − μ|, one quantile per
  calibration-quartile bin of distance to `train_fit`), normalized Mondrian
  (|y − μ| / σ_total per bin). Quantile level ⌈(n+1)(1−α)⌉/n; infinite interval if that
  exceeds n. Calibration functions never receive test data.

## Metrics

On the test set, per task, seed and method: MAE, RMSE, Gaussian NLL, Spearman(σ_total, |error|),
miscalibration area (mean |observed − expected| coverage of Gaussian intervals at levels
0.1-0.9, plus its signed version), RMSE when keeping the 100%-50% most confident molecules.
Per conformal variant and calibration source: coverage, mean and median width, coverage and width
per test-quartile bin of distance to `train_fit`. Distance: 1 − max Tanimoto similarity, Morgan
radius 2, 2048 bits.

## Statistics

- Mean ± sd over the 5 seeds.
- 95% percentile CIs from 1000 bootstrap resamples of test scaffold groups (never single
  molecules), shared across seeds, methods and variants so comparisons are paired.
- For conformal metrics, a second 95% CI that also resamples calibration scaffold groups and
  refits the conformal quantiles in every resample.
- Paired differences: deep ensemble − MC dropout; `valid` − `naive_cal`, with the number of seeds
  in which the difference is positive.
- No correction for multiple comparisons.

## Baselines

No external baseline model (e.g. random forest) is included. Comparisons are internal:
MC dropout versus deep ensemble; the models' own Gaussian intervals versus conformal intervals;
`naive_cal` versus `valid` calibration; split conformal versus Mondrian conformal.

## Stopping rules

- Training: fixed 60 epochs for every model; no early stopping.
- Runs: one CPU smoke run, one Kaggle smoke run and a pilot (all tasks, seeds 1-2, full
  configuration) checked the pipeline and timing; the full grid (all tasks, seeds 1-5) was then
  run once from a single commit and all 90 trainings succeeded. No run was repeated or discarded.
  Pilot predictions were replaced by the full run, which used the same configuration.
- Analysis: no metric, model or run was dropped after seeing results. The plan's cut order
  (Mondrian, miscalibration area/NLL, third task) was not needed.

## Deviations and post-hoc additions

Made after seeing results; reported alongside, not instead of, the planned analyses:

1. Normalized Mondrian conformal (DR-002) was added after a first look at T7 coverage numbers.
   The planned Mondrian variant (|y − μ| score) is kept unchanged.
2. The calibration-resampling CI (DR-005) was added after seeing the T9 results. The planned
   test-only CI is kept unchanged.
3. Metric definitions not fixed by the plan (DR-003) were chosen while implementing T8, before
   the full metrics were inspected.
