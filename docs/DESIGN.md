# Design decisions

## Template

### DR-NNN: <title>
- **Decision:**
- **Why:**
- **Alternatives rejected:**
- **Cost in hours:**

## Log

### DR-001: T4 model and training details not fixed by the plan
- **Decision:** (a) No validation loader; the last-epoch model is used and Lightning checkpointing is off. (b) `batch_norm=False` (chemprop MPNN default). (c) Chemprop `MveFFN` head as is: variance = softplus(output), no floor. (d) Targets standardised with train_fit statistics; chemprop's `UnscaleTransform` returns mu and variance on the original scale in eval mode only, so MC dropout enables only `nn.Dropout` modules and keeps the model in eval mode. (e) Seeds: ensemble member k of split seed s uses `s*100+k`; the MC dropout model uses `s*100+99`. (f) Epistemic variance is the population variance (ddof=0) of the K means. (g) Prediction runs by hand-written loop, not `Trainer.predict`.
- **Why:** Plan requires no checkpoint selection and fixed configuration; these are the library defaults or the smallest choice consistent with the plan.
- **Alternatives rejected:** validation loader for logging (adds nothing, risks selection); variance floor (changes the chemprop head); ddof=1 (not the mixture variance of Lakshminarayanan et al.).
- **Cost in hours:** 0.

### DR-002: two Mondrian conformal variants
- **Decision:** "mondrian" uses the score |y - mu| with one quantile per nearest-neighbour-distance bin; "mondrian_normalized" uses |y - mu| / sigma_total per bin and an interval mu +- q_bin * sigma_total. Bins are calibration-set quartiles (merged if tied). If a bin has too few calibration points for the 90% quantile, its interval is infinite.
- **Why:** the plan names "Mondrian" without a score; the user asked for both.
- **Alternatives rejected:** a single variant.
- **Cost in hours:** 0.3.

### DR-003: T8 metric definitions not fixed by the plan
- **Decision:** (a) Miscalibration area = mean |observed - expected| coverage of central Gaussian intervals mu +- z sigma_total at levels 0.1, ..., 0.9; also reported signed (negative = intervals too narrow). (b) NLL uses sigma_total. (c) Selective prediction ranks by sigma_total, keeps the ceil(f*n) smallest. (d) Coverage-by-distance uses test-set quartiles of nn_dist (same tie handling as the Mondrian bins). (e) Every results CSV carries the git commit of the code that produced it.
- **Why:** smallest definitions consistent with the plan's wording.
- **Alternatives rejected:** integrating the calibration curve over [0, 1] (needs more levels; same ranking); calibration-set bins for the test breakdown (bins would differ per calibration source).
- **Cost in hours:** 0.

### DR-004: T9 statistics
- **Decision:** (a) One set of 1000 scaffold-group resamples per task (seed 0 in config), shared by all seeds, methods and conformal variants, so comparisons are paired; test rows were checked to be identical across all prediction files. (b) A resample draws as many test scaffold groups as exist, with replacement; molecules are weighted by how often their group was drawn. (c) The CI is the 2.5-97.5 percentile of the seed-averaged metric; mean and sd are over the 5 seeds on the full test set. (d) CIs cover test-set sampling only; models, calibration sets and conformal quantiles are fixed. (e) Far-bin coverage keeps the full-test-set quartile bins fixed. (f) Infinite widths stay infinite (nearest-rank percentile); inf - inf differences are reported as NaN.
- **Why:** plan T9 (cluster bootstrap over test scaffold groups, never single molecules; paired per-seed differences).
- **Alternatives rejected:** re-fitting conformal quantiles inside each resample (would need resampling calibration groups too; not in the plan); independent resamples per seed (loses pairing).
- **Cost in hours:** 0.5.

### DR-005: second CI that also resamples calibration groups (user choice, option B)
- **Decision:** For conformal metrics, also report ci_lo_cal/ci_hi_cal: in each of the 1000 resamples the calibration set (naive_cal or valid) is resampled by scaffold group and all conformal quantiles are refit, then scored on that resample's test weights. Calibration resamples are seeded by (BOOT_SEED, split seed, source) and shared by ensemble and MC dropout of the same seed. The plan's test-only CI (ci_lo/ci_hi) is kept unchanged.
- **Why:** the main question (naive_cal vs valid calibration) depends on calibration-set sampling, which the test-only CI ignores; e.g. split-conformal width had a zero-width test-only CI.
- **Alternatives rejected:** replacing the plan's CI (user asked to report both).
- **Cost in hours:** 0.5.

### DR-006: post-hoc robustness checks of the calibration-source result
- **Decision:** All three checks were chosen after seeing the results, in response to review
  comments, and are labelled post hoc in summary.md and the README. The planned analyses are
  unchanged (every earlier number in results/ is reproduced exactly).
  (a) Size-matched control: for each task, seed and base method, split and normalized conformal
  are refit on R = 200 subsamples of `valid`, drawn without replacement, each the size of that
  seed's `naive_cal` (source "valid_sizematched"). The existing fit_* functions are reused.
  Subsamples are seeded by (SIZE_MATCH_SEED = 0, split seed) in config, so ensemble and MC
  dropout of one seed share them. Reported: test coverage, mean width and Winkler score averaged
  over the subsamples, mean ± sd over seeds, and the test-only bootstrap CI (the subsample average
  of each per-molecule value is bootstrapped like any other metric). No calibration-resampling
  CI and no paired comparison for this control.
  (b) Winkler (interval) score at alpha = 0.1, mean over test molecules of
  (hi − lo) + (2/alpha)(lo − y)[y < lo] + (2/alpha)(y − hi)[y > hi], for every conformal variant
  and calibration source: in the per-seed metrics, the seed aggregation, both CIs (test, and
  test + calibration as in DR-005) and the `valid` − `naive_cal` paired comparisons. It is not
  added to the ensemble − MC dropout comparisons. An infinite interval scores inf, and infinite
  values are handled as for mean_width (DR-004 f).
  (c) Mechanism check: median |y − μ| on `naive_cal`, `valid` and test, per task, method and
  seed, with mean ± sd over seeds. Descriptive only; no test.
- **Why:** a reviewer pointed out that (a) `naive_cal` is smaller than `valid` on every task and
  seed, so calibration-set size is confounded with calibration source; (b) higher coverage can
  come from wider intervals alone, and coverage plus width give no single measure of interval
  quality; (c) `naive_cal` shares more scaffolds with `train_fit` than test does, so its errors
  may understate test errors. The Winkler score combines width and misses in one proper scoring
  rule for central intervals.
- **Alternatives rejected:** comparing coverage at matched width. For split conformal both
  sources use the same model and a constant margin, so at matched width the two sources give
  identical intervals and the comparison is empty by construction; the same holds for normalized
  conformal (one scale factor times σ). Bootstrap resampling of `valid` to the `naive_cal` size
  (with replacement: duplicates change the quantile). A paired test for the size-matched control
  or for the median errors (adds comparisons to an analysis that already has no
  multiple-comparison correction).
- **Cost in hours:** 1.5.
