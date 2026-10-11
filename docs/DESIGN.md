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
