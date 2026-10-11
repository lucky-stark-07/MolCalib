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
