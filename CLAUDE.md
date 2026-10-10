# MolCalib: rules for Claude

- Read docs/DESIGN.md and docs/EVAL_PROTOCOL.md before changing code.
- Use PDM (see docs/DESIGN.md); run `pdm run pytest` and `pdm run ruff check` before every commit.
- Seeds fixed in configs; log the git commit hash into every results file.
- Never commit raw data (cache/, data/), checkpoints, kaggle.json, tokens, or .env files. Exception: splits/*.csv (derived TDC splits) are committed so others can reproduce results.
- Never modify docs/EVAL_PROTOCOL.md without asking the user first.
- README rules: no novelty language, cite prior work, report CIs and negative results.
- Every Kaggle job must first pass a CPU smoke test on a tiny subset.
- If unsure about a library API, read current docs instead of guessing.
- Test data is never used for training, calibration, checkpoint selection or any model choice.
- See docs/IMPLEMENTATION_PLAN.md for the task list and fixed configuration.
