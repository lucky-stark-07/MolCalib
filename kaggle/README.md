# kaggle/

`molcalib_train.ipynb` is a template. `scripts/kaggle_ctl.py` fills it in, builds the upload folders
under `cache/kaggle/` and drives the `kaggle` CLI. See `docs/IMPLEMENTATION_PLAN.md` (T5, T6).

## One-time setup (laptop)

1. Install the CLI outside the project lock: `uv tool install kaggle` (needs Python 3.11+).
   Make sure `~/.local/bin` is on `PATH`.
2. Create credentials yourself at https://www.kaggle.com/settings/api ("Generate New Token"), then
   authenticate with one of:
   - `kaggle auth login` (browser flow), or
   - `export KAGGLE_API_TOKEN=<token>` in your shell profile, or
   - save the token in `~/.kaggle/access_token`.
   Never put the token in the repo; `.gitignore` and the pre-commit hook block `kaggle.json`.
3. `export KAGGLE_USERNAME=<your kaggle username>` (not a secret; used for dataset and kernel ids).
4. Verify with `kaggle datasets list --mine`.

## Run

    make kaggle-smoke     # stage, upload datasets, push notebook with --smoke (3 epochs, caco2)
    make kaggle-status
    make kaggle-fetch     # smoke outputs -> cache/smoke/preds (never results/)
    make kaggle-fetch-full  # full outputs -> cache/preds, results/compute_*.csv
    make kaggle-full      # only after the smoke run passes; commit first (the hash is logged)

With two GPUs the notebook runs seeds 1-3 on cuda:0 and seeds 4-5 on cuda:1 (full run only).
The first push to a new dataset can fail while Kaggle is still processing it; wait and retry.
