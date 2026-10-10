# Setup report

Date: 2026-10-10. Scaffold only; no models or experiments implemented.

## Decisions taken from the user
- PDM + flat `molcalib/` layout (docs/IMPLEMENTATION_PLAN.md), not uv + src layout from the original brief.
- MIT licence replaced the Apache-2.0 file from the initial commit.
- Repo-local git email set to the author's personal address (the global git config held a work address).

## Installed versions (from `pdm list --freeze`, Python 3.11.15 via `pdm python install 3.11`)
PDM 2.27.0 (2.29.2 available), numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, scikit-learn 1.9.1, pyarrow 26.0.0,
matplotlib 3.11.2, rdkit 2026.9.1, PyTDC 0.4.1, pytest 9.1.1, ruff 0.17.0;
deep group: torch 2.14.1 (+cu130 build, see problems), chemprop 2.2.4, lightning 2.6.6.

## Verification
- `pdm install` (core + dev), `pdm run pytest` (1 passed) and `pdm run ruff check` (clean): OK.
- `pdm install -G deep`: installs, but see problems 1-3. Not worked around.

## Problems found (need your decision)
1. **torch is not a CPU build.** `torch.version.cuda` is 13.0. The `pytorch-cpu` source with `include_packages = ["torch"]` did not force the CPU wheel; the lock lists both PyPI wheels and `+cpu` wheels, and the Linux x86_64 install picked the CUDA one.
2. **chemprop resolved to 2.2.4, not 2.3.1** (PyPI latest). I did not investigate which constraint caused this.
3. **PyTDC resolved to 0.4.1, not 1.1.15** (PyPI latest). `import tdc` fails with `ModuleNotFoundError: No module named 'pkg_resources'` (setuptools is not installed). PyTDC 1.1.15 declares numpy<2, pandas<3, rdkit<2024.3.1, which conflict with the current unpinned stack.
4. Versions are not pinned in pyproject.toml; only pdm.lock fixes them. The plan says to pin chemprop and PyTDC.
5. `requirements-kaggle.txt` (plan T1) was not created, because the torch lock issue affects the export.

## Sources used
- PyPI JSON API (`pypi.org/pypi/<pkg>/json`) for latest versions and requires_python.
- pdm-project.org usage pages: dependency groups, config (`[[tool.pdm.source]]`), project (`pdm python install`). These pages do not document PyTorch indexes or `pdm export`; I used `pdm export --help` locally.
- download.pytorch.org/whl/cpu (index exists, cp311 wheels listed).
- github.com/pdm-project/setup-pdm README (v4.6; `python-version`, `cache`). actions/checkout v7.0.1 from the GitHub API.
- Not read: official TDC, Chemprop and RDKit docs pages, Kaggle CLI docs. Their API usage is unverified (no code uses them yet).

## Choices to double-check
- `requires-python = ">=3.11,<3.12"` (plan says Python 3.11; chemprop declares >=3.11,<3.15).
- lightgbm and MAPIE (in the original brief) were left out because the plan does not use them.
- Subpackages (data/models/uq/eval/plots) from the brief were not created; the plan's flat modules will be added with their tasks. `molcalib/config.py` holds only tasks, seeds and alpha.
- CI runs `pdm install` (core + dev, no deep) on ubuntu-latest, CPU only. It has not been run on GitHub.
- The pre-commit hook lives in untracked `.git/hooks/`; `scripts/check_secrets.sh` is tracked.

## Open questions
- TDC endpoint names (`caco2_wang`, `lipophilicity_astrazeneca`, `solubility_aqsoldb`) are unconfirmed, because `import tdc` fails.
- How to get a CPU-only torch and newer chemprop/PyTDC in the lock.

## Update 2026-10-10 (second pass)
- **Torch CPU vs GPU:** `torch` is no longer a direct dependency (it comes in via chemprop). The lock holds one plain version (2.14.1). Laptop: `make setup-cpu` runs `pdm install -G deep` then `scripts/install_cpu_torch.sh`, which swaps in `torch==2.14.1+cpu` from download.pytorch.org/whl/cpu (verified: `torch.version.cuda is None`). Kaggle: `make kaggle-reqs` exports `requirements-kaggle.txt` with torch/nvidia/triton lines removed, so Kaggle's preinstalled CUDA torch is kept. Not yet tested on Kaggle.
- chemprop is now 2.3.1 (was 2.2.4) after relocking with `use_uv=false` (stored in pdm.toml).
- PyTDC is still 0.4.1 (1.1.15 declares numpy<2, pandas<3, rdkit<2024.3.1). `import tdc` works once `setuptools<81` is added (pkg_resources). Installed source confirms `BenchmarkGroup.get_train_valid_split` and the three endpoint names in tdc/metadata.py. No data download was tried.
- The PyPI package is `PyTDC`; `pip install tdc` installs an unrelated project.
- Lock strategy is `inherit_metadata` (needed for `pdm export`).
- **Check:** `requirements-kaggle.txt` pins `rdkit==2026.3.6` and also `rdkit-pypi==2022.9.5` (a dependency of PyTDC 0.4.1); two packages provide the `rdkit` module, and the local env has a different rdkit build than the earlier 2026.9.1. Not yet resolved.
