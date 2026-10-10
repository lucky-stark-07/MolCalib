#!/usr/bin/env bash
# Laptop only: replace the locked torch with the CPU wheel from the PyTorch CPU index.
# The lock (pdm.lock) keeps the plain PyPI torch version so the Kaggle export is unaffected.
set -euo pipefail
ver=$(pdm run python -c "import importlib.metadata as m; print(m.version('torch').split('+')[0])")
uv pip install --python .venv/bin/python --reinstall-package torch \
  --index-url https://download.pytorch.org/whl/cpu "torch==${ver}"
