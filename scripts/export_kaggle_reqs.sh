#!/usr/bin/env bash
# Export pinned prod deps for Kaggle, WITHOUT torch: Kaggle's preinstalled CUDA torch is kept.
set -euo pipefail
pdm export --prod -G deep --without-hashes --no-extras -o requirements-kaggle.txt
sed -i -E '/^(torch==|nvidia-|triton==)/d' requirements-kaggle.txt
