.PHONY: setup-cpu kaggle-reqs test lint splits audit smoke train evaluate

test:
	pdm run pytest

lint:
	pdm run ruff check

splits:
	pdm run python scripts/make_splits.py

audit:
	pdm run python scripts/audit.py

smoke:
	pdm run python scripts/train_all.py --smoke --device cpu

train:
	pdm run python scripts/train_all.py --device cuda

evaluate:
	pdm run python scripts/evaluate.py

setup-cpu:
	pdm install -G deep
	scripts/install_cpu_torch.sh

kaggle-reqs:
	scripts/export_kaggle_reqs.sh
