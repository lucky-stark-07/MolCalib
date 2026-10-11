import importlib.util
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kaggle_ctl", ROOT / "scripts" / "kaggle_ctl.py")
ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctl)


def _code(nb_path):
    nb = json.loads(nb_path.read_text())
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_stage_smoke_and_full(tmp_path):
    out = ctl.stage("someone", full=False, allow_dirty=True, stage_dir=tmp_path / "s")
    meta = json.loads((out / "kernel" / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/molcalib-train" and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["dataset_sources"] == ["someone/molcalib-splits", "someone/molcalib-code"]
    assert 'EXTRA = [a for a in ["--smoke"] if a]' in _code(out / "kernel" / "molcalib_train.ipynb")
    assert len(list((out / "splits").glob("*.csv"))) == 15
    names = zipfile.ZipFile(out / "code" / "molcalib_code.zip").namelist()
    assert {"molcalib/__init__.py", "scripts/train_all.py", "commit.txt",
            "requirements-kaggle.txt"} <= set(names)
    assert not [n for n in names if "kaggle.json" in n or n.endswith(".env") or "cache/" in n]

    out = ctl.stage("someone", full=True, allow_dirty=True, stage_dir=tmp_path / "f")
    assert 'EXTRA = [a for a in [""] if a]' in _code(out / "kernel" / "molcalib_train.ipynb")
    assert "__EXTRA_ARGS__" not in (out / "kernel" / "molcalib_train.ipynb").read_text()
