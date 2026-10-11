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
    out = ctl.stage("someone", "smoke", allow_dirty=True, stage_dir=tmp_path / "s")
    meta = json.loads((out / "kernel" / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/molcalib-train" and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["dataset_sources"] == ["someone/molcalib-splits", "someone/molcalib-code"]
    code = _code(out / "kernel" / "molcalib_train.ipynb")
    assert '"__SMOKE__"' not in code and '"True", "caco2_wang", "1"' in code and '"caco2_wang", "1"' in code
    assert len(list((out / "splits").glob("*.csv"))) == 15
    names = zipfile.ZipFile(out / "code" / "molcalib_code.zip").namelist()
    assert {"molcalib/__init__.py", "scripts/train_all.py", "commit.txt",
            "requirements-kaggle.txt"} <= set(names)
    assert not [n for n in names if "kaggle.json" in n or n.endswith(".env") or "cache/" in n]

    out = ctl.stage("someone", "pilot", allow_dirty=True, stage_dir=tmp_path / "p")
    code = _code(out / "kernel" / "molcalib_train.ipynb")
    assert '"False", "caco2_wang lipophilicity_astrazeneca solubility_aqsoldb", "1 2"' in code
    assert 'RUN = "pilot"' in code
    assert not [k for k in ("__SMOKE__", "__TASKS__", "__SEEDS__", "__RUN__") if k in code]
    out = ctl.stage("someone", "full", seeds=[3, 4, 5], allow_dirty=True, stage_dir=tmp_path / "f")
    code = _code(out / "kernel" / "molcalib_train.ipynb")
    assert '"3 4 5"' in code and 'RUN = "custom"' in code
