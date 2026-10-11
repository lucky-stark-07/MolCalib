"""Drive Kaggle runs from the laptop through the `kaggle` CLI.

    export KAGGLE_USERNAME=<your kaggle username>   # not a secret; used for dataset/kernel ids
    python scripts/kaggle_ctl.py stage [--mode M]  # build upload folders under cache/kaggle/
    python scripts/kaggle_ctl.py upload             # create or version the two datasets
    python scripts/kaggle_ctl.py push [--mode M]   # stage + upload + push the notebook (starts the run)
    python scripts/kaggle_ctl.py status
    python scripts/kaggle_ctl.py fetch [--mode M]  # download outputs (smoke: cache/smoke/preds; otherwise cache/preds, results/)

Modes: smoke (default; 3 epochs, caco2), pilot (all tasks, seeds 1 2, full epochs), full (all seeds).
--seeds / --tasks override the mode's lists.

Authentication is the CLI's own (`kaggle auth login` or KAGGLE_API_TOKEN). This script never reads
or writes credentials.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "cache" / "kaggle"
KERNEL_SLUG = "molcalib-train"
SPLITS_SLUG = "molcalib-splits"
CODE_SLUG = "molcalib-code"
CODE_FILES = ("requirements-kaggle.txt", "pyproject.toml")
CODE_DIRS = ("molcalib", "scripts")


def user() -> str:
    u = os.environ.get("KAGGLE_USERNAME")
    if not u:
        sys.exit("set KAGGLE_USERNAME to your Kaggle username (not a secret)")
    return u


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _dataset_meta(folder: Path, owner: str, slug: str, title: str, subtitle: str) -> None:
    meta = {
        "title": title,
        "subtitle": subtitle,
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "other"}],  # Kaggle has no MIT entry; see repo LICENSE and splits/README.md
    }
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


MODES = {"smoke": (["caco2_wang"], [1]), "pilot": (None, [1, 2]), "full": (None, None)}


def stage(owner: str, mode: str = "smoke", tasks=None, seeds=None, allow_dirty: bool = False,
          root: Path = ROOT, stage_dir: Path = STAGE) -> Path:
    """Build cache/kaggle/{splits,code,kernel}. Returns the stage dir."""
    sys.path.insert(0, str(root))
    from molcalib import config

    mode_tasks, mode_seeds = MODES[mode]
    tasks = tasks or mode_tasks or list(config.TASKS)
    seeds = seeds or mode_seeds or list(config.SEEDS)
    dirty = git("status", "--porcelain")
    if dirty and mode != "smoke" and not allow_dirty:
        sys.exit("working tree has uncommitted changes; commit first (the hash is logged) "
                 "or pass --allow-dirty")
    if dirty:
        print("warning: uncommitted changes; the logged commit hash does not match this code")
    shutil.rmtree(stage_dir, ignore_errors=True)
    for d in ("splits", "code", "kernel"):
        (stage_dir / d).mkdir(parents=True)

    for f in (root / "splits").glob("*.csv"):
        shutil.copy(f, stage_dir / "splits" / f.name)
    _dataset_meta(stage_dir / "splits", owner, SPLITS_SLUG, "MolCalib splits",
                  "Scaffold splits of TDC ADMET tasks for MolCalib")

    # subdirectories are skipped by `datasets create` by default, so ship the code as one zip
    with zipfile.ZipFile(stage_dir / "code" / "molcalib_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for d in CODE_DIRS:
            for p in sorted((root / d).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    z.write(p, p.relative_to(root))
        for f in CODE_FILES:
            z.write(root / f, f)
        z.writestr("commit.txt", git("rev-parse", "HEAD") + "\n")
    _dataset_meta(stage_dir / "code", owner, CODE_SLUG, "MolCalib code",
                  "MolCalib source code snapshot as a zip file")

    nb = (root / "kaggle" / "molcalib_train.ipynb").read_text()
    for key, val in (("__SMOKE__", str(mode == "smoke")),
                     ("__TASKS__", " ".join(tasks)),
                     ("__SEEDS__", " ".join(map(str, seeds))),
                     ("__RUN__", mode if (tasks, seeds) == (list(config.TASKS), list(config.SEEDS)) or mode != "full" else "custom")):
        nb = nb.replace(key, val)
    (stage_dir / "kernel" / "molcalib_train.ipynb").write_text(nb)
    (stage_dir / "kernel" / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner}/{KERNEL_SLUG}",
        "title": "MolCalib Train",
        "code_file": "molcalib_train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{SPLITS_SLUG}", f"{owner}/{CODE_SLUG}"],
    }, indent=2))
    return stage_dir


def kaggle(*args: str, check: bool = True) -> int:
    print("+ kaggle", *args)
    if not shutil.which("kaggle"):
        sys.exit("kaggle CLI not found; install with `uv tool install kaggle`")
    return subprocess.run(["kaggle", *args], check=check).returncode


def upload(owner: str) -> None:
    for slug, folder in ((SPLITS_SLUG, "splits"), (CODE_SLUG, "code")):
        path = str(STAGE / folder)
        if kaggle("datasets", "status", f"{owner}/{slug}", check=False) == 0:
            kaggle("datasets", "version", "-p", path, "-m", f"commit {git('rev-parse', '--short', 'HEAD')}")
        else:
            kaggle("datasets", "create", "-p", path, "-t")


def push(owner: str, mode: str, timeout: int | None) -> None:
    args = ["kernels", "push", "-p", str(STAGE / "kernel"), "--accelerator", "NvidiaTeslaT4"]
    if timeout:
        args += ["-t", str(timeout)]
    kaggle(*args)
    print(f"started; check with: python scripts/kaggle_ctl.py status ({mode} run)")


def fetch(owner: str, mode: str) -> None:
    full = mode != "smoke"
    out = ROOT / "cache" / "kaggle_out"
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{owner}/{KERNEL_SLUG}", "-p", str(out), "-o")
    # smoke outputs never go into cache/preds or results/: they would be taken for finished runs
    for sub in ("cache/preds",) if full else ("cache/smoke/preds",):
        src = out / sub
        if src.is_dir():
            dst = ROOT / sub
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.parquet"):
                shutil.copy(f, dst / f.name)
    logs = sorted(out.rglob("compute_*.csv"))
    if full:
        (ROOT / "results").mkdir(exist_ok=True)
        for f in logs:
            shutil.copy(f, ROOT / "results" / f.name)
    print(f"outputs in {out}; compute logs: {[str(p) for p in logs]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["stage", "upload", "push", "status", "fetch"])
    ap.add_argument("--mode", choices=list(MODES), default="smoke")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--timeout", type=int, default=None, help="kernel run limit in seconds")
    a = ap.parse_args()
    owner = user()
    if a.cmd == "stage":
        print("staged in", stage(owner, a.mode, a.tasks, a.seeds, a.allow_dirty))
    elif a.cmd == "upload":
        upload(owner)
    elif a.cmd == "push":
        stage(owner, a.mode, a.tasks, a.seeds, a.allow_dirty)
        upload(owner)
        push(owner, a.mode, a.timeout)
    elif a.cmd == "status":
        kaggle("kernels", "status", f"{owner}/{KERNEL_SLUG}")
    elif a.cmd == "fetch":
        fetch(owner, a.mode)


if __name__ == "__main__":
    main()
