"""Pack cache/preds/*.parquet into results/preds.zip (deterministic: sorted, fixed timestamps)."""

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cache/preds")
    ap.add_argument("--out", default="results/preds.zip")
    ap.add_argument("--unpack", action="store_true", help="extract --out into --src instead")
    a = ap.parse_args()
    if a.unpack:
        zipfile.ZipFile(a.out).extractall(a.src)
        return
    files = sorted(Path(a.src).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {a.src}")
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            info = zipfile.ZipInfo(f.name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, f.read_bytes())
    print(f"{len(files)} files -> {a.out}")


if __name__ == "__main__":
    main()
