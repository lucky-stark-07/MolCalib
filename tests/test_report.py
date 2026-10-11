import shutil
from pathlib import Path

from molcalib.report import TASK_NAME, write_report

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_report_builds_from_results(tmp_path):
    for f in list(RESULTS.glob("*.csv")) + [RESULTS / "split_audit.json"]:
        shutil.copy(f, tmp_path / f.name)
    text = write_report(tmp_path).read_text()
    for name in TASK_NAME.values():
        assert name in text
    assert "n/a" not in text and "nan" not in text.lower()
    block = []  # every row of a table has as many cells as its header
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            block.append(line.count("|"))
        elif block:
            assert len(set(block)) == 1, block
            block = []
    for fig in ("fig_coverage_by_source.png", "fig_coverage_by_nn_bin.png", "fig_selective.png"):
        assert (tmp_path / fig).stat().st_size > 10_000
