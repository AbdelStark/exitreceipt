"""Keep public score tables tied to the versioned per-case reports."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/seed-robustness.json"


def test_seed_summary_matches_all_published_reports():
    expected = SUMMARY.read_bytes()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_seeds.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert SUMMARY.read_bytes() == expected
    published = json.loads(expected)
    assert [run["seed"] for run in published["runs"]] == [20260925, 20260926, 20260927]
    assert all(run["training_code"]["tracked_changes"] is False for run in published["runs"])
