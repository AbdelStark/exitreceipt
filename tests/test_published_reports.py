"""Keep public score tables tied to the versioned per-case reports."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/seed-robustness.json"
V2_SUMMARY = ROOT / "results/v2-robustness.json"
V2_SELECTED = ROOT / "results/v2-selected.json"


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


def test_v2_summary_matches_locked_seed_and_per_case_reports():
    expected_summary = V2_SUMMARY.read_bytes()
    expected_selected = V2_SELECTED.read_bytes()
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_v2.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert V2_SUMMARY.read_bytes() == expected_summary
    assert V2_SELECTED.read_bytes() == expected_selected
    published = json.loads(expected_summary)
    locked = json.loads((ROOT / "results/v2-selection.json").read_text())
    assert published["selected_seed"] == locked["selected_seed"] == 20260925
    assert published["test_tasks"] == 79
    assert published["test_templates"] == 10
    assert all(run["base"]["n"] == 158 for run in published["runs"])
