"""Model quality checks — run after `dvc repro`."""

import json
import os

import pytest

from common import REPORTS_DIR, load_params

REPORT = os.path.join(REPORTS_DIR, "evaluation_report.json")

pytestmark = pytest.mark.skipif(not os.path.exists(REPORT), reason="run `dvc repro` first")


def test_model_passes_quality_gate():
    report = json.load(open(REPORT))
    assert report["test_roc_auc"] >= load_params()["evaluate"]["min_roc_auc"]
    assert report["promoted"] is True


def test_model_catches_most_churners():
    report = json.load(open(REPORT))
    assert report["test_recall"] >= 0.65
