"""Shared helper: pick a real survey to exercise the pipeline with.

Prefers ``kelenderis/k3.dat`` (private field data, present only in the full
dev checkout - some tests pin exact values to its geometry) and falls back to
the bundled synthetic sample so the pipeline-level tests (mesh -> inversion ->
viz, not exact-value MATLAB comparisons) still run in a trimmed hand-off copy
that ships only ``sample_data/``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # sibling of python/ in the full repo
PY_ROOT = Path(__file__).resolve().parents[1]        # this python/ package root

K3 = ROOT / "kelenderis" / "k3.dat"
FALLBACK = PY_ROOT / "sample_data" / "synthetic_ws.dat"


def default_survey_path() -> Path:
    return K3 if K3.is_file() else FALLBACK
