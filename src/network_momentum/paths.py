"""Project-relative paths with no machine-specific absolute locations."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
REPLICATION_DIR = RESULTS_DIR / "replication"
EXTENSION_DIR = RESULTS_DIR / "extension"
ARTIFACTS_DIR = REPLICATION_DIR / "artifacts"
TABLES_DIR = REPLICATION_DIR / "tables"
FIGURES_DIR = REPLICATION_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORT_DIR = PROJECT_ROOT / "report"
