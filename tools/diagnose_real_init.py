"""Diagnose real adapter initialization failures.

This script sets OM_LOAD_ADAPTERS=1 and OM_REAL_STRICT=1, attempts to instantiate
Pipeline(mock=False) and captures any exception traceback to projects/smoke-report/artifacts/real_init.log.
"""
import os
import traceback
from pathlib import Path
import sys

os.environ["OM_LOAD_ADAPTERS"] = "1"
os.environ["OM_REAL_STRICT"] = "1"

# Ensure repo root is on sys.path (mirrors notebook PYTHONPATH setup)
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

out_dir = Path("projects") / "smoke-report" / "artifacts"
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "real_init.log"

try:
    from src.pipeline import Pipeline
    print("Instantiating Pipeline(mock=False)")
    p = Pipeline(mock=False)
    print("Pipeline instantiated successfully. (Adapters loaded)")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Pipeline instantiated successfully.\n")
except Exception as e:
    tb = traceback.format_exc()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(tb)
    print("Exception captured during pipeline init. See", log_path)
    raise
