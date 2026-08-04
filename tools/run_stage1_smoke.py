"""
Simple Stage 1 smoke runner for OpenMontage.

Behavior:
 - Detect GPU and report VRAM
 - Run a mock end-to-end pipeline to verify orchestration works
 - Attempt a non-mock run and record which adapters are missing (BLOCKED)
 - Write results to projects/<name>/artifacts/

Designed to be runnable in Colab or locally. Do NOT auto-install packages here; the COLAB_SETUP doc contains recommended install cells.
"""
import logging
import json
import time
import os
import traceback
from pathlib import Path

log = logging.getLogger(__name__)

PROJECTS_ROOT = Path("projects")
PROJECTS_ROOT.mkdir(exist_ok=True)


def detect_gpu():
    info = {"torch_installed": False, "cuda_available": False}
    try:
        import torch

        info["torch_installed"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            idx = 0
            name = torch.cuda.get_device_name(idx)
            total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            info.update({"device_name": name, "vram_gb": round(total, 2)})
    except Exception:
        # torch not installed or import failed
        pass
    return info


def run_mock_pipeline(project_name="smoke-mock"):
    from src.pipeline import Pipeline

    p = Pipeline(mock=True)
    return p.run(project_name)


def run_real_pipeline(project_name="smoke-real"):
    from src.pipeline import Pipeline

    p = Pipeline(mock=False)
    return p.run(project_name)


def main():
    logging.basicConfig(level=logging.INFO)
    start = time.perf_counter()

    env = detect_gpu()
    log.info("Environment: %s", env)

    results = {"env": env, "stages": {}, "timestamp": time.time()}

    out_dir = PROJECTS_ROOT / "smoke-report"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Stage: mock run (should always pass)
    try:
        log.info("Starting MOCK pipeline run...")
        r = run_mock_pipeline("smoke-mock")
        results["stages"]["mock_run"] = {"status": "VERIFIED", "summary": r}
        log.info("Mock run VERIFIED")
    except Exception as e:
        tb = traceback.format_exc()
        results["stages"]["mock_run"] = {"status": "FAILED", "error": str(e), "traceback": tb}
        log.error("Mock run failed: %s", e)
        with open(artifacts_dir / "mock_run.log", "w", encoding="utf-8") as lf:
            lf.write(tb)

    # Stage: real run (may be blocked by missing adapters or models)
    try:
        log.info("Starting REAL pipeline run (may fail if adapters or models missing)...")
        r2 = run_real_pipeline("smoke-real")
        results["stages"]["real_run"] = {"status": "VERIFIED", "summary": r2}
        log.info("Real run VERIFIED")
    except RuntimeError as re:
        tb = traceback.format_exc()
        results["stages"]["real_run"] = {"status": "BLOCKED", "error": str(re), "traceback": tb}
        log.error("Real run BLOCKED: %s", re)
        with open(artifacts_dir / "real_run.log", "w", encoding="utf-8") as lf:
            lf.write(tb)
    except Exception as e:
        tb = traceback.format_exc()
        results["stages"]["real_run"] = {"status": "FAILED", "error": str(e), "traceback": tb}
        log.error("Real run failed: %s", e)
        with open(artifacts_dir / "real_run.log", "w", encoding="utf-8") as lf:
            lf.write(tb)

    duration = time.perf_counter() - start
    results["duration_s"] = duration

    out_dir = PROJECTS_ROOT / "smoke-report"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(exist_ok=True)
    with open(out_dir / "artifacts" / "smoke_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("SMOKE RUN COMPLETE. Report written to:", out_dir / "artifacts" / "smoke_report.json")
    if results["stages"]["real_run"]["status"] == "VERIFIED":
        print("Stage 1 VERIFIED: real pipeline executed end-to-end.")
    elif results["stages"]["real_run"]["status"] == "BLOCKED":
        print("Stage 1 BLOCKED: adapters/models missing. See smoke_report.json for details.")
    else:
        print("Stage 1 NOT VERIFIED: see smoke_report.json for errors.")


if __name__ == "__main__":
    main()
