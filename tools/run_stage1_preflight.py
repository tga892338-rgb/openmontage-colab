"""Preflight checks for REAL Stage 1 on local/Colab runtime.

Performs the minimal validation before running a full real pipeline:
- Verifies torch + CUDA
- Loads planner adapter and runs one short generation
- Loads image adapter and generates one 512x512 image
- Reports adapter names, model ids, device usage, and asset paths

Does NOT run expensive multi-shot generation.
"""
import os
import time
import json
import logging
from pathlib import Path

# ensure repo root on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in os.sys.path:
    os.sys.path.insert(0, str(repo_root))

log = logging.getLogger("preflight")
logging.basicConfig(level=logging.INFO)

ARTIFACTS = Path("projects") / "preflight"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ensure adapters import allowed
os.environ.setdefault("OM_LOAD_ADAPTERS", "1")

from src.adapters.adapter_loader import get_component


def check_torch_cuda():
    try:
        import torch
        info = {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": torch.cuda.device_count(),
            "torch_file": getattr(torch, '__file__', None),
        }
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        return info
    except Exception as e:
        return {"error": str(e)}


def test_planner(idea="A lonely astronaut discovers a strange glowing forest on an abandoned planet."):
    res = {"status": "unknown"}
    comp = get_component("planner", profile="balanced")
    if comp is None:
        res.update({"status": "missing", "reason": "no planner adapter available"})
        return res
    res['adapter'] = comp.__class__.__name__
    # try to run
    try:
        start = time.time()
        plan = comp.run(idea)
        elapsed = time.time() - start
        res.update({"status": "ok", "elapsed_s": elapsed, "plan_preview": plan})
    except Exception as e:
        res.update({"status": "error", "error": str(e)})
    return res


def test_image():
    res = {"status": "unknown"}
    comp = get_component("image_generation", profile="balanced")
    if comp is None:
        res.update({"status": "missing", "reason": "no image adapter available"})
        return res
    res['adapter'] = comp.__class__.__name__
    out_dir = ARTIFACTS / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        start = time.time()
        r = comp.run("A lonely astronaut in a glowing alien forest", count=1, output_dir=str(out_dir))
        elapsed = time.time() - start
        res.update({"status": "ok", "elapsed_s": elapsed, "result": r})
        # list files
        files = [str(p) for p in out_dir.glob("**/*") if p.is_file()]
        res['files'] = files
    except Exception as e:
        res.update({"status": "error", "error": str(e)})
    return res


def main():
    out = {"timestamp": time.time()}
    out['env'] = check_torch_cuda()
    out['planner_test'] = test_planner()
    out['image_test'] = test_image()
    p = ARTIFACTS / 'preflight_report.json'
    p.write_text(json.dumps(out, indent=2))
    print("Wrote preflight report to", p)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
