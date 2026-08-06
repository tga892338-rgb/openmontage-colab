"""
Run the orchestration script and capture stdout/stderr to a log file while streaming it.
Usage:
    python scripts/run_orchestration_capture.py --project projects/first-creative-video --tts qwen [--dry-run]

Writes projects/<name>/orchestration.log and exits with the orchestration exit code.
"""
import argparse
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--project', default='projects/first-creative-video')
p.add_argument('--tts', choices=['qwen','chatterbox'], default='qwen')
p.add_argument('--dry-run', action='store_true')
args = p.parse_args()

proj = Path(args.project)
proj.mkdir(parents=True, exist_ok=True)
log_path = proj / 'orchestration.log'
cmd = [sys.executable, 'scripts/colab_run_full_pipeline.py', '--project', str(proj), '--tts', args.tts]
if args.dry_run:
    cmd.append('--dry-run')

print('Running orchestration:', ' '.join(subprocess.list2cmdline(cmd) if isinstance(cmd, list) else cmd))
print('Logging to', log_path)

with open(log_path, 'wb') as logf:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for chunk in proc.stdout:
        logf.write(chunk)
        logf.flush()
        try:
            sys.stdout.buffer.write(chunk)
        except Exception:
            sys.stdout.write(chunk.decode('utf-8', errors='replace'))
    proc.wait()
ret = proc.returncode
print('\nOrchestration exit code:', ret)
if ret != 0:
    print('--- Orchestration failed. Last 400 lines of log: ---')
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()[-400:]
        print(''.join(lines))
sys.exit(ret)
