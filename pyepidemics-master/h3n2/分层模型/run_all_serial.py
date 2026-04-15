"""串行运行三个场景的优化，然后重新生成所有论文图表。"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = r"D:\Develop\miniconda\envs\py310\python.exe"

SCENARIOS = [
    ("normal", r"常态\分层模型_常态.py"),
    ("sporadic", r"散发\分层模型_散发.py"),
    ("cluster", r"聚集\分层模型_聚集.py"),
]

def run_scenario(script_rel):
    """串行运行单个场景的优化，等待完成。"""
    script_path = ROOT / script_rel
    if not script_path.exists():
        print(f"  ERROR: {script_path} not found!")
        return False
    
    print(f"  Starting: {script_rel}")
    print(f"  Command: {PYTHON} {script_path} optim")
    
    result = subprocess.run(
        [PYTHON, str(script_path), "optim"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,  # 30 min max per scenario
    )
    
    # Print last 30 lines of output for progress
    stdout_lines = (result.stdout or "").strip().split("\n")
    stderr_lines = (result.stderr or "").strip().split("\n")
    
    print(f"  Exit code: {result.returncode}")
    if stdout_lines:
        print(f"  stdout (last 10 lines):")
        for line in stdout_lines[-10:]:
            print(f"    {line}")
    if stderr_lines:
        print(f"  stderr (last 5 lines):")
        for line in stderr_lines[-5:]:
            print(f"    {line}")
    
    return result.returncode == 0

def main():
    print("=" * 70)
    print("SERIAL OPTIMIZATION: 3 scenarios, then regenerate all figures")
    print("=" * 70)
    
    t0 = time.time()
    
    # Phase 1: Run 3 scenarios serially
    for key, script_rel in SCENARIOS:
        print(f"\n{'='*70}")
        print(f"[{key.upper()}] Running optimization...")
        print(f"{'='*70}")
        ok = run_scenario(script_rel)
        elapsed = time.time() - t0
        print(f"  Completed: {ok} (elapsed: {elapsed:.0f}s)")
        if not ok:
            print(f"  WARNING: {key} scenario failed!")
    
    total_opt_time = time.time() - t0
    print(f"\nAll optimizations completed in {total_opt_time:.0f}s")
    
    # Phase 2: Regenerate all paper figures
    print(f"\n{'='*70}")
    print("REGENERATING ALL PAPER FIGURES")
    print(f"{'='*70}")
    
    fig_script = ROOT / "paper_figures.py"
    if fig_script.exists():
        result = subprocess.run(
            [PYTHON, str(fig_script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        print(f"  Exit code: {result.returncode}")
        stdout_lines = (result.stdout or "").strip().split("\n")
        if stdout_lines:
            for line in stdout_lines[-15:]:
                print(f"    {line}")
    else:
        print(f"  ERROR: {fig_script} not found!")
    
    # Phase 3: Regenerate fig9
    fig9_script = ROOT / "run_fig9.py"
    if fig9_script.exists():
        print(f"\nRegenerating fig9...")
        result = subprocess.run(
            [PYTHON, str(fig9_script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        print(f"  Exit code: {result.returncode}")
        stdout_lines = (result.stdout or "").strip().split("\n")
        if stdout_lines:
            for line in stdout_lines[-10:]:
                print(f"    {line}")
    
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"ALL DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
