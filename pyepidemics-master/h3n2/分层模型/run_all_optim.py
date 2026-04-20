"""
串行运行三场景优化脚本（不修改任何场景参数）
用法: D:\Develop\miniconda\envs\py310\python.exe run_all_optim.py
"""
import subprocess
import sys
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
PYTHON = r"D:\Develop\miniconda\envs\py310\python.exe"

scenarios = [
    ("常态", "分层模型_常态.py"),
    ("散发", "分层模型_散发.py"),
    ("聚集", "分层模型_聚集.py"),
]

for folder, script in scenarios:
    script_path = BASE / folder / script
    cwd = BASE / folder
    print(f"\n{'='*60}")
    print(f"[{folder}] 开始优化: {script_path}")
    print(f"{'='*60}")
    sys.stdout.flush()
    result = subprocess.run(
        [PYTHON, str(script_path), "optim"],
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(f"[{folder}] 退出码: {result.returncode}")
    sys.stdout.flush()

print("\n\n===== 全部三场景优化完成 =====")

# 运行绘图
print("\n开始生成图表...")
fig_script = BASE / "paper_figures.py"
result = subprocess.run(
    [PYTHON, str(fig_script)],
    cwd=str(BASE),
    text=True,
    encoding="utf-8",
    errors="replace",
)
print(f"绘图退出码: {result.returncode}")
print("===== 完成 =====")
