"""检查三个场景数据完整性"""
import json, os
from pathlib import Path

ROOT = Path(r"d:\HP\OneDrive\Desktop\学校\课程\专业课\数学建模\课程项目\Mathematical-Modeling-Course-Project\pyepidemics-master\h3n2\分层模型")

for key, cn in [("normal","常态"),("sporadic","散发"),("cluster","聚集")]:
    d = ROOT / cn
    print(f"\n=== {cn} ===")
    print(f"  dir exists: {d.exists()}")
    
    # 列出目录中所有文件
    if d.exists():
        files = list(d.iterdir())
        for f in sorted(files):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name}: {size_kb:.1f} KB")
    
    # 读取 opt_result
    opt_path = d / f"opt_result_{key}.json"
    if opt_path.exists():
        with open(opt_path, encoding="utf-8-sig") as f:
            opt = json.load(f)
        print(f"  attack_rate: {opt.get('attack_rate','N/A')}")
        print(f"  J: {opt.get('J','N/A')}")
        print(f"  best_x: {opt.get('best_x','N/A')}")
    else:
        print(f"  opt_result: MISSING")
