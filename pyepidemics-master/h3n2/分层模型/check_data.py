"""快速检查所有数据文件完整性"""
import pandas as pd
import json
from pathlib import Path

root = Path(__file__).resolve().parent

print("=== 分场景数据检查 ===")
for s, cn in [("normal", "常态"), ("sporadic", "散发"), ("cluster", "聚集")]:
    d = root / cn
    ts_path = d / f"timeseries_{s}.csv"
    pc_path = d / f"place_contribution_{s}.csv"
    reff_path = d / f"reff_{s}.csv"
    opt_path = d / f"opt_result_{s}.json"
    
    ts = pd.read_csv(ts_path, index_col=0) if ts_path.exists() else None
    pc = pd.read_csv(pc_path, index_col=0) if pc_path.exists() else None
    reff = pd.read_csv(reff_path) if reff_path.exists() else None
    
    print(f"  {cn}: ts={ts.shape if ts is not None else 'MISSING'}", end="")
    print(f", pc={pc.shape if pc is not None else 'MISSING'}", end="")
    print(f", reff={reff.shape if reff is not None else 'MISSING'}", end="")
    
    if opt_path.exists():
        with open(opt_path, encoding="utf-8-sig") as f:
            opt = json.load(f)
        print(f", opt_J={opt.get('J', '?'):.4f}, AR={opt.get('attack_rate', 0):.4f}")
    else:
        print(", opt=MISSING")

print("\n=== 真实校园数据检查 ===")
real_dir = root.parent / "真实校园"
if real_dir.exists():
    for f in real_dir.glob("*.csv"):
        df_temp = pd.read_csv(f, nrows=3)
        print(f"  {f.name}: shape=({pd.read_csv(f).shape}), cols={list(df_temp.columns)}")
else:
    print("  真实校园目录不存在!")

print("\n=== 图表文件检查 ===")
fig_dir = root / "figures"
for f in sorted(fig_dir.glob("*.png")):
    print(f"  {f.name}: {f.stat().st_size / 1024:.0f} KB")
