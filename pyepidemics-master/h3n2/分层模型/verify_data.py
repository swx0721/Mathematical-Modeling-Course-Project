"""快速验证图表数据"""
import json
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent

print("=== 优化结果 ===")
for s, cn in [("normal", "常态"), ("sporadic", "散发"), ("cluster", "聚集")]:
    opt_path = root / cn / f"opt_result_{s}.json"
    if opt_path.exists():
        with open(opt_path, encoding="utf-8-sig") as f:
            opt = json.load(f)
        ctrl = opt.get("controls", {})
        print(f"  {cn}: J={opt.get('J',0):.4f}, AR={opt.get('attack_rate',0):.4f}, "
              f"peak_I={opt.get('peak_I',0):.0f}")
        print(f"    controls: {ctrl}")
    else:
        print(f"  {cn}: opt_result MISSING")

print("\n=== 时间序列概要 ===")
for s, cn in [("normal", "常态"), ("sporadic", "散发"), ("cluster", "聚集")]:
    ts_path = root / cn / f"timeseries_{s}.csv"
    ts = pd.read_csv(ts_path, index_col=0)
    print(f"  {cn}: days={len(ts)}, peak_I={ts['I'].max():.0f} @ day {ts['I'].idxmax()}, "
          f"final_R={ts['R'].iloc[-1]:.0f}, N={ts['N'].iloc[0]:.0f}")

print("\n=== 真实校园对比数据 ===")
real_path = root.parent / "真实校园" / "michigan_case001_compare.csv"
if real_path.exists():
    df = pd.read_csv(real_path)
    print(f"  Shape: {df.shape}")
    print(f"  Cols: {list(df.columns)}")
    # RMSE/MAE
    obs = df["cum_cases"].dropna().values
    sim = df["sim_cum_cases_scaled"].dropna().values
    n = min(len(obs), len(sim))
    if n > 0:
        import numpy as np
        err = sim[:n] - obs[:n]
        rmse = np.sqrt(np.mean(err**2))
        mae = np.mean(np.abs(err))
        print(f"  累计病例: RMSE={rmse:.2f}, MAE={mae:.2f}")
    obs2 = df["new_cases"].dropna().values
    sim2 = df["sim_new_cases_scaled"].dropna().values
    n2 = min(len(obs2), len(sim2))
    if n2 > 0:
        err2 = sim2[:n2] - obs2[:n2]
        rmse2 = np.sqrt(np.mean(err2**2))
        mae2 = np.mean(np.abs(err2))
        print(f"  日新增:   RMSE={rmse2:.2f}, MAE={mae2:.2f}")
else:
    print("  数据不存在!")

print("\n=== 图表文件 ===")
fig_dir = root / "figures"
for f in sorted(fig_dir.glob("*.png")):
    print(f"  {f.name}: {f.stat().st_size/1024:.0f}KB")
