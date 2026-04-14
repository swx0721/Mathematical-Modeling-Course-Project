#!/usr/bin/env python
"""Quick smoke test for paper-grade outputs."""

import runpy
import tempfile
from pathlib import Path

# Create temp output dir
tmpdir = Path(tempfile.mkdtemp())
print(f"📁 Output directory: {tmpdir}")

# Load module
print("Loading module...")
m = runpy.run_path("分层模型_聚集.py")

# Test run_demo (quick baseline with n_days=100)
print("\n🔧 Testing run_demo()...")
try:
    model, traj, summ = m["run_demo"](output_dir=tmpdir, n_days=100, scenario="cluster")
    print(f"  ✅ run_demo() completed")
    print(f"  ✅ Attack rate: {summ.get('attack_rate', 0):.4f}")
    print(f"  ✅ Peak I: {summ.get('peak_I', 0):.1f}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback

    traceback.print_exc()

# Check generated files
print("\n📄 Checking generated output files:")
files_csv = sorted(tmpdir.glob("*.csv"))
files_json = sorted(tmpdir.glob("*.json"))
all_files = files_csv + files_json

for f in all_files:
    size = f.stat().st_size
    print(f"  ✅ {f.name} ({size} bytes)")

print(f"\n✨ Total files generated: {len(all_files)}")
print(f"📋 Expected 9 files:")
print(f"    1. timeseries_cluster.csv")
print(f"    2. place_contribution_cluster.csv")
print(f"    3. opt_result_cluster.json")
print(f"    4. opt_history_cluster.csv (not in demo)")
print(f"    5. summary_cluster.csv")
print(f"    6. config_cluster.json")
print(f"    7. compare_cluster.csv (not in demo)")
print(f"    8. sensitivity_cluster.csv (separate)")
print(f"    9. reff_cluster.csv")

print(f"\n📊 Actual count: {len(all_files) - 3} (expected ~6 for demo run)")
