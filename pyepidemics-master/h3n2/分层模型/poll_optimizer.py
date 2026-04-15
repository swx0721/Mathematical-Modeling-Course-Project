"""轮询等待三个优化进程完成，完成后输出结果摘要。"""
import subprocess, time, json, os
from pathlib import Path

ROOT = Path(__file__).parent
SCENARIOS = [("normal", "常态"), ("sporadic", "散发"), ("cluster", "聚集")]

def check_opt_done(key):
    d = ROOT / SCENARIOS[{"normal":0,"sporadic":1,"cluster":2}[key]][1]
    p = d / f"opt_result_{key}.json"
    if not p.exists():
        return False, None
    try:
        with open(p, encoding="utf-8-sig") as f:
            data = json.load(f)
        # Check if attack_rate exists (means optimization completed)
        return "attack_rate" in data, data
    except:
        return False, None

def count_python_procs():
    try:
        r = subprocess.run(
            ['powershell', '-Command', 
             'Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count'],
            capture_output=True, text=True, timeout=10
        )
        return int(r.stdout.strip() or 0)
    except:
        return -1

def get_history_lines(key):
    d = ROOT / SCENARIOS[{"normal":0,"sporadic":1,"cluster":2}[key]][1]
    p = d / f"opt_history_{key}.csv"
    if not p.exists():
        return 0
    with open(p, encoding="utf-8-sig") as f:
        return sum(1 for _ in f) - 1  # minus header

print("=" * 60)
print("Polling optimizer processes ...")
print("=" * 60)

poll_interval = 30  # seconds
max_wait = 120 * 60  # 2 hours max

start = time.time()
while True:
    elapsed = int(time.time() - start)
    if elapsed > max_wait:
        print(f"\nTimeout after {max_wait//60} minutes!")
        break
    
    n_procs = count_python_procs()
    
    status_lines = []
    all_done = True
    for key, cn in SCENARIOS:
        done, data = check_opt_done(key)
        lines = get_history_lines(key)
        if done:
            ar = data.get("attack_rate", 0)
            j = data.get("J", 0)
            status_lines.append(f"  {cn}: DONE ({lines} iters, AR={ar:.4f}, J={j:.4f})")
        else:
            all_done = False
            status_lines.append(f"  {cn}: running ({lines} iters)")
    
    mins = elapsed // 60
    secs = elapsed % 60
    print(f"\n[{mins:02d}:{secs:02d}] Python processes: {n_procs}")
    for line in status_lines:
        print(line)
    
    if all_done and n_procs == 0:
        print("\n" + "=" * 60)
        print("ALL OPTIMIZATIONS COMPLETED!")
        print("=" * 60)
        
        # Print final results
        for key, cn in SCENARIOS:
            done, data = check_opt_done(key)
            if done:
                print(f"\n--- {cn} ---")
                for k, v in data.items():
                    if k not in ("best_x", "history"):
                        print(f"  {k}: {v}")
        break
    
    time.sleep(poll_interval)
