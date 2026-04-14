# 论文级输出系统 - 快速使用指南

## 概述

三份脚本（常态、散发、聚集）已升级为支持**论文级一次性数据采集**方案。每次运行输出 6-9 个文件，涵盖模型配置、时间序列、优化结果、指标换算等所有论文所需数据。

## 文件结构

```
聚集/                              (for 聚集 scenario)
├── timeseries_cluster.csv          # 时间序列（所有房室、人群、总体）
├── place_contribution_cluster.csv  # 场所贡献度（日新增按dorm/class/canteen/club分解）
├── summary_cluster.csv             # 摘要统计（peak_I, attack_rate等）
├── config_cluster.json             # 模型配置（beta0, sigma, gamma, populations等）
├── opt_result_cluster.json         # 优化结果（目标函数J/A/C/D + 防控参数）
├── reff_cluster.csv                # 再生数曲线（R0 + R_eff(t)采样）
├── opt_history_cluster.csv         # [仅优化] 收敛历史
└── compare_cluster.csv             # [仅优化] 优化前后对比

散发/
├── timeseries_sporadic.csv
├── place_contribution_sporadic.csv
├── ... (同上，改为 sporadic)

常态/
├── timeseries_normal.csv
├── place_contribution_normal.csv
├── ... (同上，改为 normal)
```

## 快速入门

### 1. 仅运行基线/演示（不优化）

```bash
cd 聚集/
python 分层模型_聚集.py demo
```

**输出**: 6 个文件 (timeseries + place_contribution + summary + config + opt_result + reff)

### 2. 运行完整优化流程

```bash
cd 聚集/
python 分层模型_聚集.py optim
```

**输出**: 8 个文件 (上述6个 + opt_history + compare)

### 3. 运行敏感性分析（PRCC）

```bash
cd 聚集/
python 分层模型_聚集.py sensitivity
```

**输出**: PRCC表 + sensitivity_cluster.csv

### 4. 默认完整流程

```bash
cd 聚集/
python 分层模型_聚集.py
```

**执行**: 优化 → 生成图 → 敏感性分析 → 保存所有9层输出

## 核心特性

✅ **固定随机种子** (`np.random.seed(42)`) — 所有运行可完全复现
✅ **完整时间序列** — 保存所有天数所有房室数据，支持离线作图
✅ **统一文件命名** — 包含 scenario 标识，三份脚本输出可直接对比
✅ **JSON 配置** — model_config.json 包含完整参数，支持配置重用
✅ **多层级输出** — 从原始数据到摘要统计，满足论文各类需求

## 文件内容说明

| 文件 | 行数/字段 | 用途 |
|------|---------|------|
| timeseries_*.csv | day~days | 所有房室时间序列，基础绘图 |
| place_contribution_*.csv | day + 4place列 | place-wise分解图 |
| summary_*.csv | 1行 | 报表：peak_I, attack_rate, final_R, final_Q |
| config_*.json | 嵌套dict | 参数保存与复现 |
| opt_result_*.json | 目标函数+控制 | J/A/C/D + mask_u等7维 |
| opt_history_*.csv | 优化迭代日志 | 收敛曲线，可画optimization_curve图 |
| compare_*.csv | 改进前后 | attack_rate/peak_I下降百分比 |
| reff_*.csv | day + R_eff | 再生数曲线图 |

## 与论文撰写的对应

### 图1: 时间序列对比
```python
import pandas as pd
df_normal = pd.read_csv('常态/timeseries_normal.csv')
df_sporadic = pd.read_csv('散发/timeseries_sporadic.csv')
df_cluster = pd.read_csv('聚集/timeseries_cluster.csv')
# 绘图：df['day'] vs df['I'] (3条曲线)
```

### 图2: 场所分解 (Stacked area)
```python
df = pd.read_csv('聚集/place_contribution_cluster.csv')
df.set_index('day')[['new_dorm', 'new_class', 'new_canteen', 'new_club']].plot.area()
```

### 图3: 优化效果（改进比例）
```python
df_hist = pd.read_csv('聚集/opt_history_cluster.csv')
df_hist[['A', 'C', 'D']].plot()  # 三个目标函数分量的收敛
```

### 表1: 模型参数
```python
import json
cfg = json.load(open('聚集/config_cluster.json'))
# → beta0, sigma, gamma, populations, vaccine_coverage
```

### 表2: 对比结果
```python
df_summary = pd.read_csv('聚集/summary_cluster.csv')
# → peak_I, attack_rate, + 三个分量
```

### 表3: PRCC敏感性
```python
df_prcc = pd.read_csv('聚集/sensitivity_cluster.csv')
df_prcc.sort_values('abs_prcc', ascending=False).head(10)
```

## 代码集成示例

### 示例 A: 循环三个场景生成所有输出

```python
from pathlib import Path
import pandas as pd

result_root = Path("./paper_results")
scenarios = ["normal", "sporadic", "cluster"]

for scenario in scenarios:
    script_dir = Path(f"h3n2/分层模型/{get_cn_name(scenario)}")
    # 运行脚本
    os.system(f"cd {script_dir} && python 分层模型_*.py optim")
    
    # 收集所有输出
    for file in (script_dir / "*.csv").glob("*_{scenario}.csv"):
        shutil.copy2(file, result_root / file.name)
```

### 示例 B: 快速对比三个场景

```python
import pandas as pd
from pathlib import Path

root = Path("h3n2/分层模型")
summaries = {}

for scenario_cn, scenario_en in [("常态", "normal"), ("散发", "sporadic"), ("聚集", "cluster")]:
    df = pd.read_csv(root / scenario_cn / f"summary_{scenario_en}.csv")
    summaries[scenario_en] = df
    
pd.concat(summaries).to_csv("comparison_table.csv")
```

## 关键提醒

⚠️ **随机种子固定**：所有脚本已内置 `np.random.seed(42)`，确保可复现
⚠️ **文件命名包含scenario**：timeseries_normal.csv 不会与 timeseries_cluster.csv 混淆
⚠️ **保存完整时间序列**：不仅保存摘要，每日数据都存——便于后续补充作图
⚠️ **JSON配置不可或缺**：config_*.json 保存所有参数，支持论文附录或补充材料

## 故障排查

Q: 为什么没有生成 opt_history_cluster.csv？
A: 这个文件只在 `run_optimization_demo()` 中生成。如果只跑 `run_demo()`，则无此文件。

Q: 可以修改输出目录吗？
A: 可以。调用 `run_demo(output_dir="/custom/path")` 即可。

Q: 如何确保参数一致？
A: 每个 config_*.json 都完整保存了参数。拷贝该配置文件到另一脚本即可复现。

