# 论文级输出系统改造 - 完整记录

**日期**: 2024年
**范围**: 三份脚本（聚集、散发、常态）
**改造内容**: 从"各输出各的格式"升级到"统一的论文级多层级输出方案"

## 核心改动清单

### 1️⃣ 新增函数：`save_paper_grade_outputs()`

**位置**: 每个脚本中 `save_trajectory()` 后
**作用**: 统一的多层级输出保存器
**输出层级**（9层）:

| 序号 | 文件 | 内容 | 何时生成 |
|------|------|------|---------|
| 1 | `timeseries_{scenario}.csv` | 完整时间序列（所有房室） | 总是 |
| 2 | `place_contribution_{scenario}.csv` | 场所分解的日新增 | 总是 |
| 3 | `opt_result_{scenario}.json` | 目标函数分解 + 最优控制 | 总是 |
| 4 | `opt_history_{scenario}.csv` | 优化收敛历史 | 仅当有history |
| 5 | `summary_{scenario}.csv` | 摘要统计 | 仅当有summary |
| 6 | `config_{scenario}.json` | 模型配置（可复现） | 总是 |
| 7 | `compare_{scenario}.csv` | 优化前后对比 | 仅当有history |
| 8 | `sensitivity_{scenario}.csv` | PRCC敏感性表 | 独立run_global_sensitivity_demo() |
| 9 | `reff_{scenario}.csv` | R0和有效再生数 | 总是 |

---

### 2️⃣ 改写 `run_demo()` 函数签名

**前**:
```python
def run_demo(output_dir=None, n_days=None, season_profile=None):
    # 仅保存 timeseries.csv + params.csv（旧方案）
```

**后**:
```python
def run_demo(output_dir=None, n_days=None, season_profile=None, scenario="cluster"):
    np.random.seed(42)  # 固定种子
    # 调用 save_paper_grade_outputs() 统一保存 6-7 个文件
```

**改动要点**:
- ✅ 新增 `scenario` 参数（cluster/sporadic/normal）
- ✅ 添加 `np.random.seed(42)` 保证可复现
- ✅ 用 `save_paper_grade_outputs()` 替代分散的保存逻辑

**散发脚本** (分层模型_散发.py): scenario="sporadic"
**常态脚本** (分层模型_常态.py): scenario="normal"
**聚集脚本** (分层模型_聚集.py): scenario="cluster"

---

### 3️⃣ 改写 `run_optimization_demo()` 函数签名

**前**:
```python
def run_optimization_demo(output_dir=None, n_days=None):
    # 保存 4 个文件（trajectory/params/summary/history）
```

**后**:
```python
def run_optimization_demo(output_dir=None, n_days=None, scenario="cluster"):
    np.random.seed(42)  # 固定种子
    # 调用 save_paper_grade_outputs() 统一保存 8 个文件
```

**改动要点**:
- ✅ 新增 `scenario` 参数
- ✅ 添加 `np.random.seed(42)` 
- ✅ 传 history 给 `save_paper_grade_outputs()`，额外生成 opt_history + compare

---

### 4️⃣ 增强 `if __name__ == "__main__"` 块

**新特性**:
- ✅ 全局 `np.random.seed(42)` 固定
- ✅ 命令行快速捷径支持

```bash
# 仅演示（无优化）
python 分层模型_聚集.py demo

# 仅优化（不运行sensitivity）
python 分层模型_聚集.py optim

# 仅敏感性分析
python 分层模型_聚集.py sensitivity

# 完整流程（默认）
python 分层模型_聚集.py
```

---

## 文件命名约定

所有输出文件**必须**包含 scenario 标识符，格式：

```
{功能}_{scenario}.{扩展名}

- timeseries_cluster.csv
- timeseries_sporadic.csv
- timeseries_normal.csv
- place_contribution_cluster.csv
- place_contribution_sporadic.csv
- ... (同理)
```

**好处**:
- ✅ 三份脚本输出可直接放在同一目录，不会冲突
- ✅ 便于自动化对比脚本（glob + scenario filter）
- ✅ 清晰指示每个文件的来源场景

---

## 改动文件清单

### 📝 修改的脚本（3个）

| 脚本 | 改动行数 | 改动内容 |
|------|---------|--------|
| [`分层模型/聚集/分层模型_聚集.py`](h3n2/分层模型/聚集/分层模型_聚集.py) | +200行左右 | 新增 save_paper_grade_outputs + 改 run_demo/run_optimization_demo + 增强 __main__ |
| [`分层模型/散发/分层模型_散发.py`](h3n2/分层模型/散发/分层模型_散发.py) | +200行左右 | 同上（scenario="sporadic"） |
| [`分层模型/常态/分层模型_常态.py`](h3n2/分层模型/常态/分层模型_常态.py) | +200行左右 | 同上（scenario="normal"） |

### 📄 新增文档（1个）

| 文件 | 用途 |
|------|------|
| [`PAPER_GRADE_OUTPUTS_GUIDE.md`](PAPER_GRADE_OUTPUTS_GUIDE.md) | 使用指南 |
| [`REFACTOR_SUMMARY.md`](REFACTOR_SUMMARY.md) | 本文件 — 改动记录 |

---

## 核心算法无变化

✅ 模型方程 (SVEIQR)
✅ 目标函数 (J = A + C + D)
✅ 优化算法 (two-stage: DE + L-BFGS-B)
✅ Re 计算 (next-generation matrix)
✅ PRCC 敏感性分析

**仅改**:
- 输出格式（分散 → 统一）
- 文件命名（无scenario → 含scenario）
- 保存时机（脚本末尾各自为政 → 调用统一函数）
- 种子固定（无 → 加上 seed(42)）

---

## 向后兼容性

⚠️ **旧代码兼容性**：

如果旧脚本依赖于：
- `layered_sveiqr_trajectory.csv` ← 现改为 `timeseries_cluster.csv`
- `layered_sveiqr_params.csv` ← 现改为 `config_cluster.json`

**解决**:
1. 保留旧函数的调用（run_demo 仍接受 output_dir）
2. 新增 scenario 参数（默认值保持向后兼容）
3. 建议更新所有后续处理脚本使用新文件名

---

## 验证清单 ✅

- [x] 三份脚本通过 Python 语法检查（get_errors）
- [x] `save_paper_grade_outputs` 函数可导入
- [x] `run_demo(output_dir)` 生成 6 个文件
- [x] 文件格式正确（CSV/JSON）
- [x] 固定种子工作（R_eff_t0 稳定）
- [x] 快速捷径命令正常（demo/optim/sensitivity）
- [x] 文件命名包含 scenario 标识

---

## 后续可选优化

📌 **不在本次改造范围内**，但可后续实现：

1. **Baseline vs Optimized 自动对比**
   - 同一脚本中跑两次（no control vs with control）
   - 输出差值表

2. **三场景自动汇总**
   - 在根目录创建 `run_all_scenarios.py`
   - 循环三个脚本，汇总 summary 表

3. **图表自动生成模块**
   - 独立的 `plot_from_outputs.py`
   - 从 CSV/JSON 读数据，一键生成论文所需的 6-8 张图

4. **参数输入接口**
   - 命令行参数或 YAML 配置
   - 在不编辑代码的情况下修改 beta0/season_start_day 等

---

## 使用示例

### 场景A：快速生成所有数据（用于论文写作）

```bash
cd h3n2/分层模型/聚集
python 分层模型_聚集.py optim    # 运行并保存所有输出

cd ../散发
python 分层模型_散发.py optim

cd ../常态
python 分层模型_常态.py optim

# 现在所有数据都在各自目录，可以离线作图
```

### 场景B：集中管理所有输出

```bash
# 复制所有 CSV/JSON 到论文目录
mkdir -p paper_data
cp h3n2/分层模型/*/timeseries_*.csv paper_data/
cp h3n2/分层模型/*/config_*.json paper_data/
cp h3n2/分层模型/*/reff_*.csv paper_data/

# 后续离线处理
cd paper_data
python plot_all.py  # 一键生成所有图表
```

### 场景C：复现特定参数组合

```bash
# 读取某次优化的参数
import json
cfg = json.load(open('h3n2/分层模型/聚集/config_cluster.json'))

# 在另一脚本中应用相同参数
params.beta0 = cfg['parameters']['beta0']
params.sigma = cfg['parameters']['sigma']
# ... 其他参数

model = CampusLayeredSVEIQR(..., params=params)
# 结果可完全复现，因为种子固定
```

---

## 相关链接

- 📖 详细使用指南: [`PAPER_GRADE_OUTPUTS_GUIDE.md`](PAPER_GRADE_OUTPUTS_GUIDE.md)
- 🔧 原始脚本: `h3n2/分层模型/{常态,散发,聚集}/分层模型_*.py`
- 📊 输出示例: 本次运行在各脚本目录生成的 `*.csv` 和 `*.json` 文件

---

**改造完成日期**: 20xx年x月x日
**验证状态**: ✅ 所有脚本通过检查，烟雾测试通过
**文档状态**: ✅ 完整

