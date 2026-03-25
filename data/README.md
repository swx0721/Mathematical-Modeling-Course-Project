# 数据集字段说明

## 文件清单

| 文件名 | 描述 | 行数 |
|--------|------|------|
| `china_flu_weekly.csv` | FluNet 全国流感监测原始数据（2005~2026） | 989 行 |
| `validation_set_2023_2024.csv` | SEIR 模型验证集（2023W40~2024W20） | 33 行 |
| `VIW_FNT.csv` | FluNet 官网原始下载文件（未处理） | — |
| `validation_overview.png` | 验证集可视化图（ILI% 与 H3N2 阳性率趋势） | — |

---

## `validation_set_2023_2024.csv` 字段说明

> **用途**：SEIR 模型参数校准与误差验证的主数据集。
> 时间范围：2023 年第 40 周（10 月初）~ 2024 年第 20 周（5 月中），共 **33 周**，ILI% 100% 完整。

| 字段名 | 类型 | 单位/范围 | 含义 | 备注 |
|--------|------|-----------|------|------|
| `iso_year` | int | 2023/2024 | ISO 年份 | |
| `iso_week` | int | 1~52 | ISO 周次 | W40=10月初，W01=1月初 |
| `week_start` | str | YYYY/MM/DD | 该周起始日期（周一） | |
| `h3n2_positive` | float | 份 | H3N2 阳性检出绝对数 | 全国哨点医院送检样本中的 H3N2 阳性份数 |
| `h3n2_pct` | float | 0~1 小数 | **H3N2 阳性率** | = H3N2 阳性数 / 总送检数；×100 转为百分比；峰值约 35.5%（2023W49） |
| `inf_all` | float | 份 | 流感全型总检出数 | H1N1 + H3N2 + B 型等合计 |
| `inf_pct` | float | 0~1 小数 | 流感总阳性率 | = 流感总检出 / 总送检数 |
| `ili_pct_full` | float | % | **ILI%（南北方均值）** | 流感样病例就诊百分比；= (ili_south + ili_north) / 2；**建模验证主用字段**，与 SEIR 仿真 I(t)/N 直接对比 |
| `ili_south` | float | % | 南方哨点 ILI% | 来源：中国疾控中心英文流感周报 PDF（南方省份哨点医院） |
| `ili_north` | float | % | 北方哨点 ILI% | 来源：中国疾控中心英文流感周报 PDF（北方省份哨点医院） |
| `ili_activity` | float | 1~5 等级 | ILI 活动等级 | 1=基线；2=低；3=中；4=较高流行；5=高度流行；验证集全程为 4.0 |

### 关键统计

| 指标 | 最小值 | 峰值 | 峰值周次 |
|------|--------|------|----------|
| ILI%（`ili_pct_full`） | 3.8% | 8.4% | 2024W01 |
| H3N2 阳性率（`h3n2_pct`×100） | 0.7% | 35.5% | 2023W49 |

---

## `china_flu_weekly.csv` 字段说明

> **用途**：完整历史数据，可用于季节性因子标定、参数粗估。
> 时间范围：2005W39~2026W11，共 **989 行**。

| 字段名 | 类型 | 单位/范围 | 含义 | 备注 |
|--------|------|-----------|------|------|
| `iso_year` | int | — | ISO 年份 | |
| `iso_week` | int | 1~52 | ISO 周次 | |
| `week_start` | str | YYYY/MM/DD | 该周起始日期 | |
| `spec_processed` | int | 份 | 总送检样本数 | 阳性率的分母 |
| `h3n2_positive` | float | 份 | H3N2 阳性绝对数 | 917/989 行有值 |
| `inf_all` | float | 份 | 流感全型总检出数 | |
| `inf_a` | float | 份 | 甲型流感检出数 | 含 H1N1 和 H3N2 |
| `inf_b` | float | 份 | 乙型流感检出数 | |
| `h3n2_pct` | float | 0~1 小数 | H3N2 阳性率 | 987/989 行有值 |
| `inf_pct` | float | 0~1 小数 | 流感总阳性率 | |
| `ili_activity` | float | 1~5 等级 | ILI 活动等级 | 989 行全满 |
| `ili_pct` | float | % | ILI%（原始） | **仅最近 19 周有值**（2025W45~2026W11），历史全为空 |

---

## 建模使用指南

### SEIR 模型验证流程

```
SEIR 仿真输出 I(t)/N（感染比例）
        ↓
乘以转换系数 k（ILI就诊率）→ 等价 ILI%
        ↓
与 validation_set_2023_2024.csv 的 ili_pct_full 列逐周对比
        ↓
计算 RMSE / MAE，最小化误差以校准 β 等参数
```

### 字段选用建议

| 建模目的 | 推荐字段 | 数据文件 |
|----------|----------|----------|
| 模型验证（主） | `ili_pct_full` | validation_set_2023_2024.csv |
| 流行强度参考 | `h3n2_pct` | validation_set_2023_2024.csv |
| 季节性因子标定 | `h3n2_pct`（多年均值） | china_flu_weekly.csv |
| 参数粗估 | `inf_pct`、`spec_processed` | china_flu_weekly.csv |

### 注意事项

- `h3n2_pct` 是 **0~1 小数**，使用时需 ×100 转为百分比
- 2020~2022 年数据受 COVID-19 防控影响，H3N2 几乎归零，**不建议用于参数拟合**
- `ili_south` / `ili_north` 反映地区差异，上海高校建模可酌情偏向南方数据

---

## 数据来源

- **FluNet 数据**：WHO 全球流感监测网络（[FluNet](https://www.who.int/tools/flunet)），中国大陆报告数据
- **ILI% 数据**：中国疾控中心（CCDC）英文流感周报 PDF（[ivdc.chinacdc.cn](https://ivdc.chinacdc.cn/cnic/en/Surveillance/WeeklyReport/)），南北方哨点医院数据
