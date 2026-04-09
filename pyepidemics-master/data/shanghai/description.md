# 上海 H3N2 SEIR 数据说明

## 1. 数据概览

| 指标 | 值 | 备注 |
|---|---:|---|
| 数据性质 | 基于公开周报与论文重建 | 不是官方原始病例台账 |
| 总人口 N | 24,183,300 | 上海 2017 年末常住人口 |
| 假设波次总感染率 | 3.0% | 用于把代理曲线映射为感染规模 |
| 假设波次总感染数 | 725,499 | 由总人口与攻击率得到 |
| 有效易感池初值 S0 | 6,045,825 | 约占总人口 25% |
| 峰值新感染日期 | 2017-08-21 | |
| 峰值新感染数 | 10,195 | |
| 峰值传染者日期 | 2017-08-22 | |
| 峰值传染者 I | 30,454 | |
| 平均 Rt_eff | 1.145 | 样本期均值 |

## 2. 参数设定与依据

| 参数 | 取值 | 用途 | 依据 | URL |
|---|---:|---|---|---|
| 总人口 N | 24183300 | 城市总人口基数 | 上海市 2017 年末常住人口 2418.33 万 | https://www.shanghai.gov.cn/nw9822/20200906/0001-9822_1283583.html |
| 波次总感染率 | 3.0% | 把比例曲线映射为总感染量 | 假设值。低于 WHO 对完整季节流感成年人 5%-10% 年攻击率，因为这里只取单个 H3N2 夏季波次 | https://www.who.int/publications/m/item/vaccine-preventable-diseases-surveillance-standards-influenza |
| 有效易感池占比 | 25.0% | 使季节性免疫背景下的 S 具有可解释性 | 假设值。表示进入本次波次的有效易感人口，而非全人口完全易感 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11378670/ |
| 无症状比例 | 16.0% | 感染者拆分为有症状与无症状 | Leung 等综述，pooled mean 约 16% | https://pmc.ncbi.nlm.nih.gov/articles/PMC4586318/ |
| 有症状比例 | 84.0% | 估算有症状感染数 | 由 1 - 无症状比例得到 | https://pmc.ncbi.nlm.nih.gov/articles/PMC4586318/ |
| 就诊/报告比例 | 25.0% | 由有症状感染映射为报告病例 | ILI 就诊研究显示约 24.1%-31.1% 寻求医疗，此处取 25% 基线 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9850268/ |
| 潜伏期 | 1.5 天 | SEIR 中 E->I 速率 | WHO 指出流感感染到发病约 2 天，建模取 1.5 天近似 | https://www.who.int/news-room/fact-sheets/detail/influenza-%28seasonal%EF%BC%89 |
| 传染期 | 3 天 | SEIR 中 I->R 速率 | 结合平均序列间隔约 3.6 天，简化取 3 天 | https://pmc.ncbi.nlm.nih.gov/articles/PMC3057478/ |

## 3. 数据构建流程

1. 代理曲线
使用上海 H3N2 代理阳性率曲线，锚定国家流感中心周报与上海地区文献。

2. 分配总感染数
按每日代理阳性率在波次内占比，分配假设总感染数 725,499。

3. 症状拆分
新有症状感染数 = 新感染数 × 84%。
新无症状感染数 = 新感染数 × 16%。

4. 报告病例映射
新报告病例数 = 新有症状感染数 × 25%。
该口径表示被医疗系统捕获的病例。

5. SEIR 递推
E_t = E_(t-1) + new_inf_t - sigma * E_(t-1)
I_t = I_(t-1) + sigma * E_(t-1) - gamma * I_(t-1)

6. 有效易感池
S_t 仅在有效易感池内递减。相比全人口完全易感假设，更符合季节性流感背景。

7. beta_eff 与 Rt_eff
由 incidence = beta_eff * S * I / S0_eff 反推时间变化传播强度，主要用于校准初值参考。

## 4. 字段解释与拟合建议

| 字段 | 含义 | 拟合用途建议 |
|---|---|---|
| 日期 | 日粒度时间索引 | 设为模型拟合索引 |
| 新感染数 | 当日新感染总量 | 可用于拟合 incidence |
| 潜伏者_E | 当日 E 状态 | 可作为 E 观测代理 |
| 传染者_I | 当日 I 状态 | 可作为 I 观测代理 |
| 移除者_R | 当日 R 状态 | 可用于校验恢复规模 |
| 新报告病例数 | 医疗系统观测口径 | 可作为可见病例约束 |
| beta_eff / Rt_eff | 反推传播强度 | 适合作为参数初值或先验，不建议直接做主拟合目标 |

建议优先用 新感染数 与 传染者_I 联合拟合 beta、sigma、gamma，再用 新报告病例数 验证报告率相关假设。

## 5. 参考来源

| 来源类别 | 出处 | 关键用途 | URL |
|---|---|---|---|
| 人口 | 上海市 2017 年国民经济运行情况 | 总人口 N | https://www.shanghai.gov.cn/nw9822/20200906/0001-9822_1283583.html |
| 官方周报 | 国家流感中心 2017 第30周 | 代理曲线锚点 | https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/201708/t20170807_149211.htm |
| 官方周报 | 国家流感中心 2017 第33周 | 代理曲线锚点 | https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/201708/t20170828_151328.htm |
| 官方周报 | 国家流感中心 2017 第34周 | 代理曲线锚点 | https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/201709/t20170905_151982.htm |
| 官方周报 | 国家流感中心 2017 第38周 | 代理曲线锚点 | https://ivdc.chinacdc.cn/cnic/zyzx/lgzb/201711/t20171106_154688.htm |
| 上海论文 | Emerg Microbes Infect. 2024 | 2017 上海夏季峰时间约束 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11378670/ |
| 上海论文 | Scientific Reports 2022 | 2017 上海口岸 H3N2 夏季波次约束 | https://www.nature.com/articles/s41598-022-19228-y |
| WHO | Influenza surveillance standards | 完整季节攻击率参考区间 | https://www.who.int/publications/m/item/vaccine-preventable-diseases-surveillance-standards-influenza |
| WHO | Seasonal influenza fact sheet | 感染到发病时间范围 | https://www.who.int/news-room/fact-sheets/detail/influenza-%28seasonal%EF%BC%89 |
| 综述 | Influenza asymptomatic fraction meta-analysis | 无症状比例 | https://pmc.ncbi.nlm.nih.gov/articles/PMC4586318/ |
| 实证研究 | ILI care-seeking study | 就诊/报告比例基线 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9850268/ |
| 实证研究 | Serial interval of influenza | 传染期简化依据 | https://pmc.ncbi.nlm.nih.gov/articles/PMC3057478/ |
