# 项目记忆

## 工作背景
用户正在完成一门数学建模专业课课程项目，研究内容为上海高校的H3N2流感传播建模。项目采用SVEIQR分层隔室模型，涵盖季节性与动态两种场景，设置2/6/18三种种子策略进行对比，R0≈1.28，模拟周期365天。

## 个人背景
- 用户以中文为主要沟通语言
- Python环境统一使用 `conda activate py310` 或 `conda run -n py310 python`
- 偏好绝对路径，输出形式偏好结构化表格
- 学术写作要求严格遵循正式规范，输出完整长文论文并以md格式呈现
- 参数调整习惯采用迭代调优方式

## 文件结构关键路径
- 论文文件：`D:\HP\OneDrive\Desktop\学校\课程\专业课\数学建模\课程项目\课程论文\`
- 论文正文原版：`论文正文.md`
- 论文整理版（按数模模板）：`论文_整理版.md`
- 数模论文Markdown模板：`数模论文模板.md`
- 模型代码：`pyepidemics-master/h3n2/分层模型/`
- 图表文件：`pyepidemics-master/h3n2/分层模型/figures/`（共9张png）

## 图表清单
| 文件名 | 内容 |
|--------|------|
| fig0_model_framework.png | 模型框架图（新生成） |
| fig1_seasonal_comparison.png | 季节性对比 |
| fig2_scenario_comparison.png | 三场景对比 |
| fig3_place_contribution.png | 场所贡献度 |
| fig4_optimization_comparison.png | 优化对比 |
| fig5_intervention_sensitivity.png | 干预敏感性 |
| fig6_model_validation.png | 模型验证 |
| fig7_reff_curves.png | Reff曲线 |
| fig8_optimization_convergence.png | 优化收敛 |
| fig9_season_scenario_grid.png | 季节场景网格 |
| fig10_prcc_sensitivity.png | PRCC柱状图（新生成） |

## 模型参数要点
- 基础传播率 β₀=0.256，恢复率 γ=0.20，R₀≈1.28
- 常态隔离率0.22，散发0.18，聚集0.12
- 三人群：学生26000、教师1700、后勤800
- 四场所权重：宿舍0.35、教室0.40、食堂0.15、社团0.10

## 论文DOCX生成
- 论文整理版DOCX：`课程论文\论文_整理版.docx`（约4.6MB，含11张图片）
- 转换脚本：`课程论文\md2docx.py`
- 图片生成脚本：`课程论文\gen_missing_figs.py`（生成fig0和fig10）
- 格式要求来源：`课程论文要求.txt`（上海大学数学建模课程论文格式）
- 关键格式：一级标题三号黑体左对齐，正文小四宋体1.25倍行距，三线表，通篇实心点代替句号
- 封皮页第1页、签名页第2页（预留手写）、摘要第3页、目录第4页、正文从第5页起编号
- 注意：Windows下需设 `$env:PYTHONUTF8=1` 才能正确运行含中文的Python脚本
- 待完善：公式仍为文本斜体格式，如需Word原生OMML公式需手动替换

## 写作偏好
- 论文润色时使用"去AI味"skill，去除膨胀措辞、机械枚举、规则三连等AI写作痕迹
- 学术风格保持正式但追求自然平实，避免"核心创新""重要拓展"等自夸用语
