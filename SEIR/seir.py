import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import LinearRegression

# 1. 路径处理与配置加载
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "parameters", "configs.json")

# 加载配置
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

FILE_PATHS = config["file_paths"]
SEASONAL_BASELINE = config["seasonal_baseline"]["month_baselines"]
DATA_PROC = config["data_processing"]
DATA_GEN = config["data_generation"]

# 根据配置的相对路径构建绝对路径
data_path = os.path.join(current_dir, FILE_PATHS["china_flu_weekly"])
val_path = os.path.join(current_dir, FILE_PATHS["validation_set"])

try:
    data = pd.read_csv(data_path)
    data_val = pd.read_csv(val_path)
except FileNotFoundError:
    data = pd.read_csv("china_flu_weekly.csv")
    data_val = pd.read_csv("validation_set_2023_2024.csv")

# 2. 预处理
data["week_start"] = pd.to_datetime(data["week_start"])
data["month"] = data["week_start"].dt.month

# 构造月度基线字典（从配置中读取）
monthly_baseline = {int(k): v for k, v in SEASONAL_BASELINE.items()}

# --- 关键改进 2：训练“增量模型” ---
# 我们利用验证集计算：增量 = 真实ILI% - 该月基准
data_val["month"] = pd.to_datetime(data_val["week_start"]).dt.month
data_val["baseline"] = data_val["month"].map(monthly_baseline)
data_val["ili_increment"] = data_val["ili_pct_full"] - data_val["baseline"]

# 只针对有意义的增量进行拟合
X_val = data_val[["h3n2_pct"]].values
y_inc = data_val["ili_increment"].values

# 强制截距为 0：如果没有 H3N2，增量就该是 0
model_inc = LinearRegression(fit_intercept=False)
model_inc.fit(X_val, y_inc)
slope = model_inc.coef_[0]


# 3. 定义填充函数
def predict_ili_realistic(h3n2_pct, month):
    if pd.isna(h3n2_pct):
        return np.nan

    # 基础值
    base = monthly_baseline.get(month, DATA_GEN["default_month_baseline"])

    # 增量计算：使用 1.1 次方增加爆发感的非线性（阳性率越高，斜率越大）
    increment = slope * (h3n2_pct ** DATA_GEN["h3n2_nonlinear_power"])

    final_val = base + increment

    # 限制最低值，防止噪声导致负数或过低值
    return round(
        max(DATA_GEN["min_ili_pct"], final_val),
        int(DATA_GEN["round_digits"]),
    )


# 4. 执行填充
missing_mask = data["ili_pct"].isna()
data.loc[missing_mask, "ili_pct"] = data.loc[missing_mask].apply(
    lambda row: predict_ili_realistic(row["h3n2_pct"], row["month"]), axis=1
)

# 5. 保存
output_file = os.path.join(
    current_dir, FILE_PATHS["output_dir"], "china_flu_realistic_final.csv"
)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
data.drop(columns=["month"]).to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ 还原完成！")
print(
    f"数据特征：月度基准控制在 {min(monthly_baseline.values())}% ~ {max(monthly_baseline.values())}%"
)
print(f"拟合增量斜率：{slope:.2f}")
print(f"仅填充缺失 ili_pct 数量：{int(missing_mask.sum())}")
print(f"结果已存至：{output_file}")
