import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import os
import json

# ==========================================
# 1. 路径自动兼容处理 & 加载配置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "parameters", "configs.json")

# 加载配置
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# 提取参数
MODEL_PARAMS = config["model_parameters"]
POPULATION = config["population"]
OPTIMIZATION = config["optimization"]
FILE_PATHS = config["file_paths"]
DATA_PROC = config["data_processing"]
RUNTIME = config["runtime"]

# 根据配置的相对路径构建绝对路径
data_path = os.path.join(current_dir, FILE_PATHS["realistic_data"])

# 加载数据
df = pd.read_csv(data_path)
df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")

target_col = DATA_PROC["target_col"]
if target_col not in df.columns:
    raise ValueError(f"数据中不存在目标列: {target_col}")

obs_data = df[["week_start", target_col]].copy()
obs_data[target_col] = pd.to_numeric(obs_data[target_col], errors="coerce")
obs_data = obs_data.dropna(subset=["week_start", target_col])
obs_data = obs_data[
    (obs_data[target_col] >= DATA_PROC["value_min"])
    & (obs_data[target_col] <= DATA_PROC["value_max"])
]

if len(obs_data) < 2:
    raise ValueError("可用于拟合的有效观测点不足（至少需要2个非空且有限值）。")

obs_ili = (
    obs_data[target_col].to_numpy(dtype=float) / DATA_PROC["ili_pct_scale"]
)  # 转为小数比例
t_weeks = np.arange(len(obs_ili))
t_days = t_weeks * RUNTIME["days_per_week"]


# ==========================================
# 2. 定义 SEIR 模型
# ==========================================
def seir_seasonal_model(t, y, n, b0, a, phi, sigma, gamma):
    S, E, I, R = y
    # 季节性强制函数
    beta_t = b0 * (1 + a * np.cos(2 * np.pi * (t / RUNTIME["days_per_year"]) - phi))

    dSdt = -beta_t * S * I / n
    dEdt = beta_t * S * I / n - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


# ==========================================
# 3. 拟合目标函数
# ==========================================
def objective(params_to_fit):
    b0, a, phi = params_to_fit

    # 从配置读取模型参数
    N = POPULATION["N"]["value"]
    sigma = MODEL_PARAMS["sigma"]["value"]
    gamma = MODEL_PARAMS["gamma"]["value"]
    e0_multiplier = POPULATION["initial_conditions"]["E0_multiplier"]

    # 初始状态
    I0 = max(N * obs_ili[0], 1.0)
    E0 = max(I0 * e0_multiplier, 1.0)
    S0 = max(N - I0 - E0, 0.0)
    y0 = [S0, E0, I0, 0.0]

    sol = solve_ivp(
        seir_seasonal_model,
        [0, t_days[-1]],
        y0,
        args=(N, b0, a, phi, sigma, gamma),
        t_eval=t_days,
        method=RUNTIME["ode_method"],
    )

    if (not sol.success) or (sol.y.shape[1] != len(t_days)):
        return np.full(len(obs_ili), 1e6, dtype=float)

    sim_i_pct = sol.y[2] / N
    if not np.isfinite(sim_i_pct).all():
        return np.full(len(obs_ili), 1e6, dtype=float)

    # 确保长度一致
    return (sim_i_pct - obs_ili)[: len(obs_ili)]


# ==========================================
# 4. 执行推理（使用配置的初值与边界）
# ==========================================
initial_guess = [
    OPTIMIZATION["beta0"]["initial"],
    OPTIMIZATION["a"]["initial"],
    OPTIMIZATION["phi"]["initial"],
]
bounds = (
    [
        OPTIMIZATION["beta0"]["bounds"][0],
        OPTIMIZATION["a"]["bounds"][0],
        OPTIMIZATION["phi"]["bounds"][0],
    ],
    [
        OPTIMIZATION["beta0"]["bounds"][1],
        OPTIMIZATION["a"]["bounds"][1],
        OPTIMIZATION["phi"]["bounds"][1],
    ],
)

print("正在计算最符合上海校园真实趋势的参数...")
print(f"有效观测点数量: {len(obs_ili)}")
res = least_squares(
    objective,
    initial_guess,
    bounds=bounds,
    loss=OPTIMIZATION["loss_function"],
    max_nfev=int(RUNTIME["optimizer_max_nfev"]),
)

# ==========================================
# 5. 结果展示
# ==========================================
fitted_beta0, fitted_a, fitted_phi = res.x
print("\n" + "=" * 30)
print("推理出的模型校准参数：")
print(f"基础传播率 beta0: {fitted_beta0:.4f}")
print(f"季节性振幅 a:     {fitted_a:.4f}")
print(f"季节性相位 phi:   {fitted_phi:.4f}")
print("=" * 30)

# 计算 R0（使用配置中的 gamma）
gamma_cfg = MODEL_PARAMS["gamma"]["value"]
print(f"当前校准后的基本再生数 R0 约为: {fitted_beta0/gamma_cfg:.2f}")
