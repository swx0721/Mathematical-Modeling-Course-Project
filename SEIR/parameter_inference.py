import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import os

# ==========================================
# 1. 路径自动兼容处理
# ==========================================
# 根据脚本位置自动定位数据文件
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "china_flu_realistic_final.csv")

# 加载数据
df = pd.read_csv(data_path)
df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")

target_col = "ili_pct"
if target_col not in df.columns:
    raise ValueError(f"数据中不存在目标列: {target_col}")

obs_data = df[["week_start", target_col]].copy()
obs_data[target_col] = pd.to_numeric(obs_data[target_col], errors="coerce")
obs_data = obs_data.dropna(subset=["week_start", target_col])
obs_data = obs_data[(obs_data[target_col] >= 0) & (obs_data[target_col] <= 100)]

if len(obs_data) < 2:
    raise ValueError("可用于拟合的有效观测点不足（至少需要2个非空且有限值）。")

obs_ili = obs_data[target_col].to_numpy(dtype=float) / 100.0  # 转为小数比例
t_weeks = np.arange(len(obs_ili))
t_days = t_weeks * 7


# ==========================================
# 2. 定义 SEIR 模型
# ==========================================
def seir_seasonal_model(t, y, n, b0, a, phi, sigma, gamma):
    S, E, I, R = y
    # 季节性强制函数
    beta_t = b0 * (1 + a * np.cos(2 * np.pi * (t / 365.0) - phi))

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

    N = 30000  # 上海某高校假设人数
    sigma = 1 / 1.5  # 潜伏期
    gamma = 1 / 4.0  # 传染期

    # 初始状态
    I0 = max(N * obs_ili[0], 1.0)
    E0 = max(I0 * 1.5, 1.0)
    S0 = max(N - I0 - E0, 0.0)
    y0 = [S0, E0, I0, 0.0]

    sol = solve_ivp(
        seir_seasonal_model,
        [0, t_days[-1]],
        y0,
        args=(N, b0, a, phi, sigma, gamma),
        t_eval=t_days,
        method="RK45",
    )

    if (not sol.success) or (sol.y.shape[1] != len(t_days)):
        return np.full(len(obs_ili), 1e6, dtype=float)

    sim_i_pct = sol.y[2] / N
    if not np.isfinite(sim_i_pct).all():
        return np.full(len(obs_ili), 1e6, dtype=float)

    # 确保长度一致
    return (sim_i_pct - obs_ili)[: len(obs_ili)]


# ==========================================
# 4. 执行推理
# ==========================================
initial_guess = [0.8, 0.3, 0.5]  # 稍微调高初始猜想
bounds = ([0.01, 0.0, -np.inf], [5.0, 1.0, np.inf])

print("正在计算最符合上海校园真实趋势的参数...")
print(f"有效观测点数量: {len(obs_ili)}")
res = least_squares(objective, initial_guess, bounds=bounds)

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

# 计算 R0
gamma = 1 / 4.0
print(f"当前校准后的基本再生数 R0 约为: {fitted_beta0/gamma:.2f}")
