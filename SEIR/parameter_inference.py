import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import os

# ==========================================
# 1. 路径自动兼容处理
# ==========================================
# 你提供的基础路径
BASE_DIR = r"E:\task\学校任务\大三春季\数学建模\Mathematical-Modeling-Course-Project-main\Mathematical-Modeling-Course-Project-main"

# 尝试寻找可能的文件名
possible_files = [
    "china_flu_realistic_final.csv",
]

data_path = None
for f in possible_files:
    full_path = os.path.join(BASE_DIR, f)
    if os.path.exists(full_path):
        data_path = full_path
        print(f"成功找到数据文件: {f}")
        break

if not data_path:
    raise FileNotFoundError(f"在路径 {BASE_DIR} 下未找到任何匹配的 CSV 文件，请检查文件名！")

# 加载数据
df = pd.read_csv(data_path)
df['week_start'] = pd.to_datetime(df['week_start'])

# 提取用于拟合的关键段落：2023年流感大爆发时期（特征最明显）
# 如果你的数据列名是 ili_pct_new，请确保一致
target_col = 'ili_pct_new' if 'ili_pct_new' in df.columns else 'ili_pct'
mask = (df['week_start'] >= '2023-10-01') & (df['week_start'] <= '2024-05-20')
obs_data = df.loc[mask].copy()

if obs_data[target_col].isnull().any():
    print("警告：拟合时段内存在空值，已自动剔除")
    obs_data = obs_data.dropna(subset=[target_col])

obs_ili = obs_data[target_col].values / 100.0  # 转为小数比例
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
    
    N = 30000        # 上海某高校假设人数
    sigma = 1 / 1.5  # 潜伏期
    gamma = 1 / 4.0  # 传染期
    
    # 初始状态
    I0 = N * obs_ili[0] 
    E0 = I0 * 1.5
    S0 = N - I0 - E0
    y0 = [S0, E0, I0, 0.0]
    
    sol = solve_ivp(seir_seasonal_model, [0, t_days[-1]], y0, 
                    args=(N, b0, a, phi, sigma, gamma),
                    t_eval=t_days, method='RK45')
    
    sim_i_pct = sol.y[2] / N
    # 确保长度一致
    return (sim_i_pct - obs_ili)[:len(obs_ili)]

# ==========================================
# 4. 执行推理
# ==========================================
initial_guess = [0.8, 0.3, 0.5] # 稍微调高初始猜想
bounds = ([0.01, 0.0, -np.inf], [5.0, 1.0, np.inf])

print("正在计算最符合上海校园真实趋势的参数...")
res = least_squares(objective, initial_guess, bounds=bounds)

# ==========================================
# 5. 结果展示
# ==========================================
fitted_beta0, fitted_a, fitted_phi = res.x
print("\n" + "="*30)
print("推理出的模型校准参数：")
print(f"基础传播率 beta0: {fitted_beta0:.4f}")
print(f"季节性振幅 a:     {fitted_a:.4f}")
print(f"季节性相位 phi:   {fitted_phi:.4f}")
print("="*30)

# 计算 R0
gamma = 1/4.0
print(f"当前校准后的基本再生数 R0 约为: {fitted_beta0/gamma:.2f}")