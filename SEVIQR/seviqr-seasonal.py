import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# =========================
# 1. 基本设置
# =========================
GROUPS = ["students", "teachers", "staff"]
G = len(GROUPS)

# 各群体人口
N = np.array([18000, 1800, 1200], dtype=float)

# 接触矩阵 C[g,h]
# 行表示“谁被感染”，列表示“感染来自谁”
C = np.array(
    [
        [12.0, 2.0, 1.5],  # 学生接触 学生/教师/职工
        [4.0, 3.0, 1.5],  # 教师接触 学生/教师/职工
        [3.0, 1.5, 2.0],  # 职工接触 学生/教师/职工
    ],
    dtype=float,
)


# =========================
# 2. 分阶段防控策略
# =========================
def control_level(t):
    """
    根据时间返回防控强度
    u_m: 口罩
    u_w: 通风
    u_s: 社团限流
    u_d: 线上授课/教学密度控制
    """
    # 例子：0-20天常态，20-40天散发，40天后聚集响应
    if t < 20:
        return {"u_m": 0.2, "u_w": 0.3, "u_s": 0.1, "u_d": 0.0}
    elif t < 40:
        return {"u_m": 0.5, "u_w": 0.6, "u_s": 0.4, "u_d": 0.2}
    else:
        return {"u_m": 0.8, "u_w": 0.8, "u_s": 0.8, "u_d": 0.6}


# =========================
# 3. 模型参数
# =========================
params = {
    "beta0": 0.035,  # 基线传播率
    "a": 0.20,  # 季节振幅
    "phi": 15.0,  # 季节相位
    "sigma": 1 / 1.5,  # 潜伏期约1.5天 -> E到I速率
    "gamma": 1 / 4.0,  # 传染期约4天 -> I恢复速率
    "gamma_q": 1 / 3.0,  # 隔离后恢复速率
    "eps_v": 0.45,  # 疫苗保护效力(示意值，需查H3N2文献)
    "omega_r": 1 / 180.0,  # 康复免疫衰减
    "omega_v": 1 / 150.0,  # 疫苗保护衰减
    "eta_m": 0.30,  # 口罩措施效果
    "eta_w": 0.20,  # 通风效果
    "eta_s": 0.25,  # 限流效果
    "eta_d": 0.35,  # 线上授课/密度控制效果
}

# 各群体接种速率
nu = np.array([0.001, 0.0008, 0.0005], dtype=float)

# 各群体隔离发现速率
q = np.array([0.08, 0.10, 0.10], dtype=float)


# =========================
# 4. 季节传播率与干预乘子
# =========================
def beta_t(t, p):
    season = 1.0 + p["a"] * np.cos(2 * np.pi * (t - p["phi"]) / 365.0)
    u = control_level(t)
    measure = (
        (1 - p["eta_m"] * u["u_m"])
        * (1 - p["eta_w"] * u["u_w"])
        * (1 - p["eta_s"] * u["u_s"])
        * (1 - p["eta_d"] * u["u_d"])
    )
    return p["beta0"] * season * measure


# =========================
# 5. ODE
# 状态顺序:
# [S(3), V(3), E(3), I(3), Q(3), R(3)]
# =========================
def unpack(y):
    S = y[0:G]
    V = y[G : 2 * G]
    E = y[2 * G : 3 * G]
    I = y[3 * G : 4 * G]
    Qv = y[4 * G : 5 * G]
    R = y[5 * G : 6 * G]
    return S, V, E, I, Qv, R


def sveiqr_rhs(t, y, p):
    S, V, E, I, Qv, R = unpack(y)
    bt = beta_t(t, p)

    # 感染压力 lambda_g
    infectious_ratio = I / N
    lam = bt * (C @ infectious_ratio)

    dS = -lam * S - nu * S + p["omega_r"] * R + p["omega_v"] * V
    dV = nu * S - (1 - p["eps_v"]) * lam * V - p["omega_v"] * V
    dE = lam * S + (1 - p["eps_v"]) * lam * V - p["sigma"] * E
    dI = p["sigma"] * E - (p["gamma"] + q) * I
    dQ = q * I - p["gamma_q"] * Qv
    dR = p["gamma"] * I + p["gamma_q"] * Qv - p["omega_r"] * R

    return np.concatenate([dS, dV, dE, dI, dQ, dR])


# =========================
# 6. 初值
# =========================
I0 = np.array([5.0, 0.0, 0.0])
E0 = np.array([8.0, 0.0, 0.0])
Q0 = np.zeros(G)
R0 = np.zeros(G)
V0 = np.array([5000.0, 600.0, 300.0])
S0 = N - V0 - E0 - I0 - Q0 - R0

y0 = np.concatenate([S0, V0, E0, I0, Q0, R0])


# =========================
# 7. 仿真函数
# =========================
def simulate(t_span=(0, 120), t_eval=None, p=None):
    if p is None:
        p = params
    if t_eval is None:
        t_eval = np.arange(t_span[0], t_span[1] + 1)

    sol = solve_ivp(
        fun=lambda t, y: sveiqr_rhs(t, y, p),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        vectorized=False,
        dense_output=False,
    )
    return sol


# =========================
# 8. 输出指标
# =========================
def get_outputs(sol, p=None):
    if p is None:
        p = params
    Y = sol.y
    S, V, E, I, Qv, R = unpack(Y)

    total_I = I.sum(axis=0)
    total_Q = Qv.sum(axis=0)
    total_R = R.sum(axis=0)

    # 每日新增报告病例，可近似用 q*I 或 rho*sigma*E
    new_reported = (q[:, None] * I).sum(axis=0)

    peak_I = total_I.max()
    peak_day = sol.t[np.argmax(total_I)]
    final_attack_rate = (total_R[-1] + total_Q[-1] + total_I[-1]) / N.sum()

    return {
        "t": sol.t,
        "I_total": total_I,
        "Q_total": total_Q,
        "R_total": total_R,
        "new_reported": new_reported,
        "peak_I": peak_I,
        "peak_day": peak_day,
        "final_attack_rate": final_attack_rate,
    }


# =========================
# 9. 参数拟合骨架
# 这里用每日新增病例 obs_cases 做拟合
# 你们只需要替换 obs_cases 即可
# =========================
obs_days = np.arange(0, 30)
obs_cases = np.array(
    [
        1,
        2,
        3,
        5,
        8,
        12,
        16,
        20,
        18,
        15,
        13,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        4,
        3,
        3,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        0,
    ],
    dtype=float,
)


def fit_residual(theta):
    beta0, sigma, gamma = theta
    p = params.copy()
    p["beta0"] = beta0
    p["sigma"] = sigma
    p["gamma"] = gamma

    sol = simulate(t_span=(0, int(obs_days[-1])), t_eval=obs_days, p=p)
    pred = get_outputs(sol, p)["new_reported"]
    return pred - obs_cases


# 示例：拟合3个参数
theta0 = np.array([0.035, 1 / 1.5, 1 / 4.0])
lb = np.array([0.001, 1 / 5.0, 1 / 10.0])
ub = np.array([0.20, 1 / 0.5, 1 / 1.0])

# result = least_squares(fit_residual, theta0, bounds=(lb, ub))
# print(result.x)

# =========================
# 10. 运行示例
# =========================
if __name__ == "__main__":
    sol = simulate()
    out = get_outputs(sol)
    print("Peak infected:", out["peak_I"])
    print("Peak day:", out["peak_day"])
    print("Final attack rate:", out["final_attack_rate"])
