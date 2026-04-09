import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# =========================
# 1. 配置加载
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "parameters", "configs.json")

with open(config_path, "r", encoding="utf-8") as f:
    CFG = json.load(f)

GROUPS = CFG["groups"]
G = len(GROUPS)

N = np.array(CFG["population"]["N"], dtype=float)
C = np.array(CFG["population"]["contact_matrix"], dtype=float)

if len(N) != G:
    raise ValueError("configs.json: population.N length must match groups")
if C.shape != (G, G):
    raise ValueError("configs.json: population.contact_matrix must be GxG")

params = {k: float(v) for k, v in CFG["model_parameters"].items()}
SCENARIOS = CFG["control_scenarios"]
if not SCENARIOS:
    raise ValueError("configs.json: control_scenarios cannot be empty")

SIM_CFG = CFG["simulation"]
FIT_CFG = CFG["fitting"]
OUT_CFG = CFG["output"]


# =========================
# 2. 场景参数集
# =========================
def get_scenario_by_name(name):
    for scenario in SCENARIOS:
        if scenario["id"] == name:
            return scenario
    raise ValueError(f"configs.json: scenario id not found: {name}")


def validate_scenario(scenario):
    for key in ["id", "label", "u_m", "u_w", "u_s", "u_d", "nu", "q"]:
        if key not in scenario:
            raise ValueError(f"configs.json: missing key '{key}' in scenario")
    nu_vec = np.array(scenario["nu"], dtype=float)
    q_vec = np.array(scenario["q"], dtype=float)
    if len(nu_vec) != G or len(q_vec) != G:
        raise ValueError("configs.json: scenario nu/q length must match groups")


for _scenario in SCENARIOS:
    validate_scenario(_scenario)


def scenario_id(scenario):
    return scenario["id"]


def scenario_label(scenario):
    return scenario["label"]


def get_control_level(scenario):
    """
    场景固定防控强度
    u_m: 口罩
    u_w: 通风
    u_s: 社团限流
    u_d: 线上授课/教学密度控制
    """
    return {
        "u_m": float(scenario["u_m"]),
        "u_w": float(scenario["u_w"]),
        "u_s": float(scenario["u_s"]),
        "u_d": float(scenario["u_d"]),
    }


# =========================
# 3. 季节传播率与干预乘子
# =========================
def beta_t(t, p, scenario):
    season = 1.0 + p["a"] * np.cos(2 * np.pi * (t - p["phi"]) / 365.0)
    u = get_control_level(scenario)
    measure = (
        (1 - p["eta_m"] * u["u_m"])
        * (1 - p["eta_w"] * u["u_w"])
        * (1 - p["eta_s"] * u["u_s"])
        * (1 - p["eta_d"] * u["u_d"])
    )
    return p["beta0"] * season * measure


# =========================
# 4. ODE
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


def sveiqr_rhs(t, y, p, scenario):
    S, V, E, I, Qv, R = unpack(y)
    bt = beta_t(t, p, scenario)
    nu = np.array(scenario["nu"], dtype=float)
    q = np.array(scenario["q"], dtype=float)

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
# 5. 初值
# =========================
I0 = np.array(CFG["initial_conditions"]["I0"], dtype=float)
E0 = np.array(CFG["initial_conditions"]["E0"], dtype=float)
Q0 = np.array(CFG["initial_conditions"]["Q0"], dtype=float)
R0 = np.array(CFG["initial_conditions"]["R0"], dtype=float)
V0 = np.array(CFG["initial_conditions"]["V0"], dtype=float)

for name, arr in {
    "I0": I0,
    "E0": E0,
    "Q0": Q0,
    "R0": R0,
    "V0": V0,
}.items():
    if len(arr) != G:
        raise ValueError(
            f"configs.json: initial_conditions.{name} length must match groups"
        )

S0 = N - V0 - E0 - I0 - Q0 - R0
if np.any(S0 < 0):
    raise ValueError("configs.json: invalid initial conditions, found S0 < 0")

y0 = np.concatenate([S0, V0, E0, I0, Q0, R0])


# =========================
# 6. 仿真函数
# =========================
def simulate(t_span=None, t_eval=None, p=None, scenario_name=None):
    if p is None:
        p = params
    if scenario_name is None:
        scenario_name = scenario_id(SCENARIOS[0])
    scenario = get_scenario_by_name(scenario_name)
    if t_span is None:
        t_span = (float(SIM_CFG["t_start"]), float(SIM_CFG["t_end"]))
    if t_eval is None:
        t_eval = np.arange(
            t_span[0],
            t_span[1] + float(SIM_CFG["t_step"]),
            float(SIM_CFG["t_step"]),
        )

    sol = solve_ivp(
        fun=lambda t, y: sveiqr_rhs(t, y, p, scenario),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method=SIM_CFG["ode_method"],
        vectorized=False,
        dense_output=False,
    )
    return sol


# =========================
# 7. 输出指标
# =========================
def get_outputs(sol, scenario_name, p=None):
    if p is None:
        p = params
    scenario = get_scenario_by_name(scenario_name)
    q = np.array(scenario["q"], dtype=float)
    Y = sol.y
    S, V, E, I, Qv, R = unpack(Y)

    total_I = I.sum(axis=0)
    total_Q = Qv.sum(axis=0)
    total_R = R.sum(axis=0)

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


def plot_outputs(out, save_path, title_suffix=""):
    plt.figure(figsize=(10, 6))
    plt.plot(out["t"], out["I_total"], label="Infectious (I)", linewidth=2.0)
    plt.plot(out["t"], out["new_reported"], label="New reported", linewidth=2.0)
    plt.plot(out["t"], out["Q_total"], label="Isolated (Q)", linewidth=1.8)
    plt.plot(out["t"], out["R_total"], label="Recovered (R)", linewidth=1.8)
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.title(f"SEVIQR Seasonal Simulation Curves{title_suffix}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=int(OUT_CFG["dpi"]))
    plt.close()


# =========================
# 8. 参数拟合骨架
# =========================
obs_days = np.array(FIT_CFG["obs_days"], dtype=float)
obs_cases = np.array(FIT_CFG["obs_cases"], dtype=float)

if len(obs_days) != len(obs_cases):
    raise ValueError(
        "configs.json: fitting.obs_days and fitting.obs_cases length must match"
    )


def fit_residual(theta):
    beta0, sigma, gamma = theta
    p = params.copy()
    p["beta0"] = beta0
    p["sigma"] = sigma
    p["gamma"] = gamma

    fit_scenario = FIT_CFG["scenario_id"]
    sol = simulate(
        t_span=(0, int(obs_days[-1])),
        t_eval=obs_days,
        p=p,
        scenario_name=fit_scenario,
    )
    pred = get_outputs(sol, scenario_name=fit_scenario, p=p)["new_reported"]
    return pred - obs_cases


theta0 = np.array(FIT_CFG["theta0"], dtype=float)
lb = np.array(FIT_CFG["bounds"]["lower"], dtype=float)
ub = np.array(FIT_CFG["bounds"]["upper"], dtype=float)

if not (len(theta0) == len(lb) == len(ub)):
    raise ValueError("configs.json: fitting.theta0 and bounds size must match")

# result = least_squares(fit_residual, theta0, bounds=(lb, ub))
# print(result.x)


# =========================
# 9. 运行示例
# =========================
if __name__ == "__main__":
    summary_rows = []
    plt.figure(figsize=(10, 6))

    for scenario in SCENARIOS:
        sid = scenario_id(scenario)
        slabel = scenario_label(scenario)
        sol = simulate(scenario_name=sid)
        out = get_outputs(sol, scenario_name=sid)

        summary_rows.append(
            {
                "scenario": sid,
                "scenario_label": slabel,
                "peak_I": float(out["peak_I"]),
                "peak_day": float(out["peak_day"]),
                "final_attack_rate": float(out["final_attack_rate"]),
            }
        )

        plt.plot(out["t"], out["I_total"], linewidth=2.0, label=sid)

        scenario_fig_path = os.path.join(
            current_dir,
            f"{sid}_{OUT_CFG['curve_filename']}",
        )
        plot_outputs(out, scenario_fig_path, title_suffix=f" - {sid}")

    compare_fig_path = os.path.join(current_dir, OUT_CFG["comparison_curve_filename"])
    plt.xlabel("Day")
    plt.ylabel("Infectious population")
    plt.title("SEVIQR Scenario Comparison: Infectious (I)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(compare_fig_path, dpi=int(OUT_CFG["dpi"]))
    plt.close()

    print("Scenario comparison summary:")
    print(f"{'Scenario':<20}{'Peak I':>14}{'Peak Day':>14}{'Final Attack Rate':>20}")
    for row in summary_rows:
        print(
            f"{row['scenario_label']:<20}{row['peak_I']:>14.3f}{row['peak_day']:>14.1f}{row['final_attack_rate']:>20.6f}"
        )
    print("Comparison curve saved to:", compare_fig_path)
