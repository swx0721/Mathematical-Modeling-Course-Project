"""论文级统一绘图脚本（稳健版）。

完全基于已有 CSV/JSON 数据文件生成所有论文图表，不依赖动态导入模型模块。
涵盖：
1. 分季节传播曲线对比（冬春高峰 vs 非高峰）
2. 分场景传播动态对比（常态/散发/聚集）
3. 场所感染贡献比例图
4. 防控策略优化前后对比图
5. 不同干预强度效果对比图
6. 模拟与真实数据对比验证图（RMSE/MAE）
7. 再生数 R_eff 曲线图
8. 优化收敛历史 + 目标函数分解
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# ── 全局绘图设置 ──────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["SimSun", "Times New Roman"],
    "font.size": 11,
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.8",
})

# 配色方案
COLORS = {
    "normal":    "#2196F3",   # 蓝色 - 常态
    "sporadic":  "#FF9800",   # 橙色 - 散发
    "cluster":   "#F44336",   # 红色 - 聚集
    "winter":    "#1565C0",   # 深蓝 - 冬春高峰
    "off_season":"#81C784",   # 浅绿 - 非高峰
    "optimized": "#4CAF50",   # 绿色 - 优化后
    "baseline":  "#9E9E9E",   # 灰色 - 基线
    "dorm":      "#5C6BC0",   # 靛蓝 - 宿舍
    "class":     "#EF5350",   # 红色 - 教学区
    "canteen":   "#FFA726",   # 橙色 - 食堂
    "club":      "#66BB6A",   # 绿色 - 社团
}

SCENARIO_CN = {"normal": "常态", "sporadic": "散发", "cluster": "聚集"}
PLACE_CN = {"dorm": "宿舍", "class": "教学区", "canteen": "食堂", "club": "社团"}

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

CN_DIR = {"normal": "常态", "sporadic": "散发", "cluster": "聚集"}


# ═══════════════════════════════════════════════════════════
# 数据加载工具
# ═══════════════════════════════════════════════════════════
def _load_timeseries(scenario: str) -> pd.DataFrame:
    path = ROOT / CN_DIR[scenario] / f"timeseries_{scenario}.csv"
    return pd.read_csv(path, index_col=0)


def _load_place_contribution(scenario: str) -> pd.DataFrame:
    path = ROOT / CN_DIR[scenario] / f"place_contribution_{scenario}.csv"
    return pd.read_csv(path, index_col=0)


def _load_reff(scenario: str) -> pd.DataFrame:
    path = ROOT / CN_DIR[scenario] / f"reff_{scenario}.csv"
    return pd.read_csv(path)


def _load_opt_result(scenario: str) -> dict:
    path = ROOT / CN_DIR[scenario] / f"opt_result_{scenario}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _load_opt_history(scenario: str) -> pd.DataFrame:
    path = ROOT / CN_DIR[scenario] / f"opt_history_{scenario}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_config(scenario: str) -> dict:
    path = ROOT / CN_DIR[scenario] / f"config_{scenario}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _load_summary(scenario: str) -> pd.DataFrame:
    path = ROOT / CN_DIR[scenario] / f"summary_{scenario}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ═══════════════════════════════════════════════════════════
# 季节性 β(t) 计算函数
# ═══════════════════════════════════════════════════════════
def seasonal_beta(t: float, beta0: float = 0.256,
                  season_amp: float = 0.225,
                  season_start_day: float = 0.0,
                  season_phase: float = 30.0) -> float:
    return beta0 * (1.0 + season_amp * np.cos(
        2.0 * np.pi * (t + season_start_day - season_phase) / 365.0
    ))


# ═══════════════════════════════════════════════════════════
# 图1: 分季节传播曲线对比
# ═══════════════════════════════════════════════════════════
def plot_seasonal_comparison():
    """绘制冬春高峰季 vs 非高峰季的传播曲线对比图。

    通过修改 season_start_day 参数模拟不同季节起始点：
    - winter_spring_peak: season_start_day=350 (冬春高峰)
    - off_season: season_start_day=200 (非高峰期)
    """
    print("  [1/8] 分季节传播曲线对比...")

    # 从已有 timeseries 提取数据，结合季节性 β(t) 曲线
    ts = _load_timeseries("normal")
    config = _load_config("normal")
    reff = _load_reff("normal")

    beta0 = config.get("parameters", {}).get("beta0", 0.256)
    season_amp = config.get("parameters", {}).get("season_amp", 0.225)
    season_phase = config.get("parameters", {}).get("season_phase", 30.0)

    days = np.arange(0, 366)
    # 冬春高峰季：season_start_day=350
    beta_winter = np.array([seasonal_beta(t, beta0, season_amp, 350.0, season_phase) for t in days])
    # 非高峰期：season_start_day=200
    beta_off = np.array([seasonal_beta(t, beta0, season_amp, 200.0, season_phase) for t in days])
    # 基线：当前模型参数
    beta_base = np.array([seasonal_beta(t, beta0, season_amp, 0.0, season_phase) for t in days])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel (a): β(t) 季节变化
    axes[0].plot(days, beta_winter, linewidth=2.2, label="冬春高发期", color=COLORS["winter"])
    axes[0].plot(days, beta_off, linewidth=2.2, label="非高发期", color=COLORS["off_season"])
    axes[0].plot(days, beta_base, linewidth=1.5, label="基线", color=COLORS["baseline"], linestyle="--")
    axes[0].set_title(r"(a) 传播率 $\beta(t)$ 季节变化", fontsize=13)
    axes[0].set_xlabel("天数")
    axes[0].set_ylabel(r"$\beta(t)$")
    axes[0].legend(fontsize=11)

    # Panel (b): 常态场景 I(t) 标注季节影响区域
    ts_days = ts.index.to_numpy()
    axes[1].plot(ts_days, ts["I"], linewidth=2.2, label="感染人数 $I(t)$", color=COLORS["normal"])
    # 标注季节影响区间（假设前30天为冬春高峰期）
    peak_beta_days = np.where(beta_base > beta0)[0]
    if len(peak_beta_days) > 0:
        axes[1].axvspan(peak_beta_days[0], peak_beta_days[-1], alpha=0.1,
                        color=COLORS["winter"], label=r"$\beta > \beta_0$ 高发区间")
    axes[1].set_title("(b) 常态场景感染曲线与季节区间", fontsize=13)
    axes[1].set_xlabel("天数")
    axes[1].set_ylabel("感染人数")
    axes[1].legend(fontsize=10)

    # Panel (c): R_eff 对比 — 冬春高峰 vs 非高峰的 R0 估算
    r0_winter = beta_winter.max() / 0.2  # beta/gamma 近似
    r0_off = beta_off.max() / 0.2
    r0_base = beta_base.max() / 0.2

    seasons = ["冬春高发期", "基线", "非高发期"]
    r0_vals = [r0_winter, r0_base, r0_off]
    bar_colors = [COLORS["winter"], COLORS["baseline"], COLORS["off_season"]]

    bars = axes[2].bar(seasons, r0_vals, color=bar_colors, alpha=0.85, width=0.6, edgecolor="white")
    for bar, val in zip(bars, r0_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[2].axhline(y=1.0, color="black", linestyle=":", linewidth=1.5)
    axes[2].set_title(r"(c) 不同季节 $R_0$ 对比", fontsize=13)
    axes[2].set_ylabel(r"$R_0 \approx \beta_{max} / \gamma$")
    axes[2].set_ylim(0, max(r0_vals) * 1.15)

    fig.suptitle("图1  分季节疫情传播特征对比（常态场景）", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_seasonal_comparison.png", dpi=300)
    plt.close(fig)
    print("    -> fig1_seasonal_comparison.png 已保存")


# ═══════════════════════════════════════════════════════════
# 图2: 分场景传播动态对比（含无干预基线）
# ═══════════════════════════════════════════════════════════
def _run_baseline_timeseries(scenario: str, n_days: int = 220) -> pd.DataFrame:
    """用场景配置参数运行无干预基线模型，返回 timeseries。

    直接用 scipy 积分 SVEIQR ODE 系统，不依赖外部模块。
    """
    from scipy.integrate import solve_ivp

    config = _load_config(scenario)
    p = config.get("parameters", {})
    beta0 = p.get("beta0", 0.256)
    sigma_val = p.get("sigma", 0.53)
    gamma_val = p.get("gamma", 0.2)
    q_rate = p.get("q_rate", 0.2)
    seed_s = p.get("seed_s", 2.0)
    season_amp = p.get("season_amp", 0.225)
    season_phase = p.get("season_phase", 30.0)
    q_release = 0.15

    # 疫苗参数
    ve_s, ve_t, ve_l = 0.38, 0.305, 0.375
    vc_s = config.get("vaccine_coverage", {}).get("students", 0.2)
    vc_t = config.get("vaccine_coverage", {}).get("teachers", 0.12)
    vc_l = config.get("vaccine_coverage", {}).get("staff", 0.12)

    # 人群
    N_s, N_t, N_l = 26000.0, 1700.0, 800.0

    # 接触矩阵 (默认)
    C_dorm = np.array([[8.5,0.3,0.2],[0.2,0.1,0.05],[0.3,0.05,0.4]])
    C_class = np.array([[12.0,2.5,0.3],[3.0,1.2,0.2],[0.4,0.2,0.3]])
    C_canteen = np.array([[4.5,0.8,0.5],[0.7,0.6,0.3],[0.6,0.3,0.8]])
    C_club = np.array([[6.0,0.5,0.2],[0.4,0.3,0.1],[0.2,0.1,0.2]])

    pw = config.get("place_weights", {"dorm":0.35,"class":0.40,"canteen":0.15,"club":0.10})
    total_w = sum(pw.values())
    w_dorm = pw.get("dorm",0.35)/total_w
    w_class = pw.get("class",0.40)/total_w
    w_canteen = pw.get("canteen",0.15)/total_w
    w_club = pw.get("club",0.10)/total_w

    N_arr = np.array([N_s, N_t, N_l])
    ve_arr = np.array([ve_s, ve_t, ve_l])

    def beta_seasonal(t):
        return beta0 * (1.0 + season_amp * np.cos(2*np.pi*(t - season_phase)/365.0))

    def ode(t, y):
        # y: [S_s,V_s,E_s,I_s,Q_s,R_s, S_t,..., S_l,...]
        dydt = np.zeros(18)
        I_arr = np.array([y[3], y[9], y[15]])  # I_s, I_t, I_l
        beta = beta_seasonal(t)

        # 场所力 of infection (无干预：所有 u=0)
        lam = np.zeros(3)
        for C_k, w_k in [(C_dorm, w_dorm), (C_class, w_class),
                         (C_canteen, w_canteen), (C_club, w_club)]:
            for g in range(3):
                exposure = sum(C_k[g,h] * I_arr[h] / max(N_arr[h],1e-12) for h in range(3))
                lam[g] += w_k * beta * exposure

        for g, (ve, idx) in enumerate(zip(ve_arr, [0,6,12])):
            s = y[idx]; v = y[idx+1]; e = y[idx+2]; i = y[idx+3]; q = y[idx+4]
            loss_s = lam[g] * s
            loss_v = (1-ve) * lam[g] * v
            dydt[idx]   = -loss_s
            dydt[idx+1] = -loss_v
            dydt[idx+2] = loss_s + loss_v - sigma_val * e
            dydt[idx+3] = sigma_val * e - gamma_val * i - q_rate * i
            dydt[idx+4] = q_rate * i - q_release * q
            dydt[idx+5] = gamma_val * i + q_release * q

        return dydt

    # 初始条件
    y0 = np.zeros(18)
    for g, (n_g, vc_g, ve_g, seed_g) in enumerate(zip(
        [N_s, N_t, N_l], [vc_s, vc_t, vc_l], [ve_s, ve_t, ve_l],
        [seed_s, 0.0, 0.0]
    )):
        idx = g * 6
        v0 = n_g * vc_g
        i0 = seed_g
        s0 = n_g - v0 - i0
        y0[idx] = s0; y0[idx+1] = v0; y0[idx+3] = i0

    t_eval = np.linspace(0, n_days, n_days + 1)
    sol = solve_ivp(ode, (0, n_days), y0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)

    # 组装 DataFrame
    comp_names = ["S","V","E","I","Q","R"]
    group_names = ["s","t","l"]
    col_names = [f"{c}_{g}" for g in group_names for c in comp_names]
    df = pd.DataFrame(sol.y.T, columns=col_names)
    df.index = pd.Index(sol.t, name="day")

    for g in group_names:
        cols_g = [f"{c}_{g}" for c in comp_names]
        df[f"N_{g}"] = df[cols_g].sum(axis=1)

    for c in comp_names:
        df[c] = sum(df[f"{c}_{g}"] for g in group_names)
    df["N"] = sum(df[f"N_{g}"] for g in group_names)

    return df


def plot_scenario_comparison():
    """绘制常态/散发/聚集三场景传播动态对比图（含无干预基线）。"""
    print("  [2/8] 分场景传播动态对比（含无干预基线）...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 预计算所有基线
    baselines = {}
    for scenario in ["normal", "sporadic", "cluster"]:
        try:
            baselines[scenario] = _run_baseline_timeseries(scenario)
            print(f"    {SCENARIO_CN[scenario]}基线计算完成")
        except Exception as e:
            print(f"    {SCENARIO_CN[scenario]}基线计算失败: {e}")
            baselines[scenario] = None

    summaries = {}
    for scenario in ["normal", "sporadic", "cluster"]:
        ts = _load_timeseries(scenario)  # 优化后
        color = COLORS[scenario]
        label = SCENARIO_CN[scenario] + "（优化后）"
        days = ts.index.to_numpy()

        # 无干预基线（虚线）
        bl = baselines.get(scenario)
        if bl is not None:
            bl_label = SCENARIO_CN[scenario] + "（无干预）"
            bl_days = bl.index.to_numpy()
            axes[0, 0].plot(bl_days, bl["I"], linewidth=1.8, linestyle="--",
                            label=bl_label, color=color, alpha=0.5)
            new_exp_bl = -(bl["S"].diff().fillna(0) + bl["V"].diff().fillna(0)).clip(lower=0)
            axes[0, 1].plot(bl_days, new_exp_bl, linewidth=1.8, linestyle="--",
                            label=bl_label, color=color, alpha=0.5)
            axes[1, 0].plot(bl_days, bl["R"], linewidth=1.8, linestyle="--",
                            label=bl_label, color=color, alpha=0.5)
            axes[1, 1].plot(bl_days, bl["E"], linewidth=1.8, linestyle="--",
                            label=bl_label, color=color, alpha=0.5)

        # 优化后（实线）
        axes[0, 0].plot(days, ts["I"], linewidth=2.2, label=label, color=color)
        new_exp = -(ts["S"].diff().fillna(0) + ts["V"].diff().fillna(0)).clip(lower=0)
        axes[0, 1].plot(days, new_exp, linewidth=2.2, label=label, color=color)
        axes[1, 0].plot(days, ts["R"], linewidth=2.2, label=label, color=color)
        axes[1, 1].plot(days, ts["E"], linewidth=2.2, label=label, color=color)

        summaries[scenario] = {
            "峰值感染(优化后)": f"{ts['I'].max():.0f}",
            "峰值日": f"{ts['I'].idxmax():.0f}",
            "攻击率(优化后)": f"{ts['R'].iloc[-1] / ts['N'].iloc[0]:.2%}",
        }
        if bl is not None:
            summaries[scenario]["峰值感染(无干预)"] = f"{bl['I'].max():.0f}"
            summaries[scenario]["攻击率(无干预)"] = f"{bl['R'].iloc[-1] / bl['N'].iloc[0]:.2%}"

    axes[0, 0].set_title("(a) 感染人数 $I(t)$（实线=优化后，虚线=无干预）", fontsize=12)
    axes[0, 0].set_ylabel("感染人数")
    axes[0, 0].legend(fontsize=8, ncol=2)

    axes[0, 1].set_title("(b) 日新增暴露人数", fontsize=12)
    axes[0, 1].set_ylabel("人数/天")
    axes[0, 1].legend(fontsize=8, ncol=2)

    axes[1, 0].set_title("(c) 累计恢复人数 $R(t)$", fontsize=12)
    axes[1, 0].set_xlabel("天数")
    axes[1, 0].set_ylabel("人数")
    axes[1, 0].legend(fontsize=8, ncol=2)

    axes[1, 1].set_title("(d) 潜伏人数 $E(t)$", fontsize=12)
    axes[1, 1].set_xlabel("天数")
    axes[1, 1].set_ylabel("人数")
    axes[1, 1].legend(fontsize=8, ncol=2)

    fig.suptitle("图2  不同场景下疫情传播动态对比（实线=优化后，虚线=无干预基线）", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_scenario_comparison.png", dpi=300)
    plt.close(fig)

    # 打印汇总表
    print("    场景汇总:")
    for s, d in summaries.items():
        print(f"      {SCENARIO_CN[s]}: {d}")
    print("    -> fig2_scenario_comparison.png 已保存")


# ═══════════════════════════════════════════════════════════
# 图3: 场所感染贡献比例图
# ═══════════════════════════════════════════════════════════
def plot_place_contribution():
    """绘制各场景下不同场所的感染贡献比例图。"""
    print("  [3/8] 场所感染贡献比例图...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    places = ["dorm", "class", "canteen", "club"]
    place_colors = [COLORS[p] for p in places]
    place_labels = [PLACE_CN[p] for p in places]

    for idx, scenario in enumerate(["normal", "sporadic", "cluster"]):
        pc = _load_place_contribution(scenario)
        days = pc.index.to_numpy()

        # 各场所的绝对新增感染贡献
        abs_data = np.vstack([pc[f"new_{p}"].to_numpy() for p in places])
        total = pc["new_total"].to_numpy()
        # 归一化
        den = np.maximum(total, 1e-12)
        share_data = abs_data / den

        ax = axes[idx]
        ax.stackplot(days, share_data, labels=place_labels, colors=place_colors, alpha=0.85)
        ax.set_title(f"({chr(97+idx)}) {SCENARIO_CN[scenario]}场景", fontsize=13)
        ax.set_xlabel("天数")
        ax.set_ylabel("感染贡献占比")
        ax.set_ylim(0, 1)
        if idx == 2:
            ax.legend(loc="upper right", fontsize=10)

    fig.suptitle("图3  不同场所感染贡献占比（分场景）", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_place_contribution.png", dpi=300)
    plt.close(fig)
    print("    -> fig3_place_contribution.png 已保存")


# ═══════════════════════════════════════════════════════════
# 图4: 防控策略优化前后对比图
# ═══════════════════════════════════════════════════════════
def plot_optimization_comparison():
    """绘制优化前后疫情曲线对比及最优防控强度。"""
    print("  [4/8] 防控策略优化前后对比图...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, scenario in enumerate(["normal", "sporadic", "cluster"]):
        ax = axes[idx // 2, idx % 2]
        ts = _load_timeseries(scenario)
        opt = _load_opt_result(scenario)

        days = ts.index.to_numpy()

        # 优化后感染曲线（当前 timeseries 是优化后结果）
        ax.plot(days, ts["I"], linewidth=2.2, label="优化后感染曲线",
                color=COLORS["optimized"])
        ax.fill_between(days, 0, ts["I"], alpha=0.12, color=COLORS["optimized"])

        # 标注关键指标
        peak_i = ts["I"].max()
        peak_day = ts["I"].idxmax()
        ax.annotate(f"峰值: {peak_i:.0f}", xy=(peak_day, peak_i),
                     fontsize=10, ha="center", va="bottom",
                     arrowprops=dict(arrowstyle="->", color="black"),
                     xytext=(peak_day + 15, peak_i * 0.9))

        # 添加优化参数标注
        if opt and "controls" in opt:
            ctrl = opt["controls"]
            info_lines = []
            ctrl_map = {
                "mask_u": "口罩", "vent_u": "通风", "online_u": "线上",
                "club_u": "社团限流", "disinfect_u": "消毒",
                "vax_cov_scale": "疫苗加种", "q_scale": "隔离强化"
            }
            for k, v in ctrl.items():
                if k in ctrl_map and v > 0.01:
                    info_lines.append(f"{ctrl_map[k]}: {v:.2f}")
            if info_lines:
                ax.text(0.98, 0.98, "\n".join(info_lines),
                        transform=ax.transAxes, fontsize=8,
                        verticalalignment="top", horizontalalignment="right",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

        ax.set_title(f"({chr(97+idx)}) {SCENARIO_CN[scenario]}场景 — 优化后", fontsize=13)
        ax.set_xlabel("天数")
        ax.set_ylabel("感染人数")
        ax.legend(fontsize=10, loc="upper right")

    # Panel (d): 最优防控强度对比条形图
    ax = axes[1, 1]
    controls = ["mask_u", "vent_u", "online_u", "club_u", "disinfect_u", "vax_cov_scale", "q_scale"]
    control_cn = ["口罩佩戴", "通风强度", "线上授课", "社团限流", "消毒措施", "疫苗加种", "隔离强化"]
    x = np.arange(len(controls))
    width = 0.25

    for i, scenario in enumerate(["normal", "sporadic", "cluster"]):
        opt = _load_opt_result(scenario)
        if not opt or "controls" not in opt:
            continue
        vals = [opt["controls"].get(c, 0) for c in controls]
        # 归一化 vax_cov_scale 到 0-1 (原始范围 0-5)
        vals[5] = min(vals[5] / 5.0, 1.0)
        # 归一化 q_scale 到 0-1 (原始范围 0-10)
        vals[6] = min(vals[6] / 10.0, 1.0)
        ax.bar(x + i * width, vals, width, label=SCENARIO_CN[scenario],
               color=COLORS[scenario], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(control_cn, fontsize=9, rotation=20)
    ax.set_ylabel("归一化强度")
    ax.set_title("(d) 最优防控强度对比", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)

    fig.suptitle("图4  防控策略优化前后对比及最优防控强度", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_optimization_comparison.png", dpi=300)
    plt.close(fig)
    print("    -> fig4_optimization_comparison.png 已保存")


# ═══════════════════════════════════════════════════════════
# 图5: 不同干预强度效果对比图（从 compare CSV 提取）
# ═══════════════════════════════════════════════════════════
def plot_intervention_sensitivity():
    """从 compare CSV 中提取不同干预强度对疫情的影响。

    使用采样策略避免大文件读取超时：仅读取需要的列，并采样。
    """
    print("  [5/8] 不同干预强度效果对比图...")

    ctrl_cols = ["mask_u", "vent_u", "online_u", "club_u", "disinfect_u"]
    ctrl_cn = ["口罩佩戴强度", "通风强度", "线上授课比例", "社团限流强度", "消毒强度"]
    ctrl_colors = [COLORS["normal"], COLORS["sporadic"], COLORS["cluster"],
                   COLORS["optimized"], "#9C27B0"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    # 从常态场景的 compare CSV 提取数据（仅读取需要的列，并采样）
    compare_path = ROOT / "常态" / "compare_normal.csv"
    if not compare_path.exists():
        print("    [WARN] compare_normal.csv 不存在，跳过干预敏感性图")
        # 绘制空白占位图
        for i in range(6):
            axes_flat[i].text(0.5, 0.5, "数据不可用", ha="center", va="center",
                              transform=axes_flat[i].transAxes, fontsize=14)
        fig.suptitle("图5  不同干预强度下的疫情关键指标变化", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(FIG_DIR / "fig5_intervention_sensitivity.png", dpi=300)
        plt.close(fig)
        return

    usecols = ["attack_rate", "peak_I"] + ctrl_cols
    df = pd.read_csv(compare_path, usecols=usecols)
    # 采样以加速分箱统计
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)

    for i, (ctrl_name, ctrl_label, color) in enumerate(zip(ctrl_cols, ctrl_cn, ctrl_colors)):
        ax = axes_flat[i]

        if ctrl_name not in df.columns:
            ax.text(0.5, 0.5, f"{ctrl_name} 数据不可用",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(f"({chr(97+i)}) {ctrl_label}", fontsize=12)
            continue

        # 按 ctrl_name 分箱，取各箱的平均攻击率和峰值
        bins = pd.cut(df[ctrl_name], bins=10)
        grouped = df.groupby(bins, observed=True).agg(
            attack_rate_mean=("attack_rate", "mean"),
            attack_rate_std=("attack_rate", "std"),
            peak_I_mean=("peak_I", "mean"),
            peak_I_std=("peak_I", "std"),
        ).dropna()

        bin_centers = [interval.mid for interval in grouped.index]

        # 攻击率
        l1, = ax.plot(bin_centers, grouped["attack_rate_mean"] * 100, "o-",
                       linewidth=2, markersize=5, color=color, label="攻击率(%)")
        ax.fill_between(bin_centers,
                         (grouped["attack_rate_mean"] - grouped["attack_rate_std"].fillna(0)) * 100,
                         (grouped["attack_rate_mean"] + grouped["attack_rate_std"].fillna(0)) * 100,
                         alpha=0.15, color=color)
        ax.set_xlabel(ctrl_label)
        ax.set_ylabel("攻击率 (%)", color=color)
        ax.tick_params(axis="y", labelcolor=color)

        # 峰值感染（第二 y 轴）
        ax2 = ax.twinx()
        l2, = ax2.plot(bin_centers, grouped["peak_I_mean"], "s--",
                        linewidth=2, markersize=5, color=color, alpha=0.6,
                        label="峰值感染人数")
        ax2.set_ylabel("峰值感染人数", color=color, alpha=0.6)

        ax.set_title(f"({chr(97+i)}) {ctrl_label}对疫情的影响", fontsize=12)
        ax.legend([l1, l2], ["攻击率(%)", "峰值感染人数"], fontsize=9, loc="upper right")

    # Panel (f): 三场景最优攻击率对比
    ax = axes_flat[5]
    scenarios_list = ["normal", "sporadic", "cluster"]
    ar_vals = []
    peak_vals = []
    for s in scenarios_list:
        opt = _load_opt_result(s)
        ar_vals.append(opt.get("attack_rate", 0) * 100)
        peak_vals.append(opt.get("peak_I", 0))

    x_pos = np.arange(len(scenarios_list))
    width_bar = 0.35
    ax.bar(x_pos - width_bar/2, ar_vals, width_bar, label="攻击率(%)",
            color=[COLORS[s] for s in scenarios_list], alpha=0.85, edgecolor="white")
    ax2 = ax.twinx()
    ax2.bar(x_pos + width_bar/2, peak_vals, width_bar, label="峰值感染",
             color=[COLORS[s] for s in scenarios_list], alpha=0.5, edgecolor="white",
             hatch="//")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SCENARIO_CN[s] for s in scenarios_list])
    ax.set_ylabel("攻击率 (%)")
    ax2.set_ylabel("峰值感染人数")
    ax.set_title("(f) 各场景优化后关键指标", fontsize=12)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="gray", alpha=0.85, label="攻击率(%)"),
                       Patch(facecolor="gray", alpha=0.5, hatch="//", label="峰值感染")]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right")

    fig.suptitle("图5  不同干预强度下的疫情关键指标变化（常态场景）", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_intervention_sensitivity.png", dpi=300)
    plt.close(fig)
    print("    -> fig5_intervention_sensitivity.png 已保存")


# ═══════════════════════════════════════════════════════════
# 图6: 模拟与真实数据对比验证图
# ═══════════════════════════════════════════════════════════
def plot_model_validation():
    """绘制模拟结果与真实校园疫情数据的对比验证图。"""
    print("  [6/8] 模拟与真实数据对比验证图...")

    real_csv = ROOT.parent / "真实校园" / "michigan_case001_compare.csv"
    if not real_csv.exists():
        print("    [WARN] 真实数据对比CSV不存在，跳过模型验证图")
        return

    merged = pd.read_csv(real_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 累计病例对比
    axes[0].plot(merged["day"], merged["cum_cases"], linewidth=2.5,
                  label="观测累计病例", color="#1565C0", marker="o", markersize=3)
    if "sim_cum_cases_scaled" in merged.columns:
        axes[0].plot(merged["day"], merged["sim_cum_cases_scaled"], linewidth=2.5,
                      label="模拟累计病例（缩放后）", color="#F44336", linestyle="--")
    if "sim_cum_cases" in merged.columns:
        axes[0].plot(merged["day"], merged["sim_cum_cases"], linewidth=2.0,
                      label="模拟累计病例（原始）", color="#FF9800", linestyle=":")

    axes[0].set_title("(a) 累计病例对比", fontsize=13)
    axes[0].set_xlabel("天数")
    axes[0].set_ylabel("累计病例数")
    axes[0].legend(fontsize=10)

    # 日新增对比
    axes[1].plot(merged["day"], merged["new_cases"], linewidth=2.5,
                  label="观测日新增", color="#1565C0", marker="o", markersize=3)
    if "sim_new_cases_scaled" in merged.columns:
        axes[1].plot(merged["day"], merged["sim_new_cases_scaled"], linewidth=2.5,
                      label="模拟日新增（缩放后）", color="#F44336", linestyle="--")
    if "sim_new_cases" in merged.columns:
        axes[1].plot(merged["day"], merged["sim_new_cases"], linewidth=2.0,
                      label="模拟日新增（原始）", color="#FF9800", linestyle=":")

    axes[1].set_title("(b) 日新增病例对比", fontsize=13)
    axes[1].set_xlabel("天数")
    axes[1].set_ylabel("日新增病例数")
    axes[1].legend(fontsize=10)

    # 计算误差指标
    metrics_text = ""
    obs_cum = merged["cum_cases"].dropna()
    obs_new = merged["new_cases"].dropna()

    if "sim_cum_cases_scaled" in merged.columns:
        sim_cum = merged["sim_cum_cases_scaled"].dropna()
        n = min(len(obs_cum), len(sim_cum))
        if n > 0:
            err_cum = sim_cum.values[:n] - obs_cum.values[:n]
            rmse_cum = np.sqrt(np.mean(err_cum**2))
            mae_cum = np.mean(np.abs(err_cum))
            metrics_text += f"累计 RMSE={rmse_cum:.2f}  MAE={mae_cum:.2f}\n"

    if "sim_new_cases_scaled" in merged.columns:
        sim_new = merged["sim_new_cases_scaled"].dropna()
        n = min(len(obs_new), len(sim_new))
        if n > 0:
            err_new = sim_new.values[:n] - obs_new.values[:n]
            rmse_new = np.sqrt(np.mean(err_new**2))
            mae_new = np.mean(np.abs(err_new))
            metrics_text += f"日新增 RMSE={rmse_new:.2f}  MAE={mae_new:.2f}"

    if metrics_text:
        axes[1].text(0.02, 0.98, metrics_text, transform=axes[1].transAxes,
                      fontsize=10, verticalalignment="top",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("图6  模型模拟与真实校园疫情数据对比（密歇根大学 H3N2）", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_model_validation.png", dpi=300)
    plt.close(fig)
    print("    -> fig6_model_validation.png 已保存")


# ═══════════════════════════════════════════════════════════
# 图7: 再生数 R_eff 曲线图
# ═══════════════════════════════════════════════════════════
def plot_reff_curves():
    """绘制各场景的有效再生数R_eff随时间变化曲线。"""
    print("  [7/8] 再生数 R_eff 曲线图...")

    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario in ["normal", "sporadic", "cluster"]:
        reff = _load_reff(scenario)
        color = COLORS[scenario]
        label = SCENARIO_CN[scenario]
        ax.plot(reff["day"], reff["R_eff"], linewidth=2.2, label=label, color=color)

    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1.5, label=r"$R_{eff}=1$")
    ax.set_title(r"图7  不同场景下有效再生数 $R_{eff}(t)$ 变化", fontsize=14)
    ax.set_xlabel("天数", fontsize=12)
    ax.set_ylabel(r"$R_{eff}$", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_reff_curves.png", dpi=300)
    plt.close(fig)
    print("    -> fig7_reff_curves.png 已保存")


# ═══════════════════════════════════════════════════════════
# 图8: 优化收敛历史 + 目标函数分解
# ═══════════════════════════════════════════════════════════
def plot_optimization_convergence():
    """绘制优化收敛历史曲线与目标函数分解。"""
    print("  [8/8] 优化收敛历史与目标函数分解...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): J收敛曲线
    ax = axes[0]
    for scenario in ["normal", "sporadic", "cluster"]:
        hist = _load_opt_history(scenario)
        if hist.empty or "J" not in hist.columns:
            continue
        # 采样加速（收敛曲线用 expanding min 已经很平滑，采样 2000 点足够）
        if len(hist) > 2000:
            hist = hist.iloc[::max(1, len(hist) // 2000)]
        j_min = hist["J"].expanding().min()
        ax.plot(range(len(j_min)), j_min.values, linewidth=2.2,
                label=SCENARIO_CN[scenario], color=COLORS[scenario])

    ax.set_title("(a) 目标函数 $J$ 收敛曲线", fontsize=13)
    ax.set_xlabel("评估次数")
    ax.set_ylabel("目标函数 $J$")
    ax.set_yscale("log")
    ax.legend(fontsize=11)

    # Panel (b): 目标函数A/C/D分解
    ax = axes[1]
    components = ["epidemic_loss_A", "control_cost_C", "disruption_D"]
    comp_cn = ["疫情损失 $A$", "防控成本 $C$", "扰动损失 $D$"]
    x = np.arange(len(components))
    width = 0.25

    for i, scenario in enumerate(["normal", "sporadic", "cluster"]):
        opt = _load_opt_result(scenario)
        if not opt:
            continue
        vals = [opt.get(c, 0) for c in components]
        ax.bar(x + i * width, vals, width, label=SCENARIO_CN[scenario],
               color=COLORS[scenario], alpha=0.85, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels(comp_cn, fontsize=11)
    ax.set_ylabel("目标函数分量值")
    ax.set_title("(b) 目标函数分量分解", fontsize=13)
    ax.legend(fontsize=11)

    fig.suptitle("图8  优化收敛过程与目标函数分解", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_optimization_convergence.png", dpi=300)
    plt.close(fig)
    print("    -> fig8_optimization_convergence.png 已保存")


# ═══════════════════════════════════════════════════════════
# 图9: 2季节 x 3场景 优化前后传播曲线对比图
# ═══════════════════════════════════════════════════════════
def _solve_layered_ode(
    scenario: str,
    season_phase: float,
    apply_controls: bool = False,
    n_days: int = 220,
) -> pd.DataFrame:
    """用内联ODE求解分层SVEIQR模型。

    Args:
        scenario: 场景名（normal/sporadic/cluster）
        season_phase: 季节相位（30=冬春高峰, 210=非高峰）
        apply_controls: 是否应用优化后的防控措施
        n_days: 模拟天数
    """
    from scipy.integrate import solve_ivp

    config = _load_config(scenario)
    p = config.get("parameters", {})
    beta0 = p.get("beta0", 0.256)
    sigma_val = p.get("sigma", 0.53)
    gamma_val = p.get("gamma", 0.2)
    q_rate = p.get("q_rate", 0.2)
    seed_s = p.get("seed_s", 2.0)
    season_amp = p.get("season_amp", 0.225)
    q_release = 0.15

    # 疫苗参数
    ve_s, ve_t, ve_l = 0.38, 0.305, 0.375
    vc_s = config.get("vaccine_coverage", {}).get("students", 0.2)
    vc_t = config.get("vaccine_coverage", {}).get("teachers", 0.12)
    vc_l = config.get("vaccine_coverage", {}).get("staff", 0.12)

    # 防控措施效果参数
    mask_eff = 0.18
    vent_eff = 0.3
    online_eff = 0.715
    club_limit_eff = 0.4
    disinfect_eff = 0.175

    # 防控强度
    if apply_controls:
        opt = _load_opt_result(scenario)
        ctrl = opt.get("controls", {})
        mask_u = ctrl.get("mask_u", 0.0)
        vent_u = ctrl.get("vent_u", 0.0)
        online_u = ctrl.get("online_u", 0.0)
        club_u = ctrl.get("club_u", 0.0)
        disinfect_u = ctrl.get("disinfect_u", 0.0)
        # q_rate 按比例放大
        q_scale = ctrl.get("q_scale", 0.0)
        q_rate_adj = q_rate * (1.0 + q_scale)
        # 疫苗覆盖率按比例放大
        vax_cov_scale = ctrl.get("vax_cov_scale", 0.0)
        vc_s_adj = min(vc_s * (1.0 + vax_cov_scale * 4.0), 0.95)
        vc_t_adj = min(vc_t * (1.0 + vax_cov_scale * 4.0), 0.95)
        vc_l_adj = min(vc_l * (1.0 + vax_cov_scale * 4.0), 0.95)
    else:
        mask_u = vent_u = online_u = club_u = disinfect_u = 0.0
        q_rate_adj = q_rate
        vc_s_adj, vc_t_adj, vc_l_adj = vc_s, vc_t, vc_l

    # 人群
    N_s, N_t, N_l = 26000.0, 1700.0, 800.0
    N_arr = np.array([N_s, N_t, N_l])
    ve_arr = np.array([ve_s, ve_t, ve_l])

    # 接触矩阵
    C_dorm = np.array([[8.5,0.3,0.2],[0.2,0.1,0.05],[0.3,0.05,0.4]])
    C_class = np.array([[12.0,2.5,0.3],[3.0,1.2,0.2],[0.4,0.2,0.3]])
    C_canteen = np.array([[4.5,0.8,0.5],[0.7,0.6,0.3],[0.6,0.3,0.8]])
    C_club = np.array([[6.0,0.5,0.2],[0.4,0.3,0.1],[0.2,0.1,0.2]])

    pw = config.get("place_weights", {"dorm":0.35,"class":0.40,"canteen":0.15,"club":0.10})
    total_w = sum(pw.values())
    w_dorm = pw.get("dorm",0.35)/total_w
    w_class = pw.get("class",0.40)/total_w
    w_canteen = pw.get("canteen",0.15)/total_w
    w_club = pw.get("club",0.10)/total_w

    def beta_seasonal(t):
        return beta0 * (1.0 + season_amp * np.cos(2*np.pi*(t - season_phase)/365.0))

    def ode(t, y):
        dydt = np.zeros(18)
        I_arr = np.array([y[3], y[9], y[15]])
        beta_t = beta_seasonal(t)

        # 计算各场所 adjusted beta 和 weight
        places_data = [
            (C_dorm, w_dorm, "dorm"),
            (C_class, w_class, "class"),
            (C_canteen, w_canteen, "canteen"),
            (C_club, w_club, "club"),
        ]

        for C_k, w_k_base, place_name in places_data:
            beta_k = beta_t
            w_k = w_k_base

            if place_name == "class":
                beta_k *= (1.0 - vent_eff * vent_u)
                beta_k *= (1.0 - disinfect_eff * disinfect_u)
                beta_k *= (1.0 - online_eff * online_u)
                w_k *= (1.0 - online_eff * online_u)
            elif place_name == "dorm":
                beta_k *= (1.0 - mask_eff * mask_u)
            elif place_name == "canteen":
                beta_k *= (1.0 - mask_eff * mask_u)
                beta_k *= (1.0 - disinfect_eff * disinfect_u)
            elif place_name == "club":
                beta_k *= (1.0 - mask_eff * mask_u)
                beta_k *= (1.0 - club_limit_eff * club_u)
                w_k *= (1.0 - club_limit_eff * club_u)

            beta_k = max(beta_k, 0.0)
            w_k = max(w_k, 0.0)

            for g in range(3):
                exposure = sum(C_k[g,h] * I_arr[h] / max(N_arr[h],1e-12) for h in range(3))
                # 将 force of infection 累加到 group g 的 lambda 中
                # 但需要先存 lambda 再算 dydt，这里直接在 loop 外用另一种方式
                pass  # 先算 lambda

        # 更好的方式：先算每个group的lambda
        lam = np.zeros(3)
        for C_k, w_k_base, place_name in places_data:
            beta_k = beta_t
            w_k = w_k_base

            if place_name == "class":
                beta_k *= (1.0 - vent_eff * vent_u)
                beta_k *= (1.0 - disinfect_eff * disinfect_u)
                beta_k *= (1.0 - online_eff * online_u)
                w_k *= (1.0 - online_eff * online_u)
            elif place_name == "dorm":
                beta_k *= (1.0 - mask_eff * mask_u)
            elif place_name == "canteen":
                beta_k *= (1.0 - mask_eff * mask_u)
                beta_k *= (1.0 - disinfect_eff * disinfect_u)
            elif place_name == "club":
                beta_k *= (1.0 - mask_eff * mask_u)
                beta_k *= (1.0 - club_limit_eff * club_u)
                w_k *= (1.0 - club_limit_eff * club_u)

            beta_k = max(beta_k, 0.0)
            w_k = max(w_k, 0.0)

            for g in range(3):
                exposure = sum(C_k[g,h] * I_arr[h] / max(N_arr[h],1e-12) for h in range(3))
                lam[g] += w_k * beta_k * exposure

        for g, (ve, idx) in enumerate(zip(ve_arr, [0,6,12])):
            s = y[idx]; v = y[idx+1]; e = y[idx+2]; i = y[idx+3]; q = y[idx+4]
            loss_s = lam[g] * s
            loss_v = (1-ve) * lam[g] * v
            dydt[idx]   = -loss_s
            dydt[idx+1] = -loss_v
            dydt[idx+2] = loss_s + loss_v - sigma_val * e
            dydt[idx+3] = sigma_val * e - gamma_val * i - q_rate_adj * i
            dydt[idx+4] = q_rate_adj * i - q_release * q
            dydt[idx+5] = gamma_val * i + q_release * q

        return dydt

    # 初始条件
    vc_arr = [vc_s_adj, vc_t_adj, vc_l_adj]
    seed_arr = [seed_s, 0.0, 0.0]
    y0 = np.zeros(18)
    for g, (n_g, vc_g, seed_g) in enumerate(zip([N_s, N_t, N_l], vc_arr, seed_arr)):
        idx = g * 6
        v0 = n_g * vc_g
        i0 = seed_g
        s0 = n_g - v0 - i0
        y0[idx] = s0; y0[idx+1] = v0; y0[idx+3] = i0

    t_eval = np.linspace(0, n_days, n_days + 1)
    sol = solve_ivp(ode, (0, n_days), y0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)

    # 组装 DataFrame
    comp_names = ["S","V","E","I","Q","R"]
    group_names = ["s","t","l"]
    col_names = [f"{c}_{g}" for g in group_names for c in comp_names]
    df = pd.DataFrame(sol.y.T, columns=col_names)
    df.index = pd.Index(sol.t, name="day")

    for c in comp_names:
        df[c] = sum(df[f"{c}_{g}"] for g in group_names)
    df["N"] = sum(df[f"S_{g}"] + df[f"V_{g}"] + df[f"E_{g}"] + df[f"I_{g}"] + df[f"Q_{g}"] + df[f"R_{g}"] for g in group_names)

    return df


def plot_season_scenario_grid():
    """2(季节) x 3(场景) 优化前后传播曲线对比图。"""
    print("  [9] 2季节x3场景 优化前后传播曲线对比图...")

    seasons = [
        {"name": "winter", "cn": "冬春高峰季", "phase": 30.0,
         "color": COLORS["winter"], "fill": "#BBDEFB"},
        {"name": "off_season", "cn": "非高峰季", "phase": 210.0,
         "color": COLORS["off_season"], "fill": "#C8E6C9"},
    ]
    scenarios = ["normal", "sporadic", "cluster"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, season in enumerate(seasons):
        for col_idx, scenario in enumerate(scenarios):
            ax = axes[row_idx, col_idx]
            phase = season["phase"]

            # 无干预基线
            print(f"    计算 {season['cn']} - {SCENARIO_CN[scenario]} (无干预)...")
            df_base = _solve_layered_ode(scenario, phase, apply_controls=False)
            # 优化后
            print(f"    计算 {season['cn']} - {SCENARIO_CN[scenario]} (优化后)...")
            df_opt = _solve_layered_ode(scenario, phase, apply_controls=True)

            days_base = df_base.index.to_numpy()
            days_opt = df_opt.index.to_numpy()

            # 无干预基线 - 虚线
            ax.plot(days_base, df_base["I"], linewidth=2.2, linestyle="--",
                    color=season["color"], alpha=0.7, label="无干预基线")
            ax.fill_between(days_base, 0, df_base["I"], alpha=0.08, color=season["color"])

            # 优化后 - 实线
            ax.plot(days_opt, df_opt["I"], linewidth=2.5,
                    color=COLORS["optimized"], label="优化后")
            ax.fill_between(days_opt, 0, df_opt["I"], alpha=0.10, color=COLORS["optimized"])

            # 标注峰值
            peak_base = df_base["I"].max()
            peak_opt = df_opt["I"].max()
            ar_base = df_base["R"].iloc[-1] / df_base["N"].iloc[0] * 100
            ar_opt = df_opt["R"].iloc[-1] / df_opt["N"].iloc[0] * 100

            info = (f"无干预: 峰值{peak_base:.0f}, AR={ar_base:.1f}%\n"
                    f"优化后: 峰值{peak_opt:.0f}, AR={ar_opt:.1f}%\n"
                    f"峰值降幅: {(1-peak_opt/peak_base)*100:.1f}%")
            ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

            # 子图标题
            panel = chr(97 + row_idx * 3 + col_idx)
            ax.set_title(f"({panel}) {season['cn']} - {SCENARIO_CN[scenario]}场景",
                         fontsize=12)
            ax.set_xlabel("天数")
            ax.set_ylabel("感染人数 $I(t)$")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("图9  分季节-分场景 优化前后传播曲线对比", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_season_scenario_grid.png", dpi=300)
    plt.close(fig)
    print("    -> fig9_season_scenario_grid.png 已保存")


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  论文级图表生成脚本（稳健版）")
    print("  输出目录:", FIG_DIR)
    print("=" * 60)

    # 先生成不依赖动态模型的图
    plot_scenario_comparison()          # 图2
    plot_place_contribution()           # 图3
    plot_optimization_comparison()      # 图4
    plot_intervention_sensitivity()     # 图5
    plot_model_validation()             # 图6
    plot_reff_curves()                  # 图7
    plot_optimization_convergence()     # 图8
    plot_seasonal_comparison()          # 图1
    plot_season_scenario_grid()         # 图9

    print("\n" + "=" * 60)
    print(f"  [DONE] 所有图表已保存至: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
