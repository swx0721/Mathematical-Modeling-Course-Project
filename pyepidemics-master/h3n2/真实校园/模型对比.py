"""CASE-001 Michigan University: layered model vs. real outbreak trend.

执行内容：
1. 读取高校校园 H3N2 汇总数据中的 CASE-001（密歇根大学）时间序列。
2. 基于常态分层 SVEIQR 模型，替换为密歇根文献参数进行一次仿真。
3. 输出累计病例与日新增病例的误差指标，并绘制对比图。

先用日新增峰值日对齐，自动估计 lag（真实峰日 - 模拟峰日）。
将模拟曲线按 lag 平移后，再做比例系数缩放。
在对齐后的同一时间轴上计算 MAE 和 RMSE（以及 sMAPE）。
本次运行结果（已对齐后）：

Best time lag: 12 天
Scale factor alpha: 0.258919
Cumulative RMSE: 25.087
Cumulative MAE: 17.522
Daily incidence RMSE: 8.402
Daily incidence MAE: 6.253
"""

from __future__ import annotations

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_layered_module(script_path: Path):
    spec = spec_from_file_location("layered_model_module", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模型文件: {script_path}")
    mod = module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_case001_real_series(excel_path: Path) -> pd.DataFrame:
    """读取并整理 CASE-001 真实时间序列（按日补齐）。"""
    raw = pd.read_excel(excel_path, sheet_name="CASE-001密歇根大学")
    raw = raw.rename(
        columns={
            "日期": "date",
            "新增确诊病例数": "new_cases",
            "累计确诊病例数": "cum_cases",
        }
    )

    df = raw[["date", "new_cases", "cum_cases"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["new_cases"] = pd.to_numeric(df["new_cases"], errors="coerce").fillna(0.0)
    df["cum_cases"] = pd.to_numeric(df["cum_cases"], errors="coerce").fillna(0.0)

    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    daily = df.set_index("date").reindex(full_idx)
    daily.index.name = "date"

    # The source data are not recorded daily, so we interpolate the cumulative series
    # and derive daily incidence to align with the ODE output.
    daily["cum_cases"] = daily["cum_cases"].interpolate(method="linear")
    daily["cum_cases"] = daily["cum_cases"].ffill().fillna(0.0)
    daily["new_cases"] = (
        daily["cum_cases"].diff().fillna(daily["cum_cases"]).clip(lower=0.0)
    )

    daily = daily.reset_index()
    daily["day"] = (daily["date"] - daily["date"].min()).dt.days.astype(int)
    return daily[["day", "date", "new_cases", "cum_cases"]]


def fit_scale_factor_by_final(real_values: pd.Series, sim_values: pd.Series) -> float:
    """Use the final cumulative level to define a simple amplitude scale factor."""
    y_final = float(real_values.iloc[-1])
    x_final = float(sim_values.iloc[-1])
    if abs(x_final) <= 1e-12:
        return 1.0
    return y_final / x_final


def estimate_time_lag_by_peak(
    real_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    max_abs_lag: int = 20,
) -> int:
    """Estimate lag by aligning peak day of daily incidence.

    lag > 0 means simulation is shifted later (to the right).
    """
    real_peak_day = int(real_df.loc[real_df["new_cases"].idxmax(), "day"])
    sim_peak_day = int(sim_df.loc[sim_df["sim_new_cases"].idxmax(), "day"])
    lag = real_peak_day - sim_peak_day
    return int(np.clip(lag, -int(max_abs_lag), int(max_abs_lag)))


def build_michigan_param_model(layered_module):
    """将模型参数替换为密歇根案例可用参数后，返回模型实例。"""
    params = layered_module.CampusLayeredParams()

    # 依据汇总文献参数进行替换：R0≈1.48，潜伏期≈2天，感染期≈5天，隔离期≈7天，接种率≈26.6%。
    params = replace(
        params,
        beta0=0.296,  # 近似取 beta0 = R0 * gamma = 1.48 * (1/5)
        sigma=1.0 / 2.0,
        gamma=1.0 / 5.0,
        q_release=1.0 / 7.0,
        v_cov_s=0.266,
        v_cov_t=0.266,
        v_cov_l=0.266,
        seed_s=1.0,
        seed_t=0.0,
        seed_l=0.0,
        seed_e_s=1.0,
        seed_e_t=0.0,
        seed_e_l=0.0,
        season_start_day=350.0,
    )

    # CASE-001 已知检测总人数 3121，作为观测人群近似规模。
    group_sizes = {"s": 2809.0, "t": 187.0, "l": 125.0}

    model = layered_module.CampusLayeredSVEIQR(
        group_sizes=group_sizes,
        contact_matrices=layered_module.default_contact_matrices(),
        place_weights=layered_module.default_place_weights(),
        params=params,
    )
    return model


def align_and_evaluate(sim_df: pd.DataFrame, real_df: pd.DataFrame) -> Dict[str, float]:
    """对齐仿真与真实序列，并计算误差指标。"""
    merged = pd.merge(real_df, sim_df, on="day", how="inner")

    err_cum = merged["sim_cum_cases"] - merged["cum_cases"]
    err_new = merged["sim_new_cases"] - merged["new_cases"]

    def _mape(y_true: pd.Series, y_pred: pd.Series) -> float:
        den = np.maximum(np.abs(y_true.to_numpy(dtype=float)), 1e-6)
        return float(
            np.mean(
                np.abs(
                    (y_pred.to_numpy(dtype=float) - y_true.to_numpy(dtype=float)) / den
                )
            )
        )

    def _mape_positive_only(y_true: pd.Series, y_pred: pd.Series) -> float:
        y_true_arr = y_true.to_numpy(dtype=float)
        y_pred_arr = y_pred.to_numpy(dtype=float)
        mask = y_true_arr > 1e-6
        if not np.any(mask):
            return float("nan")
        return float(
            np.mean(np.abs((y_pred_arr[mask] - y_true_arr[mask]) / y_true_arr[mask]))
        )

    def _smape(y_true: pd.Series, y_pred: pd.Series) -> float:
        y_true_arr = y_true.to_numpy(dtype=float)
        y_pred_arr = y_pred.to_numpy(dtype=float)
        den = np.maximum((np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0, 1e-6)
        return float(np.mean(np.abs(y_pred_arr - y_true_arr) / den))

    return {
        "rmse_cum": float(np.sqrt(np.mean(np.square(err_cum)))),
        "mae_cum": float(np.mean(np.abs(err_cum))),
        "mape_cum": _mape(merged["cum_cases"], merged["sim_cum_cases"]),
        "mape_pos_cum": _mape_positive_only(
            merged["cum_cases"], merged["sim_cum_cases"]
        ),
        "smape_cum": _smape(merged["cum_cases"], merged["sim_cum_cases"]),
        "rmse_new": float(np.sqrt(np.mean(np.square(err_new)))),
        "mae_new": float(np.mean(np.abs(err_new))),
        "mape_new": _mape(merged["new_cases"], merged["sim_new_cases"]),
        "mape_pos_new": _mape_positive_only(
            merged["new_cases"], merged["sim_new_cases"]
        ),
        "smape_new": _smape(merged["new_cases"], merged["sim_new_cases"]),
        "n_days": float(len(merged)),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    excel_path = (
        root / "Kimi_Agent_高校H3N2传播趋势" / "高校校园H3N2流感流行数据汇总.xlsx"
    )
    model_path = root / "常态校园模型.py"

    layered = _load_layered_module(model_path)
    real_df = load_case001_real_series(excel_path)

    model = build_michigan_param_model(layered)
    n_days = int(real_df["day"].max())
    traj = model.solve(n_days=n_days, dense_points=1)

    sim = pd.DataFrame({"day": traj.index.astype(int)})
    s0 = float(traj["S"].iloc[0])
    v0 = float(traj["V"].iloc[0])
    sim["sim_cum_cases"] = (s0 + v0) - (traj["S"].to_numpy() + traj["V"].to_numpy())
    sim["sim_new_cases"] = (
        sim["sim_cum_cases"].diff().fillna(sim["sim_cum_cases"]).clip(lower=0.0)
    )

    best_lag = estimate_time_lag_by_peak(real_df=real_df, sim_df=sim, max_abs_lag=20)

    sim_aligned = sim.copy()
    sim_aligned["day"] = sim_aligned["day"] + best_lag
    overlap_for_scale = pd.merge(real_df, sim_aligned, on="day", how="inner")
    scale_factor = fit_scale_factor_by_final(
        overlap_for_scale["cum_cases"], overlap_for_scale["sim_cum_cases"]
    )
    sim_aligned["sim_cum_cases_scaled"] = sim_aligned["sim_cum_cases"] * scale_factor
    sim_aligned["sim_new_cases_scaled"] = sim_aligned["sim_new_cases"] * scale_factor

    merged = pd.merge(real_df, sim_aligned, on="day", how="inner")
    sim_scaled = sim_aligned[
        ["day", "sim_cum_cases_scaled", "sim_new_cases_scaled"]
    ].rename(
        columns={
            "sim_cum_cases_scaled": "sim_cum_cases",
            "sim_new_cases_scaled": "sim_new_cases",
        }
    )
    metrics = align_and_evaluate(sim_df=sim_scaled, real_df=real_df)

    out_csv = root / "michigan_case001_compare.csv"
    out_png = root / "michigan_case001_compare.png"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        merged["day"], merged["cum_cases"], label="Observed cumulative", linewidth=2.2
    )
    axes[0].plot(
        merged["day"],
        merged["sim_cum_cases_scaled"],
        label=f"Sim cumulative (lag={best_lag}, x {scale_factor:.3f})",
        linewidth=2.2,
    )
    axes[0].set_title("CASE-001 Cumulative Trend Comparison")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Cases")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        merged["day"],
        merged["new_cases"],
        label="Observed daily incidence",
        linewidth=2.0,
    )
    axes[1].plot(
        merged["day"],
        merged["sim_new_cases_scaled"],
        label=f"Sim daily (lag={best_lag}, x {scale_factor:.3f})",
        linewidth=2.0,
    )
    axes[1].set_title("CASE-001 Daily Incidence Comparison")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Cases/day")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print("=== CASE-001 Michigan validation results ===")
    print(f"Best time lag (days): {best_lag}")
    print(f"Scale factor alpha: {scale_factor:.6f}")
    print(f"Aligned days: {int(metrics['n_days'])}")
    print(f"Cumulative RMSE: {metrics['rmse_cum']:.3f}")
    print(f"Cumulative MAE : {metrics['mae_cum']:.3f}")
    print(f"Cumulative sMAPE: {metrics['smape_cum']:.3%}")
    print(f"Cumulative MAPE (positive days): {metrics['mape_pos_cum']:.3%}")
    print(f"Daily incidence RMSE: {metrics['rmse_new']:.3f}")
    print(f"Daily incidence MAE : {metrics['mae_new']:.3f}")
    print(f"Daily incidence sMAPE: {metrics['smape_new']:.3%}")
    print(f"Daily incidence MAPE (positive days): {metrics['mape_pos_new']:.3%}")
    print(f"Comparison table saved to: {out_csv}")
    print(f"Comparison figure saved to: {out_png}")


if __name__ == "__main__":
    main()
