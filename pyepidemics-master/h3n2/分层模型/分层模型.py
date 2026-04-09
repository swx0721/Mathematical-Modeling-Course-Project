"""分层多人群校园 SVEIQR 模型。

本脚本在单群体校园 SVEIR 的基础上，升级为三类人群、四类场所的分层模型：

- 人群：学生、教师、后勤
- 仓室：S, V, E, I, Q, R
- 场所：宿舍、教学区、食堂、社团

当前版本只完成模型建立与情景模拟，不包含参数优化。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


GROUPS = ("s", "t", "l")
COMPARTMENTS = ("S", "V", "E", "I", "Q", "R")
PLACES = ("dorm", "class", "canteen", "club")


def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {key: float(value) / total for key, value in weights.items()}


def _as_array(matrix: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"contact matrix must be 3x3, got {arr.shape}")
    return arr


@dataclass
class CampusLayeredParams:
    """模型参数。"""

    beta0: float = 0.858
    sigma: float = 1.161
    gamma: float = 0.770

    q_rate: float = 0
    q_release: float = 0

    ve_s: float = 0.38
    ve_t: float = 0.305
    ve_l: float = 0.375
    v_cov_s: float = 0.3
    v_cov_t: float = 0.4
    v_cov_l: float = 0.4

    season_amp: float = 0.225
    season_phase: float = 30.0

    omega_v: float = 0.0
    omega_r: float = 0.0

    mask_effect = 0.18
    vent_effect = 0.3
    online_effect = 0.715
    club_limit_effect = 0.4
    disinfect_effect = 0.175
    mask_u: float = 0.0
    vent_u: float = 0.0
    online_u: float = 0.0
    club_u: float = 0.0
    disinfect_u: float = 0.0

    seed_s: float = 2.0
    seed_t: float = 0.0
    seed_l: float = 0.0
    seed_e_s: float = 2.0
    seed_e_t: float = 0.0
    seed_e_l: float = 0.0


class CampusLayeredSVEIQR:
    """三群体、四场所分层校园 SVEIQR 模型。"""

    def __init__(
        self,
        group_sizes: Mapping[str, float],
        contact_matrices: Mapping[str, Iterable[Iterable[float]]],
        place_weights: Mapping[str, float],
        params: CampusLayeredParams | None = None,
    ) -> None:
        self.group_sizes = {group: float(group_sizes[group]) for group in GROUPS}
        self.contact_matrices = {
            place: _as_array(contact_matrices[place]) for place in PLACES
        }
        self.place_weights = _normalize_weights(place_weights)
        self.params = params or CampusLayeredParams()

        self._state_names = [
            f"{comp}_{group}" for group in GROUPS for comp in COMPARTMENTS
        ]
        self._state_index = {name: idx for idx, name in enumerate(self._state_names)}

    @property
    def state_names(self) -> List[str]:
        return list(self._state_names)

    def pack_state(self, state: Mapping[str, float]) -> np.ndarray:
        return np.array(
            [float(state.get(name, 0.0)) for name in self._state_names], dtype=float
        )

    def unpack_state(self, y: np.ndarray) -> Dict[str, float]:
        return {name: float(y[idx]) for idx, name in enumerate(self._state_names)}

    def total_population(self) -> float:
        return float(sum(self.group_sizes.values()))

    def seasonal_beta(self, t: float) -> float:
        params = self.params
        return float(
            params.beta0
            * (
                1.0
                + params.season_amp
                * np.cos(2.0 * np.pi * (t - params.season_phase) / 365.0)
            )
        )

    def adjusted_place_beta(self, place: str, t: float) -> float:
        params = self.params
        beta = self.seasonal_beta(t)

        if place == "class":
            beta *= 1.0 - params.vent_effect * params.vent_u
            beta *= 1.0 - params.disinfect_effect * params.disinfect_u
            beta *= 1.0 - params.online_effect * params.online_u
        elif place == "dorm":
            beta *= 1.0 - params.mask_effect * params.mask_u
        elif place == "canteen":
            beta *= 1.0 - params.mask_effect * params.mask_u
            beta *= 1.0 - params.disinfect_effect * params.disinfect_u
        elif place == "club":
            beta *= 1.0 - params.mask_effect * params.mask_u
            beta *= 1.0 - params.club_limit_effect * params.club_u

        return max(beta, 0.0)

    def place_weight(self, place: str, t: float) -> float:
        params = self.params
        base = float(self.place_weights[place])
        if place == "class":
            base *= 1.0 - params.online_effect * params.online_u
        elif place == "club":
            base *= 1.0 - params.club_limit_effect * params.club_u
        return max(base, 0.0)

    def _group_slice(self, y: np.ndarray, group: str) -> Dict[str, float]:
        offset = GROUPS.index(group) * len(COMPARTMENTS)
        return {comp: float(y[offset + idx]) for idx, comp in enumerate(COMPARTMENTS)}

    def infectious_by_group(self, y: np.ndarray) -> Dict[str, float]:
        return {group: self._group_slice(y, group)["I"] for group in GROUPS}

    def force_of_infection(self, y: np.ndarray, t: float) -> Dict[str, float]:
        infectious = self.infectious_by_group(y)
        result = {group: 0.0 for group in GROUPS}

        for place in PLACES:
            beta_k = self.adjusted_place_beta(place, t)
            w_k = self.place_weight(place, t)
            matrix = self.contact_matrices[place]

            for g_idx, group in enumerate(GROUPS):
                exposure = 0.0
                for h_idx, source_group in enumerate(GROUPS):
                    nh = max(self.group_sizes[source_group], 1e-12)
                    exposure += matrix[g_idx, h_idx] * infectious[source_group] / nh
                result[group] += w_k * beta_k * exposure

        return result

    def force_components_by_place(
        self, y: np.ndarray, t: float
    ) -> Dict[str, Dict[str, float]]:
        infectious = self.infectious_by_group(y)
        result: Dict[str, Dict[str, float]] = {
            group: {place: 0.0 for place in PLACES} for group in GROUPS
        }

        for place in PLACES:
            beta_k = self.adjusted_place_beta(place, t)
            w_k = self.place_weight(place, t)
            matrix = self.contact_matrices[place]

            for g_idx, group in enumerate(GROUPS):
                exposure = 0.0
                for h_idx, source_group in enumerate(GROUPS):
                    nh = max(self.group_sizes[source_group], 1e-12)
                    exposure += matrix[g_idx, h_idx] * infectious[source_group] / nh
                result[group][place] = w_k * beta_k * exposure

        return result

    def vaccine_effect(self, group: str) -> float:
        if group == "s":
            return float(self.params.ve_s)
        if group == "t":
            return float(self.params.ve_t)
        if group == "l":
            return float(self.params.ve_l)
        raise KeyError(group)

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        params = self.params
        lam = self.force_of_infection(y, t)
        dydt = np.zeros_like(y, dtype=float)

        for group_idx, group in enumerate(GROUPS):
            base = group_idx * len(COMPARTMENTS)
            s = y[base + 0]
            v = y[base + 1]
            e = y[base + 2]
            i = y[base + 3]
            q = y[base + 4]

            susceptible_loss = lam[group] * s
            vaccinated_loss = (1.0 - self.vaccine_effect(group)) * lam[group] * v

            dydt[base + 0] = -susceptible_loss
            dydt[base + 1] = -vaccinated_loss
            dydt[base + 2] = susceptible_loss + vaccinated_loss - params.sigma * e
            dydt[base + 3] = params.sigma * e - params.gamma * i - params.q_rate * i
            dydt[base + 4] = params.q_rate * i - params.q_release * q
            dydt[base + 5] = params.gamma * i + params.q_release * q

        return dydt

    def initial_state(self) -> Dict[str, float]:
        params = self.params
        init: Dict[str, float] = {}
        coverage = {"s": params.v_cov_s, "t": params.v_cov_t, "l": params.v_cov_l}
        seeds_i = {"s": params.seed_s, "t": params.seed_t, "l": params.seed_l}
        seeds_e = {"s": params.seed_e_s, "t": params.seed_e_t, "l": params.seed_e_l}

        for group in GROUPS:
            n_g = self.group_sizes[group]
            v0 = n_g * coverage[group]
            e0 = seeds_e[group]
            i0 = seeds_i[group]
            q0 = 0.0
            r0 = 0.0
            s0 = n_g - v0 - e0 - i0 - q0 - r0

            init[f"S_{group}"] = s0
            init[f"V_{group}"] = v0
            init[f"E_{group}"] = e0
            init[f"I_{group}"] = i0
            init[f"Q_{group}"] = q0
            init[f"R_{group}"] = r0

        return init

    def solve(
        self,
        n_days: int,
        init_state: Mapping[str, float] | None = None,
        dense_points: int = 1,
    ) -> pd.DataFrame:
        if init_state is None:
            init_state = self.initial_state()

        y0 = self.pack_state(init_state)
        t_eval = np.linspace(0.0, float(n_days), int(n_days * dense_points) + 1)
        sol = solve_ivp(
            self.derivatives,
            (0.0, float(n_days)),
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solve failed: {sol.message}")

        data = pd.DataFrame(sol.y.T, columns=self._state_names)
        data.index = pd.Index(sol.t, name="day")

        for group in GROUPS:
            cols = [f"{comp}_{group}" for comp in COMPARTMENTS]
            data[f"N_{group}"] = data[cols].sum(axis=1)

        data["S"] = data[[f"S_{g}" for g in GROUPS]].sum(axis=1)
        data["V"] = data[[f"V_{g}" for g in GROUPS]].sum(axis=1)
        data["E"] = data[[f"E_{g}" for g in GROUPS]].sum(axis=1)
        data["I"] = data[[f"I_{g}" for g in GROUPS]].sum(axis=1)
        data["Q"] = data[[f"Q_{g}" for g in GROUPS]].sum(axis=1)
        data["R"] = data[[f"R_{g}" for g in GROUPS]].sum(axis=1)
        data["N"] = data[[f"N_{g}" for g in GROUPS]].sum(axis=1)

        return data

    def summary(self, trajectory: pd.DataFrame) -> Dict[str, float]:
        peak_i_idx = trajectory["I"].idxmax()
        peak_i_day = float(np.asarray(peak_i_idx).item())
        peak_i = float(trajectory["I"].max())
        final_r = float(trajectory["R"].iloc[-1])
        final_q = float(trajectory["Q"].iloc[-1])
        final_attack_rate = final_r / max(float(trajectory["N"].iloc[0]), 1e-12)
        return {
            "peak_I_day": peak_i_day,
            "peak_I": peak_i,
            "final_R": final_r,
            "final_Q": final_q,
            "attack_rate": final_attack_rate,
        }


def default_group_sizes() -> Dict[str, float]:
    return {"s": 26000.0, "t": 1700.0, "l": 800.0}


def default_contact_matrices() -> Dict[str, List[List[float]]]:
    return {
        "dorm": [
            [8.5, 0.3, 0.2],
            [0.2, 0.1, 0.05],
            [0.3, 0.05, 0.4],
        ],
        "class": [
            [12.0, 2.5, 0.3],
            [3.0, 1.2, 0.2],
            [0.4, 0.2, 0.3],
        ],
        "canteen": [
            [4.5, 0.8, 0.5],
            [0.7, 0.6, 0.3],
            [0.6, 0.3, 0.8],
        ],
        "club": [
            [6.0, 0.5, 0.2],
            [0.4, 0.3, 0.1],
            [0.2, 0.1, 0.2],
        ],
    }


def default_place_weights() -> Dict[str, float]:
    return {"dorm": 0.35, "class": 0.40, "canteen": 0.15, "club": 0.10}


def build_baseline_model() -> CampusLayeredSVEIQR:
    return CampusLayeredSVEIQR(
        group_sizes=default_group_sizes(),
        contact_matrices=default_contact_matrices(),
        place_weights=default_place_weights(),
        params=CampusLayeredParams(),
    )


def build_scenario_model(scenario: str = "baseline") -> CampusLayeredSVEIQR:
    params = CampusLayeredParams()
    scenario = scenario.lower().strip()

    if scenario == "baseline":
        pass
    elif scenario == "sporadic":
        params.mask_u = 0.40
        params.vent_u = 0.35
        params.club_u = 0.30
        params.disinfect_u = 0.25
        params.seed_s = 3.0
        params.seed_e_s = 2.0
    elif scenario == "cluster":
        params.mask_u = 0.80
        params.vent_u = 0.70
        params.online_u = 0.60
        params.club_u = 0.80
        params.disinfect_u = 0.60
        params.q_rate = 0.25
        params.seed_s = 10.0
        params.seed_e_s = 5.0
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    return CampusLayeredSVEIQR(
        group_sizes=default_group_sizes(),
        contact_matrices=default_contact_matrices(),
        place_weights=default_place_weights(),
        params=params,
    )


def load_shanghai_seir_anchor(csv_path: str | Path | None = None) -> Dict[str, float]:
    """读取上海数据文件，并返回可作为校园模型锚定的基准参数。

    当前实现直接返回论文里已经约定的中心值；若需要后续接入更完整的拟合结果，
    可在此函数中扩展为从参数日志中读取 beta, sigma, gamma。
    """

    _ = csv_path
    return {"beta": 0.858, "sigma": 1.161, "gamma": 0.770}


def save_trajectory(trajectory: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(output_path, encoding="utf-8-sig")


def run_demo(
    output_dir: str | Path | None = None,
) -> Tuple[CampusLayeredSVEIQR, pd.DataFrame, Dict[str, float]]:
    anchor = load_shanghai_seir_anchor()
    model = build_baseline_model()
    model.params.beta0 = anchor["beta"]
    model.params.sigma = anchor["sigma"]
    model.params.gamma = anchor["gamma"]

    trajectory = model.solve(n_days=220)
    summary = model.summary(trajectory)

    if output_dir is not None:
        output_dir = Path(output_dir)
        save_trajectory(trajectory, output_dir / "layered_sveiqr_trajectory.csv")
        pd.DataFrame([asdict(model.params)]).to_csv(
            output_dir / "layered_sveiqr_params.csv", index=False, encoding="utf-8-sig"
        )

    return model, trajectory, summary


def build_incidence_breakdown(
    model: CampusLayeredSVEIQR, trajectory: pd.DataFrame
) -> pd.DataFrame:
    """按群体和场所分解日新增暴露。"""

    rows = []
    for t, row in trajectory.iterrows():
        t_day = float(np.asarray(t).item())
        y = np.asarray([row[name] for name in model.state_names], dtype=float)
        lam_comp = model.force_components_by_place(y, t_day)

        entry: Dict[str, float] = {"day": t_day}
        total_new = 0.0

        for group in GROUPS:
            sg = float(row[f"S_{group}"])
            vg = float(row[f"V_{group}"])
            ve = model.vaccine_effect(group)
            group_new = 0.0

            for place in PLACES:
                lam_gp = lam_comp[group][place]
                new_gp_place = lam_gp * sg + (1.0 - ve) * lam_gp * vg
                entry[f"new_{group}_{place}"] = new_gp_place
                group_new += new_gp_place

            entry[f"new_{group}"] = group_new
            total_new += group_new

        entry["new_total"] = total_new
        for place in PLACES:
            entry[f"new_{place}"] = sum(entry[f"new_{g}_{place}"] for g in GROUPS)

        rows.append(entry)

    return pd.DataFrame(rows).set_index("day")


def plot_detailed_results(
    model: CampusLayeredSVEIQR,
    trajectory: pd.DataFrame,
    output_dir: str | Path | None = None,
    title_suffix: str = "Baseline",
) -> pd.DataFrame:
    """生成详细图形并可选保存到文件。"""

    plt.style.use("bmh")
    inc = build_incidence_breakdown(model, trajectory)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)

    # Panel 1: Total compartment trajectories
    for state in ["S", "V", "E", "I", "Q", "R"]:
        axes[0, 0].plot(trajectory.index, trajectory[state], label=state, linewidth=1.8)
    axes[0, 0].set_title(f"Layered SVEIQR Trajectories ({title_suffix})")
    axes[0, 0].set_ylabel("Population")
    axes[0, 0].legend(ncol=3)
    axes[0, 0].grid(alpha=0.25)

    # Panel 2: Group-specific infectious and isolated
    axes[0, 1].plot(
        trajectory.index, trajectory["I_s"], label="I_students", linewidth=2.0
    )
    axes[0, 1].plot(
        trajectory.index, trajectory["I_t"], label="I_teachers", linewidth=2.0
    )
    axes[0, 1].plot(trajectory.index, trajectory["I_l"], label="I_staff", linewidth=2.0)
    axes[0, 1].plot(
        trajectory.index, trajectory["Q_s"], "--", label="Q_students", linewidth=1.5
    )
    axes[0, 1].plot(
        trajectory.index, trajectory["Q_t"], "--", label="Q_teachers", linewidth=1.5
    )
    axes[0, 1].plot(
        trajectory.index, trajectory["Q_l"], "--", label="Q_staff", linewidth=1.5
    )
    axes[0, 1].set_title(f"Group-level I and Q Curves ({title_suffix})")
    axes[0, 1].set_ylabel("Population")
    axes[0, 1].legend(ncol=2)
    axes[0, 1].grid(alpha=0.25)

    # Panel 3: Daily new exposures
    axes[1, 0].plot(
        inc.index, inc["new_total"], label="New exposures total", linewidth=2.2
    )
    axes[1, 0].plot(inc.index, inc["new_s"], label="Students", linewidth=1.6)
    axes[1, 0].plot(inc.index, inc["new_t"], label="Teachers", linewidth=1.6)
    axes[1, 0].plot(inc.index, inc["new_l"], label="Staff", linewidth=1.6)
    axes[1, 0].set_title(f"Estimated Daily Incidence ({title_suffix})")
    axes[1, 0].set_xlabel("Day")
    axes[1, 0].set_ylabel("People/day")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    # Panel 4: Place contribution share
    stack_data = np.vstack([inc[f"new_{place}"].to_numpy() for place in PLACES])
    den = np.maximum(inc["new_total"].to_numpy(), 1e-12)
    stack_share = stack_data / den
    axes[1, 1].stackplot(
        inc.index,
        stack_share,
        labels=["Dorm", "Classroom", "Canteen", "Club"],
        alpha=0.85,
    )
    axes[1, 1].set_title(f"Place-wise Contribution Share ({title_suffix})")
    axes[1, 1].set_xlabel("Day")
    axes[1, 1].set_ylabel("Share of new exposures")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(alpha=0.25)

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"layered_sveiqr_detailed_{title_suffix.lower()}.png", dpi=180
        )

    plt.show()
    return inc


def plot_scenario_comparison(output_dir: str | Path | None = None) -> pd.DataFrame:
    """输出 baseline/sporadic/cluster 三场景对比图与汇总表。"""

    results = []
    curves = {}

    for name in ["baseline", "sporadic", "cluster"]:
        model = build_scenario_model(name)
        traj = model.solve(n_days=220)
        sum_dict = model.summary(traj)
        results.append({"scenario": name, **sum_dict})
        curves[name] = traj

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    for name, traj in curves.items():
        label = name.capitalize()
        axes[0].plot(traj.index, traj["I"], linewidth=2.0, label=label)
        new_exp = -(traj["S"].diff().fillna(0.0) + traj["V"].diff().fillna(0.0))
        new_exp = new_exp.clip(lower=0.0)
        axes[1].plot(traj.index, new_exp, linewidth=2.0, label=label)

    axes[0].set_title("Total Infectious Comparison Across Scenarios")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Infectious population")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Daily New Exposures Comparison Across Scenarios")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("People/day")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "layered_sveiqr_scenario_comparison.png", dpi=180)

    plt.show()

    summary_df = pd.DataFrame(results)
    if output_dir is not None:
        summary_df.to_csv(
            Path(output_dir) / "layered_sveiqr_scenario_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return summary_df


if __name__ == "__main__":
    model, trajectory, summary = run_demo(output_dir=Path(__file__).resolve().parent)
    print("Model state names:")
    print(", ".join(model.state_names))
    print("Summary:")
    for key, value in summary.items():
        if key.endswith("_day"):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:.4f}")

    print("Generating detailed figures...")
    plot_detailed_results(
        model=model,
        trajectory=trajectory,
        output_dir=Path(__file__).resolve().parent,
        title_suffix="Baseline",
    )

    print("Generating scenario comparison figures...")
    scenario_table = plot_scenario_comparison(
        output_dir=Path(__file__).resolve().parent
    )
    print(scenario_table)
