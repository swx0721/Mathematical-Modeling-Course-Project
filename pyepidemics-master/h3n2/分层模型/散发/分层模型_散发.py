"""分层多人群校园 SVEIQR 模型。

本脚本在单群体校园 SVEIR 的基础上，升级为三类人群、四类场所的分层模型：

- 人群：学生、教师、后勤
- 仓室：S, V, E, I, Q, R
- 场所：宿舍、教学区、食堂、社团

当前版本只完成模型建立与情景模拟，不包含参数优化。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from scipy.stats import rankdata, t as student_t
from tqdm.auto import tqdm


GROUPS = ("s", "t", "l")
COMPARTMENTS = ("S", "V", "E", "I", "Q", "R")
PLACES = ("dorm", "class", "canteen", "club")
CONTROL_NAMES = (
    "mask_u",
    "vent_u",
    "online_u",
    "club_u",
    "disinfect_u",
    "vax_cov_scale",
    "q_scale",
)
CONTROL_BOUNDS = {
    "mask_u": (0.0, 1.0),
    "vent_u": (0.0, 1.0),
    "online_u": (0.0, 1.0),
    "club_u": (0.0, 1.0),
    "disinfect_u": (0.0, 1.0),
    "vax_cov_scale": (0.0, 5.0),
    "q_scale": (0.0, 10.0),
}


@dataclass
class ObjectiveWeights:
    """目标函数中的疫情损失权重。"""

    omega1: float = 1.20
    omega2: float = 8.00
    lambda_AR: float = 1.00
    lambda_P: float = 0.10


@dataclass
class CostWeights:
    """防控成本权重。"""

    c_m: float = 1.00
    c_v: float = 0.5
    c_d: float = 2.00
    c_o: float = 1.50
    c_cl: float = 0.80
    c_vax: float = 8.0
    c_q: float = 5.50
    c_q2: float = 0.50


@dataclass
class DisruptionWeights:
    """教学秩序损失权重。"""

    d_o: float = 1.50
    d_c: float = 0.80
    d_q_policy: float = 1.50
    d_q_load: float = 30.00


SEASON_START_DAY = {
    "winter_spring_peak": 350.0,
    "off_season": 200.0,
}


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


class _TerminalEvent:
    def __init__(self, fn, direction: int = -1) -> None:
        self.fn = fn
        self.terminal = True
        self.direction = direction

    def __call__(self, t: float, y: np.ndarray) -> float:
        return float(self.fn(t, y))


@dataclass
class CampusLayeredParams:
    """模型参数。"""

    beta0: float = 0.256
    sigma: float = 0.53
    gamma: float = 0.2

    q_rate: float = 0.18
    q_release: float = 0.15
    q_scale: float = 1.0
    q_sat_k: float = 1.0
    q_gain_max: float = 3.0
    q_capacity_alpha: float = 10.0
    q_rate_cap: float = 1.0

    ve_s: float = 0.38
    ve_t: float = 0.305
    ve_l: float = 0.375
    v_cov_s: float = 0.2
    v_cov_t: float = 0.12
    v_cov_l: float = 0.12
    vax_cov_scale: float = 1.0

    season_amp: float = 0.225
    season_phase: float = 30.0
    season_start_day: float = 0.0

    omega_v: float = 0.0
    omega_r: float = 0.0

    mask_effect = 0.40
    vent_effect = 0.3
    online_effect = 0.715
    club_limit_effect = 0.4
    disinfect_effect = 0.60
    mask_u: float = 0.0
    vent_u: float = 0.0
    online_u: float = 0.0
    club_u: float = 0.0
    disinfect_u: float = 0.0

    seed_s: float = 6.0
    seed_t: float = 0.0
    seed_l: float = 0.0
    seed_e_s: float = 6.0
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
                * np.cos(
                    2.0
                    * np.pi
                    * (t + params.season_start_day - params.season_phase)
                    / 365.0
                )
            )
        )

    def adjusted_place_beta(self, place: str, t: float) -> float:
        params = self.params
        beta = self.seasonal_beta(t)

        if place == "class":
            beta *= 1.0 - params.mask_effect * params.mask_u
            beta *= 1.0 - params.vent_effect * params.vent_u
            beta *= 1.0 - params.disinfect_effect * params.disinfect_u
            beta *= 1.0 - params.online_effect * params.online_u
        elif place == "dorm":
            beta *= 1.0 - params.vent_effect * params.vent_u
        elif place == "canteen":
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
        i_total = float(sum(y[self._state_index[f"I_{g}"]] for g in GROUPS))
        i_ratio = i_total / max(self.total_population(), 1e-12)
        q_scale_nonneg = max(float(params.q_scale), 0.0)
        q_sat_k = max(float(params.q_sat_k), 1e-12)
        q_response = q_scale_nonneg / (q_sat_k + q_scale_nonneg)
        q_rate_cmd = float(params.q_rate) * (
            1.0 + float(params.q_gain_max) * q_response
        )
        q_rate_eff = q_rate_cmd / (
            1.0 + float(params.q_capacity_alpha) * max(i_ratio, 0.0)
        )
        q_rate_eff = min(max(q_rate_eff, 0.0), float(params.q_rate_cap))

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
            dydt[base + 3] = params.sigma * e - params.gamma * i - q_rate_eff * i
            dydt[base + 4] = q_rate_eff * i - params.q_release * q
            dydt[base + 5] = params.gamma * i + params.q_release * q

        return dydt

    def initial_state(self) -> Dict[str, float]:
        params = self.params
        init: Dict[str, float] = {}
        scale = float(np.clip(params.vax_cov_scale, 0.0, 10.0))
        coverage = {
            "s": float(np.clip(params.v_cov_s * scale, 0.0, 1.0)),
            "t": float(np.clip(params.v_cov_t * scale, 0.0, 1.0)),
            "l": float(np.clip(params.v_cov_l * scale, 0.0, 1.0)),
        }
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
        n_days: int | None = None,
        init_state: Mapping[str, float] | None = None,
        dense_points: int = 1,
        stop_threshold: float = 1.0,
        max_days: int = 365,
    ) -> pd.DataFrame:
        if init_state is None:
            init_state = self.initial_state()

        y0 = self.pack_state(init_state)
        horizon_days = int(max_days if n_days is None else n_days)
        horizon_days = max(horizon_days, 1)
        t_eval = np.linspace(
            0.0, float(horizon_days), int(horizon_days * dense_points) + 1
        )

        e_idx = [self._state_index[f"E_{g}"] for g in GROUPS]
        i_idx = [self._state_index[f"I_{g}"] for g in GROUPS]
        q_idx = [self._state_index[f"Q_{g}"] for g in GROUPS]

        def _epidemic_stop_event(_t: float, y: np.ndarray) -> float:
            active = float(np.sum(y[e_idx]) + np.sum(y[i_idx]) + np.sum(y[q_idx]))
            return active - float(stop_threshold)

        epidemic_stop_event = _TerminalEvent(_epidemic_stop_event, direction=-1)
        sol = solve_ivp(
            self.derivatives,
            (0.0, float(horizon_days)),
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
            events=epidemic_stop_event,
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
    scales = {"dorm": 1.10, "class": 1.08, "canteen": 1.10, "club": 1.20}
    return {
        "dorm": [
            [5.5 * scales["dorm"], 0.3 * scales["dorm"], 0.2 * scales["dorm"]],
            [0.2 * scales["dorm"], 0.1 * scales["dorm"], 0.05 * scales["dorm"]],
            [0.3 * scales["dorm"], 0.05 * scales["dorm"], 0.4 * scales["dorm"]],
        ],
        "class": [
            [6.5 * scales["class"], 2.0 * scales["class"], 0.3 * scales["class"]],
            [2.5 * scales["class"], 1.2 * scales["class"], 0.2 * scales["class"]],
            [0.4 * scales["class"], 0.2 * scales["class"], 0.3 * scales["class"]],
        ],
        "canteen": [
            [2.8 * scales["canteen"], 0.8 * scales["canteen"], 0.5 * scales["canteen"]],
            [0.7 * scales["canteen"], 0.6 * scales["canteen"], 0.3 * scales["canteen"]],
            [0.6 * scales["canteen"], 0.3 * scales["canteen"], 0.8 * scales["canteen"]],
        ],
        "club": [
            [2.0 * scales["club"], 0.5 * scales["club"], 0.2 * scales["club"]],
            [0.4 * scales["club"], 0.3 * scales["club"], 0.1 * scales["club"]],
            [0.2 * scales["club"], 0.1 * scales["club"], 0.2 * scales["club"]],
        ],
    }


def default_place_weights() -> Dict[str, float]:
    return {"dorm": 0.35, "class": 0.40, "canteen": 0.15, "club": 0.10}


def apply_season_profile(
    params: CampusLayeredParams, season_profile: str | None
) -> CampusLayeredParams:
    if season_profile is None:
        return params

    profile = season_profile.strip().lower().replace("-", "_")
    aliases = {
        "winter": "winter_spring_peak",
        "winter_spring": "winter_spring_peak",
        "peak": "winter_spring_peak",
        "off": "off_season",
        "low": "off_season",
        "non_peak": "off_season",
    }
    profile = aliases.get(profile, profile)

    if profile not in SEASON_START_DAY:
        raise ValueError(
            "season_profile must be one of: " "'winter_spring_peak', 'off_season'"
        )

    params.season_start_day = float(SEASON_START_DAY[profile])
    return params


def build_baseline_model(season_profile: str | None = None) -> CampusLayeredSVEIQR:
    params = apply_season_profile(CampusLayeredParams(), season_profile)
    return CampusLayeredSVEIQR(
        group_sizes=default_group_sizes(),
        contact_matrices=default_contact_matrices(),
        place_weights=default_place_weights(),
        params=params,
    )


def build_scenario_model(
    scenario: str = "baseline", season_profile: str | None = None
) -> CampusLayeredSVEIQR:
    params = apply_season_profile(CampusLayeredParams(), season_profile)
    scenario = scenario.lower().strip()

    if scenario != "sporadic":
        raise ValueError("only 'sporadic' scenario is supported in this script")

    return CampusLayeredSVEIQR(
        group_sizes=default_group_sizes(),
        contact_matrices=default_contact_matrices(),
        place_weights=default_place_weights(),
        params=params,
    )


def _clip_control(name: str, x: float) -> float:
    low, high = CONTROL_BOUNDS[name]
    return float(np.clip(float(x), low, high))


def _build_objective_components(
    trajectory: pd.DataFrame,
    controls: Mapping[str, float],
    objective_weights: ObjectiveWeights,
    cost_weights: CostWeights,
    disruption_weights: DisruptionWeights,
) -> Dict[str, float]:
    n0 = max(float(trajectory["N"].iloc[0]), 1e-12)
    t = trajectory.index.to_numpy(dtype=float)
    horizon = float(t[-1] - t[0]) if len(t) >= 2 else 0.0

    e_ratio = trajectory["E"].to_numpy(dtype=float) / n0
    i_ratio = trajectory["I"].to_numpy(dtype=float) / n0
    q_ratio = trajectory["Q"].to_numpy(dtype=float) / n0

    # A: 指数形式 (移除Q项) + 累计感染率 + 峰值项
    epidemic_exp = objective_weights.omega1 * (
        np.exp(objective_weights.omega2 * (e_ratio + i_ratio)) - 1.0
    )
    a_integral = float(np.trapezoid(epidemic_exp, t)) / max(horizon, 1e-12)

    # 累计感染率项
    final_r = float(trajectory["R"].iloc[-1])
    attack_rate = final_r / max(n0, 1e-12)
    a_ar = objective_weights.lambda_AR * attack_rate

    # 峰值感染人数项
    peak_i = float(trajectory["I"].max())
    peak_i_norm = peak_i / max(n0, 1e-12)
    a_p = objective_weights.lambda_P * peak_i_norm

    a = a_integral + a_ar + a_p

    # C: 一次项（线性）控制成本（时间平均）
    q_excess = max(0.0, controls["q_scale"] - 1.0)
    c = (
        cost_weights.c_m * controls["mask_u"]
        + cost_weights.c_v * controls["vent_u"]
        + cost_weights.c_d * controls["disinfect_u"]
        + cost_weights.c_o * controls["online_u"]
        + cost_weights.c_cl * controls["club_u"]
        + cost_weights.c_vax * max(0.0, controls["vax_cov_scale"] - 1.0)
        + cost_weights.c_q * q_excess
        + cost_weights.c_q2 * q_excess * q_excess
    )

    # D: 策略扰动 + 隔离负荷扰动（时间平均）
    d_policy = (
        disruption_weights.d_o * controls["online_u"]
        + disruption_weights.d_c * controls["club_u"]
        + disruption_weights.d_q_policy * controls["q_scale"]
    )
    d_load = (
        disruption_weights.d_q_load
        * float(np.trapezoid(q_ratio, t))
        / max(horizon, 1e-12)
    )
    d = d_policy + d_load

    j = a + c + d

    return {
        "A": float(a),
        "C": float(c),
        "D": float(d),
        "J": float(j),
        "peak_I_norm": float(np.max(i_ratio)),
    }


def evaluate_controls(
    controls: Mapping[str, float],
    n_days: int | None = None,
    objective_weights: ObjectiveWeights | None = None,
    cost_weights: CostWeights | None = None,
    disruption_weights: DisruptionWeights | None = None,
    base_params: CampusLayeredParams | None = None,
) -> Dict[str, float]:
    """给定防控强度，计算 A/C/D/J 和核心流行病学指标。"""

    objective_weights = objective_weights or ObjectiveWeights()
    cost_weights = cost_weights or CostWeights()
    disruption_weights = disruption_weights or DisruptionWeights()

    params = replace(base_params) if base_params is not None else CampusLayeredParams()
    for name in CONTROL_NAMES:
        low, high = CONTROL_BOUNDS[name]
        value = float(controls.get(name, getattr(params, name)))
        setattr(params, name, float(np.clip(value, low, high)))

    model = CampusLayeredSVEIQR(
        group_sizes=default_group_sizes(),
        contact_matrices=default_contact_matrices(),
        place_weights=default_place_weights(),
        params=params,
    )
    trajectory = model.solve(n_days=n_days)
    summary = model.summary(trajectory)
    components = _build_objective_components(
        trajectory=trajectory,
        controls={name: float(getattr(params, name)) for name in CONTROL_NAMES},
        objective_weights=objective_weights,
        cost_weights=cost_weights,
        disruption_weights=disruption_weights,
    )
    return {
        **components,
        **summary,
        **{name: float(getattr(params, name)) for name in CONTROL_NAMES},
    }


def optimize_interventions(
    n_days: int | None = None,
    objective_weights: ObjectiveWeights | None = None,
    cost_weights: CostWeights | None = None,
    disruption_weights: DisruptionWeights | None = None,
    base_params: CampusLayeredParams | None = None,
    maxiter: int = 1000,
    global_maxiter: int = 1000,
    global_popsize: int = 15,
    global_tol: float = 1e-3,
) -> Tuple[CampusLayeredSVEIQR, pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """优化七类防控强度，返回最优模型、轨迹、指标和搜索日志。

    采用两阶段策略：先用 differential_evolution 全局搜索，再用 L-BFGS-B 局部精修。
    """

    objective_weights = objective_weights or ObjectiveWeights()
    cost_weights = cost_weights or CostWeights()
    disruption_weights = disruption_weights or DisruptionWeights()
    params0 = replace(base_params) if base_params is not None else CampusLayeredParams()

    bounds = [CONTROL_BOUNDS[name] for name in CONTROL_NAMES]
    records: List[Dict[str, float]] = []

    def _objective(x: np.ndarray) -> float:
        controls = {
            name: _clip_control(name, x[idx]) for idx, name in enumerate(CONTROL_NAMES)
        }
        metrics = evaluate_controls(
            controls=controls,
            n_days=n_days,
            objective_weights=objective_weights,
            cost_weights=cost_weights,
            disruption_weights=disruption_weights,
            base_params=params0,
        )
        records.append(metrics)
        return float(metrics["J"])

    pbar = tqdm(
        total=int(global_maxiter),
        desc="Global optimization",
        unit="gen",
        dynamic_ncols=True,
        leave=False,
    )

    def _de_callback(_xk: np.ndarray, convergence: float) -> bool:
        _ = convergence
        pbar.update(1)
        return False

    # ---- Stage 1: Global search with differential_evolution ----
    try:
        global_result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=int(global_maxiter),
            popsize=int(global_popsize),
            tol=float(global_tol),
            rng=42,
            polish=False,
            callback=_de_callback,
        )
        if pbar.n < pbar.total:
            pbar.update(pbar.total - pbar.n)
    finally:
        pbar.close()

    # ---- Stage 2: Local refinement with L-BFGS-B ----
    local_result = minimize(
        _objective,
        global_result.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter)},
    )

    # Use the better of the two results
    if local_result.fun <= global_result.fun:
        best_x = local_result.x
        best_fun = local_result.fun
        local_success = local_result.success
        local_nit = local_result.nit
    else:
        best_x = global_result.x
        best_fun = global_result.fun
        local_success = True
        local_nit = 0

    best_controls = {
        name: _clip_control(name, best_x[idx]) for idx, name in enumerate(CONTROL_NAMES)
    }
    best_metrics = evaluate_controls(
        controls=best_controls,
        n_days=n_days,
        objective_weights=objective_weights,
        cost_weights=cost_weights,
        disruption_weights=disruption_weights,
        base_params=params0,
    )

    best_params = replace(params0)
    for name, value in best_controls.items():
        setattr(best_params, name, value)

    best_model = CampusLayeredSVEIQR(
        group_sizes=default_group_sizes(),
        contact_matrices=default_contact_matrices(),
        place_weights=default_place_weights(),
        params=best_params,
    )
    best_trajectory = best_model.solve(n_days=n_days)

    best_summary = {
        **best_metrics,
        "global_success": float(bool(global_result.success)),
        "global_nit": float(global_result.nit),
        "global_fun": float(global_result.fun),
        "optimizer_success": float(bool(local_success)),
        "optimizer_nit": float(local_nit),
        "optimizer_fun": float(best_fun),
    }
    history = pd.DataFrame(records)
    return best_model, best_trajectory, best_summary, history


def run_optimization_demo(
    output_dir: str | Path | None = None,
    n_days: int | None = None,
    scenario: str = "sporadic",
) -> Tuple[CampusLayeredSVEIQR, pd.DataFrame, Dict[str, float]]:
    """执行一次优化示例并可选保存结果（论文级多层级输出）。"""
    np.random.seed(42)  # Fix random seed for reproducibility

    base_params = CampusLayeredParams()

    model, trajectory, summary, history = optimize_interventions(
        n_days=n_days,
        base_params=base_params,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        # Save paper-grade outputs with scenario identifier
        save_paper_grade_outputs(
            model=model,
            trajectory=trajectory,
            scenario=scenario,
            output_root=output_dir,
            summary=summary,
            history=history,
        )

    return model, trajectory, summary


def save_trajectory(trajectory: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(output_path, encoding="utf-8-sig")


def save_paper_grade_outputs(
    model: CampusLayeredSVEIQR,
    trajectory: pd.DataFrame,
    scenario: str = "cluster",
    output_root: str | Path | None = None,
    summary: Dict[str, float] | None = None,
    history: pd.DataFrame | None = None,
) -> Path:
    """保存论文级多层级输出。

    Parameters:
    -----------
    model : CampusLayeredSVEIQR
        已优化/模拟的模型实例
    trajectory : pd.DataFrame
        时间序列数据
    scenario : str
        场景标识 ("normal", "sporadic", "cluster")
    output_root : Path or str
        输出根目录
    summary : dict, optional
        目标函数分解与指标摘要
    history : pd.DataFrame, optional
        优化收敛历史

    Returns:
    --------
    Path
        已创建的输出根目录
    """
    import json

    if output_root is None:
        output_root = Path.cwd()
    else:
        output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    # Layer 1: Full timeseries with all compartments
    ts_path = output_root / f"timeseries_{scenario}.csv"
    trajectory.to_csv(ts_path, encoding="utf-8-sig")

    # Layer 2: Place-wise contribution of daily new infections
    inc_df = build_incidence_breakdown(model, trajectory)
    pc_path = output_root / f"place_contribution_{scenario}.csv"
    inc_df.to_csv(pc_path, encoding="utf-8-sig")

    # Layer 3: Optimization result (objective, costs, controls)
    if summary is not None:
        opt_result = {
            "scenario": scenario,
            "objective_J": float(summary.get("J", np.nan)),
            "epidemic_loss_A": float(summary.get("A", np.nan)),
            "control_cost_C": float(summary.get("C", np.nan)),
            "disruption_D": float(summary.get("D", np.nan)),
            "attack_rate": float(summary.get("attack_rate", np.nan)),
            "peak_I": float(summary.get("peak_I", np.nan)),
            "peak_I_day": float(summary.get("peak_I_day", np.nan)),
            "controls": {
                "mask_u": float(summary.get("mask_u", model.params.mask_u)),
                "vent_u": float(summary.get("vent_u", model.params.vent_u)),
                "online_u": float(summary.get("online_u", model.params.online_u)),
                "club_u": float(summary.get("club_u", model.params.club_u)),
                "disinfect_u": float(
                    summary.get("disinfect_u", model.params.disinfect_u)
                ),
                "vax_cov_scale": float(
                    summary.get("vax_cov_scale", model.params.vax_cov_scale)
                ),
                "q_scale": float(summary.get("q_scale", model.params.q_scale)),
            },
        }
        opt_path = output_root / f"opt_result_{scenario}.json"
        with open(opt_path, "w", encoding="utf-8-sig") as f:
            json.dump(opt_result, f, indent=2, ensure_ascii=False)

    # Layer 4: Optimization history (convergence curve)
    if history is not None:
        hist_path = output_root / f"opt_history_{scenario}.csv"
        history.to_csv(hist_path, index=False, encoding="utf-8-sig")

    # Layer 5: Summary statistics
    if summary is not None:
        summary_df = pd.DataFrame([summary])
        summ_path = output_root / f"summary_{scenario}.csv"
        summary_df.to_csv(summ_path, index=False, encoding="utf-8-sig")

    # Layer 6: Model configuration (for reproducibility)
    config = {
        "scenario": scenario,
        "model_type": "LayeredSVEIQR",
        "populations": {
            "students": float(model.group_sizes["s"]),
            "teachers": float(model.group_sizes["t"]),
            "staff": float(model.group_sizes["l"]),
        },
        "parameters": {
            "beta0": float(model.params.beta0),
            "sigma": float(model.params.sigma),
            "gamma": float(model.params.gamma),
            "q_rate": float(model.params.q_rate),
            "season_start_day": float(model.params.season_start_day),
            "season_amp": float(model.params.season_amp),
            "season_phase": float(model.params.season_phase),
            "seed_s": float(model.params.seed_s),
            "seed_t": float(model.params.seed_t),
            "seed_l": float(model.params.seed_l),
        },
        "vaccine_coverage": {
            "students": float(model.params.v_cov_s),
            "teachers": float(model.params.v_cov_t),
            "staff": float(model.params.v_cov_l),
        },
        "place_weights": {k: float(v) for k, v in model.place_weights.items()},
    }
    cfg_path = output_root / f"config_{scenario}.json"
    with open(cfg_path, "w", encoding="utf-8-sig") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Layer 7: Baseline vs Optimized comparison (saved if history provided)
    if history is not None and len(history) > 0:
        compare_data = {
            "scenario": scenario,
            "initial_J": (
                float(history["J"].iloc[0]) if "J" in history.columns else np.nan
            ),
            "final_J": (
                float(history["J"].iloc[-1]) if "J" in history.columns else np.nan
            ),
            "improvements": {
                "A": (
                    float(history["A"].iloc[0] - history["A"].iloc[-1])
                    if "A" in history.columns
                    else np.nan
                ),
                "C": (
                    float(history["C"].iloc[0] - history["C"].iloc[-1])
                    if "C" in history.columns
                    else np.nan
                ),
                "D": (
                    float(history["D"].iloc[0] - history["D"].iloc[-1])
                    if "D" in history.columns
                    else np.nan
                ),
            },
        }
        history.to_csv(
            output_root / f"compare_{scenario}.csv", index=False, encoding="utf-8-sig"
        )

    # Layer 8: Sensitivity analysis results (if computed)
    # This will be saved separately when running sensitivity analysis

    # Layer 9: R0 and Re results (compute here if needed)
    t_eval = trajectory.index.to_numpy(dtype=float)
    r0 = effective_reproduction_number(model, t=t_eval[0], state=model.initial_state())

    reff_records = []
    for t_day in t_eval[:: max(1, len(t_eval) // 100)]:  # Sample ~ 100 points
        y_t = trajectory.loc[t_day].to_dict() if t_day in trajectory.index else {}
        state_t = {
            k: float(v) if not np.isnan(float(v)) else 0.0 for k, v in y_t.items()
        }
        r_eff = effective_reproduction_number(model, t=float(t_day), state=state_t)
        reff_records.append({"day": float(t_day), "R0": r0, "R_eff": r_eff})

    reff_df = pd.DataFrame(reff_records)
    re_path = output_root / f"reff_{scenario}.csv"
    reff_df.to_csv(re_path, index=False, encoding="utf-8-sig")

    return output_root


def _effective_q_rate(params: CampusLayeredParams, i_ratio: float) -> float:
    q_scale_nonneg = max(float(params.q_scale), 0.0)
    q_sat_k = max(float(params.q_sat_k), 1e-12)
    q_response = q_scale_nonneg / (q_sat_k + q_scale_nonneg)
    q_rate_cmd = float(params.q_rate) * (1.0 + float(params.q_gain_max) * q_response)
    q_rate_eff = q_rate_cmd / (
        1.0 + float(params.q_capacity_alpha) * max(float(i_ratio), 0.0)
    )
    return float(min(max(q_rate_eff, 0.0), float(params.q_rate_cap)))


def effective_reproduction_number(
    model: CampusLayeredSVEIQR,
    t: float = 0.0,
    state: Mapping[str, float] | None = None,
) -> float:
    if state is None:
        state = model.initial_state()

    params = model.params
    infectious_total = float(sum(float(state.get(f"I_{g}", 0.0)) for g in GROUPS))
    i_ratio = infectious_total / max(model.total_population(), 1e-12)
    q_eff = _effective_q_rate(params, i_ratio)
    removal = max(float(params.gamma) + q_eff, 1e-12)

    ng = len(GROUPS)
    ngm = np.zeros((ng, ng), dtype=float)

    for g_idx, g in enumerate(GROUPS):
        s_eff = float(state.get(f"S_{g}", 0.0)) + (
            1.0 - model.vaccine_effect(g)
        ) * float(state.get(f"V_{g}", 0.0))

        for h_idx, h in enumerate(GROUPS):
            kernel = 0.0
            for place in PLACES:
                beta_k = model.adjusted_place_beta(place, t)
                w_k = model.place_weight(place, t)
                c_gh = float(model.contact_matrices[place][g_idx, h_idx])
                nh = max(float(model.group_sizes[h]), 1e-12)
                kernel += w_k * beta_k * c_gh / nh
            ngm[g_idx, h_idx] = (s_eff * kernel) / removal

    eigvals = np.linalg.eigvals(ngm)
    return float(np.max(np.real(eigvals)))


def _lhs_unit(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    samples = np.zeros((n_samples, n_dim), dtype=float)
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    for idx in range(n_dim):
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        samples[:, idx] = u
    return samples


def _ols_residuals(target: np.ndarray, predictors: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=float)
    x = np.asarray(predictors, dtype=float)
    if x.size == 0:
        return y - np.mean(y)
    x_design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    return y - x_design @ coef


def _compute_prcc_table(samples_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    param_cols = [col for col in samples_df.columns if col != target_col]
    x = samples_df[param_cols].to_numpy(dtype=float)
    y = samples_df[target_col].to_numpy(dtype=float)

    x_rank = np.column_stack([rankdata(x[:, idx]) for idx in range(x.shape[1])])
    y_rank = rankdata(y)

    records: List[Dict[str, float | str]] = []
    n = x_rank.shape[0]
    p = x_rank.shape[1]

    for idx, name in enumerate(param_cols):
        z = np.delete(x_rank, idx, axis=1)
        rx = _ols_residuals(x_rank[:, idx], z)
        ry = _ols_residuals(y_rank, z)

        denom = float(np.std(rx) * np.std(ry))
        prcc = float(np.corrcoef(rx, ry)[0, 1]) if denom > 0 else np.nan

        k = p - 1
        df = max(n - k - 2, 1)
        if np.isfinite(prcc) and abs(prcc) < 1.0:
            t_val = abs(prcc) * np.sqrt(df / max(1e-12, 1.0 - prcc * prcc))
            p_value = float(2.0 * (1.0 - student_t.cdf(t_val, df)))
        elif np.isfinite(prcc):
            p_value = 0.0
        else:
            p_value = np.nan

        records.append(
            {
                "parameter": name,
                "prcc": prcc,
                "abs_prcc": float(abs(prcc)) if np.isfinite(prcc) else np.nan,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(records).sort_values("abs_prcc", ascending=False)


def global_sensitivity_analysis(
    model: CampusLayeredSVEIQR,
    n_samples: int = 300,
    t_eval: float = 0.0,
    seed: int = 42,
    param_ranges: Mapping[str, Tuple[float, float]] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if n_samples < 20:
        raise ValueError("n_samples must be >= 20 for stable global sensitivity")

    if param_ranges is None:
        param_ranges = {
            "beta0": (0.13, 0.38),
            "gamma": (0.10, 0.30),
            "q_rate": (0.08, 0.30),
            "q_scale": (0.8, 3.0),
            "ve_s": (0.25, 0.60),
            "ve_t": (0.20, 0.55),
            "ve_l": (0.20, 0.55),
            "v_cov_s": (0.05, 0.60),
            "v_cov_t": (0.05, 0.60),
            "v_cov_l": (0.05, 0.60),
            "season_amp": (0.10, 0.35),
            "mask_u": (0.0, 1.0),
            "vent_u": (0.0, 1.0),
            "online_u": (0.0, 1.0),
            "club_u": (0.0, 1.0),
            "disinfect_u": (0.0, 1.0),
        }

    param_names = list(param_ranges.keys())
    rng = np.random.default_rng(seed)
    unit = _lhs_unit(int(n_samples), len(param_names), rng)
    low = np.array([float(param_ranges[name][0]) for name in param_names], dtype=float)
    high = np.array([float(param_ranges[name][1]) for name in param_names], dtype=float)
    samples = low + (high - low) * unit

    rows: List[Dict[str, float]] = []
    for idx in range(samples.shape[0]):
        params_i = replace(model.params)
        row: Dict[str, float] = {}
        for j, name in enumerate(param_names):
            value = float(samples[idx, j])
            setattr(params_i, name, value)
            row[name] = value

        model_i = CampusLayeredSVEIQR(
            group_sizes=model.group_sizes,
            contact_matrices=model.contact_matrices,
            place_weights=model.place_weights,
            params=params_i,
        )
        row["R_eff"] = effective_reproduction_number(model_i, t=t_eval, state=None)
        rows.append(row)

    samples_df = pd.DataFrame(rows)
    prcc_df = _compute_prcc_table(samples_df, target_col="R_eff")
    return samples_df, prcc_df


def run_global_sensitivity_demo(
    output_dir: str | Path | None = None,
    n_samples: int = 300,
    season_profile: str | None = None,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    model = build_scenario_model("sporadic", season_profile=season_profile)

    r_eff_t0 = effective_reproduction_number(model, t=0.0, state=None)
    samples_df, prcc_df = global_sensitivity_analysis(
        model=model,
        n_samples=n_samples,
        t_eval=0.0,
        seed=42,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"R_eff_t0": r_eff_t0}]).to_csv(
            output_dir / "layered_sveiqr_reff_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        samples_df.to_csv(
            output_dir / "layered_sveiqr_global_sensitivity_samples.csv",
            index=False,
            encoding="utf-8-sig",
        )
        prcc_df.to_csv(
            output_dir / "layered_sveiqr_global_sensitivity_prcc.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return r_eff_t0, samples_df, prcc_df


# 冬春高发季：season_profile="winter_spring_peak"
# 非高发季：season_profile="off_season"
def run_demo(
    output_dir: str | Path | None = None,
    n_days: int | None = None,
    season_profile: str | None = None,
    scenario: str = "sporadic",
) -> Tuple[CampusLayeredSVEIQR, pd.DataFrame, Dict[str, float]]:
    np.random.seed(42)  # Fix random seed for reproducibility
    model = build_scenario_model("sporadic", season_profile=season_profile)

    trajectory = model.solve(n_days=n_days)
    summary = model.summary(trajectory)

    if output_dir is not None:
        output_dir = Path(output_dir)
        # Save paper-grade outputs
        save_paper_grade_outputs(
            model=model,
            trajectory=trajectory,
            scenario=scenario,
            output_root=output_dir,
            summary=summary,
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
    title_suffix: str = "Sporadic",
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
    """输出 sporadic 场景曲线与汇总表。"""

    model = build_scenario_model("sporadic")
    traj = model.solve(n_days=None)
    results = [{"scenario": "sporadic", **model.summary(traj)}]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    axes[0].plot(traj.index, traj["I"], linewidth=2.0, label="Sporadic")
    new_exp = -(traj["S"].diff().fillna(0.0) + traj["V"].diff().fillna(0.0))
    new_exp = new_exp.clip(lower=0.0)
    axes[1].plot(traj.index, new_exp, linewidth=2.0, label="Sporadic")

    axes[0].set_title("Total Infectious (Sporadic)")
    axes[0].set_xlabel("Day")
    axes[0].set_ylabel("Infectious population")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Daily New Exposures (Sporadic)")
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
    import sys

    # Fix global random seed for reproducibility
    np.random.seed(42)

    out_dir = Path(__file__).resolve().parent
    scenario = "sporadic"  # This script is for sporadic scenario

    # Parse command-line arguments for quick shortcuts
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "demo":
            print(f"Running baseline demo ({scenario} scenario)...")
            model, traj, summ = run_demo(
                output_dir=out_dir,
                n_days=None,
                scenario=scenario,
            )
            print(f"  Attack rate: {summ.get('attack_rate', np.nan):.4f}")
            print(f"  Peak I: {summ.get('peak_I', np.nan):.1f}")
            sys.exit(0)
        elif cmd == "optim":
            print(f"Running optimization ({scenario} scenario)...")
            model, traj, summ = run_optimization_demo(
                output_dir=out_dir,
                n_days=None,
                scenario=scenario,
            )
            print(f"  Objective J: {summ.get('J', np.nan):.6f}")
            print(f"  Attack rate: {summ.get('attack_rate', np.nan):.4f}")
            sys.exit(0)
        elif cmd == "sensitivity":
            print(f"Running sensitivity analysis ({scenario} scenario)...")
            run_global_sensitivity_demo(
                output_dir=out_dir,
                n_samples=300,
            )
            sys.exit(0)

    # Default: Run full pipeline
    print(f"[{scenario.upper()}] Running intervention optimization...")
    opt_model, opt_trajectory, opt_summary = run_optimization_demo(
        output_dir=out_dir,
        n_days=None,
        scenario=scenario,
    )

    print("\n===== Optimization Summary =====")
    for key in [
        "J",
        "A",
        "C",
        "D",
        "attack_rate",
        "peak_I",
        "mask_u",
        "vent_u",
        "online_u",
        "club_u",
        "disinfect_u",
        "vax_cov_scale",
        "q_scale",
        "global_success",
        "global_nit",
        "global_fun",
        "optimizer_success",
        "optimizer_nit",
        "optimizer_fun",
    ]:
        if key in opt_summary:
            print(f"  {key}: {opt_summary[key]:.6f}")

    print("\nGenerating optimized figures...")
    plot_detailed_results(
        model=opt_model,
        trajectory=opt_trajectory,
        output_dir=out_dir,
        title_suffix="Optimized",
    )

    print("\nRunning R_eff and global sensitivity analysis...")
    r_eff_t0, _, prcc_df = run_global_sensitivity_demo(
        output_dir=out_dir,
        n_samples=300,
    )
    print(f"  R_eff_t0: {r_eff_t0:.6f}")
    print("  Top PRCC parameters:")
    for _, row in prcc_df.head(5).iterrows():
        print(
            f"    {row['parameter']}: PRCC={row['prcc']:.4f}, "
            f"p={row['p_value']:.4g}"
        )

    print(f"\n✅ Paper-grade outputs saved to: {out_dir}")
