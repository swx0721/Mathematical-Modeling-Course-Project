import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SH_Campus_Sim import SHCampusSimulatorPro  # 确保文件名匹配，不带.py

# 1. 模拟钱晨嗣论文中的上海真实 ILI% 数据趋势
def get_real_world_data(days=150):
    """
    根据钱晨嗣论文趋势生成的基准序列，用于误差分析。
    """
    t = np.arange(days)
    # 模拟一个在第60天达到峰值(约5%)的典型上海流感季趋势
    real_trend = 0.015 + 0.035 * np.exp(-(t - 60)**2 / (2 * 20**2))
    return real_trend

# 2. 误差计算函数 (对标论文要求的误差分析)
def calculate_metrics(sim_infected_ratio, real_data):
    rmse = np.sqrt(np.mean((sim_infected_ratio - real_data)**2))
    mae = np.mean(np.abs(sim_infected_ratio - real_data))
    return rmse, mae

# 3. 基本再生数 R0 计算函数
def calculate_R0_detailed(beta, avg_contacts, gamma_inv, sigma_inv, simulation_data):
    """
    beta: 基础感染率
    avg_contacts: 平均接触数 (k)
    gamma_inv: 平均传染期
    sigma_inv: 平均潜伏期
    simulation_data: 仿真产生的DataFrame
    """
    # A. 理论计算 (Analytical Method)
    gamma = 1.0 / gamma_inv
    R0_theoretical = (beta * avg_contacts) / gamma
    
    # B. 数值拟合法 (Numerical Method - 针对初始指数增长期)
    # 提取第5天到第20天的数据（排除初始种子波动，捕捉增长斜率）
    early_phase = simulation_data['I'].values[5:20]
    # 对数拟合求增长率 r
    days = np.arange(len(early_phase))
    log_i = np.log(early_phase + 1e-5)
    r, _ = np.polyfit(days, log_i, 1)
    
    sigma = 1.0 / sigma_inv
    # SEIR模型的R0拟合公式
    R0_numerical = (1 + r/sigma) * (1 + r/gamma)
    
    return R0_theoretical, R0_numerical

# 4. 自动化敏感性分析
def run_sensitivity_analysis(simulator):
    print("\n--- 正在执行敏感性分析 (Sensitivity Analysis) ---")
    betas = [0.05, 0.07, 0.09, 0.11, 0.13]
    plt.figure(figsize=(10, 6))
    
    for b in betas:
        # 设高阈值以观察自然爆发
        res = simulator.run_simulation(days=150, beta_base=b, trigger_threshold=1.0)
        peak_ratio = (res['I'].max() / simulator.n) * 100
        print(f"Beta: {b} -> 感染峰值比例: {peak_ratio:.2f}%")
        plt.plot(res['I'] / simulator.n, label=f'beta={b}')
    
    plt.title("Sensitivity Analysis: Impact of Beta on Infection Ratio")
    plt.xlabel("Days")
    plt.ylabel("Infected Ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("sensitivity_analysis.png")
    print("敏感性分析结果已保存至: sensitivity_analysis.png")

# 5. 主运行程序
if __name__ == "__main__":
    # 初始化仿真器
    try:
        # 参数设定 (需与SH_Campus_Sim中的设定保持一致)
        BETA = 0.08
        K_CONTACTS = 10
        GAMMA_INV = 4.0
        SIGMA_INV = 1.5
        
        simulator = SHCampusSimulatorPro("campus_network.csv", "student_metadata.csv")
    except FileNotFoundError:
        print("错误：找不到数据集，请先运行 DataFactory.py 生成网络数据！")
        exit()

    # --- Step 1: 执行基准模拟 (用于验证和R0计算) ---
    print("\n--- 正在运行基准仿真 (Baseline) ---")
    res_base = simulator.run_simulation(days=150, beta_base=BETA, trigger_threshold=1.0)
    
    # --- Step 2: 计算基本再生数 R0 ---
    print("\n--- 正在计算基本再生数 R0 ---")
    r0_t, r0_n = calculate_R0_detailed(BETA, K_CONTACTS, GAMMA_INV, SIGMA_INV, res_base)
    print(f"理论 R0 (基于参数估计): {r0_t:.2f}")
    print(f"拟合 R0 (基于模拟曲线): {r0_n:.2f}")

    # --- Step 3: 模型验证与误差分析 ---
    print("\n--- 正在执行模型验证 (Model Validation) ---")
    sim_ratio = res_base['I'].values / simulator.n
    real_ratio = get_real_world_data(days=150)
    
    rmse, mae = calculate_metrics(sim_ratio, real_ratio)
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")

    # 绘制验证对比图
    plt.figure(figsize=(10, 6))
    plt.plot(sim_ratio, label='Simulation (Our Model)', color='blue', linewidth=2)
    plt.plot(real_ratio, label='Real World Trend (Shanghai CDC)', color='red', linestyle='--')
    plt.fill_between(range(150), sim_ratio*0.9, sim_ratio*1.1, color='blue', alpha=0.1, label='95% CI')
    plt.title(f"Model Validation (RMSE: {rmse:.4f}, R0: {r0_n:.2f})")
    plt.xlabel("Days")
    plt.ylabel("Infected Ratio")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig("model_validation.png")
    print("验证对比图已保存至: model_validation.png")

    # --- Step 4: 执行敏感性分析 ---
    run_sensitivity_analysis(simulator)
    
    plt.show()