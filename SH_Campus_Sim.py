import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SHCampusSimulatorPro:
    def __init__(self, network_file, metadata_file):
        self.edges_orig = pd.read_csv(network_file)
        self.metadata = pd.read_csv(metadata_file)
        self.n = len(self.metadata)
        self.reset()

    def reset(self):
        """重置模拟状态"""
        self.status = np.zeros(self.n)
        # 初始化疫苗接种者 (S -> R)
        self.status[self.metadata['is_vaccinated'] == 1] = 3
        self.timer = np.zeros(self.n)
        self.active_edges = self.edges_orig.copy()

    def run_simulation(self, days=180, beta_base=0.08, sigma_inv=1.5, gamma_inv=4.0, 
                       trigger_threshold=0.01, interventions=None):
        """
        beta_base: 调低至0.08左右以延缓爆发时间，使峰值出现在40-60天
        trigger_threshold: 触发干预的感染比例阈值（如0.01代表1%学生感染时开启）
        """
        self.reset()
        history = {'S': [], 'E': [], 'I': [], 'R': []}
        
        # 初始感染者 (种子病例)
        patient_zero = np.random.choice(np.where(self.status==0)[0], 3)
        self.status[patient_zero] = 1
        
        # 记录干预是否已激活
        policy_activated = False

        for t in range(days):
            # 1. 计算当前感染比例
            current_inf_ratio = np.sum(self.status == 2) / self.n
            
            # 2. 动态触发逻辑：当感染人数达到阈值时开启干预
            current_beta_factor = 1.0
            if interventions and current_inf_ratio >= trigger_threshold:
                if not policy_activated:
                    # print(f"Day {t}: 感染率达到{current_inf_ratio:.2%}, 触发防控措施")
                    policy_activated = True
                
                # --- A. 参数级干预 (改变传播概率) ---
                if interventions.get('mask_on'):
                    current_beta_factor *= 0.65  # 强制口罩降低35%风险
                if interventions.get('ventilation'):
                    current_beta_factor *= 0.9   # 通风降低10%风险
                
                # --- B. 拓扑级干预 (改变网络连接) ---
                if interventions.get('online_classes'):
                    # 模拟线上授课：随机切断70%的社交边（保留宿舍等必要接触）
                    mask = np.random.rand(len(self.active_edges)) > 0.7
                    self.active_edges = self.active_edges[mask]
            
            # 3. 季节性因子 (正弦波动，对标钱晨嗣论文)
            # 假设 t=0 是11月初，t=60是1月初（高峰期）
            season_factor = 1.0 + 0.25 * np.cos(2 * np.pi * (t - 60) / 365)
            current_beta = beta_base * current_beta_factor * season_factor
            
            new_status = self.status.copy()
            
            # 4. 网络传播逻辑 (基于当前活跃边)
            infected_indices = np.where(self.status == 2)[0]
            if len(infected_indices) > 0:
                # 提取与感染者相关的接触者
                rel_edges = self.active_edges[
                    self.active_edges['source'].isin(infected_indices) | 
                    self.active_edges['target'].isin(infected_indices)
                ]
                
                # 获取潜在受害者名单
                targets = np.concatenate([rel_edges['source'].values, rel_edges['target'].values])
                unique_targets = np.unique(targets)
                
                for n in unique_targets:
                    if self.status[n] == 0:
                        # 简化计算：按接触频率和beta计算感染概率
                        if np.random.rand() < current_beta:
                            new_status[n] = 1
            
            # 5. 状态转移与免疫衰减 (R -> S 重复感染风险)
            for i in range(self.n):
                if self.status[i] == 1: # E -> I
                    self.timer[i] += 1
                    if self.timer[i] >= sigma_inv:
                        new_status[i] = 2
                        self.timer[i] = 0
                elif self.status[i] == 2: # I -> R
                    self.timer[i] += 1
                    if self.timer[i] >= gamma_inv:
                        new_status[i] = 3
                        self.timer[i] = 0
                elif self.status[i] == 3: # R -> S (模拟免疫失效)
                    if np.random.rand() < 0.001: # 极低概率重复感染
                        new_status[i] = 0
            
            self.status = new_status
            history['S'].append(np.sum(self.status == 0))
            history['E'].append(np.sum(self.status == 1))
            history['I'].append(np.sum(self.status == 2))
            history['R'].append(np.sum(self.status == 3))
            
        return pd.DataFrame(history)

# --- 绘图与对比分析 ---
if __name__ == "__main__":
    # 请确保已运行过 DataFactory.py 生成 csv
    sim = SHCampusSimulatorPro("campus_network.csv", "student_metadata.csv")
    
    # 场景1：自然传播 (Baseline)
    res_base = sim.run_simulation(days=150, beta_base=0.08, trigger_threshold=1.0) # 设高阈值不触发
    
    # 场景2：动态干预 (当感染达1%时开启口罩和通风)
    res_policy = sim.run_simulation(days=150, beta_base=0.08, trigger_threshold=0.01,
                                   interventions={'mask_on': True, 'ventilation': True})

    # 场景3：极强干预 (达1%时增加线上授课)
    res_strong = sim.run_simulation(days=150, beta_base=0.08, trigger_threshold=0.01,
                                    interventions={'mask_on': True, 'online_classes': True})

    plt.figure(figsize=(12, 6))
    plt.plot(res_base['I'], label='Scenario A: No Control (Baseline)', color='blue', linewidth=2)
    plt.plot(res_policy['I'], label='Scenario B: Masks & Vent (Trigger at 1%)', color='orange', linestyle='--')
    plt.plot(res_strong['I'], label='Scenario C: Online Classes (Trigger at 1%)', color='red')
    
    plt.axhline(y=sim.n * 0.01, color='gray', linestyle=':', label='Trigger Threshold (1%)')
    plt.title("H3N2 Campus Spread with Dynamic Intervention & Seasonality")
    plt.xlabel("Days (Starting from Nov.)")
    plt.ylabel("Active Infections (I)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()