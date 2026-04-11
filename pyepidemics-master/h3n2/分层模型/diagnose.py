"""诊断脚本：分析优化结果全为0的根本原因"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\HP\OneDrive\Desktop\学校\课程\专业课\数学建模\课程项目\Mathematical-Modeling-Course-Project\pyepidemics-master\h3n2\分层模型\常态')
from 分层模型_常态 import *

# 构建基线模型
model = build_baseline_model()
anchor = load_shanghai_seir_anchor()
model.params.beta0 = anchor['beta']
model.params.sigma = anchor['sigma']
model.params.gamma = anchor['gamma']

traj = model.solve(n_days=220)
summary = model.summary(traj)
n0 = traj['N'].iloc[0]

print('=== 无防控基线结果 ===')
print(f'总人数: {n0:.0f}')
print(f'攻击率: {summary["attack_rate"]:.4f}')
print(f'峰值感染日: {summary["peak_I_day"]:.1f}')
print(f'峰值感染人数: {summary["peak_I"]:.1f}')
print(f'最终康复: {summary["final_R"]:.1f}')

# 各防控措施单独最大效果
print('\n=== 各防控措施单独最大效果(=1.0) ===')
for ctrl_name in CONTROL_NAMES:
    controls = {n: 0.0 for n in CONTROL_NAMES}
    controls[ctrl_name] = 1.0
    metrics = evaluate_controls(controls, n_days=220)
    ar = metrics["attack_rate"]
    pi = metrics["peak_I"]
    j_val = metrics["J"]
    a_val = metrics["A"]
    c_val = metrics["C"]
    d_val = metrics["D"]
    print(f'  {ctrl_name}=1.0: attack_rate={ar:.4f}, peak_I={pi:.1f}, A={a_val:.6f}, C={c_val:.6f}, D={d_val:.6f}, J={j_val:.6f}')

# 全防控效果
print('\n=== 全部最大防控(全部=1.0) ===')
controls = {n: 1.0 for n in CONTROL_NAMES}
metrics = evaluate_controls(controls, n_days=220)
print(f'  attack_rate={metrics["attack_rate"]:.4f}, peak_I={metrics["peak_I"]:.1f}')
print(f'  A={metrics["A"]:.6f}, C={metrics["C"]:.6f}, D={metrics["D"]:.6f}, J={metrics["J"]:.6f}')

# 分析目标函数分量
ow = ObjectiveWeights()
cw = CostWeights()
dw = DisruptionWeights()
print(f'\n=== 常态权重 ===')
print(f'  w1(感染)={ow.w1}, w2(成本)={ow.w2}, w3(教学损失)={ow.w3}')
print(f'  alpha1={ow.alpha1}, alpha2={ow.alpha2}')
print(f'  c_m={cw.c_m}, c_v={cw.c_v}, c_o={cw.c_o}, c_c={cw.c_c}, c_d={cw.c_d}')
print(f'  d_o={dw.d_o}, d_c={dw.d_c}')

# 无防控 vs 全防控 vs 仅线上
controls0 = {n: 0.0 for n in CONTROL_NAMES}
m0 = evaluate_controls(controls0, n_days=220)
print(f'\n无防控: A={m0["A"]:.6f}, C={m0["C"]:.6f}, D={m0["D"]:.6f}, J={m0["J"]:.6f}')
print(f'  attack_rate={m0["attack_rate"]:.6f}, peak_I_norm={m0["peak_I_norm"]:.6f}')

controls_on = {n: 0.0 for n in CONTROL_NAMES}
controls_on['online_u'] = 1.0
mo = evaluate_controls(controls_on, n_days=220)
print(f'仅线上: A={mo["A"]:.6f}, C={mo["C"]:.6f}, D={mo["D"]:.6f}, J={mo["J"]:.6f}')
print(f'  attack_rate={mo["attack_rate"]:.6f}, peak_I_norm={mo["peak_I_norm"]:.6f}')

# 关键：防控措施能否改变攻击率？
print('\n=== 防控措施对攻击率的边际效果 ===')
for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    controls = {n: level for n in CONTROL_NAMES}
    metrics = evaluate_controls(controls, n_days=220)
    print(f'  所有措施={level:.1f}: attack_rate={metrics["attack_rate"]:.4f}, peak_I={metrics["peak_I"]:.1f}, A={metrics["A"]:.6f}, C={metrics["C"]:.6f}, D={metrics["D"]:.6f}, J={metrics["J"]:.6f}')

# 计算 beta0 的有效 R0
print('\n=== 有效再生数估算 ===')
# lambda_g(t=0) = sum_k w_k * beta_k(0) * sum_h C[g,h] * I_h / N_h
# 有效 R0 ≈ beta0 / gamma * 加权接触数
for place in PLACES:
    w = model.place_weights[place]
    C = model.contact_matrices[place]
    # 主要传播来自学生->学生
    c_ss = C[0, 0]
    print(f'  {place}: weight={w:.3f}, C_s_s={c_ss:.1f}, w*C_s_s={w*c_ss:.3f}')
total_eff = sum(model.place_weights[p] * model.contact_matrices[p][0,0] for p in PLACES)
R0_est = model.params.beta0 / model.params.gamma * total_eff
print(f'  估算 R0 ≈ beta0/gamma * sum(w*C_s_s) = {model.params.beta0}/{model.params.gamma} * {total_eff:.3f} = {R0_est:.2f}')
