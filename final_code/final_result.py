import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import math

# ==========================================
# 0. 全局物理常数
# ==========================================
R_E = 6371e3  # 地球半径 (米)
GM = 3.986004418e14  # 地球引力参数 (m^3/s^2)


# ==========================================
# 1. 计算最大接入角 phi_max
# ==========================================
def calculate_phi_max(R, h, gamma_deg):
    """计算最大访问角 phi_max (公式 1)"""
    print(f"gamma_deg={gamma_deg}")
    gamma_rad = np.radians(gamma_deg)
    gamma_0 = np.arcsin(R / (R + h)) * 2
    print(f"gamma_0={np.degrees(gamma_0)}")
    if gamma_0 > gamma_rad:
        sin_arg = (R + h) / R * np.sin(gamma_rad / 2)
        phi_max = np.arcsin(sin_arg) - gamma_rad / 2
    else:
        phi_max = np.arccos(R / (R + h))
    # 当 h_km 过低时可能发生，确保返回地球视界角
    return phi_max


# ==========================================
# 2. 平均接入时间 (使用双重积分)
# ==========================================
def calculate_average_access_time(R, h, GM, phi_max, N):
    """
    计算平均接入时间 T_bar。
    通过对 theta 进行关于 phi 和 delta 的双重积分求期望。
    """
    # 轨道周期相关的系数
    T_coeff = np.sqrt((R + h) ** 3 / GM)

    # 辅助函数: 根据球面几何计算中心角 theta
    def get_theta(delta, phi):
        arg = np.cos(phi_max) * np.cos(phi) + np.sin(phi_max) * np.sin(phi) * np.cos(delta)
        arg = np.clip(arg, -1.0, 1.0)  # 数值保护
        return np.arccos(arg)

    # 1. 内层积分 (对 delta: 0 -> 2pi)
    def inner_integral(phi):
        # 被积函数: theta(delta, phi) * f_delta(delta)
        # 其中 f_delta = 1 / (2*pi)
        func = lambda delta: get_theta(delta, phi) * (1.0 / (2 * np.pi))
        res, _ = quad(func, 0, 2 * np.pi)
        return res

    # 2. 最近卫星接入角 PDF
    def pdf_phi_0(phi, N):
        """
        最近卫星接入角的概率密度函数 f(phi)
        """
        return (N / 2) * np.sin(phi) * np.exp(-(N / 2) * (1 - np.cos(phi)))

    # 3. 外层积分 (对 phi: 0 -> phi_max)
    def outer_integral(phi):
        # 被积函数: E[theta|phi] * f_phi(phi)
        return inner_integral(phi) * pdf_phi_0(phi, N)

    result, error = quad(outer_integral, 0, phi_max)
    T_average = T_coeff * result
    print(f"T_average={T_average}")
    return T_average


# ==========================================
# 4. 几何覆盖面积 S
# ==========================================
def calculate_S(R1, R2, R_search):
    """
    计算球冠表面积。
    R1: 被搜索层轨道半径 (计算层)
    R2: 搜索发起层轨道半径 (感知层)
    """
    # 几何约束项: R_search^2 - (R2 - R1)^2
    term = R_search ** 2 - (R2 - R1) ** 2
    if term < 0:
        return 0.0  # 搜索半径小于层间距，无法覆盖任何区域
    return (np.pi * R1 / R2) * term


# ==========================================
# 5. MPPP 稀疏化与有效卫星数 N_eff
# ==========================================
def calculate_p_available(lambda_task, mu, T_pass_avg):
    """
    计算卫星空闲概率 (Thinning Probability)。
    注意: 所有时间单位必须统一 (秒)。
    """
    if mu == 0: return 0.0
    rho = lambda_task / mu

    # 排队论稳定性检查: 如果 lambda >= mu，系统过载，概率为0
    if rho >= 1.0:
        return 0.0

    # 指数项: -(mu - lambda) * (T_pass / 5)
    exponent = -(mu - lambda_task) * (T_pass_avg / 5.0)
    p = 1 - rho * np.exp(exponent)
    return max(0.0, p)


def calculate_N_eff(lambda_eff, S):
    """
    计算覆盖区域内的预期卫星数 (向下取整)。
    """
    val = lambda_eff * S
    return math.floor(val + 1e-9)  # 加微小值防止浮点精度误差


# ==========================================
# 6. Amdahl 加速比
# ==========================================
def amdahl_speedup(f, N_eff):
    """
    计算并行加速比。
    f: 串行比例 (0-1)
    """
    if N_eff < 1:
        return 0.0
    return 1.0 / (f + (1.0 - f) / N_eff)


# ==========================================
# 7. 反解最小卫星数 N_min
# ==========================================
def calculate_min_satellites(f, lambda_task, mu, tau, P_task):
    """
    根据目标概率 P_task 反解所需的卫星数量 N。
    """
    if P_task >= 1.0: return float('inf')
    if mu <= lambda_task: return float('inf')  # 系统不稳

    # 公式推导的中间项 X
    ln_term = np.log(1 - P_task)
    # 注意: 这里的 tau 单位必须是秒
    term_X = (lambda_task / mu) - (1.0 / (mu * tau)) * ln_term

    denominator = 1 - f * term_X
    if denominator <= 0:
        return float('inf')  # 需要无限个卫星

    numerator = (1 - f) * term_X
    N_val = numerator / denominator

    if N_val <= 0: return 1  # 至少需要1颗
    return math.ceil(N_val)


# ==========================================
# 8. 计算搜索半径 R_search
# ==========================================
def calculate_search_radius(N_min, R1, R2, lambda_eff):
    """
    根据所需卫星数 N_min 反推搜索半径。
    """
    if lambda_eff <= 0:
        return float('inf')  # 密度为0，半径无穷大

    term1 = (N_min * R2) / (lambda_eff * np.pi * R1)
    term2 = (R2 - R1) ** 2
    return np.sqrt(term1 + term2)


def plot_results(N_sen, N_com, h_compute, h_sensing, gamma_beam_deg, tau_factor, lambda_task, mu_service, f, P):
    # [步骤 1] 计算最大接入角 phi_max
    phi_m = calculate_phi_max(R_E, h_sensing, gamma_beam_deg)
    print(f"\n[1] 最大接入角 (phi_max):{np.degrees(phi_m):.2f}度")

    # [步骤 2] 计算平均接入时间
    avg_time = calculate_average_access_time(R_E, h_sensing, GM, phi_m, N_sen)
    print(f"[2] 平均接入时间: {avg_time:.4f} 秒")
    # [步骤 2.5] 确定截止时间 tau (根据用户设定的系数)
    tau_deadline = tau_factor * avg_time
    print(f"    -> 设定截止时间 (tau): {tau_deadline:.4f} 秒 (平均时间的 {tau_factor} 倍)")
    # [步骤 3] 计算 MPPP 可用性
    p_avail = calculate_p_available(lambda_task, mu_service, avg_time)
    lambda_real = N_com / (4 * np.pi * (R_E + h_compute) ** 2)
    print(f"    实际卫星密度: {lambda_real:.2e} sat/m^2")
    lambda_eff = lambda_real * p_avail
    print(f"[3] 卫星可用概率: {p_avail:.4f}")
    print(f"    有效卫星密度: {lambda_eff:.2e} sat/m^2")
    # [步骤 4] 反解所需最少卫星数 N_min
    N_min = calculate_min_satellites(f, lambda_task, mu_service, tau_deadline, P)
    print(f"[4] 满足条件所需最少卫星数: {N_min}")
    # [步骤 5] 计算搜索半径 R_search
    R_comp = R_E + h_compute
    R_sens = R_E + h_sensing
    R_s = calculate_search_radius(N_min, R_comp, R_sens, lambda_eff)
    if math.isinf(R_s):
        print(f"[5] 搜索半径: 无穷大 (Infinity)")
        print("原因: 有效密度过低或所需卫星数过多，无法在有限半径内找到足够卫星。")
    else:
        print(f"[5] 搜索半径: {R_s / 1000:.2f} km")

    return R_s, N_min


# -------------------------------------------------
# A. 参数配置区
# -------------------------------------------------

# 1. 轨道参数
h_compute = 550e3  # 计算层高度 (米)
h_sensing = 1000e3  # 感知层高度 (米)

# 2. 波束参数
gamma_list = np.linspace(80, 120, 40)
# 3. 卫星总数 (用于密度计算)
N_total_sensing = 300  # 感知层总星数 (用于接入角分布)
N_total_compute = 1000  # 计算层总星数 (用于计算资源密度)

# 4. 任务与排队参数 (输入您指定的数值)
input_lambda = 5.0  # 任务到达率数值
input_mu = 10.0  # 任务完成率数值
# 单位换算 (统一转换为 "秒")
# 1小时 = 3600秒
lambda_task = input_lambda / 3600.0
mu_service = input_mu / 3600.0

# 5. 时间限制系数 (截止时间 = 系数 * 平均接入时间)
tau_factor = 0.9

target_P = 0.9  # 目标完成率
# -------------------------------------------------
# result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, gamma_beam_deg, tau_factor, lambda_task,mu_service, param_f, target_P)
# print("搜索半径:", result[0], "km")
# print("所需卫星数:", result[1])
r_result = []
r_result2 = []
r_result3 = []
n_result = []
n_result2 = []
n_result3 = []
param_f = 0.05  # 串行比例 (10%)
param_f2 = 0.15
param_f3 = 0.1

for i in gamma_list:
    result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, i, tau_factor, lambda_task,
                          mu_service, param_f, target_P)
    r_result.append(result[0])
    n_result.append(result[1])
    result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, i, tau_factor, lambda_task,
                          mu_service, param_f2, target_P)
    r_result2.append(result[0])
    n_result2.append(result[1])
    result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, i, tau_factor, lambda_task,
                          mu_service, param_f3, target_P)
    r_result3.append(result[0])
    n_result3.append(result[1])

plt.figure(1)
plt.plot(gamma_list, r_result, label=f'f={param_f}')
plt.plot(gamma_list, r_result2, label=f'f={param_f2}')
plt.plot(gamma_list, r_result3, label=f'f={param_f3}')
plt.legend()
# 只要设定上限，下限自动
plt.ylim(top=2 * r_result3[0])
plt.xlabel(fr'Beam Width ($^\circ$)')
plt.ylabel('Search Radius (m)')
plt.grid()
plt.figure(2)
plt.plot(gamma_list, n_result, label=f'f={param_f}')
plt.plot(gamma_list, n_result2, label=f'f={param_f2}')
plt.plot(gamma_list, n_result3, label=f'f={param_f3}')
plt.legend()
plt.ylim(0, top=2 * n_result3[0])
plt.xlabel(fr'Beam Width ($^\circ$)')
plt.ylabel('Number of Satellites')
plt.grid()
# plt.show()

f_list = np.linspace(0.1, 0.15, 30)
tau_factor_1 = 0.85
tau_factor_2 = 0.9
tau_factor_3 = 0.95
gamma_beam_deg = 90.0  # 波束宽度 (度)
n_result4 = []
n_result5 = []
n_result6 = []
for f in f_list:
    result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, gamma_beam_deg, tau_factor_1,
                          lambda_task,
                          mu_service, f, target_P)
    # r_result.append(result[0])
    n_result4.append(result[1])
    result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, gamma_beam_deg, tau_factor_2,
                          lambda_task,
                          mu_service, f, target_P)
    n_result5.append(result[1])
    result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, gamma_beam_deg, tau_factor_3,
                          lambda_task,
                          mu_service, f, target_P)
    n_result6.append(result[1])
plt.figure(3)
plt.plot(f_list, n_result4, label=fr'$\tau$={tau_factor_1}')
plt.plot(f_list, n_result5, label=fr'$\tau$={tau_factor_2}')
plt.plot(f_list, n_result6, label=fr'$\tau$={tau_factor_3}')
plt.legend()
# plt.ylim(0, top=2 * n_result6[0])
plt.xlabel(fr'Serial Ratio ($f$)')
plt.ylabel('Number of Satellites')
plt.grid()
plt.show()
