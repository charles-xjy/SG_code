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

        # 验证计算
        S_final = calculate_S(R_comp, R_sens, R_s)
        N_check = calculate_N_eff(lambda_eff, S_final)

        print(f"\n[验证环节]")
        print(f"  -> 实际获取卫星数: {N_check}")

        speedup = amdahl_speedup(param_f, N_check)
        mu_new = mu_service * speedup

        final_prob = 0.0
        if mu_new > lambda_task:
            final_prob = 1 - np.exp(-(mu_new - lambda_task) * tau_deadline)
        print(f"  -> 最终任务完成率: {final_prob:.4f} (目标: {target_P})")

    return R_s, N_min


# -------------------------------------------------
# A. 参数配置区
# -------------------------------------------------

# 1. 轨道参数
h_compute = 300e3  # 计算层高度 (米)
h_sensing = 600e3  # 感知层高度 (米)

# 2. 波束参数
gamma_beam_deg = 110.0  # 波束宽度 (度)
gamma_list = np.linspace(110, 130, 20)
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

param_f = 0.1  # 串行比例 (10%)
target_P = 0.90  # 目标完成率 (95%)
# -------------------------------------------------
# result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, gamma_beam_deg, tau_factor, lambda_task,mu_service, param_f, target_P)
# print("搜索半径:", result[0], "km")
# print("所需卫星数:", result[1])
r_result = []
r_result2 = []
r_result3 =  []
n_result = []
n_result2 = []
n_result3 = []
for i in gamma_list:
    result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, i, tau_factor, lambda_task,
                          mu_service, param_f, target_P)
    r_result.append(result[0])
    n_result.append(result[1])
    result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, i, tau_factor, lambda_task,
                          mu_service, 0.05, target_P)
    r_result2.append(result[0])
    n_result2.append(result[1])
    result = plot_results(N_total_sensing, N_total_compute, h_compute, h_sensing, i, tau_factor, lambda_task,
                          mu_service, 0.15, target_P)
    r_result3.append(result[0])
    n_result3.append(result[1])


# plt.figure(1)
# plt.plot(gamma_list, r_result, label='f=0.1')
# plt.plot(gamma_list, r_result2, label='f=0.05')
# plt.plot(gamma_list, r_result3, label='f=0.15')
# plt.legend()
# plt.xlabel(fr'Beam Width ($^\circ$)')
# plt.ylabel('Search Radius (m)')
# plt.grid()
# plt.figure(2)
# plt.plot(gamma_list, n_result, label='f=0.1')
# plt.plot(gamma_list, n_result2, label='f=0.05')
# plt.plot(gamma_list, n_result3, label='f=0.15')
# plt.legend()
# plt.xlabel(fr'Beam Width ($^\circ$)')
# plt.ylabel('Number of Satellites')
# plt.grid()
# plt.show()





# ==============================================================================
#                 以下为蒙特卡洛验证模块 (直接粘贴在原代码后)
# ==============================================================================

print("\n" + "=" * 40)
print("   开始蒙特卡洛仿真 (Monte Carlo Verification)")
print("=" * 40)


# --- 1. 蒙特卡洛辅助函数 ---

def generate_random_satellites(R, N):
    """在半径为 R 的球面上均匀生成 N 个随机卫星坐标"""
    # 使用正态分布生成各向同性的向量，然后归一化
    points = np.random.normal(size=(N, 3))
    norms = np.linalg.norm(points, axis=1)
    return (points / norms[:, np.newaxis]) * R


def run_mc_point(R_search_theory, gamma_deg, params, trials=2000):
    """
    针对给定的 搜索半径(理论值) 和 波束角，
    运行 trials 次仿真，返回：
    1. 实际平均能找到的有效卫星数 (N_mc)
    2. 实际测得的任务成功率 (P_sim)
    """
    # 解包参数
    h_c, h_s = params['h_c'], params['h_s']
    N_tot_c, N_tot_s = params['N_tot_c'], params['N_tot_s']
    lam, mu, f = params['lambda'], params['mu'], params['f']
    tau_factor = params['tau_factor']

    R_c = R_E + h_c
    R_s_orbit = R_E + h_s

    # 1. 重算中间变量 (为了获取准确的 tau 和 p_avail)
    #    注意：必须调用您原有的函数来保持逻辑一致
    phi_m = calculate_phi_max(R_E, h_s, gamma_deg)
    T_avg = calculate_average_access_time(R_E, h_s, GM, phi_m, N_tot_s)
    tau_deadline = tau_factor * T_avg
    p_avail = calculate_p_available(lam, mu, T_avg)

    # 2. 仿真循环
    success_count = 0
    total_eff_sats = 0

    # 为了加速，假设观测卫星固定在北极点 (0, 0, R_s_orbit)
    sens_pos = np.array([0, 0, R_s_orbit])

    for _ in range(trials):
        # A. 生成计算层卫星分布
        sat_pos = generate_random_satellites(R_c, N_tot_c)

        # B. 几何筛选 (Distance & Angle)
        # B1. 距离判断
        dists = np.linalg.norm(sat_pos - sens_pos, axis=1)
        in_range = dists <= R_search_theory

        # B2. 角度判断 (波束限制)
        # 计算卫星相对于地心的纬度角，判断是否在 phi_max 内
        # 简单方法：计算 sat_pos 与 sens_pos 的地心夹角 theta
        # dot_product = z_coord * R_s_orbit (因为 sens 在 z 轴)
        # cos_theta = z / R_c
        cos_theta = sat_pos[:, 2] / R_c
        # 甚至不需要 arccos，直接比较 cos 值 (phi_max 越小，cos 越大)
        in_beam = cos_theta >= np.cos(phi_m)

        # 几何上可见的卫星索引
        candidates_idx = np.where(in_range & in_beam)[0]
        num_geo = len(candidates_idx)

        if num_geo == 0:
            continue  # 失败：无卫星覆盖

        # C. MPPP 稀疏化 (模拟忙闲状态)
        # 对这 num_geo 个卫星，每个以 p_avail 概率可用
        # 生成随机数对比
        rand_vals = np.random.rand(num_geo)
        num_eff = np.sum(rand_vals < p_avail)

        # 记录一下平均有效卫星数用于绘图
        total_eff_sats += num_eff

        if num_eff == 0:
            continue  # 失败：卫星都在忙

        # D. 并行计算加速 (Amdahl)
        speedup = amdahl_speedup(f, num_eff)
        mu_new = mu * speedup

        # E. 判定任务是否成功
        # 方式: 计算在 tau_deadline 内完成的概率 (减少方差)
        if mu_new > lam:
            prob_success = 1.0 - np.exp(-(mu_new - lam) * tau_deadline)
            # 这里我们累加概率，等价于做无穷多次伯努利实验的平均
            success_count += prob_success
        else:
            # 服务率低于到达率，不稳定，成功率趋近0(在有限时间内可能完成但概率极低)
            pass

    avg_P_sim = success_count / trials
    avg_N_mc = total_eff_sats / trials

    return avg_N_mc, avg_P_sim


# --- 2. 逆向求解 Wrapper ---

def find_radius_matching_target(target_P, gamma, params, R_guess):
    """
    逆向思路：
    我们不验证 P 是否达标，而是去寻找“蒙特卡洛仿真中达到 0.9 成功率”所需的真实半径 R。
    如果 R_mc 和 R_theory 重合，说明理论准确。
    """
    if math.isinf(R_guess) or R_guess == 0:
        return np.nan, np.nan

    # 二分查找范围
    low = R_guess * 0.5
    high = R_guess * 1.5

    best_R = np.nan
    best_N = np.nan
    min_diff = 1.0

    # 快速搜索 10 次
    for i in range(10):
        mid_R = (low + high) / 2
        N_mc, P_sim = run_mc_point(mid_R, gamma, params, trials=800)  # trials适中以平衡速度

        if abs(P_sim - target_P) < min_diff:
            min_diff = abs(P_sim - target_P)
            best_R = mid_R
            best_N = N_mc

        if P_sim < target_P:
            # 成功率不够，需要扩大半径
            low = mid_R
        else:
            high = mid_R

    return best_R, best_N


# --- 3. 执行验证循环 ---

# 封装参数
mc_params = {
    'h_c': h_compute, 'h_s': h_sensing,
    'N_tot_c': N_total_compute, 'N_tot_s': N_total_sensing,
    'lambda': lambda_task, 'mu': mu_service, 'f': param_f,  # 使用 f=0.1 那组
    'tau_factor': tau_factor
}

mc_r_results = []
mc_n_results = []

# 这里的 r_result 是您代码中计算出的 f=0.1 对应的理论半径列表
# 我们直接遍历 gamma_list 和 r_result
print(f"正在验证 f={param_f}, Target_P={target_P} 的曲线...")

for idx, gamma in enumerate(gamma_list):
    theory_R = r_result[idx]  # 获取您的理论结果

    # 运行蒙特卡洛寻找匹配半径
    # 注意：如果理论计算本身就是 inf，我们跳过
    if math.isinf(theory_R):
        mc_r_results.append(np.nan)
        mc_n_results.append(np.nan)
        print(f"Gamma={gamma:.1f}: Theory=Inf -> MC=Skip")
    else:
        # 使用“逆向求解”找到蒙特卡洛所需的半径
        # 解释：如果理论是对的，那么使得 MC=0.9 的半径应该等于理论半径
        val_R, val_N = find_radius_matching_target(target_P, gamma, mc_params, theory_R)
        mc_r_results.append(val_R)
        mc_n_results.append(val_N)
        print(f"Gamma={gamma:.1f}: Theory R={theory_R / 1000:.1f}km | MC R={val_R / 1000:.1f}km")

# --- 4. 绘图对比 ---

# 创建新图或叠加在原图上
# 为了清晰，我们重新画图，包含理论线和仿真点

plt.figure(figsize=(12, 5))

# 子图1：搜索半径
plt.subplot(1, 2, 1)
# 您的理论曲线 (f=0.1)
plt.plot(gamma_list, np.array(r_result) / 1000, 'b-', label='Theory (f=0.1)', linewidth=2)
# 蒙特卡洛验证点
plt.plot(gamma_list, np.array(mc_r_results) / 1000, 'ro', label='Monte Carlo Sim', markersize=6, fillstyle='none',
         markeredgewidth=2)

plt.xlabel(r'Beam Width ($^\circ$)')
plt.ylabel('Search Radius (km)')
plt.title('Validation: Search Radius')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 子图2：卫星数量
plt.subplot(1, 2, 2)
# 您的理论曲线 (f=0.1)
plt.plot(gamma_list, n_result, 'b-', label='Theory (f=0.1)', linewidth=2)
# 蒙特卡洛验证点
plt.plot(gamma_list, mc_n_results, 'ro', label='Monte Carlo Sim', markersize=6, fillstyle='none', markeredgewidth=2)

plt.xlabel(r'Beam Width ($^\circ$)')
plt.ylabel('Number of Satellites')
plt.title('Validation: Req. Satellites')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()