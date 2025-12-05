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
    print(f"phi_max={np.degrees(phi_max)}")
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
def calculate_search_radius(N_min, R1, R2, h1, h2, lambda_eff):
    """
    根据所需卫星数 N_min 反推搜索半径。
    """
    if lambda_eff <= 0:
        return float('inf')  # 密度为0，半径无穷大

    term1 = (N_min * R2) / (lambda_eff * np.pi * R1)
    term2 = (h2 - h1) ** 2
    return np.sqrt(term1 + term2)


# ==========================================
# 主函数: 参数配置与流程控制
# ==========================================
def main():
    # -------------------------------------------------
    # A. 参数配置区
    # -------------------------------------------------

    # 1. 轨道参数
    h_compute = 300e3  # 计算层高度 (米)
    h_sensing = 600e3  # 感知层高度 (米)

    # 2. 波束参数
    gamma_beam_deg = 130.0  # 波束宽度 (度)

    # 3. 卫星总数 (用于密度计算)
    N_total_sensing = 200  # 感知层总星数 (用于接入角分布)
    N_total_compute = 1000  # 计算层总星数 (用于计算资源密度)

    # 4. 任务与排队参数 (输入您指定的数值)
    input_lambda = 5.0  # 任务到达率数值
    unit_lambda = 'hour'  # 单位: 'hour' (个/小时)

    input_mu = 10.0  # 任务完成率数值
    unit_mu = 'hour'  # 单位: 'hour' (个/小时)

    # 时间限制系数 (截止时间 = 系数 * 平均接入时间)
    tau_factor = 0.9

    param_f = 0.1  # 串行比例 (10%)
    target_P = 0.90  # 目标完成率 (95%)
    # -------------------------------------------------

    # B. 自动单位换算 (统一转换为 "秒")
    # 1小时 = 3600秒
    lambda_task = input_lambda / 3600.0
    mu_service = input_mu / 3600.0

    print(f"--- 参数换算结果 ---")
    print(f"任务到达率 (λ): {input_lambda}/{unit_lambda} -> {lambda_task:.6f} 个/秒")
    print(f"任务服务率 (μ): {input_mu}/{unit_mu}    -> {mu_service:.6f} 个/秒")

    # 检查系统稳定性
    if lambda_task >= mu_service:
        print("\n[错误] 系统过载! 到达率 λ ({:.4f}) >= 服务率 μ ({:.4f})".format(lambda_task, mu_service))
        return

    # C. 几何参数计算
    R_comp = R_E + h_compute
    R_sens = R_E + h_sensing

    # 计算层原始密度
    lambda_real = N_total_compute / (4 * np.pi * R_comp ** 2)

    # -------------------------------------------------
    # D. 仿真执行流程
    # -------------------------------------------------

    # [步骤 1] 计算最大接入角 phi_max
    # 注意：这里会用到上面更新后的 gamma_0
    phi_m = calculate_phi_max(R_E, h_sensing, gamma_beam_deg)
    print(f"\n[1] 最大接入角 (phi_max):{np.degrees(phi_m):.2f}度")

    # [步骤 2] 计算平均接入时间
    avg_time = calculate_average_access_time(R_E, h_sensing, GM, phi_m, N_total_sensing)
    print(f"[2] 平均接入时间: {avg_time:.4f} 秒")

    # [步骤 2.5] 确定截止时间 tau (根据用户设定的系数)
    tau_deadline = tau_factor * avg_time
    print(f"    -> 设定截止时间 (tau): {tau_deadline:.4f} 秒 (平均时间的 {tau_factor} 倍)")

    # [步骤 3] 计算 MPPP 可用性
    p_avail = calculate_p_available(lambda_task, mu_service, avg_time)
    lambda_eff = lambda_real * p_avail
    print(f"[3] 卫星可用概率: {p_avail:.4f}")
    print(f"    有效卫星密度: {lambda_eff:.2e} sat/m^2")

    if p_avail == 0:
        print("警告: 可用概率为0，仿真提前结束。")
        return

    # [步骤 4] 反解所需最少卫星数 N_min
    N_min = calculate_min_satellites(param_f, lambda_task, mu_service, tau_deadline, target_P)
    print(f"[4] 满足条件所需最少卫星数: {N_min}")

    # [步骤 5] 计算搜索半径 R_search
    R_s = calculate_search_radius(N_min, R_comp, R_sens, h_compute, h_sensing, lambda_eff)

    # -------------------------------------------------
    # E. 结果验证
    # -------------------------------------------------
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


# 运行主程序
if __name__ == "__main__":
    main()
