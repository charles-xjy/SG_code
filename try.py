import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from scipy.optimize import fsolve  # 用于求解 theta_A(t) 的非线性方程

# --- 1. 物理常量和系统参数 ---
G = 6.67430e-11
M_earth = 5.972e24
R = 6371.0e3  # Radius of Earth in meters (6371 km)


# --- 2. 核心几何函数 ---

def calculate_phi_max(h_km, gamma_deg):
    """计算最大访问角 phi_max (公式 1)"""
    h = h_km * 1000
    gamma_rad = np.radians(gamma_deg)
    try:
        sin_arg = (R + h) / R * np.sin(gamma_rad / 2)
        if sin_arg > 1:
            phi_max = np.arccos(R / (R + h))
        else:
            phi_max = np.arcsin(sin_arg) - gamma_rad / 2
        phi_max_h = np.arccos(R / (R + h))
        return min(phi_max, phi_max_h)
    except ValueError:
        return np.arccos(R / (R + h))


def T_factor_func(h_km):
    """轨道周期时间因子 T_factor = sqrt((R+h)^3 / (GM))"""
    h = h_km * 1000
    return np.sqrt(((R + h) ** 3) / (G * M_earth))


# --- Z. 自定义积分器 (纯 Python实现) ---

def custom_quad_integrate_pure_python(func, a, b, n=1000):
    """
    使用纯 Python 实现的梯形法则近似计算一重定积分。
    I = integral_a^b f(t) dt
    """
    if n < 2:
        return 0.0

    h_step = (b - a) / (n - 1)
    t_points = np.linspace(a, b, n)

    integral_sum = 0.0
    for i in range(n - 1):
        f_i = func(t_points[i])
        f_i_plus_1 = func(t_points[i + 1])
        integral_sum += (f_i + f_i_plus_1)

    final_result = integral_sum * (h_step / 2.0)
    return final_result


# --- 3. 理论任务完成概率 $P_{TC}$ (公式 16) ---

def theta_A_of_t_equation(theta_A, t, T_factor):
    """求解中心角 theta_A(t) 的非线性方程：t = T_factor * theta_A"""
    return T_factor * theta_A - t


def calculate_P_TC_theory_custom_int(h_km, N, gamma_deg, mu, n_points=1000):
    """
    计算理论任务完成概率 P_TC (公式 16)
    P_TC = integral_0^infinity (mu * exp(-mu * t)) * exp(-(N/2) * (1 - cos(theta_A(t)))) dt
    theta_A(t) 需要通过 fsolve 求解。
    """
    h = h_km * 1000
    phi_max = calculate_phi_max(h_km, gamma_deg)
    T_factor = T_factor_func(h_km)

    # 定义接入时间 CDF 的补函数 (Survival function)
    def P_access_success(t):
        if t <= 0:
            return 0.0

        # 求解 theta_A(t): 假设在 0 到 pi 之间
        # fsolve 可能会返回警告，此处选择忽略
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            try:
                # 初始猜测值为 t / T_factor (近似值)
                theta_A = fsolve(theta_A_of_t_equation, x0=t / T_factor, args=(t, T_factor))[0]

                # 确保 theta_A 在有效范围内 [0, phi_max]
                theta_A = np.clip(theta_A, 0, phi_max)

            except Exception:
                # 如果求解失败，可能是在 phi_max 以外
                if t > T_factor * phi_max:
                    theta_A = phi_max
                else:
                    return 0.0  # 理论上不应发生

        if theta_A >= phi_max:
            # 当 t >= T_factor * phi_max 时，1 - F_TA(t) = exp(-(N/2) * (1 - cos(phi_max)))
            cos_theta_val = np.cos(phi_max)
        else:
            cos_theta_val = np.cos(theta_A)

        # P(T_A > t) = exp(-(N/2) * (1 - cos(theta_A)))
        return np.exp(-(N / 2) * (1 - cos_theta_val))

    # 定义被积函数 I(t)
    # I(t) = f_Tp(t) * P(T_A > t) = (mu * exp(-mu * t)) * P(T_A > t)
    def integrand_ptc(t):
        if t < 0:
            return 0.0
        # 如果 mu*t 过大，exp(-mu*t) 会溢出/下溢，但我们关注的是 t > 0 的区域
        return mu * np.exp(-mu * t) * P_access_success(t)

    # 积分上限：理论上是无穷大，但由于指数衰减，我们取一个足够大的值
    T_max_approx = T_factor * phi_max * 1.5

    result = custom_quad_integrate_pure_python(integrand_ptc,
                                               a=0,
                                               b=T_max_approx,
                                               n=n_points)
    return result


# --- 4. 蒙特卡洛仿真核心函数 ---

def monte_carlo_simulation_ptc(h_km, N_satellites, gamma_deg, mu, K_samples):
    """执行 P_TC 的蒙特卡洛仿真"""
    h = h_km * 1000
    phi_max = calculate_phi_max(h_km, gamma_deg)
    T_factor = T_factor_func(h_km)

    # 1. 采样任务处理时间 T_p ~ Exp(mu)
    T_p_samples = np.random.exponential(scale=1 / mu, size=K_samples)

    # 2. 采样接入时间 T_A
    # 采样 phi_0 ~ f_phi_0(phi)
    U = np.random.uniform(0, 1, K_samples)
    cos_phi = 1 + (2 / N_satellites) * np.log(1 - U)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi_0_samples = np.arccos(cos_phi)

    # T_A = T_factor * phi_0
    T_A_samples = T_factor * phi_0_samples

    # 3. 任务完成概率是 P(T_p <= T_A)
    # 任务完成（Success）的样本数量
    success_count = np.sum(T_p_samples <= T_A_samples)

    P_TC_sim = success_count / K_samples
    return P_TC_sim


# --- 5. 绘图函数：复现图 5 (P_TC vs. h, 多个 mu) ---

def plot_P_TC_multiple_mu(H_list, N_sat, gamma_deg, Mu_list, K_samples=50000, N_points=100):
    """
    绘制不同卫星高度 h 下的任务完成概率 P_TC 对比图，包含多条 mu 曲线。
    """
    results = {}

    # 定义绘图颜色和标记
    colors = ['blue', 'red', 'green', 'purple']

    # 1. 创建图表
    plt.figure(figsize=(10, 7))

    # 外部循环：迭代不同的 mu 值
    for i, mu in enumerate(Mu_list):
        theory_ptc = []
        sim_ptc = []

        # 内部循环：迭代不同的 h 值
        for h_km in H_list:
            # 1. 理论 P_TC (自定义一重积分) - ANA
            print(f"Calculating ANA P_TC for h={h_km}km, mu={mu}...")
            ptc_theory = calculate_P_TC_theory_custom_int(h_km, N_sat, gamma_deg, mu, n_points=N_points)
            theory_ptc.append(ptc_theory)

            # 2. 蒙特卡洛 P_TC - MC
            print(f"Running MC P_TC Simulation for h={h_km}km, mu={mu}...")
            ptc_sim = monte_carlo_simulation_ptc(h_km, N_sat, gamma_deg, mu, K_samples)
            sim_ptc.append(ptc_sim)

        # 存储结果
        results[mu] = {'ANA': theory_ptc, 'MC': sim_ptc}

        # 绘制曲线
        color = colors[i % len(colors)]

        # ANA 曲线 (实线, 圆点)
        plt.plot(H_list, theory_ptc,
                 label=fr'ANA ($\mu={mu}$ task/s)',
                 linestyle='-', marker='o', markersize=4, color=color)

        # MC 曲线 (虚线, 叉号)
        plt.plot(H_list, sim_ptc,
                 label=fr'MC ($\mu={mu}$ task/s)',
                 linestyle='--', marker='x', markersize=4, color=color)

    # --- 设置图表样式 ---
    plt.xlabel('Satellite Altitude $h$ (km)')
    plt.ylabel(r'Probability of Task Completion $P_{TC}$')
    plt.title(fr'Task Completion Probability vs. $h$ ($N={N_sat}$, $\gamma={gamma_deg}^\circ$, $\mu$ varied)')

    plt.legend()
    plt.grid(True, linestyle='--')

    # 确保 X 轴刻度清晰
    plt.xticks(H_list)
    plt.ylim(0, 1.05)  # 概率范围
    plt.show()

    # --- 打印结果表格 ---
    print("\n--- 任务完成概率 $P_{TC}$ 对比表 (横轴 h, 多 $\mu$ 曲线) ---")
    print(fr"固定参数: N={N_sat}, $\gamma={gamma_deg}^\circ$, K_samples={K_samples}, N_points={N_points}")
    print("-------------------------------------------------------")
    print(f"{'h (km)':<6}", end=' | ')
    for mu in Mu_list:
        print(f"ANA $\mu={mu}$", end=' | ')
        print(f"MC $\mu={mu}$", end=' | ')
    print()
    print("-------------------------------------------------------")

    for i, h_km in enumerate(H_list):
        print(f"{h_km:<6}", end=' | ')
        for mu in Mu_list:
            ana = results[mu]['ANA'][i]
            mc = results[mu]['MC'][i]
            print(f"{ana:>11.4f}", end=' | ')
            print(f"{mc:>11.4f}", end=' | ')
        print()
    print("-------------------------------------------------------")


# --- 6. 执行验证（复现图 5 参数） ---

H_list_fig5 = [1000, 1400, 1800, 2200, 2600, 3000]  # 变化的高度列表 (km)
Mu_list = [0.01, 0.05, 0.1]  # 变化的任务处理速率 (task/s)
N_sat_fig5 = 1000  # 固定卫星总数 N
gamma_fig5 = 90  # 固定波束宽度 gamma (degrees)
K_sim = 200000  # 蒙特卡洛样本数 (提高精度)
N_int_points = 500  # 自定义积分点数

# 运行 P_TC 验证
plot_P_TC_multiple_mu(H_list_fig5, N_sat_fig5, gamma_fig5, Mu_list, K_sim, N_int_points)
