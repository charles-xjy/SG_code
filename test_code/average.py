import numpy as np
import matplotlib.pyplot as plt

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
        # 当 h_km 过低时可能发生，确保返回地球视界角
        return np.arccos(R / (R + h))


# --- Z. 自定义双重积分器 (纯 Python实现，使用梯形法则) ---

def custom_dbl_integrate_pure_python(func, a, b, c, d, nx=1000, ny=1000):
    """
    使用纯 Python 实现的梯形法则近似计算双重定积分。
    I = integral_a^b [ integral_c^d f(y, x) dy ] dx
    """
    if nx < 2 or ny < 2:
        return 0.0

    # 1. 定义外层积分 (x) 步长和网格点
    x_step = (b - a) / (nx - 1)
    x_points = [a + i * x_step for i in range(nx)]

    # 2. 定义内层积分 (y) 步长和网格点
    y_step = (d - c) / (ny - 1)
    y_points = [c + j * y_step for j in range(ny)]

    I_y_list = []

    # --- 外层循环：迭代 x 点，计算 I_y(x) ---
    for x_i in x_points:
        inner_integral_sum = 0.0

        # 内层积分（梯形法则 for y）
        for j in range(1, ny):
            y_j = y_points[j]
            y_j_minus_1 = y_points[j - 1]

            f_j = func(y_j, x_i)
            f_j_minus_1 = func(y_j_minus_1, x_i)

            inner_integral_sum += (f_j + f_j_minus_1)

        I_y_i = inner_integral_sum * (y_step / 2.0)
        I_y_list.append(I_y_i)

    # --- 外层积分（梯形法则 for x） ---
    outer_integral_sum = 0.0

    for i in range(1, nx):
        outer_integral_sum += (I_y_list[i] + I_y_list[i - 1])

    final_result = outer_integral_sum * (x_step / 2.0)

    return final_result


# --- 3. 理论平均访问时间 $\overline{T}_{AB}$ (公式 12) ---

def calculate_average_T_AB_theory_custom_int(h_km, N, gamma_deg, nx=1000, ny=1000):
    """
    计算理论平均访问时间 T_AB_bar (公式 12)，使用自定义纯 Python 积分函数。
    """
    h = h_km * 1000  # 转换为米
    phi_max = calculate_phi_max(h_km, gamma_deg)

    # 预计算常数项
    T_factor = np.sqrt(((R + h) ** 3) / (G * M_earth))
    const_factor_corrected = T_factor * (1 / (2 * np.pi)) * (N / 2)

    # 被积函数 I(delta, phi)
    # func(inner_var, outer_var) -> func(delta, phi)
    def integrand(delta, phi):
        pdf_phi_0_term = np.sin(phi) * np.exp(-(N / 2) * (1 - np.cos(phi)))

        cos_theta_arg = (np.cos(phi_max) * np.cos(phi) +
                         np.sin(phi_max) * np.sin(phi) * np.cos(delta))

        cos_theta_arg = np.clip(cos_theta_arg, -1.0, 1.0)
        theta_val = np.arccos(cos_theta_arg)

        return pdf_phi_0_term * theta_val * const_factor_corrected

    # 积分限：phi 外部 (0 到 phi_max), delta 内部 (0 到 2*pi)
    result = custom_dbl_integrate_pure_python(integrand,
                                              a=0, b=phi_max,  # outer: phi
                                              c=0, d=2 * np.pi,  # inner: delta
                                              nx=nx, ny=ny)

    return result


# --- 4. 蒙特卡洛仿真核心函数 ---

def sample_phi_0(N_satellites, K_samples):
    """使用逆变换采样法从 f_phi_0(phi) 抽样 phi_0"""
    U = np.random.uniform(0, 1, K_samples)
    cos_phi = 1 + (2 / N_satellites) * np.log(1 - U)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi_0_samples = np.arccos(cos_phi)
    return phi_0_samples


def monte_carlo_simulation(h_km, N_satellites, gamma_deg, K_samples):
    """执行蒙特卡洛仿真，返回 T_AB 的样本数组"""
    h = h_km * 1000
    phi_max = calculate_phi_max(h_km, gamma_deg)

    phi_0_samples = sample_phi_0(N_satellites, K_samples)
    delta_samples = np.random.uniform(0, 2 * np.pi, K_samples)

    # 计算中心角 theta (公式 5)
    cos_theta_arg = (np.cos(phi_max) * np.cos(phi_0_samples) +
                     np.sin(phi_max) * np.sin(phi_0_samples) * np.cos(delta_samples))
    cos_theta_arg = np.clip(cos_theta_arg, -1.0, 1.0)
    theta_samples = np.arccos(cos_theta_arg)

    # 计算访问时间 T_AB (公式 6)
    time_factor = np.sqrt(((R + h) ** 3) / (G * M_earth))
    T_AB_samples = time_factor * theta_samples

    return T_AB_samples


# --- 5. 绘图函数：复现图 4 (T_AB_bar vs. h) ---
# --- 绘图部分 ---
plt.figure()


def plot_average_T_AB_validation_varying_H(H_list, N_sat, gamma_deg, K_samples=500000, N_points=1000):
    """
    绘制不同卫星高度 h 下的平均访问时间 T_AB_bar 对比图
    """
    theory_means = []
    sim_means = []

    for h_km in H_list:
        # 1. 理论平均值 (自定义双重积分) - ANA
        print(f"Calculating ANA Average T_AB (Custom Pure Python) for h={h_km}km with {N_points}x{N_points} grid...")
        mean_theory = calculate_average_T_AB_theory_custom_int(h_km, N_sat, gamma_deg, nx=N_points, ny=N_points)
        theory_means.append(mean_theory)

        # 2. 蒙特卡洛平均值 (样本均值) - MC
        print(f"Running MC Simulation for h={h_km}km with {K_samples} samples...")
        T_AB_samples = monte_carlo_simulation(h_km, N_sat, gamma_deg, K_samples)
        mean_sim = np.mean(T_AB_samples)
        sim_means.append(mean_sim)
    plt.plot(H_list, theory_means, label=f'ANA (Custom Pure Python, {N_points} pts)')
    plt.plot(H_list, sim_means, label=f'MC ({K_samples} samples)', linestyle='--', markersize=5,
             marker='.')
    # 更改横轴为高度 h (km)
    plt.xlabel('Satellite Altitude $h$ (km)')
    # 纵轴为平均访问时间
    plt.ylabel(r'Average Access Time $\overline{T}_{AB}$ (s)')
    # 更改标题，固定参数为 N 和 gamma
    plt.title(fr'Average Access Time vs. $h$ ($N={N_sat}$, $\gamma={gamma_deg}^\circ$)')
    plt.legend()
    plt.grid(True, linestyle='--')

    # 打印结果表格
    print("\n--- 结果对比表 (横轴为高度 h) ---")
    print(fr"固定参数: N={N_sat}, $\gamma={gamma_deg}^\circ$")
    print(f"自定义积分点数: {N_points} x {N_points}")
    print("---------------------------------------")
    print(" h (km) | 理论平均值 (ANA) (s) | 仿真平均值 (MC) (s) ")
    print("---------------------------------------")
    for h_km, ana, mc in zip(H_list, theory_means, sim_means):
        print(f"{h_km:<6} | {ana:>20.4f} | {mc:>17.4f}")
    print("---------------------------------------")


# --- 6. 执行验证（复现图 4 参数 - 更改横轴为 H） ---

# 假设复现图 4 的参数：
H_list_fig4 = np.linspace(1000, 2800, 50)
N_sat_fig4 = 800  # 固定卫星总数 N
gamma_fig4 = 90  # 固定波束宽度 gamma (degrees)
K_sim = 5000  # 蒙特卡洛样本数
N_int_points = 50  # 自定义积分点数

# 运行平均访问时间验证
plot_average_T_AB_validation_varying_H(H_list_fig4, N_sat_fig4, gamma_fig4, K_sim, N_int_points)
plot_average_T_AB_validation_varying_H(H_list_fig4, N_sat_fig4, 95, K_sim, N_int_points)
plot_average_T_AB_validation_varying_H(H_list_fig4, N_sat_fig4, 100, K_sim, N_int_points)
plt.show()
