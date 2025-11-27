import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- 1. 物理常量和系统参数 ---
# --- 物理常量 (近似值，确保与论文使用的值一致) ---
# G: Gravitational constant (m^3 kg^-1 s^-2)
# M_earth: Mass of the Earth (kg)
# R: Radius of the Earth (m)
G = 6.67430e-11
M_earth = 5.972e24
R = 6371.0e3  # Radius of Earth in meters (6371 km)


# --- 2. 核心函数定义（与理论计算部分相同） ---
def calculate_phi_max(h_km, gamma_deg):
    """
    计算最大访问角 phi_max (公式 1)
    """
    h = h_km * 1000  # 转换为米
    gamma_rad = np.radians(gamma_deg)

    # 仅使用由波束宽度决定的第一个条件，除非它超出物理限制
    try:
        sin_arg = (R + h) / R * np.sin(gamma_rad / 2)
        if sin_arg > 1:
            # 超过波束宽度限制，使用由轨道高度决定的第二种情况
            phi_max = np.arccos(R / (R + h))
        else:
            phi_max = np.arcsin(sin_arg) - gamma_rad / 2

        # 确保 phi_max 不超过高度决定的最大可能角度
        phi_max_h = np.arccos(R / (R + h))
        return min(phi_max, phi_max_h)

    except ValueError:
        return np.arccos(R / (R + h))


def pdf_phi_0(phi, N):
    """用户访问角 PDF f_phi_0(phi) (公式 2)"""
    return (N / 2) * np.sin(phi) * np.exp(-(N / 2) * (1 - np.cos(phi)))


def cdf_T_AB_theory(t, h_km, N, gamma_deg):
    """
    理论访问时间 CDF F_T_AB(t) (公式 10)
    """
    h = h_km * 1000  # 转换为米
    phi_max = calculate_phi_max(h_km, gamma_deg)

    # 预计算常数项: sqrt(GM/(R+h)^3)
    const_term = np.sqrt((G * M_earth) / ((R + h) ** 3))

    # 定义积分函数 f(phi)
    def integrand(phi):
        term1 = pdf_phi_0(phi, N)

        # arccos 内部的参数 u
        num = np.cos(const_term * t) - np.cos(phi_max) * np.cos(phi)
        den = np.sin(phi_max) * np.sin(phi)

        # 约束 u in [-1, 1] for arccos
        # 避免 phi 接近 0 或 phi_max 时分母接近 0 导致的数值不稳定
        if np.sin(phi) < 1e-6 or np.sin(phi_max) < 1e-6:
            u = -1.0
        else:
            u = np.clip(num / den, -1.0, 1.0)

        # P(cos_delta > u) = 1/pi * arccos(u) (公式 9)
        term2 = (1 / np.pi) * np.arccos(u)

        return term1 * term2

    # 执行数值积分 (从 0 到 phi_max)
    result, error = quad(integrand, 0, phi_max, limit=100)
    return result


# --- 3. 蒙特卡洛仿真核心函数 ---

def sample_phi_0(N_satellites, K_samples):
    """
    使用逆变换采样法从 f_phi_0(phi) 抽样 phi_0
    phi_0 = arccos(1 + 2/N * ln(1-U)), 其中 U ~ Uniform(0, 1)
    """
    # 抽取均匀分布的随机数 U (0 到 1)
    U = np.random.uniform(0, 1, K_samples)

    # 应用逆变换公式
    # 1 - U 保证 ln 内部为正且小于 1
    cos_phi = 1 + (2 / N_satellites) * np.log(1 - U)

    # cos_phi 必须在 [-1, 1] 范围内，虽然理论上应该在范围内，但浮点运算可能导致微小偏差
    cos_phi = np.clip(cos_phi, -1.0, 1.0)

    phi_0_samples = np.arccos(cos_phi)
    return phi_0_samples


def monte_carlo_simulation(h_km, N_satellites, gamma_deg, K_samples=100000):
    """
    执行蒙特卡洛仿真，返回 T_AB 的样本数组
    """
    h = h_km * 1000  # 转换为米
    phi_max = calculate_phi_max(h_km, gamma_deg)

    # 1. 抽样随机变量
    phi_0_samples = sample_phi_0(N_satellites, K_samples)
    delta_samples = np.random.uniform(0, 2 * np.pi, K_samples)  # delta ~ Uniform(0, 2pi)

    # 2. 计算中心角 theta (公式 5)
    cos_theta_arg = (np.cos(phi_max) * np.cos(phi_0_samples) +
                     np.sin(phi_max) * np.sin(phi_0_samples) * np.cos(delta_samples))

    # 约束 cos_theta_arg 在 [-1, 1]
    cos_theta_arg = np.clip(cos_theta_arg, -1.0, 1.0)

    theta_samples = np.arccos(cos_theta_arg)

    # 3. 计算访问时间 T_AB (公式 6)
    # T_AB = sqrt((R+h)^3 / GM) * theta
    time_factor = np.sqrt(((R + h) ** 3) / (G * M_earth))
    T_AB_samples = time_factor * theta_samples

    return T_AB_samples


# --- 4. 绘图函数：复现图 2 示例 ---
def plot_cdf_validation(h_km_list, N_satellites, gamma_deg, K_samples=100000):
    """
    绘制理论 CDF 和蒙特卡洛仿真 CDF
    """
    times = np.linspace(50, 600, 100)  # 时间范围，参考图 2

    plt.figure()

    for h_km in h_km_list:
        # 理论 CDF
        cdf_theory_values = [cdf_T_AB_theory(t, h_km, N_satellites, gamma_deg) for t in times]
        plt.plot(times, cdf_theory_values, label=f'Theo $h={h_km}km$')
        # 蒙特卡洛仿真
        T_AB_samples = monte_carlo_simulation(h_km, N_satellites, gamma_deg, K_samples)
        # 计算仿真 CDF
        cdf_sim_values = [np.sum(T_AB_samples < t) / K_samples for t in times]
        plt.plot(times, cdf_sim_values, marker='.', linestyle='--', markersize=6,
                 label=f'Sim $h={h_km}km$')

    plt.xlabel('Time(s)')
    plt.ylabel('CDF of the access time')
    plt.title(fr'Access Time CDF: Theory vs. Monte Carlo Simulation (N={N_satellites}, $\gamma={gamma_deg}^\circ$)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.ylim(0, 1.05)
    plt.show()


# --- 4. 绘图函数：复现图 3 (N变化) ---
def plot_cdf_validation_varying_N(N_list, h_km, gamma_deg, K_samples=200000):
    """
    绘制不同卫星总数 N 下的理论 CDF 和蒙特卡洛仿真 CDF
    """
    # 调整时间范围以匹配图 3
    times = np.linspace(50, 400, 100)

    plt.figure()

    # 遍历不同的 N 值
    for N_sat in N_list:
        # 理论 CDF (实线)
        print(f"Calculating Theory CDF for N={N_sat}...")
        cdf_theory_values = [cdf_T_AB_theory(t, h_km, N_sat, gamma_deg) for t in times]
        plt.plot(times, cdf_theory_values, label=f'Theo $N={N_sat}$', linestyle='-')

        # 蒙特卡洛仿真 (虚线/点线)
        print(f"Running Simulation for N={N_sat} with {K_samples} samples...")
        T_AB_samples = monte_carlo_simulation(h_km, N_sat, gamma_deg, K_samples)

        # 计算仿真 CDF
        cdf_sim_values = [np.sum(T_AB_samples < t) / K_samples for t in times]
        plt.plot(times, cdf_sim_values, marker='.', linestyle='--', markersize=6,
                 label=f'Sim $N={N_sat}$')

    plt.xlabel('Time (s)')
    plt.ylabel('CDF of the access time')

    # 标题使用原始 f-string 修复转义警告，并显示固定参数 h 和 gamma
    plt.title(fr'Access Time CDF: Varying $N$ ($h={h_km}km$, $\gamma={gamma_deg}^\circ$)')

    plt.legend()
    plt.grid(True, linestyle='--')
    plt.ylim(0, 1.05)
    plt.xlim(times.min(), times.max())
    plt.show()


# --- 5. 执行验证（复现图 2 参数） ---
# 图 2 参数
N_fig2 = 800
gamma_fig2 = 100  # degrees
heights_fig2 = [1000, 1250, 1500]  # km
K_sim = 200000  # 增加样本数以提高仿真精度
# 图 3 参数
N_list_fig3 = [400, 800, 1600]
h_fig3 = 1000  # km (固定高度)
gamma_fig3 = 100  # degrees (固定波束宽度)

# 运行验证
plot_cdf_validation(heights_fig2, N_fig2, gamma_fig2, K_sim)
plot_cdf_validation_varying_N(N_list_fig3, h_fig3, gamma_fig3, K_sim)
