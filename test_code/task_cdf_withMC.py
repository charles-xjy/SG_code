import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d

# ==========================================
# 1. 物理常数与系统参数 (System Parameters)
# ==========================================
R_earth = 6371e3  # 地球半径 (m)
G = 6.67430e-11  # 万有引力常数
M_earth = 5.972e24  # 地球质量 (kg)

# 论文推断参数 (用户修改后的参数)
h = 1000e3  # 轨道高度 1000 km
gamma_deg = 95  # 波束宽度 90度
gamma = np.radians(gamma_deg)

# 仿真变量
mu_range = np.linspace(0.01, 0.03, 30)
N_values = [150, 500, 1000]

# 排队论参数 (推断)
n_servers = 2  # 服务节点数 n
lam = 0.009  # 任务到达率 lambda

# 数值积分参数
PHI_POINTS = 500  # 内部积分 (F_Tab) 的采样点数
T_POINTS = 800  # 外部积分 (P_Completion) 的采样点数

# 蒙特卡洛参数
NUM_MC_SAMPLES = 100000  # MC 仿真样本数 M


# ==========================================
# 2. 辅助函数：几何与运动 (Geometry & Motion)
# ==========================================
def get_satellite_velocity(h):
    return np.sqrt(G * M_earth / (R_earth + h))


def get_phi_max(h, gamma):
    # 公式 (64)
    term = R_earth / (R_earth + h)
    gamma_0 = 2 * np.arcsin(np.clip(term, -1, 1))

    if gamma < gamma_0:
        val = (R_earth + h) / R_earth * np.sin(gamma / 2.0)
        phi_max_val = np.arcsin(np.clip(val, -1, 1)) - gamma / 2.0
        return phi_max_val
    else:
        return np.arccos(R_earth / (R_earth + h))


# 计算常数
v_sat = get_satellite_velocity(h)
phi_max = get_phi_max(h, gamma)
omega = v_sat / (R_earth + h)  # 角速度
max_possible_time = 2 * phi_max / omega  # 外部积分上限


# ==========================================
# 3. 核心公式实现 (Core Formulas) - 解析解部分
# ==========================================

def pdf_phi0(phi, N):
    # 用户接入角度的PDF, 公式 (69)
    return (N / 2.0) * np.sin(phi) * np.exp(-(N / 2.0) * (1 - np.cos(phi)))


def cdf_delta_term(t, phi, phi_max_val, omega_val):
    denom = np.sin(phi_max_val) * np.sin(phi)
    num = np.cos(omega_val * t) - np.cos(phi_max_val) * np.cos(phi)

    zero_denom_mask = np.isclose(denom, 0)
    u = np.zeros_like(phi)

    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.divide(num, denom, out=u, where=~zero_denom_mask)

    u_clipped = np.clip(u, -1.0, 1.0)
    result = np.arccos(u_clipped) / np.pi
    boundary_value = 1.0 if t == 0 else 0.0
    result = np.where(zero_denom_mask, boundary_value, result)
    return result


def calculate_access_time_cdf(t, N, phi_grid):
    # 公式 (10) - 使用梯形法则手动积分
    if t >= max_possible_time:
        return 1.0

    p_cond = cdf_delta_term(t, phi_grid, phi_max, omega)
    pdf_val = pdf_phi0(phi_grid, N)
    integrand_values = p_cond * pdf_val

    result = np.trapezoid(integrand_values, phi_grid)
    return result


def get_stationary_pi0(n, rho):
    # M/M/n 排队论 pi_0, 公式 (14)/(208)
    sum_part = 0
    for k in range(n):
        sum_part += (n * rho) ** k / math.factorial(k)

    last_term = (n * rho) ** n / (math.factorial(n) * (1 - rho))
    return 1.0 / (sum_part + last_term)


def get_pi_n(n, rho, pi0):
    # M/M/n 排队论 pi_n, 公式 (13)
    return ((n * rho) ** n / math.factorial(n)) * pi0


def pdf_sojourn_time(t, mu, n, lam):
    # M/M/n 驻留时间 (Sojourn Time) PDF, f_Ts(t)
    # 公式 (16)
    rho = lam / (n * mu)
    if rho >= 1:
        return np.zeros_like(t)

    pi0 = get_stationary_pi0(n, rho)
    pi_n = get_pi_n(n, rho, pi0)

    # 系数 K
    denom = (1 - rho) * (n * mu - lam - mu)
    K = (mu ** 2 * pi_n) / denom

    exp_mu = np.exp(-mu * t)

    if np.isclose(denom, 0):  # 处理临界情况
        # 洛必达法则导出的特殊形式 (仅用于避免数值不稳定)
        pdf_val = exp_mu * (n * mu ** 2 * pi_n * t + mu - n * mu * pi_n)
    else:
        exp_eta = np.exp(-(n * mu - lam) * t)
        part1 = (mu + K) * exp_mu
        part2 = (n * mu ** 2 * pi_n) * exp_eta / (n * mu - lam - mu)
        pdf_val = part1 - part2

    return np.clip(pdf_val, 0, None)


# ==========================================
# 4. 解析解主循环 (Analytical Solution)
# ==========================================
print("--- Starting Analytical Calculation ---")
analytical_results = {}
analytical_F_Tab_grids = {}  # 存储 F_Tab 网格用于 MC 采样

# 预先生成 phi 积分网格 (内层积分)
phi_grid = np.linspace(0, phi_max, PHI_POINTS)
# 预先生成 t 积分网格 (外层积分)
t_grid = np.linspace(0, max_possible_time, T_POINTS)

for N in N_values:
    probs = []

    # 预计算 F_Tab(t) 对 grid points t 的值
    F_Tab_grid = np.array([calculate_access_time_cdf(t, N, phi_grid) for t in t_grid])
    analytical_F_Tab_grids[N] = F_Tab_grid

    for mu in mu_range:
        rho = lam / (n_servers * mu)
        if rho >= 1.0:
            probs.append(0.0)
            continue

        f_Ts_grid = pdf_sojourn_time(t_grid, mu, n_servers, lam)

        # 任务完成概率 P = Integral [ (1 - F_Ts) * f_Tab ] dt
        # 原始公式: P = Integral [ (1 - F_Tab) * f_Ts ] dt
        # 由于我们计算的是 P(T_AB > T_s)，其结果是 Integral [ (1 - F_Tab) * f_Ts ] dt
        integrand_values = (1.0 - F_Tab_grid) * f_Ts_grid
        prob = np.trapezoid(integrand_values, t_grid)
        probs.append(prob)

    analytical_results[N] = probs
print("--- Analytical Calculation Finished ---")

# ==========================================
# 5. 蒙特卡洛仿真部分 (Monte Carlo Simulation)
# ==========================================
print("\n--- Starting Monte Carlo Simulation ---")


# 辅助函数 5.1: M/M/n 队列仿真 (生成 Sojourn Time T_s 样本)
def simulate_mmc_sojourn(lam, mu, n, M):
    # 模拟 M/M/n 队列
    inter_arrival_times = np.random.exponential(1.0 / lam, M)
    service_times = np.random.exponential(1.0 / mu, M)

    # 记录每个服务台的可用时间
    server_available_time = np.zeros(n)
    sojourn_times = []

    current_time = 0.0

    for i in range(M):
        # 任务到达
        current_time += inter_arrival_times[i]
        arrival_time = current_time

        # 找到最早可用的服务台
        server_index = np.argmin(server_available_time)
        server_ready_time = server_available_time[server_index]

        # 计算等待时间
        start_service_time = max(arrival_time, server_ready_time)
        wait_time = start_service_time - arrival_time

        # 服务完成时间
        completion_time = start_service_time + service_times[i]

        # 更新服务台可用时间
        server_available_time[server_index] = completion_time

        # 驻留时间 T_s = 等待时间 + 服务时间
        sojourn_times.append(completion_time - arrival_time)

    return np.array(sojourn_times)


# 辅助函数 5.2: 逆变换采样 (生成 Access Time T_AB 样本)
def get_Tab_samples(M, t_grid, F_Tab_grid):
    # 使用解析解计算出的 CDF (F_Tab_grid) 进行逆变换采样

    # 建立 CDF 的反函数 (插值)
    # F_Tab_grid 的值从 0 变化到 1
    F_Tab_func = interp1d(F_Tab_grid, t_grid, kind='linear', fill_value=(t_grid[0], t_grid[-1]), bounds_error=False)

    # 生成 M 个均匀分布样本 u ~ U(0, 1)
    u_samples = np.random.rand(M)

    # 逆变换: T_AB = F_Tab^{-1}(u)
    Tab_samples = F_Tab_func(u_samples)

    return Tab_samples


# MC 结果存储
mc_results = {}

for N in N_values:
    mc_probs = []
    F_Tab_grid_N = analytical_F_Tab_grids[N]  # 从解析解中获取 F_Tab

    for mu in mu_range:
        rho = lam / (n_servers * mu)
        if rho >= 1.0:
            mc_probs.append(0.0)
            continue

        # 1. 生成 T_AB 样本 (逆变换采样)
        Tab_samples = get_Tab_samples(NUM_MC_SAMPLES, t_grid, F_Tab_grid_N)

        # 2. 生成 T_s 样本 (M/M/n 仿真)
        # 注意：这里需要对每个 mu 重新进行 T_s 仿真
        Ts_samples = simulate_mmc_sojourn(lam, mu, n_servers, NUM_MC_SAMPLES)

        # 3. 计算及时完成概率: P(T_AB > T_s)
        completion_count = np.sum(Tab_samples > Ts_samples)
        prob = completion_count / NUM_MC_SAMPLES
        mc_probs.append(prob)

    mc_results[N] = mc_probs
    print(f"MC finished for N={N}")

print("--- Monte Carlo Simulation Finished ---")

# ==========================================
# 6. 绘图对比 (Plotting Comparison)
# ==========================================

plt.figure()
markers = ['^', 's', 'o']
line_styles = ['-', '--']  # 解析解用实线，MC用虚线

for i, N in enumerate(N_values):
    # 解析解 (实线)
    plt.plot(mu_range, analytical_results[N],
             linestyle=line_styles[0],
             label=f'Analytical (N={N})',
             linewidth=1.5,
             )

    # 蒙特卡洛 (虚线)
    plt.plot(mu_range, mc_results[N],
             marker=markers[i],
             markersize=5,
             linestyle=line_styles[1],
             label=f'MC Sim ({N})',
             linewidth=1,
             )

plt.title(r'Comparison of Analytical and Monte Carlo Results for Task Completion CDF')
plt.xlabel(r'Average service rate ($\mu$)')
plt.ylabel('CDF of timely task completion')
plt.grid(True)
plt.legend()
plt.show()
