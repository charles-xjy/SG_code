import numpy as np
import matplotlib.pyplot as plt
import math  # <-- 修复：导入 math 模块

# ==========================================
# 1. 物理常数与系统参数 (System Parameters)
# ==========================================
R_earth = 6371e3  # 地球半径 (m)
G = 6.67430e-11  # 万有引力常数
M_earth = 5.972e24  # 地球质量 (kg)

# 论文推断参数
h = 1000e3  # 轨道高度 1000 km
gamma_deg = 90  # 波束宽度 100度
gamma = np.radians(gamma_deg)

# 仿真变量
mu_range = np.linspace(0.010, 0.024, 30)
N_values = [200, 400, 800, 1600]

# 排队论参数 (推断)
n_servers = 3  # 服务节点数 n
lam = 0.0195  # 任务到达率 lambda

# 数值积分参数
PHI_POINTS = 500  # 内部积分 (F_Tab) 的采样点数
T_POINTS = 800  # 外部积分 (P_Completion) 的采样点数


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
# 3. 核心公式实现 (Core Formulas)
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

    # 修复：使用 np.trapezoid 替代 np.trapz
    result = np.trapezoid(integrand_values, phi_grid)
    return result


def get_stationary_pi0(n, rho):
    # M/M/n 排队论 pi_0, 公式 (14)/(208)
    sum_part = 0
    for k in range(n):
        # 修复：使用 math.factorial
        sum_part += (n * rho) ** k / math.factorial(k)

    last_term = (n * rho) ** n / (math.factorial(n) * (1 - rho))
    return 1.0 / (sum_part + last_term)


def get_pi_n(n, rho, pi0):
    # M/M/n 排队论 pi_n, 公式 (13)
    # 修复：使用 math.factorial
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

    if denom == 0:
        pdf_val = exp_mu * (n * mu ** 2 * pi_n * t + mu - n * mu * pi_n)
    else:
        exp_eta = np.exp(-(n * mu - lam) * t)
        part1 = (mu + K) * exp_mu
        part2 = (n * mu ** 2 * pi_n) * exp_eta / (n * mu - lam - mu)
        pdf_val = part1 - part2

    return np.clip(pdf_val, 0, None)


# ==========================================
# 4. 主计算循环 (Main Loop)
# ==========================================
results = {}

# 预先生成 phi 积分网格 (内层积分)
phi_grid = np.linspace(0, phi_max, PHI_POINTS)
# 预先生成 t 积分网格 (外层积分)
t_grid = np.linspace(0, max_possible_time, T_POINTS)

for N in N_values:
    probs = []

    # 预计算 F_Tab(t) 对 grid points t 的值
    F_Tab_grid = np.array([calculate_access_time_cdf(t, N, phi_grid) for t in t_grid])

    for mu in mu_range:
        # 检查系统稳定性
        rho = lam / (n_servers * mu)
        if rho >= 1.0:
            probs.append(0.0)
            continue

        # 1. 计算 f_Ts(t) 对 grid points t 的值
        f_Ts_grid = pdf_sojourn_time(t_grid, mu, n_servers, lam)

        # 2. 计算任务完成概率的被积函数
        integrand_values = F_Tab_grid * f_Ts_grid

        # 3. 使用梯形法则进行最终积分
        prob = np.trapezoid(integrand_values, t_grid)
        probs.append(1 - prob)

    results[N] = probs

# ==========================================
# 5. 绘图 (Plotting)
# ==========================================

plt.figure()
markers = ['^', '>', '<', 'o']

for i, N in enumerate(N_values):
    plt.plot(mu_range, results[N],
             marker=markers[i],
             markersize=6,
             label=f'N = {N}',
             linewidth=1.5)

plt.title(r'CDF of timely task completion (Fig. 5 Reproduction)')
plt.xlabel(r'Average service rate ($\mu$)')
plt.ylabel('CDF of timely task completion')
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', alpha=0.7)

# plt.xlim(0.0095, 0.0245)
# plt.ylim(0.68, 0.98)

plt.tight_layout()
plt.show()
