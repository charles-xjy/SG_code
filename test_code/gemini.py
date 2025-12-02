import numpy as np
import math

# ==========================================
# 0. 兼容性处理 (NumPy 2.0+)
# ==========================================
# 如果当前 numpy 版本没有 trapezoid (旧版本), 则将 trapz 赋值给 trapezoid
# 如果是新版本 (2.0+), 直接使用 trapezoid, 避免 trapz 的警告
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

# ==========================================
# 1. 系统参数 (System Parameters)
# ==========================================
R_earth = 6371e3
G = 6.67430e-11
M_earth = 5.972e24
h = 1000e3
gamma_deg = 100
gamma = np.radians(gamma_deg)

# 仿真参数
mu = 0.03  # 单颗卫星服务率
lam = 0.08  # 任务到达率
N_constellation = 1000  # 星座总卫星数 (影响接入时间分布)

# 计算 M/M/n 系统的最小稳定卫星数 (n*mu > lambda)
n_mmn = int(np.ceil(lam / mu))
if n_mmn * mu <= lam: n_mmn += 1

print(f"System Load: λ={lam}, μ={mu}")
print(f"For M/M/n, using n={n_mmn} servers (Total Capacity={n_mmn * mu:.2f})")

# 为了公平对比，M/M/1 应该拥有相同的总服务能力
mu_mm1 = n_mmn * mu
print(f"For M/M/1, using mu={mu_mm1:.2f} (Equivalent Total Capacity)")

# 积分设置
PHI_POINTS = 200
T_POINTS = 500


# ==========================================
# 2. 几何与运动辅助函数 (Geometry & Motion)
# ==========================================
def get_satellite_velocity(h):
    return np.sqrt(G * M_earth / (R_earth + h))


def get_phi_max(h, gamma):
    term = R_earth / (R_earth + h)
    gamma_0 = 2 * np.arcsin(np.clip(term, -1, 1))
    if gamma < gamma_0:
        val = (R_earth + h) / R_earth * np.sin(gamma / 2.0)
        return np.arcsin(np.clip(val, -1, 1)) - gamma / 2.0
    else:
        return np.arccos(R_earth / (R_earth + h))


v_sat = get_satellite_velocity(h)
phi_max = get_phi_max(h, gamma)
omega = v_sat / (R_earth + h)
max_possible_time = 2 * phi_max / omega


# ==========================================
# 3. 核心公式 (Core Formulas)
# ==========================================
def pdf_phi0(phi, N):
    return (N / 2.0) * np.sin(phi) * np.exp(-(N / 2.0) * (1 - np.cos(phi)))


def cdf_delta_term(t, phi, phi_max_val, omega_val):
    # 避免分母为 0
    denom = np.sin(phi_max_val) * np.sin(phi)
    denom = np.where(np.isclose(denom, 0), 1e-9, denom)

    num = np.cos(omega_val * t) - np.cos(phi_max_val) * np.cos(phi)
    u = num / denom
    u_clipped = np.clip(u, -1.0, 1.0)

    return np.arccos(u_clipped) / np.pi


def calculate_access_time_cdf_vectorized(t_array, N, phi_grid):
    # 向量化计算 F_Tab(t)
    results = []
    pdf_val = pdf_phi0(phi_grid, N)

    for t in t_array:
        if t >= max_possible_time:
            results.append(1.0)
        elif t <= 0:
            results.append(0.0)
        else:
            p_cond = cdf_delta_term(t, phi_grid, phi_max, omega)
            integrand = p_cond * pdf_val
            # 【修复】使用 np.trapezoid 替代 np.trapz
            results.append(np.trapezoid(integrand, phi_grid))

    return np.array(results)


def get_stationary_pi0(n, rho):
    sum_part = 0
    for k in range(n):
        sum_part += (n * rho) ** k / math.factorial(k)
    last_term = (n * rho) ** n / (math.factorial(n) * (1 - rho))
    return 1.0 / (sum_part + last_term)


def get_pi_n(n, rho, pi0):
    return ((n * rho) ** n / math.factorial(n)) * pi0


def mmn_pdf_sojourn_time(t, mu, n, lam):
    rho = lam / (n * mu)
    if rho >= 1: return np.zeros_like(t)

    pi0 = get_stationary_pi0(n, rho)
    pi_n = get_pi_n(n, rho, pi0)
    denom = (1 - rho) * (n * mu - lam - mu)

    pdf_vals = []
    for val_t in t:
        if abs(denom) < 1e-9:  # 奇异点处理 (n*mu - lam - mu = 0)
            term = (n * mu ** 2 * val_t * pi_n + mu - n * mu * pi_n) * np.exp(-mu * val_t)
            pdf_vals.append(term)
        else:
            K = (mu ** 2 * pi_n) / denom
            term1 = (mu + K) * np.exp(-mu * val_t)
            term2 = K * np.exp(-(n * mu - lam) * val_t)
            pdf_vals.append(term1 - term2)

    return np.array(pdf_vals)


def mm1_pdf_sojourn_time(t, mu, lam):
    # 标准 M/M/1 响应时间 PDF
    if lam >= mu: return np.zeros_like(t)
    return (mu - lam) * np.exp(-(mu - lam) * t)


# ==========================================
# 4. 主计算过程 (Main Execution)
# ==========================================

# 1. 建立积分网格
phi_grid = np.linspace(1e-5, phi_max, PHI_POINTS)
t_grid = np.linspace(0, max_possible_time, T_POINTS)

# 2. 计算接入时间分布 F_Tab (通用)
# 这一步依赖于星座规模 N_constellation
F_Tab_grid = calculate_access_time_cdf_vectorized(t_grid, N_constellation, phi_grid)

# 3. 计算 M/M/n 任务完成概率
f_Ts_mmn = mmn_pdf_sojourn_time(t_grid, mu, n_mmn, lam)
# 积分计算 P(Ts < Tab) = 1 - P(Ts > Tab) ???
# 注意: 论文公式 (17) P(Complete) = P(Tab > Ts) = 1 - Integral(F_Tab(y) * f_Ts(y) dy)
integrand_mmn = F_Tab_grid * f_Ts_mmn
# 【修复】使用 np.trapezoid 替代 np.trapz
prob_fail_mmn = np.trapezoid(integrand_mmn, t_grid)
prob_success_mmn = 1 - prob_fail_mmn

# 4. 计算 M/M/1 任务完成概率 (高能单星)
f_Ts_mm1 = mm1_pdf_sojourn_time(t_grid, mu_mm1, lam)
integrand_mm1 = F_Tab_grid * f_Ts_mm1
# 【修复】使用 np.trapezoid 替代 np.trapz
prob_fail_mm1 = np.trapezoid(integrand_mm1, t_grid)
prob_success_mm1 = 1 - prob_fail_mm1

# ==========================================
# 5. 输出结果 (Output Results)
# ==========================================
print("-" * 30)
print(f"Results for N_constellation={N_constellation}")
print("-" * 30)
print(f"M/M/{n_mmn} Completion Prob: {prob_success_mmn:.4f}")
print(f"M/M/1 Completion Prob: {prob_success_mm1:.4f}")
print("-" * 30)