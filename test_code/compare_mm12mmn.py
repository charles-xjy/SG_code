import numpy as np
from scipy.special import factorial


# ==========================================
# 1. 通用 M/M/N 理论计算函数 (用于 MMN 和 MM1)
# ==========================================

def calculate_P0(lam, mu, N):
    """ 计算 M/M/N 系统空闲概率 P0。 """
    rho = lam / (N * mu)
    if rho >= 1.0: return 0.0

    sum_term = 0
    for k in range(N):
        sum_term += ((lam / mu) ** k) / factorial(k)

    n_term = ((lam / mu) ** N) / (factorial(N) * (1 - rho))
    P0 = 1.0 / (sum_term + n_term)
    return P0


def calculate_Pn_busy(lam, mu, N, P0):
    """ 计算 M/M/N 系统所有 N 个服务器都忙碌的概率。 """
    rho = lam / (N * mu)
    if rho >= 1.0: return 1.0

    Pn_busy = (P0 * ((lam / mu) ** N)) / (factorial(N) * (1 - rho))
    return Pn_busy


def mmn_waiting_time_cdf(t, lam_total, mu, N):
    """ 计算 M/M/N (集中式) 的等待时间 Tq 的 CDF。 """
    rho = lam_total / (N * mu)
    if rho >= 1.0: return 0.0

    P0 = calculate_P0(lam_total, mu, N)
    Pn_busy = calculate_Pn_busy(lam_total, mu, N, P0)

    beta = N * mu - lam_total

    if t < 0: return 0.0

    # F_q(t) = (1 - Pn_busy) + Pn_busy * (1 - e^(-beta * t))
    cdf_value = (1 - Pn_busy) + Pn_busy * (1 - np.exp(-beta * t))
    return cdf_value


def mm1_waiting_time_cdf(t, lam_prime, mu):
    """ 计算单个 M/M/1 的等待时间 Tq 的 CDF。 """
    rho_prime = lam_prime / mu
    if rho_prime >= 1.0: return 0.0

    beta = mu - lam_prime  # 衰减参数

    if t < 0: return 0.0

    # F_q(t) = (1 - rho') + rho' * (1 - e^(-beta * t))
    cdf_value = (1 - rho_prime) + rho_prime * (1 - np.exp(-beta * t))
    return cdf_value


# ==========================================
# 2. 迭代求解函数
# ==========================================

def find_min_servers_comparison(target_t, target_p, lam_total, mu):
    """
    寻找满足目标概率 P 的最小服务器数量 N，并对比两种模型。
    """
    # N 必须满足 N*mu > lambda_total 才能稳定

    N_min_stable = int(np.ceil(lam_total / mu))
    if N_min_stable * mu == lam_total:
        N_min_stable += 1
    N = max(1, N_min_stable)
    MAX_N = N_min_stable + 15

    N_mmc = None
    N_n_mm1 = None

    print(f"\n--- 寻找满足 P(Tq ≤ {target_t:.2f}小时) ≥ {target_p * 100:.0f}% 的最小 N ---")
    print(f"总到达率 λ={lam_total:.2f}, 单服务率 μ={mu:.2f}, 最小稳定 N={N_min_stable}")
    print("-" * 50)

    while N <= MAX_N and (N_mmc is None or N_n_mm1 is None):
        # --- 场景 1: M/M/N (单体大模型/协同) ---
        cdf_mmc = mmn_waiting_time_cdf(target_t, lam_total, mu, N)
        if N_mmc is None and cdf_mmc >= target_p:
            N_mmc = N
        # --- 场景 2: N x M/M/1 (分解小模型/隔离) ---
        # 此时，单个队列的到达率 lam_prime 依赖于 N
        """
        这里是修改代码的部分
        """
        lam_prime = lam_total
        mu_prime = mu * N ** 2
        cdf_mm1_single = mm1_waiting_time_cdf(target_t, lam_prime, mu_prime)
        # 整个任务完成的概率是单个 CDF 的 N 次方
        cdf_n_mm1 = cdf_mm1_single ** N

        if N_n_mm1 is None and cdf_n_mm1 >= target_p:
            N_n_mm1 = N
        print(f"N={N}: MMN CDF={cdf_mmc:.4f}, (N x MM1) CDF={cdf_n_mm1:.4f}")
        N += 1
    print("-" * 50)
    # 输出结果
    result = {
        'N_mmc': N_mmc,
        'N_n_mm1': N_n_mm1
    }

    if N_mmc and N_n_mm1:
        print(f"结论: M/M/N 仅需 N={N_mmc}。 N x M/M/1 需 N={N_n_mm1}。")
    elif N_mmc:
        print(f"结论: M/M/N 仅需 N={N_mmc}。 N x M/M/1 在此范围内不满足。")
    elif N_n_mm1:
        print(f"结论: N x M/M/1 需 N={N_n_mm1}。 M/M/N 在此范围内不满足。")

    return result


# ==========================================
# 3. 运行示例
# ==========================================

# 场景参数
LAMBDA_TOTAL = 5.9  # 总任务到达率 (例如：每小时 4 个任务)
MU_SINGLE = 2  # 单个模型的服务率 (例如：每小时 1 个任务)

# 优化目标
TARGET_T = 0.1  # 给定时间限制 (T/小时)
TARGET_P = 0.90  # 目标概率 (P)

# 求解最小 N
min_N_required = find_min_servers_comparison(TARGET_T, TARGET_P, LAMBDA_TOTAL, MU_SINGLE)
