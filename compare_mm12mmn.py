import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 仿真参数设置
# ==========================================
# 模拟的总任务数 (M)
NUM_MC_SAMPLES = 500000
# 服务台数量 (n) - 假设有 5 个服务器/卫星
N_SERVERS = 5
# 总任务到达率 (lambda)
LAMBDA_TOTAL = 0.1  # 例如，每秒 0.1 个任务到达整个系统
# 单个服务器的服务率 (mu)
MU_SINGLE = 0.1  # 例如，每个服务器每秒服务 0.1 个任务

# 稳定性检查: 总服务能力 (n*mu) 必须大于总到达率 (lambda)
if N_SERVERS * MU_SINGLE <= LAMBDA_TOTAL:
    print("WARNING: System is unstable (n*mu <= lambda). Results will show infinite queue.")


# ==========================================
# 2. M/M/n 仿真函数 (集中式队列)
# ==========================================

def simulate_mmc_sojourn(lam, mu, n, M):
    """
    模拟 M/M/n 队列（集中式队列，n个服务器共享一个等待区）。
    lam: 总到达率
    mu: 单个服务器的服务率
    n: 服务器数量
    M: 样本数量
    """
    # 随机输入：到达间隔和服务时间
    inter_arrival_times = np.random.exponential(1.0 / lam, M)
    service_times = np.random.exponential(1.0 / mu, M)

    # 系统状态：记录每个服务台的可用时间
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

        # 服务开始时间：取决于任务到达时间和服务器就绪时间
        start_service_time = max(arrival_time, server_ready_time)

        # 服务完成时间
        completion_time = start_service_time + service_times[i]

        # 更新服务台可用时间
        server_available_time[server_index] = completion_time

        # 驻留时间 T_s = 离开时间 - 到达时间
        sojourn_times.append(completion_time - arrival_time)

    return np.array(sojourn_times)


# ==========================================
# 3. n x M/M/1 仿真函数 (分布式队列)
# ==========================================

def simulate_n_mm1_sojourn(lam_total, mu, n, M):
    """
    模拟 n 个独立的 M/M/1 队列（分布式队列），总任务流被平均分配。
    lam_total: 总到达率
    mu: 单个服务器的服务率
    n: 服务器数量
    M: 样本数量（总任务数）
    """
    # 关键：每个 M/M/1 队列的到达率 lam_prime = lam_total / n
    lam_prime = lam_total / n

    # 模拟 M 个任务在其中一个 M/M/1 系统中的驻留时间，
    # 因为所有 n 个 M/M/1 系统是独立的且参数相同，它们的 T_s 分布是完全一样的。

    # 随机输入：到达间隔和服务时间
    inter_arrival_times = np.random.exponential(1.0 / lam_prime, M)
    service_times = np.random.exponential(1.0 / mu, M)

    # M/M/1 只需要一个服务器状态跟踪
    server_available_time = 0.0
    sojourn_times = []

    current_time = 0.0

    for i in range(M):
        # 任务到达
        current_time += inter_arrival_times[i]
        arrival_time = current_time

        # 服务开始时间
        start_service_time = max(arrival_time, server_available_time)

        # 服务完成时间
        completion_time = start_service_time + service_times[i]

        # 更新服务器可用时间
        server_available_time = completion_time

        # 驻留时间 T_s
        sojourn_times.append(completion_time - arrival_time)

    return np.array(sojourn_times)


# ==========================================
# 4. 主执行与结果计算
# ==========================================

# 1. M/M/n 仿真
Ts_mmc = simulate_mmc_sojourn(LAMBDA_TOTAL, MU_SINGLE, N_SERVERS, NUM_MC_SAMPLES)

# 2. n x M/M/1 仿真
# 每个 M/M/1 系统处理 LAMBDA_TOTAL / N_SERVERS 的到达率
Ts_n_mm1 = simulate_n_mm1_sojourn(LAMBDA_TOTAL, MU_SINGLE, N_SERVERS, NUM_MC_SAMPLES)


# 计算 CDF (累积分布函数)
def calculate_cdf(data):
    # 排序数据
    sorted_data = np.sort(data)
    # 累积概率
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf


Ts_mmc_sorted, cdf_mmc = calculate_cdf(Ts_mmc)
Ts_n_mm1_sorted, cdf_n_mm1 = calculate_cdf(Ts_n_mm1)

# ==========================================
# 5. 绘图对比
# ==========================================

plt.figure(figsize=(10, 6))

# 绘制 M/M/n 结果
plt.plot(Ts_mmc_sorted, cdf_mmc,
         label=f'M/M/{N_SERVERS} (Cooperative, $\\rho={LAMBDA_TOTAL / (N_SERVERS * MU_SINGLE):.2f}$)',
         color='blue',
         linewidth=2)

# 绘制 n x M/M/1 结果
rho_prime = (LAMBDA_TOTAL / N_SERVERS) / MU_SINGLE
plt.plot(Ts_n_mm1_sorted, cdf_n_mm1,
         label=f'${N_SERVERS}$ x M/M/1 (Non-Cooperative, $\\rho\'={rho_prime:.2f}$)',
         color='red',
         linestyle='--',
         linewidth=2)

plt.title('Comparison of Sojourn Time CDF: M/M/n vs. n x M/M/1')
plt.xlabel('Sojourn Time $T_s$ (s)')
# 修复 Matplotlib LaTeX 兼容性问题：使用 r'' 原始字符串和 \leq 替代 \le
plt.ylabel(r'CDF $P(T_s \leq t)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

# 限制 x 轴，以突出分布的差异（通常 M/M/n 的尾部更短）
max_time = np.max([Ts_mmc_sorted[int(0.99 * NUM_MC_SAMPLES)], Ts_n_mm1_sorted[int(0.99 * NUM_MC_SAMPLES)]]) * 1.5
plt.xlim(0, max_time)
plt.ylim(0, 1.05)

plt.show()
