import numpy as np


def calculate_orbital_radii(R_E, h1, h2):
    """
    计算轨道半径

    参数:
        R_E: 地球半径 (km)
        h1: 下层卫星高度 (km)
        h2: 上层卫星高度 (km)

    返回:
        R1, R2: 下层和上层轨道半径 (km)
    """
    R1 = R_E + h1
    R2 = R_E + h2
    return R1, R2


def check_system_stability(lambda_arrival, mu):
    """
    检查M/M/1系统稳定性

    参数:
        lambda_arrival: 任务到达率 (tasks/s)
        mu: 服务率 (tasks/s)

    返回:
        rho: 系统利用率
    """
    rho = lambda_arrival / mu
    if rho >= 1:
        raise ValueError(f"系统不稳定！rho = {rho:.4f} >= 1，必须满足 lambda < mu")
    return rho


def calculate_availability_probability(rho, mu, lambda_arrival, T_pass):
    """
    计算MPPP可用概率

    参数:
        rho: 系统利用率
        mu: 服务率 (tasks/s)
        lambda_arrival: 任务到达率 (tasks/s)
        T_pass: 平均过境时间 (s)

    返回:
        p_available: 卫星可用概率
    """
    p_available = 1 - rho * np.exp(-(mu - lambda_arrival) * T_pass / 2)
    return p_available


def calculate_effective_density(lambda_real, p_available):
    """
    计算有效卫星密度

    参数:
        lambda_real: 实际卫星密度 (satellites/km^2)
        p_available: 卫星可用概率

    返回:
        lambda_eff: 有效卫星密度 (satellites/km^2)
    """
    lambda_eff = lambda_real * p_available
    return lambda_eff


def calculate_min_satellites(lambda_arrival, mu, tau, P_task, f):
    """
    计算所需的最小卫星数量

    参数:
        lambda_arrival: 任务到达率 (tasks/s)
        mu: 服务率 (tasks/s)
        tau: 任务完成时间限制 (s)
        P_task: 期望任务完成概率
        f: 不可并行部分比例

    返回:
        N_continuous: 连续值的卫星数量
        N_min: 最小卫星数量（整数）
    """
    # 计算目标加速比
    term1 = lambda_arrival / mu
    term2 = (1 / (mu * tau)) * np.log(1 - P_task)
    S_amdahl_target = term1 - term2

    # 根据Amdahl定律反解N
    numerator = (1 - f) * S_amdahl_target
    denominator = 1 - f * S_amdahl_target

    if denominator <= 0:
        raise ValueError("参数组合不可行！无法达到目标完成概率，请调整参数")

    N_continuous = numerator / denominator
    N_min = int(np.ceil(N_continuous))

    return N_continuous, N_min


def calculate_amdahl_speedup(N, f):
    """
    计算Amdahl加速比

    参数:
        N: 卫星数量
        f: 不可并行部分比例

    返回:
        S_amdahl: 加速比
    """
    S_amdahl = 1 / (f + (1 - f) / N)
    return S_amdahl


def calculate_actual_probability(N, f, mu, lambda_arrival, tau):
    """
    计算实际任务完成概率

    参数:
        N: 卫星数量
        f: 不可并行部分比例
        mu: 服务率 (tasks/s)
        lambda_arrival: 任务到达率 (tasks/s)
        tau: 任务完成时间限制 (s)

    返回:
        P_actual: 实际完成概率
        mu_eff: 有效服务率
    """
    S_amdahl = calculate_amdahl_speedup(N, f)
    mu_eff = mu * S_amdahl
    P_actual = 1 - np.exp(-(mu_eff - lambda_arrival) * tau)
    return P_actual, mu_eff


def calculate_search_radius(N_min, lambda_eff, R1, R2, h1, h2):
    """
    计算搜索半径

    参数:
        N_min: 最小卫星数量
        lambda_eff: 有效卫星密度 (satellites/km^2)
        R1: 下层轨道半径 (km)
        R2: 上层轨道半径 (km)
        h1: 下层卫星高度 (km)
        h2: 上层卫星高度 (km)

    返回:
        R_search: 搜索半径 (km)
    """
    R_search = np.sqrt(
        (N_min * R2) / (lambda_eff * np.pi * R1) + (h2 - h1) ** 2
    )
    return R_search


def calculate_coverage_area(R_search, R1, R2):
    """
    计算覆盖面积（球冠公式）

    参数:
        R_search: 搜索半径 (km)
        R1: 下层轨道半径 (km)
        R2: 上层轨道半径 (km)

    返回:
        S: 覆盖面积 (km^2)
    """
    cos_theta = (R1 ** 2 + R2 ** 2 - R_search ** 2) / (2 * R1 * R2)
    S = 2 * np.pi * R1 ** 2 * (1 - cos_theta)
    return S


def run_satellite_simulation(R_E=6371, h1=500, h2=600,
                             lambda_arrival=0.5, mu=1.0,
                             P_task=0.95, tau=10, f=0.1,
                             lambda_real=0.001, T_pass=600,
                             verbose=True):
    """
    运行完整的卫星协同计算仿真

    参数:
        R_E: 地球半径 (km)，默认6371
        h1: 下层卫星高度 (km)，默认500
        h2: 上层卫星高度 (km)，默认600
        lambda_arrival: 任务到达率 (tasks/s)，默认0.5
        mu: 单卫星服务率 (tasks/s)，默认1.0
        P_task: 期望任务完成概率，默认0.95
        tau: 任务完成时间限制 (s)，默认10
        f: 不可并行部分比例，默认0.1
        lambda_real: 实际卫星密度 (satellites/km^2)，默认0.001
        T_pass: 平均过境时间 (s)，默认600
        verbose: 是否打印详细信息，默认True

    返回:
        results: 包含所有计算结果的字典
    """
    if verbose:
        print("=" * 60)
        print("卫星协同计算仿真系统")
        print("=" * 60)

        # 显示输入参数
        print("\n输入参数:")
        print(f"  轨道参数: R_E={R_E} km, h1={h1} km, h2={h2} km")
        print(f"  排队论参数: lambda={lambda_arrival} tasks/s, mu={mu} tasks/s, f={f}")
        print(f"  任务需求: P_task={P_task}, tau={tau} s")
        print(f"  卫星密度: lambda_real={lambda_real} sats/km^2, T_pass={T_pass} s")

    try:
        # 1. 计算轨道半径
        R1, R2 = calculate_orbital_radii(R_E, h1, h2)

        # 2. 检查系统稳定性
        rho = check_system_stability(lambda_arrival, mu)

        # 3. 计算可用概率
        p_available = calculate_availability_probability(rho, mu, lambda_arrival, T_pass)

        # 4. 计算有效密度
        lambda_eff = calculate_effective_density(lambda_real, p_available)

        # 5. 计算最小卫星数量
        N_continuous, N_min = calculate_min_satellites(lambda_arrival, mu, tau, P_task, f)

        # 6. 计算实际性能
        P_actual, mu_eff = calculate_actual_probability(N_min, f, mu, lambda_arrival, tau)
        S_amdahl = calculate_amdahl_speedup(N_min, f)

        # 7. 计算搜索半径
        R_search = calculate_search_radius(N_min, lambda_eff, R1, R2, h1, h2)

        # 8. 计算覆盖面积
        S = calculate_coverage_area(R_search, R1, R2)

        # 9. 计算区域内有效卫星数
        N_eff = int(np.floor(lambda_eff * S))

        # 保存结果
        results = {
            'R1': R1,
            'R2': R2,
            'rho': rho,
            'p_available': p_available,
            'lambda_eff': lambda_eff,
            'N_continuous': N_continuous,
            'N_min': N_min,
            'S_amdahl': S_amdahl,
            'mu_eff': mu_eff,
            'P_actual': P_actual,
            'R_search': R_search,
            'S': S,
            'N_eff': N_eff
        }

        if verbose:
            # 显示结果
            print("\n计算结果:")
            print(f"  系统利用率 rho = {rho:.4f}")
            print(f"  可用概率 p_available = {p_available:.4f}")
            print(f"  有效卫星密度 lambda_eff = {lambda_eff:.6f} sats/km^2")
            print(f"\n  所需最小卫星数量 N_min = {N_min}")
            print(f"     (连续值 N = {N_continuous:.4f})")
            print(f"\n  实际加速比 S_Amdahl = {S_amdahl:.4f}x")
            print(f"  有效服务率 mu_eff = {mu_eff:.4f} tasks/s")
            print(f"  实际完成概率 P_actual = {P_actual:.4f} ({P_actual * 100:.2f}%)")
            print(f"  目标完成概率 P_task = {P_task:.4f} ({P_task * 100:.2f}%)")
            print(f"\n  搜索半径 R_search = {R_search:.2f} km")
            print(f"  覆盖面积 S = {S:.2f} km^2")
            print(f"  区域内有效卫星数 N_eff = {N_eff}")
            print("\n" + "=" * 60)

        return results

    except Exception as e:
        if verbose:
            print(f"\n错误：{str(e)}")
        return None


# 使用示例
if __name__ == "__main__":

    # 示例1：基本使用 - 使用默认参数
    print("示例1：使用默认参数\n")
    results = run_satellite_simulation()

    # 示例2：自定义参数
    print("\n\n示例2：自定义参数 - 更高的任务要求\n")
    results2 = run_satellite_simulation(
        lambda_arrival=0.6,  # 更高的到达率
        mu=1.2,  # 更高的服务率
        P_task=0.99,  # 更高的目标概率（99%）
        tau=8,  # 更短的时间限制（8秒）
        f=0.15  # 更大的不可并行比例
    )

    # 示例3：只获取结果不打印详细信息
    print("\n\n示例3：批量计算不同参数组合\n")
    print("P_task\ttau\tN_min\tP_actual")
    print("-" * 40)

    for P in [0.90, 0.95, 0.99]:
        for t in [8, 10, 12]:
            results = run_satellite_simulation(
                P_task=P,
                tau=t,
                verbose=False  # 不打印详细信息
            )
            if results:
                print(f"{P:.2f}\t{t}\t{results['N_min']}\t{results['P_actual']:.4f}")