import numpy as np


class SatelliteSimulation:
    """卫星协同计算仿真系统"""

    def __init__(self):
        # 轨道参数
        self.R_E = 6371  # 地球半径 (km)
        self.h1 = 500  # 下层卫星高度 (km)
        self.h2 = 800  # 上层卫星高度 (km)

        # 排队论参数
        self.lambda_arrival = 0.5 / 3600  # 任务到达率 (tasks/s)
        self.mu = 1.0 / 3600  # 单卫星服务率 (tasks/s)

        # 任务参数
        self.P_task = 0.8  # 期望任务完成概率
        self.tau = 1000  # 任务完成时间限制 (s)
        self.f = 0.1  # 不可并行部分比例

        # 卫星密度参数
        self.lambda_real = 0.001  # 实际卫星密度 (satellites/km²)
        self.T_pass = 600  # 平均过境时间 (s)

        # 计算结果
        self.results = {}

    def calculate_orbital_radii(self):
        """计算轨道半径"""
        R1 = self.R_E + self.h1
        R2 = self.R_E + self.h2
        return R1, R2

    def check_stability(self):
        """检查M/M/1系统稳定性"""
        rho = self.lambda_arrival / self.mu
        if rho >= 1:
            raise ValueError(f"系统不稳定！ρ = {rho:.4f} >= 1，必须满足 λ < μ")
        return rho

    def calculate_availability(self, rho):
        """计算MPPP可用概率"""
        p_available = 1 - rho * np.exp(-(self.mu - self.lambda_arrival) * self.T_pass / 2)
        return p_available

    def calculate_effective_density(self, p_available):
        """计算有效卫星密度"""
        lambda_eff = self.lambda_real * p_available
        return lambda_eff

    def calculate_min_satellites(self):
        """计算所需的最小卫星数量 N_min"""
        # 计算目标加速比
        term1 = self.lambda_arrival / self.mu
        term2 = (1 / (self.mu * self.tau)) * np.log(1 - self.P_task)
        S_amdahl_target = term1 - term2

        # 根据Amdahl定律反解N
        numerator = (1 - self.f) * S_amdahl_target
        denominator = 1 - self.f * S_amdahl_target

        if denominator <= 0:
            raise ValueError("参数组合不可行！无法达到目标完成概率，请调整参数")

        N_continuous = numerator / denominator
        N_min = int(np.ceil(N_continuous))

        return N_continuous, N_min

    def calculate_amdahl_speedup(self, N):
        """计算Amdahl加速比"""
        S_amdahl = 1 / (self.f + (1 - self.f) / N)
        return S_amdahl

    def calculate_actual_probability(self, N):
        """计算实际任务完成概率"""
        S_amdahl = self.calculate_amdahl_speedup(N)
        mu_eff = self.mu * S_amdahl
        P_actual = 1 - np.exp(-(mu_eff - self.lambda_arrival) * self.tau)
        return P_actual, mu_eff

    def calculate_search_radius(self, N_min, lambda_eff):
        """计算搜索半径"""
        R1, R2 = self.calculate_orbital_radii()

        R_search = np.sqrt(
            (N_min * R2) / (lambda_eff * np.pi * R1) + (self.h2 - self.h1) ** 2
        )
        return R_search

    def calculate_coverage_area(self, R_search):
        """计算覆盖面积"""
        R1, R2 = self.calculate_orbital_radii()

        # 使用球冠公式
        cos_theta = (R1 ** 2 + R2 ** 2 - R_search ** 2) / (2 * R1 * R2)
        S = 2 * np.pi * R1 ** 2 * (1 - cos_theta)

        return S

    def run_simulation(self):
        """运行完整仿真"""
        print("=" * 60)
        print("卫星协同计算仿真系统")
        print("=" * 60)

        # 显示输入参数
        print("\n【输入参数】")
        print(f"  轨道参数: R_E={self.R_E} km, h1={self.h1} km, h2={self.h2} km")
        print(f"  排队论参数: λ={self.lambda_arrival} tasks/s, μ={self.mu} tasks/s, f={self.f}")
        print(f"  任务需求: P_task={self.P_task}, τ={self.tau} s")
        print(f"  卫星密度: λ_real={self.lambda_real} sats/km², T_pass={self.T_pass} s")

        try:
            # 1. 计算轨道半径
            R1, R2 = self.calculate_orbital_radii()

            # 2. 检查系统稳定性
            rho = self.check_stability()

            # 3. 计算可用概率
            p_available = self.calculate_availability(rho)

            # 4. 计算有效密度
            lambda_eff = self.calculate_effective_density(p_available)

            # 5. 计算最小卫星数量
            N_continuous, N_min = self.calculate_min_satellites()

            # 6. 计算实际性能
            P_actual, mu_eff = self.calculate_actual_probability(N_min)
            S_amdahl = self.calculate_amdahl_speedup(N_min)

            # 7. 计算搜索半径
            R_search = self.calculate_search_radius(N_min, lambda_eff)

            # 8. 计算覆盖面积
            S = self.calculate_coverage_area(R_search)

            # 9. 计算区域内有效卫星数
            N_eff = int(np.floor(lambda_eff * S))

            # 保存结果
            self.results = {
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

            # 显示结果
            print("\n【计算结果】")
            print(f"  系统利用率 ρ = {rho:.4f}")
            print(f"  可用概率 p_available = {p_available:.4f}")
            print(f"  有效卫星密度 λ_eff = {lambda_eff:.6f} sats/km²")
            print(f"\n  ★ 所需最小卫星数量 N_min = {N_min}")
            print(f"     (连续值 N = {N_continuous:.4f})")
            print(f"\n  实际加速比 S_Amdahl = {S_amdahl:.4f}×")
            print(f"  有效服务率 μ_eff = {mu_eff:.4f} tasks/s")
            print(f"  实际完成概率 P_actual = {P_actual:.4f} ({P_actual * 100:.2f}%)")
            print(f"  目标完成概率 P_task = {self.P_task:.4f} ({self.P_task * 100:.2f}%)")
            print(f"\n  搜索半径 R_search = {R_search:.2f} km")
            print(f"  覆盖面积 S = {S:.2f} km²")
            print(f"  区域内有效卫星数 N_eff = {N_eff}")

            print("\n" + "=" * 60)

            return self.results

        except Exception as e:
            print(f"\n错误：{str(e)}")
            return None


# 使用示例
if __name__ == "__main__":
    # 创建仿真实例
    sim = SatelliteSimulation()

    # 运行仿真
    results = sim.run_simulation()

