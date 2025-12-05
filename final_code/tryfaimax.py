import numpy as np


def calculate_phi_max(h, gamma, R=6371, is_degree=False):
    """
    计算最大接入角 phi_max

    参数:
    h : float
        卫星高度 (单位: km，需与 R 保持一致)
    gamma : float or np.array
        波束宽度 (默认单位: 弧度。如果输入是角度，请设置 is_degree=True)
    R : float, optional
        地球半径 (默认 6371 km)
    is_degree : bool, optional
        如果 gamma 输入的是角度(degrees)，请设为 True，函数会自动转为弧度计算

    返回:
    phi_max : float or np.array
        最大接入角 (单位: 弧度)
    """

    # 1. 如果输入是角度，先转为弧度
    if is_degree:
        gamma_rad = np.deg2rad(gamma)
    else:
        gamma_rad = gamma

    # 2. 计算阈值 gamma_0
    # 公式: gamma_0 = arcsin(R / (R + h)) / 2
    gamma_0 = np.arcsin(R / (R + h)) * 2
    # 3. 准备两个情况的计算公式
    # --- 情况 A: gamma < gamma_0 ---
    # 公式: arcsin( (R+h)/R * sin(gamma/2) ) - gamma/2
    # 注意：需防止数值误差导致 arcsin 参数略微超过 1
    term_inside = ((R + h) / R) * np.sin(gamma_rad / 2)
    # np.clip 用于处理浮点数精度误差，防止报错 (例如 1.000000001)
    term_inside = np.clip(term_inside, -1.0, 1.0)
    phi_case1 = np.arcsin(term_inside) - (gamma_rad / 2)

    # --- 情况 B: gamma >= gamma_0 ---
    # 公式: arccos( R / (R+h) )
    phi_case2 = np.arccos(R / (R + h))
    # 4. 根据条件选择结果 (支持数组输入)
    # 如果 gamma_rad < gamma_0 返回 phi_case1，否则返回 phi_case2
    phi_max = np.where(gamma_rad < gamma_0, phi_case1, phi_case2)

    return phi_max


# ==========================================
# 使用示例
# ==========================================

h_val = 500  # 高度 1000 km
gamma_val_deg = 133  # 波束宽度 30 度

# 计算结果 (输入角度，输出也是弧度，如果想看角度需要转换)
result_rad = calculate_phi_max(h_val, gamma_val_deg, is_degree=True)
result_deg = np.degrees(result_rad)

print(f"输入: h={h_val} km, gamma={gamma_val_deg}°")
print(f"计算出的 phi_max (弧度): {result_rad:.4f}")
print(f"计算出的 phi_max (角度): {result_deg:.4f}°")

# 验证临界值 gamma_0
g0 = np.arcsin(6371 / (6371 + h_val))
print(f"当前高度下的临界角 gamma_0: {np.degrees(g0):.4f}°")
