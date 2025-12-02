import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_two_layer_satellite_geometry():
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # --- 1. 参数设置 (为了可视化，比例经过夸张处理) ---
    R_earth = 40.0  # 地球半径 (基准)
    h1 = 10.0  # 底层卫星高度
    h2 = 25.0  # 顶层卫星高度 (遥感卫星)

    R1 = R_earth + h1  # 底层轨道半径
    R2 = R_earth + h2  # 顶层轨道半径

    # 假设搜索半径 (斜距) R_search
    # 我们逆向设定一个合理的张角 theta 来反推 R_search，方便绘图
    theta_coverage = np.pi / 5  # 覆盖的半圆心角 (36度)

    # 计算对应的 R_search (利用余弦定理验证)
    # d^2 = R1^2 + R2^2 - 2*R1*R2*cos(theta)
    R_search = np.sqrt(R1 ** 2 + R2 ** 2 - 2 * R1 * R2 * np.cos(theta_coverage))

    # --- 2. 绘制底层轨道球面 (R1) - 也就是"内凹"的目标面 ---
    # 我们只画出一部分球面，避免遮挡，但这部分要体现曲率
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, theta_coverage * 1.3, 30)  # 画得比覆盖范围稍大一点

    x_R1 = R1 * np.outer(np.sin(v), np.cos(u))
    y_R1 = R1 * np.outer(np.sin(v), np.sin(u))
    z_R1 = R1 * np.outer(np.cos(v), np.ones_like(u))

    # 绘制淡蓝色的底层球面网格
    ax.plot_wireframe(x_R1, y_R1, z_R1, color='#4A90E2', alpha=0.2, linewidth=0.5)

    # --- 3. 绘制实际覆盖的球冠 (R1 Surface Cap) ---
    v_cap = np.linspace(0, theta_coverage, 20)
    x_cap = R1 * np.outer(np.sin(v_cap), np.cos(u))
    y_cap = R1 * np.outer(np.sin(v_cap), np.sin(u))
    z_cap = R1 * np.outer(np.cos(v_cap), np.ones_like(u))

    # 用半透明面填充覆盖区域
    ax.plot_surface(x_cap, y_cap, z_cap, color='#5DADE2', alpha=0.4, shade=False)

    # 绘制球冠的边缘圈 (蓝色虚线)
    x_border = R1 * np.sin(theta_coverage) * np.cos(u)
    y_border = R1 * np.sin(theta_coverage) * np.sin(u)
    z_border = np.full_like(x_border, R1 * np.cos(theta_coverage))
    ax.plot(x_border, y_border, z_border, color='#2980B9', linestyle='--', linewidth=1.5)

    # --- 4. 绘制底层卫星 (分布在球冠上) ---
    np.random.seed(10)
    num_sats = 15
    # 随机生成球面点
    rand_v = np.arccos(1 - np.random.rand(num_sats) * (1 - np.cos(theta_coverage)))
    rand_u = 2 * np.pi * np.random.rand(num_sats)

    xs = R1 * np.sin(rand_v) * np.cos(rand_u)
    ys = R1 * np.sin(rand_v) * np.sin(rand_u)
    zs = R1 * np.cos(rand_v)
    ax.scatter(xs, ys, zs, c='#2980B9', marker='o', s=30, label='Computing Sat ($h_1$)')

    # --- 5. 绘制顶层卫星 (R2) ---
    ax.scatter([0], [0], [R2], c='#E74C3C', marker='^', s=120, edgecolors='black', label='Remote Sensing Sat ($h_2$)',
               zorder=10)

    # --- 6. 关键几何连线与标注 ---

    # 6.1 地心 O
    ax.scatter([0], [0], [0], c='black', marker='x', s=50)
    ax.text(0, 0, -2, r'$O$ (Earth Center)', fontsize=12)

    # 6.2 轨道半径连线 R1 和 R2
    # 画 R2 线 (地心到顶层卫星)
    ax.plot([0, 0], [0, 0], [0, R2], color='black', linestyle='-.', linewidth=1)
    ax.text(0.5, 0.5, R2 * 0.8, r'$R_2$', fontsize=12)

    # 画 R1 线 (地心到球冠边缘某点)
    border_idx = 0
    ax.plot([0, x_border[border_idx]], [0, y_border[border_idx]], [0, z_border[border_idx]], color='black',
            linestyle='-.', linewidth=1)
    ax.text(x_border[border_idx] / 2, y_border[border_idx] / 2, z_border[border_idx] / 2, r'$R_1$', fontsize=12)

    # 6.3 搜索半径 R_search (斜线：顶层卫星 -> 球冠边缘)
    ax.plot([0, x_border[border_idx]], [0, y_border[border_idx]], [R2, z_border[border_idx]], color='#E67E22',
            linewidth=2)
    mid_x = x_border[border_idx] / 2
    mid_z = (R2 + z_border[border_idx]) / 2
    ax.text(mid_x, y_border[border_idx] / 2, mid_z, r'$R_{search}$', color='#E67E22', fontsize=12, fontweight='bold')

    # 6.4 标注夹角 Theta (覆盖角)
    # 简单画一个小弧度示意
    # 这里就不画复杂的弧线了，直接在接近地心处标注
    ax.text(2, 0, 5, r'$\theta$', fontsize=14, color='black')

    # --- 7. 视角调整 ---
    ax.set_xlim(-R1, R1)
    ax.set_ylim(-R1, R1)
    ax.set_zlim(0, R2 + 5)

    ax.set_axis_off()
    ax.view_init(elev=10, azim=10)  # 侧视图，最能体现层级高度差

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_two_layer_satellite_geometry()