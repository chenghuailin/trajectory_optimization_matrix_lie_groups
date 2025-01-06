import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 绘图
import sys
sys.path.append('..')
sys.path.append('.')
from traoptlibrary.traopt_utilis import euler2quat, se3_vee
from manifpy import SE3, SE3Tangent


# 定义参考位置和姿态
p0 = np.array([1.0, 1.0, -1.0])
quat_ref = Rotation.from_euler('zyx', [30.0, 10.0, 0.0], degrees=True).as_quat()
q_ref_mnf = SE3(position=p0, quaternion=quat_ref)

# 定义 th_z 和 th_y 的角度范围
angle_min, angle_max, angle_step = -360.0, 360.0, 1.0
angle_range_z = np.arange(angle_min, angle_max + angle_step, angle_step)
angle_range_y = np.arange(angle_min, angle_max + angle_step, angle_step)

# 创建 th_z 和 th_y 的二维网格
angle_z, angle_y = np.meshgrid(angle_range_z, angle_range_y)

# 初始化用于存储误差范数的二维数组
errs_left_norm = np.zeros_like(angle_z)
errs_right_norm = np.zeros_like(angle_z)  # 如果需要，可计算
errs_cmptb_norm = np.zeros_like(angle_z)  # 如果需要，可计算

# 计算误差范数
print("开始计算误差范数...")

for i in range(angle_z.shape[0]):
    for j in range(angle_z.shape[1]):
        th_z = angle_z[i, j]
        th_y = angle_y[i, j]
        
        # 计算当前角度对应的四元数
        quat0 = Rotation.from_euler('zyx', [th_z, th_y, 0.0], degrees=True).as_quat()
        
        # 创建 SE3 变换
        q_mnf = SE3(position=p0, quaternion=quat0)
        
        # 计算左误差
        error_left = q_mnf.lminus(q_ref_mnf)
        errs_left_norm[i, j] = np.linalg.norm(error_left.coeffs())
        
        # 如果需要计算右误差和兼容误差，请确保 SE3 类中有相应的方法
        # error_right = q_ref_mnf.rminus(q_mnf)
        # errs_right_norm[i, j] = np.linalg.norm(error_right.coeffs())
        
        # error_cmptb = q_mnf.compatible_minus(q_ref_mnf)  # 替换为实际方法
        # errs_cmptb_norm[i, j] = np.linalg.norm(error_cmptb)
    
    # 可选：每隔一定行数打印进度
    if i % 50 == 0:
        print(f"已完成 {i}/{angle_z.shape[0]} 行。")

print("误差范数计算完成。")

# 绘制交互式 3D 曲面图（Left Error Norm）
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 为了节省内存和提升绘图速度，可以适当减少绘图点的密度
stride = 8  # 调整步长以控制绘图密度
surf = ax.plot_surface(
    angle_z[::stride, ::stride],
    angle_y[::stride, ::stride],
    errs_left_norm[::stride, ::stride],
    cmap='viridis',
    edgecolor='none',
    alpha=0.8
)

# 设置坐标轴标签和标题
ax.set_xlabel(r'Z-axis angle $\theta_z$ ($^\circ$)')
ax.set_ylabel(r'Y-axis angle $\theta_y$ ($^\circ$)')
ax.set_zlabel('Left Error Norm')
ax.set_title(r'Left Error Norm $||\text{Log}(R(\theta_z,\theta_y)R_{ref}^{-1})||_2$, $\theta_x = 0^{\circ}$')

# 添加颜色条，关联到 surf 对象
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Left Error Norm')

plt.show()

# 如果需要绘制右误差和兼容误差的 3D 图，可以参考以下代码：
"""
# 绘制右误差 3D 曲面图
fig_right = plt.figure(figsize=(14, 10))
ax_right = fig_right.add_subplot(111, projection='3d')
surf_right = ax_right.plot_surface(
    angle_z[::stride, ::stride],
    angle_y[::stride, ::stride],
    errs_right_norm[::stride, ::stride],
    cmap='plasma',
    edgecolor='none',
    alpha=0.8
)
ax_right.set_xlabel('th_z (degrees)', fontsize=12)
ax_right.set_ylabel('th_y (degrees)', fontsize=12)
ax_right.set_zlabel('Right Error Norm', fontsize=12)
ax_right.set_title('Right Error Norm as a Function of th_z and th_y', fontsize=14)
fig_right.colorbar(surf_right, ax=ax_right, shrink=0.5, aspect=10, label='Right Error Norm')
plt.show()

# 绘制兼容误差 3D 曲面图
fig_cmptb = plt.figure(figsize=(14, 10))
ax_cmptb = fig_cmptb.add_subplot(111, projection='3d')
surf_cmptb = ax_cmptb.plot_surface(
    angle_z[::stride, ::stride],
    angle_y[::stride, ::stride],
    errs_cmptb_norm[::stride, ::stride],
    cmap='inferno',
    edgecolor='none',
    alpha=0.8
)
ax_cmptb.set_xlabel('th_z (degrees)', fontsize=12)
ax_cmptb.set_ylabel('th_y (degrees)', fontsize=12)
ax_cmptb.set_zlabel('Compatible Error Norm', fontsize=12)
ax_cmptb.set_title('Compatible Error Norm as a Function of th_z and th_y', fontsize=14)
fig_cmptb.colorbar(surf_cmptb, ax=ax_cmptb, shrink=0.5, aspect=10, label='Compatible Error Norm')
plt.show()
"""
