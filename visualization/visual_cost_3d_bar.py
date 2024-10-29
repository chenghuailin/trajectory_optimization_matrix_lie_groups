import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # 启用3D绘图
import os
from manifpy import SE3, SE3Tangent  # 使用 manifpy 的 SE3 类

# 声明 surf 为全局变量
surf = None

def main():
    global surf  # 声明 surf 为全局变量，以便在 update 函数中访问和修改

    # 定义参考位置和姿态
    p0 = np.array([1.0, 1.0, -1.0])
    quat_ref = Rotation.from_euler('zyx', [60.0, 60.0, 0.0], degrees=True).as_quat()
    q_ref_mnf = SE3(p0, quat_ref)
    
    # 定义 th_z、th_y 和 th_x 的角度范围
    angle_min, angle_max, angle_step = -360.0, 360.0, 5.0  # 步长为30度，您可根据需要调整
    angle_range_z = np.arange(angle_min, angle_max + angle_step, angle_step)
    angle_range_y = np.arange(angle_min, angle_max + angle_step, angle_step)
    angle_range_x = np.arange(angle_min, angle_max + angle_step, angle_step)
    
    # 创建 th_z、th_y 和 th_x 的三维网格
    angle_z_grid, angle_y_grid, angle_x_grid = np.meshgrid(angle_range_z, angle_range_y, angle_range_x, indexing='ij')
    
    # 初始化用于存储误差范数的三维数组
    errs_left_norm = np.zeros_like(angle_z_grid)
    
    # 计算误差范数
    print("开始计算误差范数...")
    
    for i in range(angle_z_grid.shape[0]):
        for j in range(angle_z_grid.shape[1]):
            for k in range(angle_z_grid.shape[2]):
                th_z_val = angle_z_grid[i, j, k]
                th_y_val = angle_y_grid[i, j, k]
                th_x_val = angle_x_grid[i, j, k]
                
                # 计算当前角度对应的四元数
                quat0 = Rotation.from_euler('zyx', [th_z_val, th_y_val, th_x_val], degrees=True).as_quat()
                
                # 创建 SE3 变换
                q_mnf = SE3(p0, quat0)
                
                # 计算左误差
                error_left = q_mnf.lminus(q_ref_mnf)
                
                # 计算四元数的欧几里得范数作为误差范数
                errs_left_norm[i, j, k] = np.linalg.norm(error_left.coeffs())
        
        # 可选：打印进度
        if (i+1) % 20 == 0 or (i+1) == angle_z_grid.shape[0]:
            print(f"已完成 {i+1}/{angle_z_grid.shape[0]} 个 th_z。")
    
    print("误差范数计算完成。")
    
    # 可视化部分
    # 为了交互性，选择一个初始的 th_x 值
    initial_th_x = angle_range_x[0]
    
    # 创建图形和3D坐标轴
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 选择初始 th_x 的数据并重塑为二维
    idx_initial = np.where(angle_range_x == initial_th_x)[0][0]
    th_z_2d = angle_z_grid[:, :, idx_initial]
    th_y_2d = angle_y_grid[:, :, idx_initial]
    errs_2d = errs_left_norm[:, :, idx_initial]
    
    # 初始绘制曲面图
    surf = ax.plot_surface(
        th_z_2d,
        th_y_2d,
        errs_2d,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8
    )
    
    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, pad=0.1)
    cbar.set_label('Left Error Norm')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('th_z (degrees)', fontsize=12)
    ax.set_ylabel('th_y (degrees)', fontsize=12)
    ax.set_zlabel('Left Error Norm', fontsize=12)
    ax.set_title(f'Left Error Norm for th_x = {initial_th_x}°', fontsize=14)
    
    # 调整图形布局以容纳滑块
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    # 创建滑块轴
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='th_x (°)',
        valmin=angle_min,
        valmax=angle_max,
        valinit=initial_th_x,
        valstep=angle_step,
        color='lightblue'
    )
    
    # 定义滑块更新函数
    def update(val):
        global surf  # 声明为全局变量，以便在函数内部修改
        th_x_val = slider.val
        # 更新标题
        ax.set_title(f'Left Error Norm for th_x = {th_x_val}°', fontsize=14)
        
        # 移除旧的曲面图
        surf.remove()
        
        # 选择新的 th_x 数据并重塑为二维
        idx = np.where(angle_range_x == th_x_val)[0]
        if len(idx) == 0:
            print(f"th_x = {th_x_val}° 不在角度范围内。")
            return
        idx = idx[0]
        
        th_z_2d_new = angle_z_grid[:, :, idx]
        th_y_2d_new = angle_y_grid[:, :, idx]
        errs_2d_new = errs_left_norm[:, :, idx]
        
        # 绘制新的曲面图
        surf = ax.plot_surface(
            th_z_2d_new,
            th_y_2d_new,
            errs_2d_new,
            cmap='viridis',
            edgecolor='none',
            alpha=0.8
        )
        
        # 更新颜色条
        cbar.update_normal(surf)
        
        # 刷新图形
        fig.canvas.draw_idle()
    
    # 绑定滑块事件
    slider.on_changed(update)
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    main()
