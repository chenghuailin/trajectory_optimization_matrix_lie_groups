import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # 启用3D绘图
from manifpy import SE3, SE3Tangent  # 使用 manifpy 的 SE3 类
from joblib import Parallel, delayed  # 并行计算
import time

# 声明 surf 为全局变量
surf = None

def compute_error(i, j, k, angle_z_grid, angle_y_grid, angle_x_grid, p0, quat_ref):
    """
    计算给定角度组合下的误差范数。
    """
    # 在每个调用中重新创建 q_ref_mnf，避免传递不可序列化的 SE3 对象
    q_ref_mnf = SE3(p0, quat_ref)
    
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
    error_norm = np.linalg.norm(error_left.coeffs())
    
    return (i, j, k, error_norm)

def main():
    global surf  # 声明 surf 为全局变量，以便在 update 函数中访问和修改

    # 定义参考位置和姿态
    p0 = np.array([1.0, 1.0, -1.0])
    quat_ref = Rotation.from_euler('zyx', [60.0, 60.0, 0.0], degrees=True).as_quat()
    
    # 定义 th_z、th_y 和 th_x 的角度范围
    angle_min, angle_max, angle_step = -360.0, 360.0, 5.0  # 步长为5度
    angle_range_z = np.arange(angle_min, angle_max + angle_step, angle_step)
    angle_range_y = np.arange(angle_min, angle_max + angle_step, angle_step)
    angle_range_x = np.arange(angle_min, angle_max + angle_step, angle_step)
    
    # 创建 th_z、th_y 和 th_x 的三维网格
    angle_z_grid, angle_y_grid, angle_x_grid = np.meshgrid(angle_range_z, angle_range_y, angle_range_x, indexing='ij')
    
    # 初始化用于存储误差范数的三维数组
    errs_left_norm = np.zeros_like(angle_z_grid)
    
    # 计算误差范数
    print("开始并行计算误差范数...")
    start_time = time.time()
    
    # 使用并行计算
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(compute_error)(i, j, k, angle_z_grid, angle_y_grid, angle_x_grid, p0, quat_ref)
        for i in range(angle_z_grid.shape[0])
        for j in range(angle_z_grid.shape[1])
        for k in range(angle_z_grid.shape[2])
    )
    
    # 填充 errs_left_norm 数组
    for res in results:
        i, j, k, error_norm = res
        errs_left_norm[i, j, k] = error_norm
    
    end_time = time.time()
    print(f"误差范数计算完成。耗时: {end_time - start_time:.2f} 秒")
    
    # 可视化部分
    # 定义动画的帧数
    frames = len(angle_range_x)
    
    # 创建图形和3D坐标轴
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 调整子图以为按钮留出空间
    plt.subplots_adjust(bottom=0.2)
    
    # 初始 th_x 的数据
    current_frame = [0]  # 使用列表使其在回调中可变
    is_paused = [False]   # 使用列表使其在回调中可变
    
    th_z_2d = angle_z_grid[:, :, current_frame[0]]
    th_y_2d = angle_y_grid[:, :, current_frame[0]]
    errs_2d = errs_left_norm[:, :, current_frame[0]]
    
    # 确定颜色条的全局最小值和最大值
    global_min = np.min(errs_left_norm)
    global_max = np.max(errs_left_norm)
    
    # 创建归一化对象
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=global_min, vmax=global_max)
    
    # 初始绘制曲面图，使用固定的归一化
    surf = ax.plot_surface(
        th_z_2d,
        th_y_2d,
        errs_2d,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8,
        norm=norm
    )
    
    # 添加颜色条，使用固定的归一化
    cbar = fig.colorbar(surf, ax=ax, pad=0.1)
    cbar.set_label('Left Error Norm')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('th_z (degrees)', fontsize=12)
    ax.set_ylabel('th_y (degrees)', fontsize=12)
    ax.set_zlabel('Left Error Norm', fontsize=12)
    title = ax.set_title(f'Left Error Norm for th_x = {angle_range_x[current_frame[0]]}°', fontsize=14)
    
    # 设置坐标轴范围（可选）
    ax.set_xlim(angle_min, angle_max)
    ax.set_ylim(angle_min, angle_max)
    ax.set_zlim(global_min, global_max)
    
    # 定义更新函数
    def update(frame):
        # 更新当前帧
        current_frame[0] = frame % frames  # 确保循环
        
        # 移除旧的曲面图
        global surf
        surf.remove()
        
        # 获取当前 th_x 的值
        th_x_val = angle_range_x[current_frame[0]]
        
        # 选择新的 th_x 数据并重塑为二维
        th_z_2d_new = angle_z_grid[:, :, current_frame[0]]
        th_y_2d_new = angle_y_grid[:, :, current_frame[0]]
        errs_2d_new = errs_left_norm[:, :, current_frame[0]]
        
        # 绘制新的曲面图，使用固定的归一化
        surf = ax.plot_surface(
            th_z_2d_new,
            th_y_2d_new,
            errs_2d_new,
            cmap='viridis',
            edgecolor='none',
            alpha=0.8,
            norm=norm
        )
        
        # 更新标题
        title.set_text(f'Left Error Norm for th_x = {th_x_val}°')
        
        return surf,

    # 创建动画
    anim = FuncAnimation(
        fig, update, frames=frames, interval=100, blit=False
    )
    
    # 定义按钮的位置和大小
    axplay = plt.axes([0.1, 0.05, 0.1, 0.075])
    axforward = plt.axes([0.21, 0.05, 0.1, 0.075])
    axbackward = plt.axes([0.32, 0.05, 0.1, 0.075])
    
    # 创建按钮
    bplay = Button(axplay, 'Pause')
    bforward = Button(axforward, 'Forward')
    bbackward = Button(axbackward, 'Backward')
    
    # 定义按钮回调函数
    def play_pause(event):
        if is_paused[0]:
            anim.event_source.start()
            bplay.label.set_text('Pause')
            is_paused[0] = False
        else:
            anim.event_source.stop()
            bplay.label.set_text('Play')
            is_paused[0] = True
        plt.draw()
    
    def forward(event):
        if is_paused[0]:
            next_frame = (current_frame[0] + 1) % frames
            update(next_frame)
            plt.draw()
    
    def backward(event):
        if is_paused[0]:
            next_frame = (current_frame[0] - 1) % frames
            update(next_frame)
            plt.draw()
    
    # 绑定按钮事件
    bplay.on_clicked(play_pause)
    bforward.on_clicked(forward)
    bbackward.on_clicked(backward)
    
    # 显示图形并保持交互
    plt.show()

if __name__ == "__main__":
    main()
