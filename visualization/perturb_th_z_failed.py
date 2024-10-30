import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from traoptlibrary.traopt_controller import iLQR_Tracking_SE3
import numpy as np
from traoptlibrary.traopt_dynamics import SE3Dynamics
from traoptlibrary.traopt_cost import ErrorStateSE3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_utilis import se3_hat, quatpos2SE3, rotmpos2SE3
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import time
import contextlib

# 定义一个上下文管理器来禁止打印输出
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        try:
            sys.stdout = devnull
            yield
        finally:
            sys.stdout = old_stdout

def perturb_x0(value_range, flag_name):
    """
    根据 flag name 按照range生成对应的x0的一个list 作为后续并行化优化的输入
    flag name 可以是 th_z, th_y, th_x, w_x, w_y, w_z, p_x, p_y, p_z, v_x, v_y, v_z
    """
    x0_list = []
    for value in value_range:
        # 设置默认值
        th_z = 0.
        th_y = 0.
        th_x = 0.
        w_x = 0.
        w_y = 0.
        w_z = 1.
        
        p_x = 0. 
        p_y = 0.
        p_z = 0.
        v_x = 2.0
        v_y = 0.
        v_z = 0.2
        
        # 根据 flag_name 调整对应参数
        if flag_name == "th_z":
            th_z = value
        elif flag_name == "th_y":
            th_y = value
        elif flag_name == "th_x":
            th_x = value
        elif flag_name == "w_x":
            w_x = value
        elif flag_name == "w_y":
            w_y = value
        elif flag_name == "w_z":
            w_z = value
        elif flag_name == "p_x":
            p_x = value
        elif flag_name == "p_y":
            p_y = value
        elif flag_name == "p_z":
            p_z = value
        elif flag_name == "v_x":
            v_x = value
        elif flag_name == "v_y":
            v_y = value
        elif flag_name == "v_z":
            v_z = value
        else:
            raise ValueError(f"未知的 flag name: {flag_name}")
        
        # 生成初始 SE3
        R0 = Rotation.from_euler(
            'zyx', [th_z, th_y, th_x], degrees=True
        ).as_matrix()
        p0 = np.array([1. + p_x, 1. + p_y, -1. + p_z])
        q0 = rotmpos2SE3(R0, p0)
        w0 = np.array([w_x, w_y, w_z]) 
        v0 = np.array([v_x, v_y, v_z])
        xi0 = np.concatenate((w0, v0))
        
        x0 = [q0, xi0]
        x0_list.append(x0)
    
    return x0_list 

def run_optimization( x0 ):
    global ilqr, us_init
    with suppress_stdout():
            xs_ilqr, _, _, _, _, _ = ilqr.fit(
                x0, us_init, n_iterations=200, on_iteration=lambda *args, **kwargs: None
            )
    return (x0, xs_ilqr)

def main():
    
    # 时间步长和模拟时长
    dt = 0.01
    Nsim = int(14/dt)   # Simulation horizon
    
    # ====================
    # Inertia Matrix
    # ====================
    
    m = 1
    Ib = np.diag([0.5, 0.7, 0.9])
    J = np.block([
        [Ib, np.zeros((3, 3))],
        [np.zeros((3, 3)), m * np.identity(3)]
    ])
    
    # =====================================================
    # Tracking Reference Generation 
    # =====================================================
    
    quat0_ref = np.array([1, 0, 0, 0])
    p0_ref = np.array([0, 0, 0])
    w0_ref = np.array([0, 0, 1]) * 1
    v0_ref = np.array([1, 0, 0.1]) * 2
    
    q0_ref = quatpos2SE3(np.concatenate((quat0_ref, p0_ref)))
    xi0_ref = np.concatenate((w0_ref, v0_ref))
    
    q_ref = np.zeros((Nsim + 1, 4, 4))  # SE(3)
    q_ref[0] = q0_ref
    xi_ref = np.zeros((Nsim + 1, 6,)) 
    xi_ref[0] = xi0_ref
    
    X = q0_ref.copy()
    
    for i in range(Nsim):
        xi_ref_rt = xi0_ref.copy()
        X = X @ expm(se3_hat(xi_ref_rt) * dt)
        q_ref[i + 1] = X.copy()
        xi_ref[i + 1] = xi_ref_rt.copy()
    
    # =====================================================
    # Solver Setup
    # =====================================================
    
    N = Nsim
    HESSIANS = False
    action_size = 6
    state_size = 12
    debug_dyn = {"vel_zero": False}

    global ilqr, us_init
    
    # =====================================================
    # Dynamics Instantiation
    # =====================================================
    
    dynamics = SE3Dynamics(J, dt, hessians=HESSIANS, debug=debug_dyn)
    
    # =====================================================
    # Cost Instantiation
    # =====================================================
    
    Q = np.diag([ 
        10., 10., 10., 1., 1., 1.,
        1., 1., 1., 1., 1., 1. 
    ])
    P = np.diag([
        10., 10., 10., 1., 1., 1.,
        1., 1., 1., 1., 1., 1.  
    ]) * 10
    R = np.identity(6) * 1e-5
    cost = ErrorStateSE3TrackingQuadraticGaussNewtonCost(
        Q, R, P, q_ref, xi_ref
    )
    
    # =====================================================
    # Solver Instantiation
    # =====================================================
    
    ilqr = iLQR_Tracking_SE3(dynamics, cost, N, 
                             hessians=HESSIANS,
                             rollout='nonlinear')
    
    # =====================================================
    # Setup
    # =====================================================
    
    th_z_range = np.arange(-160, 160, 0.5)  # 调整步长和范围    
    results_th_z = {
        'th_z': [],
        'trajectories_th_z': []
    }

    us_init = np.zeros((N, action_size,))
    x0_list = perturb_x0( th_z_range, "th_z" )
    
    # =====================================================
    # 并行化优化过程
    # =====================================================

    print("开始并行化遍历 th_z_range 并运行优化")
    start_time = time.time()
    
    # 使用 joblib 进行并行化
    n_jobs = -1  # 使用所有可用的CPU核心

    parallel_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_optimization)(x0) for x0 in x0_list
    )
    
    # 处理并行化的结果
    for i, res in enumerate(parallel_results):
        if len(res) == 2:
            _, trajectory = res
            if trajectory is not None:
                final_positions = np.array([x[0][:3, 3] for x in trajectory])
                th_z_val = th_z_range[i]
                results_th_z['th_z'].append(th_z_val)
                results_th_z['trajectories_th_z'].append(final_positions)
            else:
                print(f"th_z = {th_z_val}° 优化失败: 无轨迹返回")
        else:
            th_z_val, trajectory, error_msg = res
            print(f"th_z = {th_z_val}° 优化失败: {error_msg}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"整体优化时间: {total_time:.2f} 秒")
    
    # =====================================================
    # 保存结果到 visualization 文件夹
    # =====================================================
    
    visualization_dir = 'visualization'
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
        print(f"创建文件夹: {visualization_dir}")
    
    save_path = os.path.join(visualization_dir, 'results_perturb_th_z.npz')
    np.savez_compressed(save_path, th_z=np.array(results_th_z['th_z']),
                        trajectories=np.array(results_th_z['trajectories_th_z']))
    print(f"优化结果已保存到 {save_path}")

    # =====================================================
    # 可视化部分
    # =====================================================
    
    # 定义颜色映射
    cmap = cm.get_cmap('viridis')
    
    # 绘制轨迹
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(vmin=min(results_th_z['th_z']), vmax=max(results_th_z['th_z']))
    
    for i, traj in enumerate(results_th_z['trajectories']):
        color = cmap(norm(results_th_z['th_z'][i]))
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=1)
    
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(results_th_z['th_z'])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label('th_z (degrees)')
    
    ax.set_title('Trajectories with varying th_z')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    # =====================================================
    # 结束
    # =====================================================
    print("所有优化和可视化完成")

if __name__ == "__main__":
    main()
