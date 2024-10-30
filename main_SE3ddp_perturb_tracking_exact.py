from traoptlibrary.traopt_controller import iLQR_Tracking_SE3
import numpy as np
from traoptlibrary.traopt_dynamics import SE3Dynamics
from traoptlibrary.traopt_cost import ErrorStateSE3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_utilis import se3_hat, quatpos2SE3, parallel_SE32manifSE3, rotmpos2SE3
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import time
import os
import sys
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

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged, grad_wrt_input_norm,
                alpha, mu, J_hist, xs_hist, us_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("Iteration", iteration_count, info, J_opt, grad_wrt_input_norm, alpha, mu)

def run_optimization(initial_conditions, ilqr, us_init, N, n_iterations=200):
    with suppress_stdout():  # 禁止 fit 函数的打印输出
        xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr, grad_hist_ilqr = \
            ilqr.fit(initial_conditions, us_init, n_iterations=n_iterations, on_iteration=on_iteration)
    return xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr, grad_hist_ilqr

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
    
    th_z_range = np.arange(160, 200, 0.5)  # 根据需要调整步长和范围    
    results_th_z = {
        'th_z': [],
        'trajectories': []
    }
    
    # =====================================================
    # 遍历 th_z_range 并运行优化
    # =====================================================
    
    print("开始遍历 th_z_range 并运行优化")

    for idx, th_z in enumerate(th_z_range):
        if idx % 50 == 0:
            print(f"当前处理th_z={th_z} ({idx}/{len(th_z_range)})")
        try:
            # 其他角度保持为初始值
            th_y = 0
            th_x = 0
            # 角速度保持为初始值
            w_x = 0
            w_y = 0
            w_z = 1.0  # 原始的 w_z
            
            # 生成初始SE3
            R0 = Rotation.from_euler(
                'zyx', [th_z, th_y, th_x], degrees=True
            ).as_matrix()
            p0 = np.array([1., 1., -1.])
            q0 = rotmpos2SE3(R0, p0)
            w0 = np.array([w_x, w_y, w_z]) 
            v0 = np.array([2., 0., 0.2])
            xi0 = np.concatenate((w0, v0))
            
            x0 = [q0, xi0]
            us_init = np.zeros((N, action_size,))
            
            # 运行优化并抑制打印输出
            xs_ilqr, us_ilqr, _, _, _, _ = run_optimization(x0, ilqr, us_init, N)
            
            # 提取位置
            final_positions = np.array([x[0][:3, 3] for x in xs_ilqr])
            results_th_z['th_z'].append(th_z)
            results_th_z['trajectories'].append(final_positions)
            print(f"th_z={th_z} 优化完成")
        except Exception as e:
            print(f"th_z={th_z} 优化失败: {e}")
    
    # =====================================================
    # 保存结果到 visualization 文件夹
    # =====================================================
    
    visualization_dir = 'visualization'
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
        print(f"创建文件夹: {visualization_dir}")
    
    save_path = os.path.join(visualization_dir, 'results_perturb_th_z.npz')
    np.savez_compressed(save_path, th_z=np.array(results_th_z['th_z']),
                        trajectories=np.array(results_th_z['trajectories']))
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
