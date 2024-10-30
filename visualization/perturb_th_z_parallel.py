import sys
import os
from joblib import Parallel, delayed
import contextlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
import time

# Define a context manager to suppress stdout
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
    with suppress_stdout():  # Suppress fit function's print output
        xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr, grad_hist_ilqr = \
            ilqr.fit(initial_conditions, us_init, n_iterations=n_iterations, on_iteration=on_iteration)
    return xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr, grad_hist_ilqr

def optimize_for_th_z(th_z, q_ref, xi_ref, dynamics_params, cost_params, N, action_size, HESSIANS, debug_dyn):
    try:
        # Unpack dynamics and cost parameters
        J, dt = dynamics_params
        Q, R, P = cost_params

        # Other angles remain at initial values
        th_y = 0
        th_x = 0
        # Angular velocity remains at initial values
        w_x = 0
        w_y = 0
        w_z = 1.0  # Original w_z

        # Generate initial SE3
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

        # Instantiate dynamics
        dynamics = SE3Dynamics(J, dt, hessians=HESSIANS, debug=debug_dyn)

        # Instantiate cost
        cost = ErrorStateSE3TrackingQuadraticGaussNewtonCost(
            Q, R, P, q_ref, xi_ref
        )

        # Instantiate iLQR
        ilqr = iLQR_Tracking_SE3(dynamics, cost, N, 
                                 hessians=HESSIANS,
                                 rollout='nonlinear')

        # Run optimization
        xs_ilqr, us_ilqr, _, _, _, _ = run_optimization(x0, ilqr, us_init, N)

        # Extract position trajectory
        final_positions = np.array([x[0][:3, 3] for x in xs_ilqr])

        print(f"th_z={th_z} optimization completed")
        return th_z, final_positions
    except Exception as e:
        print(f"th_z={th_z} optimization failed: {e}")
        return th_z, None

def main():
    # Time step and simulation duration
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

    # Precompute dynamics and cost parameters to pass to the parallel function
    dynamics_params = (J, dt)
    cost_params = (Q, R, P)

    # =====================================================
    # Setup th_z_range
    # =====================================================
    th_z_range = np.arange(150, 210, 0.5)  # Adjust step size and range as needed    

    # =====================================================
    # Run Parallel Optimization
    # =====================================================
    print("Starting parallel optimization over th_z_range")

    # Number of parallel jobs, set to -1 to use all available cores
    num_jobs = -1

    # Execute parallel optimization
    results = Parallel(n_jobs=num_jobs, verbose=5)(
        delayed(optimize_for_th_z)(
            th_z, q_ref, xi_ref, dynamics_params, cost_params, N, action_size, HESSIANS, debug_dyn
        ) for th_z in th_z_range
    )

    # =====================================================
    # Collect Results
    # =====================================================
    results_th_z = {
        'th_z': [],
        'trajectories': []
    }

    for th_z, traj in results:
        if traj is not None:
            results_th_z['th_z'].append(th_z)
            results_th_z['trajectories'].append(traj)
        else:
            print(f"th_z={th_z} has no trajectory due to optimization failure.")

    # =====================================================
    # Save Results to Visualization Folder
    # =====================================================
    visualization_dir = 'results'
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
        print(f"Created directory: {visualization_dir}")

    save_path = os.path.join(visualization_dir, 'results_perturb_th_z.npz')
    np.savez_compressed(save_path, th_z=np.array(results_th_z['th_z']),
                        trajectories=np.array(results_th_z['trajectories']))
    print(f"Optimization results saved to {save_path}")

    # =====================================================
    # Visualization
    # =====================================================
    # Define color mapping
    cmap = cm.get_cmap('viridis')

    # Plot trajectories
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
    # Completion Message
    # =====================================================
    print("All optimizations and visualizations are complete.")

if __name__ == "__main__":
    main()
