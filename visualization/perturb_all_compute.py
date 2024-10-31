import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from joblib import Parallel, delayed
import contextlib
import numpy as np
from traoptlibrary.traopt_controller import iLQR_Tracking_SE3
from traoptlibrary.traopt_dynamics import SE3Dynamics
from traoptlibrary.traopt_cost import ErrorStateSE3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_utilis import se3_hat, quatpos2SE3, rotmpos2SE3
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle

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

def optimize_parameter(param_name, param_value, dynamics_params, cost_params, q_ref, xi_ref, N, action_size, HESSIANS, debug_dyn):
    try:
        # Initialize parameters with default values
        params = {
            'th_z': 0.0,
            'th_y': 0.0,
            'th_x': 0.0,
            'w_x': 0.0,
            'w_y': 0.0,
            'w_z': 1.0,
            'p_x': 1.0,
            'p_y': 1.0,
            'p_z': -1.0,
            'v_x': 0.2,
            'v_y': 0.0,
            'v_z': 2.0
        }

        # Update only the specific parameter
        params[param_name] = param_value

        # Extract parameters
        th_z = params['th_z']
        th_y = params['th_y']
        th_x = params['th_x']
        w_x = params['w_x']
        w_y = params['w_y']
        w_z = params['w_z']
        p_x = params['p_x']
        p_y = params['p_y']
        p_z = params['p_z']
        v_x = params['v_x']
        v_y = params['v_y']
        v_z = params['v_z']

        # Generate initial SE3
        R0 = Rotation.from_euler(
            'zyx', [th_z, th_y, th_x], degrees=True
        ).as_matrix()
        p0 = np.array([p_x, p_y, p_z])
        q0 = rotmpos2SE3(R0, p0)
        w0 = np.array([w_x, w_y, w_z]) 
        v0 = np.array([v_x, v_y, v_z])
        xi0 = np.concatenate((w0, v0))

        x0 = [q0, xi0]
        us_init = np.zeros((N, action_size,))

        # Unpack dynamics and cost parameters
        J, dt = dynamics_params
        Q, R, P = cost_params

        # Instantiate dynamics
        dynamics = SE3Dynamics(J, dt, hessians=HESSIANS, debug=debug_dyn)

        # Instantiate cost
        cost = ErrorStateSE3TrackingQuadraticGaussNewtonCost(Q, R, P, q_ref, xi_ref)

        # Instantiate iLQR
        ilqr = iLQR_Tracking_SE3(dynamics, cost, N, 
                                 hessians=HESSIANS,
                                 rollout='nonlinear')

        # Run optimization
        xs_ilqr, us_ilqr, _, _, _, _ = run_optimization(x0, ilqr, us_init, N)

        print(f"Parameter {param_name}={param_value} optimization completed")
        return param_name, param_value, xs_ilqr

    except Exception as e:
        print(f"Parameter {param_name}={param_value} optimization failed: {e}")
        return param_name, param_value, None

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

    # =====================================================
    # Setup Parameter Ranges
    # =====================================================
    parameter_ranges = {
        'th_z': np.arange(150., 210., 0.5),
        'th_y': np.arange(150., 210., 0.5),
        'th_x': np.arange(150., 210., 0.5),
        'w_x': np.arange(-3, 3, 0.05),
        'w_y': np.arange(-2, 2, 0.05),
        'w_z': np.arange(-1., 1., 0.025) + 1.,
        'p_x': np.arange(-50., 50., 2.),
        'p_y': np.arange(-50., 50., 2.),
        'p_z': np.arange(-50., 50., 2.),
        'v_x': np.arange(-10., 10., .5),
        'v_y': np.arange(-10., 10., .5),
        'v_z': np.arange(-10., 10., .5),
    }

    results = {
        'params':{
            'th_z': [],
            'th_y': [],
            'th_x': [],
            'w_x': [],
            'w_y': [],
            'w_z': [],
            'p_x': [],
            'p_y': [],
            'p_z': [],
            'v_x': [],
            'v_y': [],
            'v_z': []
        },
        'trajectories': {
            'th_z': [],
            'th_y': [],
            'th_x': [],
            'w_x': [],
            'w_y': [],
            'w_z': [],
            'p_x': [],
            'p_y': [],
            'p_z': [],
            'v_x': [],
            'v_y': [],
            'v_z': []
        }
    }

    # =====================================================
    # Precompute Dynamics and Cost Parameters
    # =====================================================
    dynamics_params = (J, dt)
    cost_params = (Q, R, P)

    # =====================================================
    # Run Parallel Optimization for Each Parameter in Batches
    # =====================================================
    print("Starting parallel optimization for multiple parameters")
    start_time = time.time()

    # Number of parallel jobs, set to -1 to use all available cores
    num_jobs = -1

    # Iterate over each parameter and optimize all its values in parallel
    for param_name, param_values in parameter_ranges.items():
        print(f"Starting parallel optimization for parameter: {param_name}")
        results_all = Parallel(n_jobs=num_jobs, verbose=5)(
            delayed(optimize_parameter)(
                param_name, param_value, dynamics_params, cost_params, q_ref, xi_ref, 
                N, action_size, HESSIANS, debug_dyn
            ) for param_value in param_values
        )

        # =====================================================
        # Collect Results
        # =====================================================
        for param_name_res, param_value, traj in results_all:
            if traj is not None:
                results['params'][param_name_res].append(param_value)
                results['trajectories'][param_name_res].append(traj)
            else:
                print(f"Parameter {param_name_res}={param_value} has no trajectory due to optimization failure.")

    print(f"Optimization for all parameters finished! Time used:{time.time()-start_time}")
    # =====================================================
    # Save Results to Visualization Folder
    # =====================================================
    visualization_dir = 'visualization/results'
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
        print(f"Created directory: {visualization_dir}")

    save_path = os.path.join(visualization_dir, f'results_parameters.pkl')

    # # Convert lists to numpy arrays for saving
    # np_save_dict = {}
    # for key in results['params']:
    #     np_save_dict[key] = np.array(results['params'][key])
    # for key in results['trajectories']:
    #     np_save_dict[key] = np.array(results['trajectories'][key])
    # np.savez_compressed(save_path, **np_save_dict)

    # Use pickle to save the whole results dictionary
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Optimization results saved to {save_path}")

    # =====================================================
    # Visualization
    # =====================================================
    # Define color mapping for each parameter separately
    for param_name in parameter_ranges.keys():
        if len(results['params'][param_name]) == 0:
            print(f"No successful results for parameter {param_name}, skipping visualization.")
            continue

        # Define color mapping
        cmap = cm.get_cmap('viridis')
        norm = plt.Normalize(vmin=min(results['params'][param_name]), vmax=max(results['params'][param_name]))

        # Plot trajectories
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, traj in enumerate(results['trajectories'][param_name]):
            color = cmap(norm(results['params'][param_name][i]))
            pos = np.array([traj_step[0][:3, 3] for traj_step in traj])
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=1)

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(results['params'][param_name])
        cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
        cbar.set_label(f'{param_name} (units)')  # 替换为实际单位，如度或米

        ax.set_title(f'Trajectories with varying {param_name}')
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
