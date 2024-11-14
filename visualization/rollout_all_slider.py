import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib.widgets import Slider
from traoptlibrary.traopt_utilis import se3_hat, quatpos2SE3, SE32manifSE3, \
    parallel_SE32manifSE3, rotm2euler
from scipy.linalg import expm

def load_results(save_path):
    with open(save_path, 'rb') as f:
        results = pickle.load(f)
    return results

save_path = 'visualization/results/results_perturb_all_parameters.pkl'
results_sol = load_results(save_path)

save_path = 'visualization/results/results_rollout_all_parameters.pkl'
results_rollout = load_results(save_path)

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

Nsim = 1400
dt = 0.01

quat0_ref = np.array([1, 0, 0, 0])
p0_ref = np.array([0, 0, 0])
w0_ref = np.array([0, 0, 1]) * 1
v0_ref = np.array([1, 0, 0.1]) * 2
q0_ref = quatpos2SE3( np.concatenate((quat0_ref, p0_ref)) )
xi0_ref = np.concatenate((w0_ref, v0_ref))

q_ref = np.zeros((Nsim + 1, 4, 4))  # SE(3)
q_ref[0] = q0_ref
xi_ref = np.zeros((Nsim + 1, 6,)) 
xi_ref[0] = xi0_ref

X = q0_ref.copy()
for i in range(Nsim):
    xi_ref_rt = xi0_ref.copy()
    X = X @ expm( se3_hat( xi_ref_rt ) * dt)
    q_ref[i + 1] = X.copy()
    xi_ref[i + 1] = xi_ref_rt.copy()

# q_ref_mnf = parallel_SE32manifSE3(q_ref)
q_ref_mnf = [ SE32manifSE3(q) for q in q_ref]

for param_name in parameter_ranges.keys():
    if len(results_sol['params'][param_name]) == 0 or len(results_rollout['params'][param_name]) == 0:
        print(f"No successful results for parameter {param_name}, skipping visualization.")
        continue

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(131, projection='3d')
    plt.subplots_adjust(bottom=0.25)  # Reserve space for the slider

    # Initialize parameter index
    initial_index = 0

    # Plot reference trajectory
    pos = np.array([q_ref[i][:3, 3] for i in range(Nsim+1)])
    line_sol, = ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='blue', linewidth=2, label='reference')

    # Plot initial solved trajectory
    traj_sol = results_sol['trajectories'][param_name][initial_index]
    pos = np.array([traj_step[0][:3, 3] for traj_step in traj_sol])
    line_sol, = ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='red', linewidth=2, label='solution')

    # Plot initial rollout trajectory
    traj_rollout = results_rollout['trajectories'][param_name][initial_index]
    pos = np.array([traj_step[0][:3, 3] for traj_step in traj_rollout])
    line_rollout, = ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='purple', linewidth=2, label='initial rollout')

    # Set labels and title
    fig.suptitle(f'Trajectory for {param_name} = {results_sol["params"][param_name][initial_index]:.2f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Trajectory Comparison')

    # Create axis for the slider
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])  # [left, bottom, width, height]

    # Create the slider: use integer step size to map to parameter value list indices
    slider = Slider(
        ax=ax_slider,
        label=param_name,
        valmin=0,
        valmax=len(results_sol['params'][param_name]) - 1,
        valinit=initial_index,
        valstep=1,
        valfmt='%0.0f'
    )

    ax2 = fig.add_subplot(132)
    R_errs = [ ( SE32manifSE3( 
                        results_rollout['trajectories'][param_name][initial_index][i][0] 
                    ).compose( q_ref_mnf[i].inverse() ) ) .rotation() for i in range(Nsim+1) ]
    if param_name == 'th_z' or param_name == 'w_z':
        order = 'zyx'
        title_order = '(Z-Y-X)'
    elif param_name == 'th_x' or param_name == 'w_x':
        # order = 'xzy'
        # title_order = '(X-Z-Y)'
        order = 'xyz'
        title_order = '(X-Y-Z)'
    elif param_name == 'th_y' or param_name == 'w_y':
        # order = 'yxz'
        # title_order = '(Y-X-Z)'
        order = 'yzx'
        title_order = '(Y-Z-X)'
    else:
        order = None
        title_order = '(Z-Y-X)'
    euler_errs = np.array([ rotm2euler(m, order) for m in R_errs ])

    ax_euler_z_rollout, = ax2.plot(range(Nsim+1), euler_errs[:, 0], color='red', linewidth=2, label='1-axis')
    ax_euler_x_rollout, = ax2.plot(range(Nsim+1), euler_errs[:, 1], color='blue', linewidth=2, label='2-axis')
    ax_euler_y_rollout, = ax2.plot(range(Nsim+1), euler_errs[:, 2], color='olive', linewidth=2, label='3-axis')
    ax2.grid(True)
    ax2.set_ylim([-190,190])
    ax2.legend()
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Angle')
    ax2.set_title(f'Euler Angle Error of Initial Rollout $X\\bar{{X}}^{{-1}}$ {title_order}')

    ax3 = fig.add_subplot(133)
    R_errs = [ ( SE32manifSE3( 
                        results_sol['trajectories'][param_name][initial_index][i][0] 
                    ) * q_ref_mnf[i].inverse()).rotation() for i in range(Nsim+1) ]
    euler_errs = np.array([ rotm2euler(m, order) for m in R_errs ])
    ax_euler_z_sol, = ax3.plot(range(Nsim+1), euler_errs[:, 0], color='red', linewidth=2, label='1-axis')
    ax_euler_x_sol, = ax3.plot(range(Nsim+1), euler_errs[:, 1], color='blue', linewidth=2, label='2-axis')
    ax_euler_y_sol, = ax3.plot(range(Nsim+1), euler_errs[:, 2], color='olive', linewidth=2, label='3-axis')
    ax3.grid(True)
    ax3.set_ylim([-190,190])
    ax3.legend()
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Angle')
    ax3.set_title(f'Euler Angle Error of Solution $X\\bar{{X}}^{{-1}}$ {title_order}')

    # Update function, called when the slider value changes
    def update(val):
        index = int(slider.val)

        traj_sol = results_sol['trajectories'][param_name][index]
        pos = np.array([traj_step[0][:3, 3] for traj_step in traj_sol])
        line_sol.set_data(pos[:, 0], pos[:, 1])
        line_sol.set_3d_properties(pos[:, 2])

        traj_rollout = results_rollout['trajectories'][param_name][index]
        pos = np.array([traj_step[0][:3, 3] for traj_step in traj_rollout])
        line_rollout.set_data(pos[:, 0], pos[:, 1])
        line_rollout.set_3d_properties(pos[:, 2])

        fig.suptitle(f'Trajectory for {param_name} = {results_sol["params"][param_name][index]:.2f}')

        R_errs = [ ( SE32manifSE3( 
                    results_rollout['trajectories'][param_name][index][i][0] 
                ) * q_ref_mnf[i].inverse()).rotation() for i in range(Nsim+1) ]
        euler_errs = np.array([ rotm2euler(m) for m in R_errs ])
        ax_euler_z_rollout.set_data(range(Nsim+1), euler_errs[:, 0])
        ax_euler_x_rollout.set_data(range(Nsim+1), euler_errs[:, 1])
        ax_euler_y_rollout.set_data(range(Nsim+1), euler_errs[:, 2])

        R_errs = [ ( SE32manifSE3( 
                    results_sol['trajectories'][param_name][index][i][0] 
                ) * q_ref_mnf[i].inverse()).rotation() for i in range(Nsim+1) ]
        euler_errs = np.array([ rotm2euler(m) for m in R_errs ])
        ax_euler_z_sol.set_data(range(Nsim+1), euler_errs[:, 0])
        ax_euler_x_sol.set_data(range(Nsim+1), euler_errs[:, 1])
        ax_euler_y_sol.set_data(range(Nsim+1), euler_errs[:, 2])

        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slider.on_changed(update)

    plt.show()
