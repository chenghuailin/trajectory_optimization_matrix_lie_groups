import sys
import os
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib.widgets import Slider

def load_results(save_path):
    with open(save_path, 'rb') as f:
        results = pickle.load(f)
    return results

save_path = 'visualization/results/results_parameters.pkl'
results = load_results(save_path)

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

for param_name in parameter_ranges.keys():
    if len(results['params'][param_name]) == 0:
        print(f"No successful results for parameter {param_name}, skipping visualization.")
        continue

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)  # Reserve space for the slider

    # Initialize parameter index
    initial_index = 0

    # Plot initial trajectory
    traj = results['trajectories'][param_name][initial_index]
    pos = np.array([traj_step[0][:3, 3] for traj_step in traj])
    line, = ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='red', linewidth=2)

    # Set labels and title
    ax.set_title(f'Trajectory for {param_name} = {results["params"][param_name][initial_index]:.2f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Create axis for the slider
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])  # [left, bottom, width, height]

    # Create the slider: use integer step size to map to parameter value list indices
    slider = Slider(
        ax=ax_slider,
        label=param_name,
        valmin=0,
        valmax=len(results['params'][param_name]) - 1,
        valinit=initial_index,
        valstep=1,
        valfmt='%0.0f'
    )

    # Update function, called when the slider value changes
    def update(val):
        index = int(slider.val)
        traj = results['trajectories'][param_name][index]
        pos = np.array([traj_step[0][:3, 3] for traj_step in traj])

        # Update line data
        line.set_data(pos[:, 0], pos[:, 1])
        line.set_3d_properties(pos[:, 2])

        # Update title
        ax.set_title(f'Trajectory for {param_name} = {results["params"][param_name][index]:.2f}')

        fig.canvas.draw_idle()

    # Connect the slider to the update function
    slider.on_changed(update)

    plt.show()
