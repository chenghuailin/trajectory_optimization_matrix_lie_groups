import sys
import os
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib.widgets import Button, Slider
import matplotlib.animation as animation
from traoptlibrary.traopt_utilis import se3_hat, quatpos2SE3, parallel_SE32manifSE3,\
    rotm2euler, manifse32se3, rotmpos2SE3
from scipy.linalg import expm

# =====================================================
# Setup
# =====================================================

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


# =====================================================
# Tracking Reference Generation 
# =====================================================

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

    # You can try some time-varying twists here:
    # xi_ref_rt[0] = np.sin(i / 20) * 2
    # xi_ref_rt[4] = np.cos(np.sqrt(i)) * 1
    # xi_ref_rt[5] = 1  # np.sin(np.sqrt(i)) * 1

    X = X @ expm( se3_hat( xi_ref_rt ) * dt)

    # Store the reference SE3 configuration
    q_ref[i + 1] = X.copy()

    # Store the reference twists
    xi_ref[i + 1] = xi_ref_rt.copy()

# =====================================================
# Tracking Reference Generation 
# =====================================================

class TrajectoryAnimator:
    def __init__(self, param_name, param_values, trajectories):
        self.param_name = param_name
        self.param_values = param_values
        self.trajectories = trajectories
        self.num_frames = len(param_values)
        self.current_frame = 0
        self.is_paused = False

        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.25)  # Reserve space for slider and buttons

        # Plot initial trajectory
        traj = self.trajectories[self.current_frame]
        pos = np.array([traj_step[0][:3, 3] for traj_step in traj])
        self.line, = self.ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='red', linewidth=2)

        # Set labels and title
        self.ax.set_title(f'Trajectory for {self.param_name} = {self.param_values[self.current_frame]:.2f}')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Create axis for the slider
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])  # [left, bottom, width, height]
        self.slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=self.current_frame,
            valstep=1,
            valfmt='%0.0f'
        )
        self.slider.on_changed(self.update_frame)

        # Create axes for buttons
        ax_prev = plt.axes([0.15, 0.05, 0.1, 0.04])
        ax_pause = plt.axes([0.4, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.65, 0.05, 0.1, 0.04])

        self.button_prev = Button(ax_prev, 'Previous')
        self.button_pause = Button(ax_pause, 'Pause')
        self.button_next = Button(ax_next, 'Next')

        self.button_prev.on_clicked(self.prev_frame)
        self.button_pause.on_clicked(self.toggle_pause)
        self.button_next.on_clicked(self.next_frame)

        # Set up animation
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, frames=self.num_frames, 
            interval=200, blit=False, repeat=True
        )

    def update_frame(self, val):
        self.current_frame = int(self.slider.val)
        self.update_plot()

    def prev_frame(self, event):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.slider.set_val(self.current_frame)

    def next_frame(self, event):
        if self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.slider.set_val(self.current_frame)

    def toggle_pause(self, event):
        if self.is_paused:
            self.ani.event_source.start()
            self.button_pause.label.set_text('Pause')
        else:
            self.ani.event_source.stop()
            self.button_pause.label.set_text('Play')
        self.is_paused = not self.is_paused

    def animate(self, frame):
        if not self.is_paused:
            self.current_frame = frame
            self.slider.set_val(self.current_frame)
            self.update_plot()
        return self.line,

    def update_plot(self):
        traj = self.trajectories[self.current_frame]
        pos = np.array([traj_step[0][:3, 3] for traj_step in traj])

        # Update line data
        self.line.set_data(pos[:, 0], pos[:, 1])
        self.line.set_3d_properties(pos[:, 2])

        # Update title
        self.ax.set_title(f'Trajectory for {self.param_name} = {self.param_values[self.current_frame]:.2f}')

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

for param_name in parameter_ranges.keys():
    if len(results['params'][param_name]) == 0:
        print(f"No successful results for parameter {param_name}, skipping visualization.")
        continue

    # Get parameter values and corresponding trajectories
    param_values = results['params'][param_name]
    trajectories = results['trajectories'][param_name]

    # Create and show animation
    animator = TrajectoryAnimator(param_name, param_values, trajectories)
    animator.show()
