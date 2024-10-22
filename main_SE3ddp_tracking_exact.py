import jax
from traoptlibrary.traopt_controller import iLQR_Tracking_SE3
import numpy as np
from jax import random
from traoptlibrary.traopt_dynamics import SE3Dynamics
from traoptlibrary.traopt_cost import ErrorStateSE3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_utilis import se3_hat, quatpos2SE3, parallel_SE32manifSE3
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
import time

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged,grad_wrt_input_norm,
                  alpha, mu, J_hist, xs_hist, us_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("Iteration", iteration_count, info, J_opt, grad_wrt_input_norm, alpha, mu)

seed = 24234156
key = random.key(seed)
jax.config.update("jax_enable_x64", True)

dt = 0.01
Nsim = 1400   # Simulation horizon
# Nsim = 400   

# ====================
# Inertia Matrix
# ====================

m = 1
Ib = np.diag([ 0.5,0.7,0.9 ])
J = np.block([
    [Ib, np.zeros((3, 3))],
    [np.zeros((3, 3)), m * np.identity(3)]
])

# =====================================================
# Reference Generation (Also the tracking reference)
# =====================================================

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
# Setup
# =====================================================

N = Nsim
HESSIANS = False
action_size = 6
state_size = 12
debug_dyn = {"vel_zero": False}

# =====================================================
# Dynamics Instantiation
# =====================================================

print("Dynamics Instatiation")
dynamics = SE3Dynamics(J, dt, hessians=HESSIANS, debug=debug_dyn)
print("Dynamics Instatiation Finished")

# =====================================================
# Cost Instantiation
# =====================================================

# This cost penalizes both error deviation and velocity (both on Lie algebra)

Q = np.diag([ 
    10., 10., 10., 1., 1., 1.,
    1., 1., 1., 1., 1., 1. 
])
P = np.diag([
    10., 10., 10., 1., 1., 1.,
    1., 1., 1., 1., 1., 1.  
]) * 10
# Q = np.diag([ 
#     10., 10., 10., 1., 1., 1.,
#     0., 0., 0., 0., 0., 0. 
# ]) 
# P = Q * 10
R = np.identity(6) * 1e-5

print("Cost Instatiation")
start_time = time.time() 
cost = ErrorStateSE3TrackingQuadraticGaussNewtonCost( Q, R, P, q_ref, xi_ref)
end_time = time.time() 
print("Cost Instantiation Finished")
print(f"Cost instantiation took {end_time - start_time:.4f} seconds")

# =====================================================
# Solver Instantiation
# =====================================================

quat0 = np.array([1., 0., 0., 0.])
p0 = np.array([-1., -1., -0.2])
q0 = quatpos2SE3( np.concatenate((quat0, p0)) )

w0 = np.array([0., 0., 0.1]) 
v0 = np.array([0.1, 0.1, 0.1])
xi0 = np.concatenate((w0, v0))

x0 = [ q0, xi0 ]
print(x0)

us_init = np.zeros((N, action_size,))

ilqr = iLQR_Tracking_SE3(dynamics, cost, N, 
                            hessians=HESSIANS,
                            rollout='nonlinear')

xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr = \
        ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)

xs_ilqr_mnf = parallel_SE32manifSE3([x[0] for x in xs_ilqr])
qref_ilqr_mnf = parallel_SE32manifSE3(q_ref)

# =====================================================
# Visualization by State
# =====================================================

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
for j in range( action_size ):
    ax1.plot( us_ilqr[:,j], label = 'Input '+str(j) )
ax1.set_title('iLQR Final Input')
ax1.set_xlabel('TimeStep')
ax1.set_ylabel('Input')
ax1.legend()
ax1.grid()

plt.figure(2)
plt.plot(J_hist_ilqr, label='ilqr')
plt.title('Cost Comparison')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.grid()


# =====================================================
# Visualization Trajectory with Vector
# =====================================================

interval_plot = int((Nsim + 1) / 40)
lim = 5

# Initialize the plot
fig1 = plt.figure(3)
ax1 = fig1.add_subplot(111, projection='3d')

# Define an initial vector and plot on figure
initial_vector = np.array([1, 0, 0])  # Example initial vector
ax1.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], color='g', label='Initial Vector')

# Loop through quaternion data to plot rotated vectors
for i in range(0, Nsim + 1, interval_plot):  

    # =========== 1. Plot the reference trajectory ===========

    se3_matrix = q_ref[i]

    rot_matrix = se3_matrix[:3,:3]
    rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
    # Extract the position 
    position = se3_matrix[:3, 3]

    # Plot the rotated vector
    ax1.quiver(position[0], position[1], position[2],
              rotated_vector[0], rotated_vector[1], rotated_vector[2],
              color='b', length=1, label='Reference Trajectory' if i == 0 else '')
    
    # =========== 2. Plot the simulated final configuration trajectory ===========

    # se3_matrix = q_ref[i] @ expm( se3_hat( xs_ilqr[i, :6]) )
    se3_matrix = xs_ilqr[i][0]
    
    rot_matrix = se3_matrix[:3,:3]  # Get the rotation matrix from the quaternion
    rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector

    position = se3_matrix[:3, 3]
    
    # Plot the rotated vector
    ax1.quiver(position[0], position[1], position[2],
              rotated_vector[0], rotated_vector[1], rotated_vector[2],
              color='r', length=1, label='Final Configuration' if i == 0 else '')


# Set the limits for the axes

ax1.set_xlim([-lim, lim]) 
ax1.set_ylim([-lim, lim])
ax1.set_zlim([-lim, lim])
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# =====================================================
# Visualization Final Trajectory with Vector as Animation
# =====================================================
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plotting is enabled

# Extract positions from the reference and final trajectories
ref_positions = q_ref[:, :3, 3]  # Reference trajectory positions (Nsim+1, 3)
final_positions = np.array([x[0][:3, 3] for x in xs_ilqr])  # Final trajectory positions (Nsim+1, 3)

# Define the frame interval and total frames for smoother animation
frame_interval = 5  # Adjust as needed for smoother or faster animation
frames = range(0, Nsim + 1, frame_interval)

# Initialize the plot
fig_anim = plt.figure(figsize=(12, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

# Plot the reference trajectory as a static blue line
ax_anim.plot(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2],
            label='Reference Trajectory', color='blue', linewidth=2)

# Initialize the final trajectory line (to be updated in animation)
final_traj_line, = ax_anim.plot([], [], [], label='Final Trajectory', color='red', linewidth=2)

# Initialize orientation vectors (quivers) for both reference and final trajectories
initial_vector = np.array([1, 0, 0])  # Example initial vector for orientation

ref_quiver = None    # Quiver for reference trajectory's current orientation
final_quiver = None  # Quiver for final trajectory's current orientation

# Initialize current position markers (optional)
ref_point, = ax_anim.plot([], [], [], 'o', color='blue', label='Reference Position')
final_point, = ax_anim.plot([], [], [], 'o', color='red', label='Final Position')

# Set plot limits based on the reference and final trajectories
all_positions = np.vstack((ref_positions, final_positions))
max_range = np.array([all_positions[:, 0].max()-all_positions[:, 0].min(),
                      all_positions[:, 1].max()-all_positions[:, 1].min(),
                      all_positions[:, 2].max()-all_positions[:, 2].min()]).max() / 2.0

mid_x = (all_positions[:, 0].max()+all_positions[:, 0].min()) * 0.5
mid_y = (all_positions[:, 1].max()+all_positions[:, 1].min()) * 0.5
mid_z = (all_positions[:, 2].max()+all_positions[:, 2].min()) * 0.5

ax_anim.set_xlim(mid_x - max_range, mid_x + max_range)
ax_anim.set_ylim(mid_y - max_range, mid_y + max_range)
ax_anim.set_zlim(mid_z - max_range, mid_z + max_range)

# Set labels and title
ax_anim.set_xlabel('X')
ax_anim.set_ylabel('Y')
ax_anim.set_zlabel('Z')
ax_anim.set_title('Final Trajectory Animation with Reference')
ax_anim.grid()

# Create proxy artists for the legend to avoid duplicate entries
from matplotlib.lines import Line2D
# legend_elements = [
#     Line2D([0], [0], color='blue', lw=2, label='Reference Trajectory'),
#     Line2D([0], [0], color='red', lw=2, label='Final Trajectory'),
#     Line2D([0], [0], marker='o', color='w', label='Reference Position',
#            markerfacecolor='blue', markersize=8),
#     Line2D([0], [0], marker='o', color='w', label='Final Position',
#            markerfacecolor='red', markersize=8),
#     Line2D([0], [0], color='blue', lw=2, label='Reference Orientation'),
#     Line2D([0], [0], color='red', lw=2, label='Final Orientation'),
# ]
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Reference Trajectory'),
    Line2D([0], [0], color='red', lw=2, label='Final Trajectory')
]
ax_anim.legend(handles=legend_elements)

def init_anim():
    """
    Initialize the animation by clearing the final trajectory line and current points.
    Also, remove any existing quivers.
    """
    final_traj_line.set_data([], [])
    final_traj_line.set_3d_properties([])
    
    ref_point.set_data([], [])
    ref_point.set_3d_properties([])
    
    final_point.set_data([], [])
    final_point.set_3d_properties([])
    
    global ref_quiver, final_quiver
    if ref_quiver:
        ref_quiver.remove()
        ref_quiver = None
    if final_quiver:
        final_quiver.remove()
        final_quiver = None
    
    return final_traj_line, ref_point, final_point

def update_anim(frame):
    """
    Update function for the animation.
    - Updates the final trajectory line up to the current frame.
    - Updates the orientation quivers for both reference and final trajectories.
    - Updates the current position markers.
    """
    global ref_quiver, final_quiver
    
    # Update the final trajectory line
    traj = final_positions[:frame + 1]
    final_traj_line.set_data(traj[:, 0], traj[:, 1])
    final_traj_line.set_3d_properties(traj[:, 2])

    # Update current position markers
    final_position = traj[-1]
    final_point.set_data(final_position[0], final_position[1])
    final_point.set_3d_properties(final_position[2])
    
    # Update reference position marker
    ref_position = ref_positions[frame]
    ref_point.set_data(ref_position[0], ref_position[1])
    ref_point.set_3d_properties(ref_position[2])
    
    # Update the reference trajectory's orientation quiver
    se3_ref = q_ref[frame]
    rot_ref = se3_ref[:3, :3]
    rotated_ref_vector = rot_ref @ initial_vector
    pos_ref = se3_ref[:3, 3]
    
    # Remove previous reference quiver if it exists
    if ref_quiver:
        ref_quiver.remove()
    
    # Plot new reference quiver
    ref_quiver = ax_anim.quiver(
        pos_ref[0], pos_ref[1], pos_ref[2],
        rotated_ref_vector[0], rotated_ref_vector[1], rotated_ref_vector[2],
        color='blue', length=0.5, normalize=True
    )
    
    # Update the final trajectory's orientation quiver
    se3_final = xs_ilqr[frame][0]
    rot_final = se3_final[:3, :3]
    rotated_final_vector = rot_final @ initial_vector
    pos_final = se3_final[:3, 3]
    
    # Remove previous final quiver if it exists
    if final_quiver:
        final_quiver.remove()
    
    # Plot new final quiver
    final_quiver = ax_anim.quiver(
        pos_final[0], pos_final[1], pos_final[2],
        rotated_final_vector[0], rotated_final_vector[1], rotated_final_vector[2],
        color='red', length=0.5, normalize=True
    )
    
    # Update the plot title to show the current frame
    ax_anim.set_title(f'Final Trajectory Animation with Reference\nFrame: {frame}/{Nsim}')
    
    return final_traj_line, ref_quiver, final_quiver, ref_point, final_point

# Create the animation
ani_final_traj2 = animation.FuncAnimation(
    fig_anim, update_anim, frames=frames,
    init_func=init_anim, blit=False,
    interval=50,  # Time between frames in milliseconds
    repeat=True    # Repeat the animation indefinitely
)


# =====================================================
# Visualization Error with Vector as Animation
# =====================================================
import matplotlib.animation as animation

# Set the sampling interval for frames
interval_plot = max(int((Nsim + 1) / 200), 1)  # Adjust as needed
lim = 1  # Axis display range

# Initialize the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define an initial vector
initial_vector = np.array([1, 0, 0])  # Example initial vector

# Plot the initial vector (green)
initial_quiver = ax.quiver(
    0, 0, 0,
    initial_vector[0], initial_vector[1], initial_vector[2],
    color='g', length=1, normalize=True, label='Initial Vector'
)

# Initialize annotation for the current stage
stage_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)

# Set the limits for the axes
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Error Trajectory Animation')
ax.legend()
ax.grid()

# Prepare frame indices for the animation
indices = range(0, Nsim + 1, interval_plot) if interval_plot > 0 else range(Nsim + 1)

# Initialize a list to store current error quivers
error_quivers = []

def update(num):
    """
    Update function for the animation.
    Removes previous error quivers and plots the current error vector.
    Updates the stage annotation.
    """
    # Remove existing error quivers
    while error_quivers:
        quiv = error_quivers.pop()
        quiv.remove()

    # Compute the error SE3 matrix
    se3_matrix = (qref_ilqr_mnf[num].inverse() * xs_ilqr_mnf[num]).transform()

    # Extract rotation matrix and position vector
    rot_matrix = se3_matrix[:3, :3]
    rotated_vector = rot_matrix @ initial_vector  # Apply rotation to the initial vector
    position = se3_matrix[:3, 3]

    # Plot the error trajectory vector (red)
    err_quiver = ax.quiver(
        position[0], position[1], position[2],
        rotated_vector[0], rotated_vector[1], rotated_vector[2],
        color='r', length=1, normalize=True,
        label='Error Trajectory' if num == indices.start else ""
    )
    error_quivers.append(err_quiver)

    # Update stage annotation
    stage_text.set_text(f'Stage: {num}/{Nsim}')

    return error_quivers + [stage_text]

# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=indices,
    interval=100,  
    blit=False,
    repeat_delay=500,
    repeat=True 
)

# Optional: Save the animation as a video file
# ani.save('error_trajectory_animation.mp4', writer='ffmpeg', fps=10)

# # =====================================================
# # Plotting
# # =====================================================

# Display the plot
plt.show()