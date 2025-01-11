import jax
from traoptlibrary.traopt_controller import iLQR_Tracking_SO3_MS
import numpy as np
from jax import random
from traoptlibrary.traopt_dynamics import SO3Dynamics, Pendulum3dDyanmics
from traoptlibrary.traopt_cost import SO3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_utilis import rotm2euler
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import time
from manifpy import SO3, SO3Tangent
import sys
sys.path.append("visualization/rerun")
from rerun_loader_urdf import URDFLogger
import rerun as rr

def on_iteration(iteration_count, xs, us, J_opt, accepted, 
                converged, defect_norm, grad_wrt_input_norm,
                alpha, mu, J_hist, xs_hist, us_hist, grad_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    grad_hist.append(grad_wrt_input_norm.copy())

    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("Iteration", iteration_count, \
          info, J_opt, defect_norm, \
          grad_wrt_input_norm, alpha, mu)

seed = 24234156
key = random.key(seed)
jax.config.update("jax_enable_x64", True)

# =====================================================
# Tracking Reference Generation 
# =====================================================

# Nsim = int(14/dt)   # Simulation horizon

# q0_ref = Rotation.from_euler(
#         'zyx', [ 0., 0., 0. ], degrees=True
#         ).as_matrix()
# xi_ref = np.array([0, 0, 1]) * 1

# q_ref = np.zeros((Nsim + 1, 4, 4))  # SE(3)
# q_ref[0] = q0_ref
# xi_ref = np.zeros((Nsim + 1, 6,)) 
# xi_ref[0] = xi0_ref

# X = q0_ref.copy()

# for i in range(Nsim):

#     xi_ref_rt = xi0_ref.copy()

#     # You can try some time-varying twists here:
#     # xi_ref_rt[0] = np.sin(i / 20) * 2
#     # xi_ref_rt[4] = np.cos(np.sqrt(i)) * 1
#     # xi_ref_rt[5] = 1  # np.sin(np.sqrt(i)) * 1

#     X = X @ expm( se3_hat( xi_ref_rt ) * dt)

#     # Store the reference SE3 configuration
#     q_ref[i + 1] = X.copy()

#     # Store the reference twists
#     xi_ref[i + 1] = xi_ref_rt.copy()


# R0 = Rotation.from_euler(
#     'zyx', [ 0., 0., 0. ], degrees=True
#     ).as_matrix()
# w0 = np.array([0., 0., 0.1]) 
# x0 = [ R0, w0 ]

# =====================================================
# Other Reference Import
# =====================================================

# path_to_reference_file = \
#     'visualization/optimized_trajectories/path_3dpendulum_8shape.npy'
    
# with open( path_to_reference_file, 'rb' ) as f:
#     q_ref = np.load(f)
#     xi_ref = np.load(f)
#     dt = np.load(f)

# Nsim = q_ref.shape[0] - 1
# print("Horizon of dataset is", Nsim)

# q0 = SO3( Rotation.from_matrix(q_ref[0]).as_quat() ) 
# xi0 = SO3Tangent( xi_ref[0] )
# x0 = [ q0, xi0 ]

# =====================================================
# Swing Up Task
# =====================================================

ONLY_TERMINAL = False

Nsim = 80
dt = 0.025

q_ref = Rotation.from_euler('x', 180., degrees=True).as_matrix()
xi_ref = np.array([0.,0.,0.])
q_ref = np.tile(q_ref, (Nsim+1,1,1))
xi_ref = np.tile(xi_ref,(Nsim+1,1))

q0 = SO3( Rotation.from_euler('y', 50., degrees=True).as_quat() )
# q0 = SO3.Identity()
xi0 = SO3Tangent( [0.,0.,0.] )
x0 = [ q0, xi0 ]

# =====================================================
# Setup
# =====================================================

J = np.diag([ 0.5,0.7,0.9 ])
m = 1
length = 0.5

N = Nsim # horizon, note the state length = horizon + 1
HESSIANS = False
action_size = 3
state_size = 6
debug_dyn = {"vel_zero": False}

# =====================================================
# Dynamics Instantiation
# =====================================================

print("Dynamics Instatiation")
dynamics = Pendulum3dDyanmics(J, m, length, dt, hessians=HESSIANS, debug=debug_dyn)
# dynamics = SO3Dynamics(J, dt, hessians=HESSIANS, debug=debug_dyn)
print("Dynamics Instatiation Finished")

# =====================================================
# Cost Instantiation
# =====================================================

Q = np.diag([ 
    10., 10., 10., 1., 1., 1.,
])
P = Q * 10
R = np.identity(3) * 1e-5

# Q = np.diag([ 
#     10., 10., 10., 1., 1., 1.,
# ]) * 1000
# P = Q * 10
# R = np.identity(3) * 1e-5

print("Cost Instatiation")
start_time = time.time() 
cost = SO3TrackingQuadraticGaussNewtonCost( Q, R, P, q_ref, xi_ref )
end_time = time.time() 
print("Cost Instantiation Finished")
print(f"Cost instantiation took {end_time - start_time:.4f} seconds")

# =====================================================
# Solver Instantiation
# =====================================================

print(f'Initial State:\n{x0[0].rotation()}')

us_init = np.zeros((N, action_size,))

ilqr = iLQR_Tracking_SO3_MS(dynamics, cost, N, 
                            q_ref, xi_ref,
                            hessians=HESSIANS,
                            line_search=True,
                            rollout='nonlinear')

xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr, grad_hist_ilqr = \
        ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)

# =====================================================
# Visualization Preparation
# =====================================================

err_ilqr = [cost._err(x, i) for i, x in enumerate(xs_ilqr)]

q_ilqr = [x[0] for x in xs_ilqr]
q_quat_ilqr = [ Rotation.from_matrix(x.rotation()).as_quat() for x in q_ilqr ] 
qref_ilqr = [ SO3(Rotation.from_matrix(x).as_quat()) for x in q_ref]

norm_q_err = np.array([np.linalg.norm(err_ilqr[i][0],ord=2) for i in range( len(err_ilqr) )])
norm_vel_err = np.array([np.linalg.norm(err_ilqr[i][1],ord=2) for i in range( len(err_ilqr) )])

q_euler_ilqr = np.array([ rotm2euler(x[0].rotation(), order='zyx') for x in xs_ilqr ] )
vel_ilqr = np.array([ x[1].coeffs() for x in xs_ilqr ])

pendulum_length = 1.2
updown_vector = np.array([0., 0., -pendulum_length]).reshape(3,1)

rod_pos_sol = np.array([rotm.rotation() @ updown_vector for rotm in q_ilqr]).reshape(N+1, 3)
rod_pos_ref = np.array([rotm.rotation() @ updown_vector for rotm in qref_ilqr]).reshape(N+1, 3)

# =====================================================
# Pendulum Translation Integration
# =====================================================

vel_cm = np.zeros((Nsim+1, 3))
pos_cm = np.zeros((Nsim+1, 3))
pos_pivot = np.zeros((Nsim+1, 3))

q_rotm_ilqr = np.array([ x.rotation() for x in q_ilqr])
g_acc = dynamics.g

for i in range( Nsim ):
    # vel_cm[i+1] = vel_cm[i] + dt * ( q_rotm_ilqr[i].T @ ( g_acc + us_ilqr[i] ) )
    vel_cm[i+1] = vel_cm[i] + dt * ( q_rotm_ilqr[i].T @ (  us_ilqr[i] ) )
    pos_cm[i+1] = pos_cm[i] + dt * vel_cm[i]

down_vec = np.array([0,0,-1.])
rho = length / 2 * down_vec

for i in range(Nsim+1):
    pos_pivot[i] = pos_cm[i] + q_rotm_ilqr[i] @ (-1 * rho)

# =====================================================
# Rerun Logging
# =====================================================

rr.init("pendulum_animation", spawn=True, recording_id="3d_inverted_pendulum")

pendulum_urdf_path = "./visualization/rerun/3d_inverted_pendulum.urdf"
urdf_logger = URDFLogger(pendulum_urdf_path, None)
urdf_logger.entity_path_prefix = f"solution_ms/pendulum_urdf"
urdf_logger.log()

for step in range(N):

    rr.set_time_seconds( "sim_time", dt * step )

    # rr.log(
    #     f"solution_ms/rod_position",
    #     rr.Points3D(
    #         rod_pos_sol[step] #,
    #         # colors=vel_mapped_color,
    #     ),
    # )

    rr.log(
        f"solution_ms/pivot_accleration",
        rr.Arrows3D(
            vectors=us_ilqr[step],
            origins=pos_pivot[step],
        ),
    )

    rr.log(
        f"solution_ms/pendulum_urdf",
        rr.Transform3D(
            translation=np.array(pos_pivot[step]),
            rotation=rr.Quaternion(xyzw=q_quat_ilqr[step]),
            axis_length=1.0,
        ),
    )

print("Rerun logging finished")

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
plt.subplot(211)
plt.plot(J_hist_ilqr, label='ilqr')
plt.title('Cost Comparison')
plt.ylabel('Cost')
plt.legend()
plt.grid()
plt.subplot(212)
plt.plot(grad_hist_ilqr, label='ilqr')
plt.title('Gradient Comparison')
plt.xlabel('Iteration')
plt.ylabel('Gradient')
plt.legend()
plt.grid()

plt.figure(3)
plt.plot(norm_q_err, label='Configuration Error')
plt.plot(norm_vel_err, label='Velocity Error')
plt.title('Error Norm Evolution')
plt.xlabel('Iteration')
plt.ylabel('Norm')
plt.legend()
plt.grid()

plt.figure(4)
plt.suptitle('Error Evolution')
plt.subplot(121)
for j in range(3):
    plt.plot( [err_ilqr[i][0][j] for i in range(len(err_ilqr))] )
plt.title('Configuration Error')
plt.legend(['th_x','th_y','th_z'])
plt.xlabel('Iteration')
plt.grid()

plt.subplot(122)
for j in range(3):
    plt.plot( [err_ilqr[i][1][j] for i in range(len(err_ilqr))] )
plt.title('Velocity Error')
plt.legend(['w_x','w_y','w_z'])
plt.xlabel('Iteration')
plt.grid()

plt.figure(5)
plt.suptitle("iLQR Final State")

plt.subplot(121)
for i in range(3):
    plt.plot( q_euler_ilqr[:,i] )
plt.title('Euler Angle (z-x-y) - World Frame')
plt.ylabel('Degree')
plt.legend(['Z-Axis','X-Axis','Y-Axis'])
plt.grid()

plt.subplot(122)
for i in range(3):
    plt.plot( vel_ilqr[:,i] )
plt.title('Angular Velocity - Body Frame')
plt.xlabel('Iteration')
plt.ylabel('Degree/s')
plt.legend(['$omega_x$','$omega_y$','$omega_z$'])
plt.grid()


fig1 = plt.figure(6)
plt.suptitle('Final iLQR Trajecotry')
ax2 = fig1.add_subplot(111, projection='3d')
ax2.plot(rod_pos_ref[:, 0], rod_pos_ref[:, 1], rod_pos_ref[:, 2],
            label='Reference Trajectory', color='blue', linewidth=2)
ax2.plot(rod_pos_sol[:, 0], rod_pos_sol[:, 1], rod_pos_sol[:, 2],
            label='Final Trajectory', color='red', linewidth=2)
ax2.legend()
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# =====================================================
# Visualization Final Trajectory with Vector as Animation
# =====================================================

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # Enable 3D plotting
from matplotlib.widgets import Button
from matplotlib.lines import Line2D

# Define frame interval for smoother animation
frame_interval = 5
frames = range(0, Nsim + 1, frame_interval)

# Initialize the figure and subplots
fig_anim = plt.figure(figsize=(14, 6))
ax_anim_err = fig_anim.add_subplot(121, projection='3d')
ax_anim_traj = fig_anim.add_subplot(122, projection='3d')

# Plot the static reference trajectory
ax_anim_traj.plot(rod_pos_ref[:, 0], rod_pos_ref[:, 1], rod_pos_ref[:, 2],
                 label='Reference Trajectory', color='blue', linewidth=2)

# Initialize the final trajectory line
final_traj_line, = ax_anim_traj.plot([], [], [], label='Final Trajectory', color='red', linewidth=2)

# Define an initial vector for orientation
initial_vector = np.array([0, 0, 1])

# Initialize quivers and points
ref_quiver = None
final_quiver = None
ref_point, = ax_anim_traj.plot([], [], [], 'o', color='blue', label='Reference Position')
final_point, = ax_anim_traj.plot([], [], [], 'o', color='red', label='Final Position')

# Set plot limits based on trajectories
all_positions = np.vstack((rod_pos_ref, rod_pos_sol))
max_range = (all_positions.max(axis=0) - all_positions.min(axis=0)).max() / 2.0
mid_points = (all_positions.max(axis=0) + all_positions.min(axis=0)) * 0.5

ax_anim_traj.set_xlim(mid_points[0] - max_range, mid_points[0] + max_range)
ax_anim_traj.set_ylim(mid_points[1] - max_range, mid_points[1] + max_range)
ax_anim_traj.set_zlim(mid_points[2] - max_range, mid_points[2] + max_range)

# Set labels and title
ax_anim_traj.set_xlabel('X')
ax_anim_traj.set_ylabel('Y')
ax_anim_traj.set_zlabel('Z')
ax_anim_traj.set_title('Final Trajectory Animation with Reference')
ax_anim_traj.grid()

# Create legend
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='Reference Trajectory'),
    Line2D([0], [0], color='red', lw=2, label='Final Trajectory')
]
ax_anim_traj.legend(handles=legend_elements)

# Initialize error trajectory plot
lim = 1  # Axis display range
ax_anim_err.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2],
                  color='blue', length=1, normalize=True, label='Reference Vector')
stage_text = ax_anim_err.text2D(0.03, 0.75, "", transform=ax_anim_err.transAxes, fontsize=7)

# Set limits and labels for error plot
ax_anim_err.set_xlim([-lim, lim])
ax_anim_err.set_ylim([-lim, lim])
ax_anim_err.set_zlim([-lim, lim])
ax_anim_err.set_xlabel('X')
ax_anim_err.set_ylabel('Y')
ax_anim_err.set_zlabel('Z')
ax_anim_err.set_title('Configuration Error Animation')
ax_anim_err.grid()
ax_anim_err.legend()

# Initialize error quivers list
error_quivers = []

# Control variables for animation
current_frame = 0
is_paused = False

def update_anim(frame):
    """Update function for the animation."""
    global ref_quiver, final_quiver, current_frame
    
    # Update final trajectory line
    traj = rod_pos_sol[:frame + 1]
    final_traj_line.set_data(traj[:, 0], traj[:, 1])
    final_traj_line.set_3d_properties(traj[:, 2])

    # Update position markers
    final_pos = traj[-1]
    final_point.set_data([final_pos[0]], [final_pos[1]])
    final_point.set_3d_properties([final_pos[2]])
    
    # Update reference trajectory markers
    ref_pos = rod_pos_ref[frame]
    ref_point.set_data([ref_pos[0]], [ref_pos[1]])
    ref_point.set_3d_properties([ref_pos[2]])
    
    # Update reference quiver   
    if ref_quiver:
        ref_quiver.remove()
    ref_quiver = ax_anim_traj.quiver(
        0, 0, 0,
        ref_pos[0], ref_pos[1], ref_pos[2],
        color='blue', length=0.5, normalize=True
    )
    
    # Update final quiver
    if final_quiver:
        final_quiver.remove()
    final_quiver = ax_anim_traj.quiver(
        0, 0, 0,
        final_pos[0], final_pos[1], final_pos[2],
        color='red', length=0.5, normalize=True
    )
    
    # Update title with current frame
    ax_anim_traj.set_title(f'Final Trajectory Animation with Reference\nFrame: {frame}/{Nsim}')
    
    # ------------------------------------------------------------------

    # Update error quivers
    while error_quivers:
        quiv = error_quivers.pop()
        quiv.remove()
    
    rot_error = ( qref_ilqr[frame].inverse() * q_ilqr[frame] ).rotation()
    rotated_error_vec = rot_error @ initial_vector
    
    err_quiver = ax_anim_err.quiver(
        0.,0.,0.,
        rotated_error_vec[0], rotated_error_vec[1], rotated_error_vec[2],
        color='red', length=1, normalize=True,
        label='Error Trajectory' if frame == frames[0] else ""
    )
    error_quivers.append(err_quiver)

    lminus_error = ( q_ilqr[frame].lminus( qref_ilqr[frame] ) ).coeffs()
    rminus_error = (  q_ilqr[frame].rminus( qref_ilqr[frame] ) ).coeffs()
    lminus_error_norm = np.linalg.norm(lminus_error,ord=2)
    rminus_error_norm = np.linalg.norm(rminus_error,ord=2)

    # Update title with current frame
    ax_anim_err.set_title(f'Error Trajectory Animation\nFrame: {frame}/{Nsim}')
    
    # Update stage annotation
    stage_text.set_text(f'Left-Error Norm:{lminus_error_norm}\n  {lminus_error}\
                        \nRight-Error Norm:{rminus_error_norm}\n  {rminus_error}')

    # ------------------------------------------------------------------

    # Update frame index
    current_frame = frame
    
    return final_traj_line, ref_quiver, final_quiver, ref_point, final_point, err_quiver, stage_text

# Create the animation
ani_final_traj = animation.FuncAnimation(
    fig_anim, 
    update_anim, 
    frames=frames,
    blit=False,
    interval=0.01,  # Time between frames in milliseconds
    repeat=True    # Repeat the animation indefinitely
)


# Add interactive buttons
# Pause/Play Button
ax_pause = plt.axes([0.7, 0.01, 0.1, 0.05])
btn_pause = Button(ax_pause, 'Pause/Play')

# Previous Frame Button
ax_prev = plt.axes([0.81, 0.01, 0.1, 0.05])
btn_prev = Button(ax_prev, 'Previous')

# Next Frame Button
ax_next = plt.axes([0.59, 0.01, 0.1, 0.05])
btn_next = Button(ax_next, 'Next')

def toggle_pause(event):
    """Toggle pause/play of the animation."""
    global is_paused
    if is_paused:
        ani_final_traj.event_source.start()
    else:
        ani_final_traj.event_source.stop()
    is_paused = not is_paused

def prev_frame(event):
    """Go to the previous frame."""
    global current_frame
    if current_frame > 0:
        current_frame = max(0, current_frame - frame_interval)
        update_anim(current_frame)
        plt.draw()

def next_frame(event):
    """Go to the next frame."""
    global current_frame
    if current_frame < Nsim:
        current_frame = min(Nsim, current_frame + frame_interval)
        update_anim(current_frame)
        plt.draw()

# Connect buttons to their callback functions
btn_pause.on_clicked(toggle_pause)
btn_prev.on_clicked(prev_frame)
btn_next.on_clicked(next_frame)


# # =====================================================
# # Plotting Display
# # =====================================================

plt.show()