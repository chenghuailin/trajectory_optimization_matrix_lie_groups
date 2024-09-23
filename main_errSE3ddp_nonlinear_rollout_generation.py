from traoptlibrary.traopt_controller import iLQR_ErrorState_NonlinearRollout
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from traoptlibrary.traopt_dynamics import ErrorStateSE3NonlinearRolloutAutoDiffDynamics
from traoptlibrary.traopt_cost import ErrorStateSE3GenerationQuadratic1stOrderAutodiffCost
from traoptlibrary.traopt_utilis import skew, unskew, se3_hat, se3_vee, quatpos2SE3, euler2quat, quat2rotm, vec_SE32quatpos
from scipy.linalg import expm, logm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def on_iteration(iteration_count, xs, us, qs, xis, J_opt,
                accepted, converged, grad_wrt_input_norm,
                alpha, mu, J_hist, 
                xs_hist, us_hist, qs_hist, xis_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    qs_hist.append(qs.copy())
    xis_hist.append(xis.copy())
    
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print(f"Iteration:{iteration_count}, {info}, cost:{J_opt}, alpha:{alpha}, mu:{mu}, grad_wrt_input_norm:{grad_wrt_input_norm}")


seed = 24234156
key = random.key(seed)
jax.config.update("jax_enable_x64", True)

# ====================
# Inertia Matrix
# ====================

m = 1
Ib = np.diag([ 0.5,0.7,0.9 ])
J = np.block([
    [Ib, np.zeros((3, 3))],
    [np.zeros((3, 3)), m * np.identity(3)]
])

# ========================================
# Initialization for Trajecotory Generation
# ========================================

N = 400
Nsim = N
dt = 0.01
HESSIANS = False
action_size = 6
state_size = 12
debug_dyn = {"vel_zero": False}

# 1. Goal

# [roll, pitch, yaw] angles in radians, 
# Yaw-pitch-roll rotation order (ZYX convention)
euler_goal = [0.,  0., jnp.pi / 4]
quat_goal = euler2quat(euler_goal) 
pos_goal = np.array([ 10., 10., 10. ])
q_goal = quatpos2SE3( np.concatenate((quat_goal, pos_goal)) )

# 2. Intial State

# [roll, pitch, yaw] angles in radians, 
# Yaw-pitch-roll rotation order (ZYX convention)
euler0 = [0.,  jnp.pi / 3., jnp.pi / 3.]
quat0 = euler2quat(euler0) 
p0 = np.array([0.5, 0, 0])
w0 = np.array([0.5, 0, 0])
v0 = np.array([0.5, 0, 0])

q0 = np.block([
    [ Quaternion(quat0).rotation_matrix, p0.reshape(-1,1) ],
    [ np.zeros((1,3)),1 ],
])
xi0 = np.array([0.,0.,0.,1.,0.,0.])

# us_init = np.zeros((N, action_size,))
row = np.array([0., 0., 0., 1., 1., 1.])
us_init = np.tile(row, (N, 1))

# =====================================================
# Dynamics Instantiation
# =====================================================

dynamics = ErrorStateSE3NonlinearRolloutAutoDiffDynamics(J, us_init, q0, 
                                                         xi0, dt, hessians=HESSIANS, 
                                                         debug=debug_dyn)
X_ref = vec_SE32quatpos(dynamics.q_ref)

# =====================================================
# Cost Instantiation
# =====================================================

# Q = np.identity(6) * 1e4
# P = np.identity(6) * 1e9
# R = np.identity(6) * 1e1

Q = np.identity(6) * 1
P = np.identity(6) * 1e5
R = np.identity(6) * 1e1
cost = ErrorStateSE3GenerationQuadratic1stOrderAutodiffCost( Q,R,P, X_ref, q_goal)

# =====================================================
# Solver Instantiation
# =====================================================

ilqr = iLQR_ErrorState_NonlinearRollout(dynamics, cost, N, hessians=HESSIANS)

xs_ilqr, us_ilqr, qs_ilqr, \
    J_hist_ilqr, xs_hist_ilqr, \
    us_hist_ilqr, qs_hist_ilqr, xis_hist = ilqr.fit(np.zeros((12,1)),
                                                us_init, n_iterations=200, 
                                                tol_J=1e-3,
                                                on_iteration=on_iteration)


# =====================================================
# Final Result Visualization with Vector
# =====================================================

interval_plot = int((Nsim + 1) / 100)
lim = 15

# Initialize the plot
fig1 = plt.figure(3)
ax1 = fig1.add_subplot(111, projection='3d')

# Define an initial vector and plot on figure
initial_vector = np.array([1, 0, 0])  # Example initial vector
ax1.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], 
           color='g', label='Initial Vector')
goal_vector = quat2rotm( quat_goal ) @ initial_vector
ax1.quiver(pos_goal[0], pos_goal[1], pos_goal[2], 
           goal_vector[0], goal_vector[1], goal_vector[2], 
           color='y', label='Goal Vector')


# Loop through quaternion data to plot rotated vectors
for i in range(0, Nsim + 1, interval_plot):  

    # =========== 1. Plot the first nominal trajectory ===========

    rot_matrix = qs_hist_ilqr[0, i, :3, :3]  # Get the rotation matrix from the quaternion
    rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
    # Extract the position 
    position = qs_hist_ilqr[0, i, :3, 3]

    # Plot the rotated vector
    ax1.quiver(position[0], position[1], position[2],
              rotated_vector[0], rotated_vector[1], rotated_vector[2],
              color='b', length=1, label='Initial Nominal Reference' if i == 0 else '')
    
    # =========== 2. Plot the final nominal trajectory ===========

    rot_matrix = qs_ilqr[i, :3, :3]  # Extract the quaternion from the X_ref data
    rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
    # Extract the position 
    position = qs_ilqr[i, :3, 3]

    # Plot the rotated vector
    ax1.quiver(position[0], position[1], position[2],
              rotated_vector[0], rotated_vector[1], rotated_vector[2],
              color='m', length=1, label='Final Nominal Reference' if i == 0 else '')
    
    # # =========== 3. Plot the simulated error-state configuration trajectory ===========

    # se3_matrix = qs_ilqr[i] @ expm( se3_hat( xs_ilqr[i, :6]) )
    
    # rot_matrix = se3_matrix[:3,:3]  # Get the rotation matrix from the quaternion
    # rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector

    # position = se3_matrix[:3, 3]
    
    # # Plot the rotated vector
    # ax1.quiver(position[0], position[1], position[2],
    #           rotated_vector[0], rotated_vector[1], rotated_vector[2],
    #           color='r', length=1, label='Error-State Configuration' if i == 0 else '')


# Set the limits for the axes

ax1.set_xlim([-lim, lim]) 
ax1.set_ylim([-lim, lim])
ax1.set_zlim([-lim, lim])
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# # # =====================================================
# # # Plotting
# # # =====================================================

# # Display the plot
plt.show()
























# # =====================================================
# # Visualization by State
# # =====================================================

# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(121)
# ax2 = fig1.add_subplot(122)

# for j in range( state_size ):
#     ax1.plot( xs_ilqr[:,j], label = 'State '+str(j) )
# ax1.set_title('iLQR Final Trajectory')
# ax1.set_xlabel('TimeStep')
# ax1.set_ylabel('State')
# ax1.legend()
# ax1.grid()

# for j in range( action_size ):
#     ax2.plot( us_ilqr[:,j], label = 'Input '+str(j) )
# ax2.set_title('iLQR Final Input')
# ax2.set_xlabel('TimeStep')
# ax2.set_ylabel('Input')
# ax2.legend()
# ax2.grid()

# plt.figure(2)
# plt.plot(J_hist_ilqr, label='ilqr')
# plt.title('Cost Comparison')
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.legend()
# plt.grid()


# # =====================================================
# # Final Result Visualization with Vector
# # =====================================================

# interval_plot = int((Nsim + 1) / 40)
# lim = 15

# # Initialize the plot
# fig1 = plt.figure(3)
# ax1 = fig1.add_subplot(111, projection='3d')

# # Define an initial vector and plot on figure
# initial_vector = np.array([1, 0, 0])  # Example initial vector
# ax1.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], 
#            color='g', label='Initial Vector')
# goal_vector = quat2rotm( quat_goal ) @ initial_vector
# ax1.quiver(pos_goal[0], pos_goal[1], pos_goal[2], 
#            goal_vector[0], goal_vector[1], goal_vector[2], 
#            color='y', label='Goal Vector')


# # Loop through quaternion data to plot rotated vectors
# for i in range(0, Nsim + 1, interval_plot):  

#     # =========== 1. Plot the first nominal trajectory ===========

#     rot_matrix = quat2rotm(X_ref[i, :4, 0])  # Get the rotation matrix from the quaternion
#     rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
#     # Extract the position 
#     position = X_ref[i, 4:, 0]

#     # Plot the rotated vector
#     ax1.quiver(position[0], position[1], position[2],
#               rotated_vector[0], rotated_vector[1], rotated_vector[2],
#               color='b', length=1, label='Initial Nominal Reference' if i == 0 else '')
    
#     # =========== 2. Plot the final nominal trajectory ===========

#     rot_matrix = quat2rotm(Xref_hist_ilqr[-1, i, :4, 0])  # Extract the quaternion from the X_ref data
#     rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
#     # Extract the position 
#     position = Xref_hist_ilqr[-1, i, 4:, 0]

#     # Plot the rotated vector
#     ax1.quiver(position[0], position[1], position[2],
#               rotated_vector[0], rotated_vector[1], rotated_vector[2],
#               color='m', length=1, label='Final Nominal Reference' if i == 0 else '')
    
#     # =========== 3. Plot the simulated error-state configuration trajectory ===========

#     se3_matrix = quatpos2SE3( Xref_hist_ilqr[-1, i, :, 0] ) @ expm( se3_hat( xs_ilqr[i, :6]) )
    
#     rot_matrix = se3_matrix[:3,:3]  # Get the rotation matrix from the quaternion
#     rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector

#     position = se3_matrix[:3, 3]
    
#     # Plot the rotated vector
#     ax1.quiver(position[0], position[1], position[2],
#               rotated_vector[0], rotated_vector[1], rotated_vector[2],
#               color='r', length=1, label='Error-State Configuration' if i == 0 else '')


# # Set the limits for the axes

# ax1.set_xlim([-lim, lim]) 
# ax1.set_ylim([-lim, lim])
# ax1.set_zlim([-lim, lim])
# ax1.legend()
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')

# # =====================================================
# # Nominal Reference Evolution Visualization with Vector
# # =====================================================

# fig1 = plt.figure(4)
# ax1 = fig1.add_subplot(111, projection='3d')

# interval_plot = int((Nsim + 1) / 30)
# lim = 15

# # Define an initial vector and plot on figure
# initial_vector = np.array([1, 0, 0])  # Example initial vector
# ax1.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], 
#            color='g', label='Initial Vector')
# goal_vector = quat2rotm( quat_goal ) @ initial_vector
# ax1.quiver(pos_goal[0], pos_goal[1], pos_goal[2], 
#            goal_vector[0], goal_vector[1], goal_vector[2], 
#            color='r', label='Goal Vector')

# # Normalize to create a color map for Xref curves
# norm = Normalize(vmin=0, vmax=Xref_hist_ilqr.shape[0] * 1)  # Adjust the normalization range for more difference
# cmap = plt.colormaps['plasma']  # Choose a colormap with more contrast

# # Loop through quaternion data to plot rotated vectors
# for i in range( Xref_hist_ilqr.shape[0] ):
#     color = cmap(norm(i)) 
#     for j in range(0, Nsim + 1, interval_plot):  
#         quat = Quaternion(Xref_hist_ilqr[i, j, :4, 0])  # Extract the quaternion from the X_ref data
#         rot_matrix = quat.rotation_matrix  # Get the rotation matrix from the quaternion
#         rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
        
#         # Extract the position 
#         position = Xref_hist_ilqr[i, j, 4:, 0]

#         # Plot the rotated vector
#         ax1.quiver(position[0], position[1], position[2],
#                 rotated_vector[0], rotated_vector[1], rotated_vector[2],
#                 color=color, length=1, label='Iteration '+str(i) if j == 0 else '')
    

# # Set the limits for the axes

# ax1.set_title('Nominal Trajectory Revolution')
# ax1.set_xlim([-lim, lim]) 
# ax1.set_ylim([-lim, lim])
# ax1.set_zlim([-lim, lim])
# ax1.legend()
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')


# # =====================================================
# # Configuration Trajectory Evolution Visualization with Vector
# # =====================================================

# fig1 = plt.figure(5)
# ax1 = fig1.add_subplot(111, projection='3d')

# interval_plot = int((Nsim + 1) / 40)
# lim = 15

# # Define an initial vector and plot on figure
# initial_vector = np.array([1, 0, 0])  # Example initial vector
# ax1.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], 
#            color='g', label='Initial Vector')
# goal_vector = quat2rotm( quat_goal ) @ initial_vector
# ax1.quiver(pos_goal[0], pos_goal[1], pos_goal[2], 
#            goal_vector[0], goal_vector[1], goal_vector[2], 
#            color='r', label='Goal Vector')

# # Normalize to create a color map for Xref curves
# norm = Normalize(vmin=0, vmax=Xref_hist_ilqr.shape[0] * 1)  # Adjust the normalization range for more difference
# cmap = plt.colormaps['plasma']  # Choose a colormap with more contrast

# # Loop through quaternion data to plot rotated vectors
# for i in range( Xref_hist_ilqr.shape[0]-1 ):
#     color = cmap(norm(i)) 
#     for j in range(0, Nsim + 1, interval_plot):  

#         se3_matrix = quatpos2SE3( Xref_hist_ilqr[i, j, :, 0] ) @ expm( se3_hat( xs_hist_ilqr[i+1, j, :6]) )

#         rot_matrix = se3_matrix[:3,:3]
#         rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
        
#         # Extract the position 
#         position = se3_matrix[:3, 3]

#         # Plot the rotated vector
#         ax1.quiver(position[0], position[1], position[2],
#                 rotated_vector[0], rotated_vector[1], rotated_vector[2],
#                 color=color, length=1, label='Iteration '+str(i) if j == 0 else '')
    

# # Set the limits for the axes

# ax1.set_title('Configuration Trajectory Revolution')

# ax1.set_xlim([-lim, lim]) 
# ax1.set_ylim([-lim, lim])
# ax1.set_zlim([-lim, lim])
# ax1.legend()
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')

# # # =====================================================
# # # Plotting
# # # =====================================================

# # Display the plot
# plt.show()