from traoptlibrary.traopt_controller import iLQR, iLQR_Generation_ErrorState_Approx_LinearRollout
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from traoptlibrary.traopt_dynamics import ErrorStateSE3ApproxLinearRolloutDynamics, ErrorStateSE3ApproxNonlinearRolloutDynamics
from traoptlibrary.traopt_cost import ErrorStateSE3ApproxGenerationQuadraticAutodiffCost
from traoptlibrary.traopt_utilis import se3_hat, quatpos2SE3, euler2quat, quat2rotm, vec_SE32quatpos
from scipy.linalg import expm, logm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def on_iteration(iteration_count, xs, us, qs, xis, J_opt,
                accepted, converged, changed, grad_wrt_input_norm,
                alpha, mu, J_hist, 
                xs_hist, us_hist, qs_hist, xis_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    qs_hist.append(qs.copy())
    xis_hist.append(xis.copy())

    info = "converged" if converged else ("accepted" if accepted else "failed")
    info_change = "changed" if changed else "unchanged"
    print(f"Iteration:{iteration_count}, {info}, {info_change}, \
        grad_wrt_input_norm:{grad_wrt_input_norm}, cost:{J_opt}, alpha:{alpha}, mu:{mu}")

seed = 24234156
key = random.key(seed)
jax.config.update("jax_enable_x64", True)

dt = 0.01
Nsim = 400   # Simulation horizon

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
# Goal Configuration For Trajecotory Generation
# ========================================

# [roll, pitch, yaw] angles in radians, 
# Yaw-pitch-roll rotation order (ZYX convention)
euler_goal = [0.,  0., jnp.pi / 4]
quat_goal = euler2quat(euler_goal) 

pos_goal = np.array([ 10., 10., 10. ])
q_goal = quatpos2SE3( np.concatenate((quat_goal, pos_goal)) )

# =====================================================
# Nominal Reference Generation
# =====================================================

quat0_ref = np.array([1, 0, 0, 0])
p0_ref = np.array([0, 0, 0])

euler_devia = np.array([np.pi/4,np.pi/4,np.pi/4])
w0_ref = ( euler_goal + euler_devia )  / (Nsim * dt)
print("w0_ref is", w0_ref)

pos_devia = np.array([1, 1,-1])
v0_ref = (pos_goal + pos_devia) / (Nsim * dt)
print("v0_ref is", v0_ref)

q0_ref = np.block([
    [ quat2rotm(quat0_ref), p0_ref.reshape(-1,1) ],
    [ np.zeros((1,3)),1 ],
])
X = q0_ref.copy()

xid_ref = np.concatenate((w0_ref, v0_ref))

q_ref = np.zeros((Nsim + 1, 4, 4))  # SE(3)
xi_ref = np.zeros((Nsim + 1, 6,)) 

q_ref[0] = q0_ref.copy()
xi_ref[0] = xid_ref

for i in range(Nsim):

    X = X @ expm( se3_hat( xid_ref ) * dt)

    # Store the reference SE3 configuration
    q_ref[i + 1] = X.copy()

    # Store the reference twists
    xi_ref[i + 1] = xid_ref.copy()

q_ref = jnp.array(q_ref)
xi_ref = jnp.array(xi_ref)

# =====================================================
# Setup
# =====================================================

N = Nsim
HESSIANS = False
action_size = 6
state_size = 12
debug_dyn = {"vel_zero": False, "derivative_compare": False}

# =====================================================
# Dynamics Instantiation
# =====================================================

dynamics = ErrorStateSE3ApproxLinearRolloutDynamics(J, q_ref, xi_ref, dt, 
                                                      hessians=HESSIANS, 
                                                      debug=debug_dyn,
                                                      autodiff_dyn=False)
X_ref = vec_SE32quatpos(dynamics.q_ref)

# =====================================================
# Cost Instantiation
# =====================================================

Q = np.identity(6) * 1
P = np.identity(6) * 1e5
R = np.identity(6) * 1e1

cost = ErrorStateSE3ApproxGenerationQuadraticAutodiffCost( Q,R,P, X_ref, q_goal)

# =====================================================
# Solver Instantiation
# =====================================================

us_init = np.zeros((N, action_size,))

ilqr = iLQR_Generation_ErrorState_Approx_LinearRollout(dynamics, cost, N, 
                                                        hessians=HESSIANS)

xs_ilqr, us_ilqr, qs_ilqr, J_hist_ilqr, \
    xs_hist_ilqr, us_hist_ilqr, \
    qs_hist_ilqr, xis_hist_ilqr  = ilqr.fit(np.zeros((12,1)), 
                                       us_init, n_iterations=200, 
                                       tol_J=1e-8, 
                                       on_iteration=on_iteration)


# =====================================================
# Visualization by State
# =====================================================

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)

for j in range( state_size ):
    ax1.plot( xs_ilqr[:,j], label = 'State '+str(j) )
ax1.set_title('iLQR Final Trajectory')
ax1.set_xlabel('TimeStep')
ax1.set_ylabel('State')
ax1.legend()
ax1.grid()

for j in range( action_size ):
    ax2.plot( us_ilqr[:,j], label = 'Input '+str(j) )
ax2.set_title('iLQR Final Input')
ax2.set_xlabel('TimeStep')
ax2.set_ylabel('Input')
ax2.legend()
ax2.grid()

plt.figure(2)
plt.plot(J_hist_ilqr, label='ilqr')
plt.title('Cost Comparison')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.grid()


# # =====================================================
# # Final Result Visualization with Vector
# # =====================================================

interval_plot = int((Nsim + 1) / 40)
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
           color='r', label='Goal Vector')

nonlinear_dynamics = ErrorStateSE3ApproxNonlinearRolloutDynamics(J, us_ilqr, q0_ref, 
                                                                xid_ref, dt, hessians=HESSIANS, 
                                                                debug=debug_dyn)

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

    # se3_matrix = qs_ilqr[i] @ expm( se3_hat( xs_ilqr[i, :6]) )

    rot_matrix = qs_ilqr[i, :3, :3]  # Extract the quaternion from the X_ref data
    rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
    # Extract the position 
    position = qs_ilqr[i, :3, 3]

    # Plot the rotated vector
    ax1.quiver(position[0], position[1], position[2],
              rotated_vector[0], rotated_vector[1], rotated_vector[2],
              color='m', length=1, label='Final Nominal Reference' if i == 0 else '')
    
    # =========== 3. Plot the nonlinear rollout trajectory ===========

    se3_matrix = nonlinear_dynamics.q_ref[i]

    rot_matrix = se3_matrix[:3,:3]
    rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
    # Extract the position 
    position = se3_matrix[:3, 3]

    # Plot the rotated vector
    ax1.quiver(position[0], position[1], position[2],
            rotated_vector[0], rotated_vector[1], rotated_vector[2],
            color='c', length=1, label='Nonlinear Rollout' if i == 0 else '')

# Set the limits for the axes

ax1.set_xlim([-lim, lim]) 
ax1.set_ylim([-lim, lim])
ax1.set_zlim([-lim, lim])
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')


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


# # =====================================================
# # Plotting
# # =====================================================

# Display the plot
plt.show()