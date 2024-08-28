from traoptlibrary.traopt_controller import iLQR
import numpy as np
import jax.numpy as jnp
from jax import random
from traoptlibrary.traopt_dynamics import ErrorStateSE3AutoDiffDynamics
from traoptlibrary.traopt_cost import ErrorStateSE3LieAlgebraAutoDiffQuadraticCost
from traoptlibrary.traopt_utilis import skew, unskew, se3_hat
from scipy.linalg import expm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged, alpha, mu, J_hist, xs_hist, us_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    info = "converged" if converged else ("accepted" if accepted else "failed")
    # final_state = xs[-1]
    # print("iteration", iteration_count, info, J_opt, "\n", final_state, "\n", alpha, mu)
    print("iteration", iteration_count, info, J_opt, alpha, mu)

seed = 24234156
key = random.key(seed)

dt = 0.01
Nsim = 200   # Simulation horizon

# ====================
# Inertia Matrix
# ====================

# m = 1
# Ib = jnp.diag(jnp.array([0.5,0.7,0.9]))
# J = jnp.block([
#     [Ib, jnp.zeros((3, 3))],
#     [jnp.zeros((3, 3)), m * jnp.identity(3)]
# ])

m = 1
Ib = np.diag([ 0.5,0.7,0.9 ])
J = np.block([
    [Ib, np.zeros((3, 3))],
    [np.zeros((3, 3)), m * np.identity(3)]
])

    
# =====================================================
# Reference Generation
# =====================================================

# q0_ref = jnp.array([1, 0, 0, 0])
# p0_ref = jnp.array([0, 0, 0])
# w0_ref = jnp.array([0, 0, 1]) * 1
# v0_ref = jnp.array([1, 0, 0.1]) * 2

# # x0_ref and X should be kept the same
# x0_ref = jnp.concatenate((q0_ref, p0_ref))
# # X = np.eye(4) # SE(3)
# X0 = np.block([
#     [ Quaternion(q0_ref).rotation_matrix, p0_ref.reshape(-1,1) ],
#     # [ jnp.array(Quaternion(q0_ref).rotation_matrix), p0_ref.reshape(-1,1) ],
#     [ np.zeros((1,3)),1 ],
# ])
# X = X0.copy()

# xid_ref = jnp.concatenate((w0_ref, v0_ref))

# X_ref = jnp.zeros((Nsim + 1, 7, 1))  # 7 because of [quat(4) + position(3)]
# xi_ref = jnp.zeros((Nsim + 1, 6, 1)) 

# X_ref = X_ref.at[0].set(x0_ref.reshape(7, 1))
# xi_ref = xi_ref.at[0].set(xid_ref.reshape(6, 1))

q0_ref = np.array([1, 0, 0, 0])
p0_ref = np.array([0, 0, 0])
w0_ref = np.array([0, 0, 1]) * 1
v0_ref = np.array([1, 0, 0.1]) * 2

# x0_ref and X should be kept the same
x0_ref = np.concatenate((q0_ref, p0_ref))
# X = np.eye(4) # SE(3)
X0 = np.block([
    [ Quaternion(q0_ref).rotation_matrix, p0_ref.reshape(-1,1) ],
    [ np.zeros((1,3)),1 ],
])
X = X0.copy()

xid_ref = np.concatenate((w0_ref, v0_ref))

X_ref = np.zeros((Nsim + 1, 7, 1))  # 7 because of [quat(4) + position(3)]
xi_ref = np.zeros((Nsim + 1, 6, 1)) 

X_ref[0] = x0_ref.reshape(7,1)
xi_ref[0] = xid_ref.reshape(6,1)

for i in range(Nsim):

    xid_ref_rt = xid_ref.copy()

    # You can try some time-varying twists here:
    # xid_ref_rt[0] = np.sin(i / 20) * 2
    # xid_ref_rt[4] = np.cos(np.sqrt(i)) * 1
    # xid_ref_rt[5] = 1  # np.sin(np.sqrt(i)) * 1

    # Xi = jnp.block([
    #     [skew(xid_ref_rt[:3]), xid_ref_rt[3:6].reshape(3, 1)],
    #     [jnp.zeros((1, 3)), 0]
    # ])

    Xi = np.block([
        [skew(xid_ref_rt[:3]), xid_ref_rt[3:6].reshape(3, 1)],
        [np.zeros((1, 3)), 0]
    ])

    X = X @ expm(Xi * dt)

    # Extract rotation matrix and position vector
    rot_matrix = X[:3, :3]
    position = X[:3, 3]

    # Convert rotation matrix to quaternion
    quaternion = Quaternion(matrix=rot_matrix)
    quat = quaternion.elements

    # Store the reference trajectory (quaternion + position)
    X_ref[i + 1] = np.concatenate((quat, position)).reshape(7,1)
    # X_ref = X_ref.at[i + 1].set(jnp.concatenate((quat, position)).reshape(7,1))

    # Store the reference twists
    xi_ref[i + 1] = xid_ref_rt.reshape(6,1)
    # xi_ref = xi_ref.at[i + 1].set(jnp.concatenate((quat, position)).reshape(7,1))

X_ref = jnp.array(X_ref)
xi_ref = jnp.array(xi_ref)

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

dynamics = ErrorStateSE3AutoDiffDynamics(J, X_ref, xi_ref, dt, hessians=HESSIANS, debug=debug_dyn)

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
R = np.identity(6) * 1e1

cost = ErrorStateSE3LieAlgebraAutoDiffQuadraticCost( Q, R, P, xi_ref )

# =====================================================
# Solver Instantiation
# =====================================================

x0 = jnp.zeros((state_size,))
us_init = np.zeros((N, action_size,))

ilqr = iLQR(dynamics, cost, N, hessians=HESSIANS)

xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr = \
        ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)


# =====================================================
# Visualization by State
# =====================================================

plt.figure(1)
for j in range( state_size ):
    plt.plot( xs_ilqr[:,j], label = 'State '+str(j) )
plt.title('ILQR Final Trajectory')
plt.xlabel('TimeStep')
plt.ylabel('State')
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(J_hist_ilqr, label='ilqr')
# plt.plot(J_hist_ddp, label='ddp')
plt.title('Cost Comparison')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.grid()


# =====================================================
# Visualization with Vector
# =====================================================

# interval_plot = int((Nsim + 1) / 40)
# lim = 5

# # Initialize the plot
# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(121, projection='3d')
# ax2 = fig1.add_subplot(122, projection='3d')

# # Define an initial vector and plot on figure
# initial_vector = np.array([1, 0, 0])  # Example initial vector
# ax1.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], color='g', label='Initial Vector')
# ax2.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], color='g', label='Initial Vector')

# # Loop through quaternion data to plot rotated vectors
# for i in range(0, Nsim + 1, interval_plot):  

#     # =========== 1. Plot the reference trajectory ===========

#     quat = Quaternion(X_ref[i, :4, 0])  # Extract the quaternion from the X_ref data
#     rot_matrix = quat.rotation_matrix  # Get the rotation matrix from the quaternion
#     rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
#     # Extract the position 
#     position = X_ref[i, 4:, 0]
#     rotated_vector = rotated_vector + position

#     # Plot the rotated vector
#     ax1.quiver(position[0], position[1], position[2],
#               rotated_vector[0], rotated_vector[1], rotated_vector[2],
#               color='b', length=1, label='Rotated Vector' if i == 0 else '')

#     # =========== 2. Plot the simulated velocity ===========
    
#     # se3_matrix = expm( se3_hat( x_sim_list[i, 6:, :] ))
#     # rot_matrix = se3_matrix[:3,:3]  # Get the rotation matrix from the quaternion
#     # rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector

#     # position = se3_matrix[:3, 3]
#     # rotated_vector = rotated_vector + position
    
#     # # Plot the rotated vector
#     # ax2.quiver(position[0], position[1], position[2],
#     #           rotated_vector[0], rotated_vector[1], rotated_vector[2],
#     #           color='b', length=1, label='Rotated Vector' if i == 0 else '')
    
#     # =========== 3. Plot the simulated error-state configuration trajectory ===========

#     se3_matrix = np.block([
#         [Quaternion(X_ref[i, :4]).rotation_matrix, X_ref[i, 4:].reshape(3, 1)],
#         [ np.zeros((1,3)), 1 ],
#     ]) @ expm( se3_hat(x_sim_list[i, :6, :]) )
    
#     rot_matrix = se3_matrix[:3,:3]  # Get the rotation matrix from the quaternion
#     rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector

#     position = se3_matrix[:3, 3]
#     rotated_vector = rotated_vector + position
    
#     # Plot the rotated vector
#     ax2.quiver(position[0], position[1], position[2],
#               rotated_vector[0], rotated_vector[1], rotated_vector[2],
#               color='b', length=1, label='Rotated Vector' if i == 0 else '')


# # Set the limits for the axes

# ax1.set_xlim([-lim, lim]) 
# ax1.set_ylim([-lim, lim])
# ax1.set_zlim([-lim, lim])
# ax1.legend()
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')

# ax2.set_xlim([-lim, lim])  
# ax2.set_ylim([-lim, lim])
# ax2.set_zlim([-lim, lim])
# ax2.legend()
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')


# # =====================================================
# # Plotting
# # =====================================================

# Display the plot
plt.show()