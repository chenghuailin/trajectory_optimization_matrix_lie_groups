from traoptlibrary.traopt_controller import iLQR, iLQR_ErrorState
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from traoptlibrary.traopt_dynamics import ErrorStateSE3AutoDiffDynamics
from traoptlibrary.traopt_cost import ErrorStateSE3TrackingQuadratic2ndOrderAutodiffCost, AutoDiffCost
from traoptlibrary.traopt_cost import ErrorStateSE3GenerationQuadratic1stOrderAutodiffCost
from traoptlibrary.traopt_utilis import skew, unskew, se3_hat, se3_vee, quatpos2SE3
from scipy.linalg import expm, logm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged,grad_wrt_input_norm,
                  alpha, mu, J_hist, xs_hist, us_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    info = "converged" if converged else ("accepted" if accepted else "failed")
    # final_state = xs[-1]
    # print("iteration", iteration_count, info, J_opt, "\n", final_state, "\n", alpha, mu)
    print("Iteration", iteration_count, info, J_opt, grad_wrt_input_norm, alpha, mu)

seed = 24234156
key = random.key(seed)
jax.config.update("jax_enable_x64", True)

dt = 0.01
Nsim = 1400   # Simulation horizon

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

q0_ref = np.array([1, 0, 0, 0])
p0_ref = np.array([0, 0, 0])
w0_ref = np.array([0, 0, 1]) * 1
v0_ref = np.array([1, 0, 0.1]) * 2

# x0_ref and X should be kept the same
x0_ref = np.concatenate((q0_ref, p0_ref))
# X = np.eye(4) # SE(3)
X0_ref = np.block([
    [ Quaternion(q0_ref).rotation_matrix, p0_ref.reshape(-1,1) ],
    [ np.zeros((1,3)),1 ],
])
X = X0_ref.copy()

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

    # Store the reference twists
    xi_ref[i + 1] = xid_ref_rt.reshape(6,1)

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

# ------------- A. Second order cost as Sangli's paper ----------------
# This cost penalizes both error deviation and velocity of error (both on Lie algebra)
# (The same as the Sangli's paper)

Q = np.diag([ 
    10., 10., 10., 1., 1., 1.,
    1., 1., 1., 1., 1., 1. 
])
P = np.diag([
    10., 10., 10., 1., 1., 1.,
    1., 1., 1., 1., 1., 1.  
]) * 10
R = np.identity(6) * 1e-5

cost = ErrorStateSE3TrackingQuadratic2ndOrderAutodiffCost( Q, R, P, xi_ref )

# ------------- B. First order cost penalty  ------------- 
# This cost only penalize the Lie algebra element of the error state, so only first order

# def l(x,u,i):
#     R = np.identity(6) * 1e-5
#     Q = jnp.diag( jnp.array([10., 10., 10., 1., 1., 1.,
#                              0., 0., 0., 0., 0., 0. ]) )

#     return u.T @ R @ u + x.T @ Q @ x

# def l_terminal(x,i):
#     Q = jnp.diag( jnp.array([10., 10., 10., 1., 1., 1.,
#                              0., 0., 0., 0., 0., 0. ]) ) * 10
    
#     return x.T @ Q @ x

# cost = AutoDiffCost( l, l_terminal, state_size, action_size )

# ------------- C. Full error-state penalty  ------------- 
# This penalize the error state deviation, and the velocity of configuration
# 
# Doesn't really work, cuz the reference configuration has velocity itself, 
# which shouldn't be penalized if to track the reference

# def l(x,u,i):
#     R = np.identity(6) * 1e-5
#     Q = jnp.diag( jnp.array([10., 10., 10., 1., 1., 1.,
#                              1., 1., 1., 1., 1., 1. ]) )

#     return u.T @ R @ u + x.T @ Q @ x

# def l_terminal(x,i):
#     Q = jnp.diag( jnp.array([10., 10., 10., 1., 1., 1.,
#                              1., 1., 1., 1., 1., 1. ]) ) * 10
    
#     return x.T @ Q @ x

# cost = AutoDiffCost( l, l_terminal, state_size, action_size )

# =====================================================
# Solver Instantiation
# =====================================================

q0 = np.array([1., 0., 0., 0.])
p0 = np.array([-1., -1., 0.2])
# w0 = np.array([0, 0, 1]) * 1
# v0 = np.array([1, 0, 0.1]) * 2
w0 = np.array([0., 0., 0.]) 
v0 = np.array([0., 0.5, 0.])
X0 = np.block([
    [ Quaternion(q0).rotation_matrix, p0.reshape(-1,1) ],
    [ np.zeros((1,3)),1 ],
])
x0 = np.concatenate(( se3_vee(logm( np.linalg.inv(X0_ref) @ X0 )), w0, v0))
print(x0)
x0 = jnp.array(x0)

us_init = np.zeros((N, action_size,))

# ilqr = iLQR(dynamics, cost, N, hessians=HESSIANS)
ilqr = iLQR_ErrorState(dynamics, cost, N, 
                       hessians=HESSIANS, tracking=True)

xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr = \
        ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)


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
# plt.plot(J_hist_ddp, label='ddp')
plt.title('Cost Comparison')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.grid()


# =====================================================
# Visualization with Vector
# =====================================================

interval_plot = int((Nsim + 1) / 40)
lim = 5

# Initialize the plot
fig1 = plt.figure(3)
ax1 = fig1.add_subplot(111, projection='3d')
# ax2 = fig1.add_subplot(122, projection='3d')

# Define an initial vector and plot on figure
initial_vector = np.array([1, 0, 0])  # Example initial vector
ax1.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], color='g', label='Initial Vector')
# ax2.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], color='g', label='Initial Vector')

# Loop through quaternion data to plot rotated vectors
for i in range(0, Nsim + 1, interval_plot):  

    # =========== 1. Plot the reference trajectory ===========

    quat = Quaternion(X_ref[i, :4, 0])  # Extract the quaternion from the X_ref data
    rot_matrix = quat.rotation_matrix  # Get the rotation matrix from the quaternion
    rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
    # Extract the position 
    position = X_ref[i, 4:, 0]

    # Plot the rotated vector
    ax1.quiver(position[0], position[1], position[2],
              rotated_vector[0], rotated_vector[1], rotated_vector[2],
              color='b', length=1, label='Reference Configuration' if i == 0 else '')
    
    # =========== 2. Plot the simulated error-state configuration trajectory ===========

    # se3_matrix = np.block([
    #     [Quaternion(X_ref[i, :4, 0]).rotation_matrix, X_ref[i, 4:].reshape(3, 1)],
    #     [ np.zeros((1,3)), 1 ],
    # ]) @ expm( se3_hat( xs_ilqr[i, :6]) )

    se3_matrix = quatpos2SE3( X_ref[i] ) @ expm( se3_hat( xs_ilqr[i, :6]) )
    
    rot_matrix = se3_matrix[:3,:3]  # Get the rotation matrix from the quaternion
    rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector

    position = se3_matrix[:3, 3]
    
    # Plot the rotated vector
    ax1.quiver(position[0], position[1], position[2],
              rotated_vector[0], rotated_vector[1], rotated_vector[2],
              color='r', length=1, label='Error-State Configuration' if i == 0 else '')


# Set the limits for the axes

ax1.set_xlim([-lim, lim]) 
ax1.set_ylim([-lim, lim])
ax1.set_zlim([-lim, lim])
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

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