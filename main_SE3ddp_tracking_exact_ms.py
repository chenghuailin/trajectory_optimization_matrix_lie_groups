import jax
from traoptlibrary.traopt_controller import iLQR_Tracking_SE3_MS
import numpy as np
from jax import random
from traoptlibrary.traopt_dynamics import SE3Dynamics
from traoptlibrary.traopt_cost import ErrorStateSE3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_utilis import skew, unskew, se3_hat, se3_vee, quatpos2SE3
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

ilqr = iLQR_Tracking_SE3_MS(dynamics, cost, N, 
                            q_ref, xi_ref, 
                            hessians=HESSIANS,
                            rollout='nonlinear')

xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr = \
        ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)


# =====================================================
# Visualization by State
# =====================================================

fig1 = plt.figure(1)
ax2 = fig1.add_subplot(111)


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


# =====================================================
# Visualization with Vector
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
    
    # =========== 2. Plot the simulated error-state configuration trajectory ===========

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


# # =====================================================
# # Plotting
# # =====================================================

# Display the plot
plt.show()