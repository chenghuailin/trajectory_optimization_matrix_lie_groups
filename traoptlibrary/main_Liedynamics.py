from traopt_controller import iLQR
import numpy as np
from jax import random
from traopt_dynamics import ErrorStateSE3AutoDiffDynamics
from traopt_manifold import skew, unskew, se3hat
from scipy.linalg import expm
# from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import matplotlib.pyplot as plt


seed = 24234156
key = random.key(seed)

dt = 0.01
Nsim = 600   # Simulation horizon

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
# Reference Generation
# =====================================================

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

X_ref = np.zeros((Nsim + 1, 7))  # 7 because of [quat(4) + position(3)]
xi_ref = np.zeros((Nsim + 1, 6)) 

X_ref[0] = x0_ref
xi_ref[0] = xid_ref

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
    X_ref[i + 1] = np.concatenate((quat, position))

    # Store the reference twists
    xi_ref[i + 1] = xid_ref_rt


# =====================================================
# Dynamics Simulation & Validation
# =====================================================

dyn_se3 = ErrorStateSE3AutoDiffDynamics( J, X_ref, xi_ref, dt )

Nsim = 30
x0 = np.zeros((12,1))
u = xid_ref.reshape(6,1)
# print(u)
x_sim_list = np.zeros((Nsim,12,1))

x_sim_list[0] = x0
for i in range(Nsim-1):
    x_sim_list[i+1] = dyn_se3.f( x_sim_list[i], u, i )


# =====================================================
# Reference Visualization
# =====================================================

# Initialize the plot
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')

# Define an initial vector
initial_vector = np.array([1, 0, 0])  # Example initial vector

# Plot the initial vector
ax1.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], color='g', label='Initial Vector')

# Loop through quaternion data to plot rotated vectors
for i in range(0, Nsim + 1, int((Nsim + 1) / 50)):  # Plot every 50th quaternion for visualization
    quat = Quaternion(X_ref[i, :4])  # Extract the quaternion from the X_ref data
    rot_matrix = quat.rotation_matrix  # Get the rotation matrix from the quaternion
    rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
    # Choose either one of them
    # 1. Assume position is [0, 0, 0] for simplicity in this example
    # position = np.array([0, 0, 0])  

    # 2. Extract the position 
    position = X_ref[i, 4:]
    rotated_vector = rotated_vector + position
    
    # Plot the rotated vector
    ax.quiver(position[0], position[1], position[2],
              rotated_vector[0], rotated_vector[1], rotated_vector[2],
              color='b', length=1, label='Rotated Vector' if i == 0 else '')

# Set the limits for the axes
lim = 4
ax1.set_xlim([-lim, lim])  # Adjusted limits for better visualization
ax1.set_ylim([-lim, lim])
ax1.set_zlim([-lim, lim])
# ax.autoscale()

# Add a legend to the plot
ax1.legend()

# Set axis labels
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')


# =====================================================
# Simulation Visualization
# =====================================================

# Initialize the plot
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, projection='3d')

# Define an initial vector
initial_vector = np.array([1, 0, 0])  # Example initial vector

# Plot the initial vector
ax2.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], color='g', label='Initial Vector')

# Loop through quaternion data to plot rotated vectors
for i in range(0, Nsim + 1, int((Nsim + 1) / 50)):  # Plot every 50th quaternion for visualization
    
    se3_matrix = expm(  )
    rot_matrix = quat.rotation_matrix  # Get the rotation matrix from the quaternion
    rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
    
    # Choose either one of them
    # 1. Assume position is [0, 0, 0] for simplicity in this example
    # position = np.array([0, 0, 0])  

    # 2. Extract the position 
    position = X_ref[i, 4:]
    rotated_vector = rotated_vector + position
    
    # Plot the rotated vector
    ax2.quiver(position[0], position[1], position[2],
              rotated_vector[0], rotated_vector[1], rotated_vector[2],
              color='b', length=1, label='Rotated Vector' if i == 0 else '')

# Set the limits for the axes
lim = 4
ax1.set_xlim([-lim, lim])  # Adjusted limits for better visualization
ax1.set_ylim([-lim, lim])
ax1.set_zlim([-lim, lim])
# ax.autoscale()

# Add a legend to the plot
ax1.legend()

# Set axis labels
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')


# =====================================================
# Plotting
# =====================================================

# Display the plot
plt.show()