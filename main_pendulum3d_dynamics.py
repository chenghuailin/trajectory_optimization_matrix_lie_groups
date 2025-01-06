import numpy as np
import matplotlib.pyplot as plt
from traoptlibrary.traopt_dynamics import Pendulum3dDyanmics
from manifpy import SO3, SO3Tangent
from scipy.spatial.transform import Rotation 

J = np.diag([ 0.5,0.7,0.9 ])
m = 1
length = 0.5
dt = 0.01

HESSIANS = False
action_size = 3
state_size = 6
debug_dyn = {"vel_zero": False}

# =================================================================================
# Simulation
# =================================================================================

dynamics = Pendulum3dDyanmics(J, m, length, dt, hessians=HESSIANS, debug=debug_dyn)

q0 = SO3(Rotation.from_euler('xy', [10., 0.], degrees=True).as_quat())
xi0 = SO3Tangent( np.array([0.,0.,0.]) )
x0 = [ q0, xi0 ]

Nsim = 320
x_list = []

x = x0.copy()
x_list.append(x)

for i in range(Nsim):
    u = np.array([0,0,0])
    x = dynamics.fd_euler( x, u, i )
    x_list.append(x)

# =================================================================================
# Visualization 
# =================================================================================

rotations = [x[0].rotation() for x in x_list]

# Initialize the plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Define the initial pendulum vector
pendulum_length = length
initial_vector = np.array([0., 0., -pendulum_length]).reshape(3,1)

# Compute the rotated pendulum positions
rod_pos = np.array([rotm @ initial_vector for rotm in rotations]).reshape(Nsim+1, 3)

# Plot the trajectory
ax.plot(rod_pos[:,0], rod_pos[:,1], rod_pos[:,2], color="b")
ax.set_title("Trajectory $SO(3)$")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim([-length, length])
ax.set_ylim([-length, length])
ax.set_zlim([-length, length])
plt.show()