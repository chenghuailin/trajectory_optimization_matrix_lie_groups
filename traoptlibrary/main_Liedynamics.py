from traopt_controller import iLQR
import numpy as np
from jax import random
from traopt_dynamics import ErrorStateSE3AutoDiffDynamics, skew
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R


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


# ====================
# Reference Generation
# ====================

q0_ref = np.array([1, 0, 0, 0])
p0_ref = np.array([0, 0, 0])
w0_ref = np.array([0, 0, 1]) * 1
v0_ref = np.array([1, 0, 0.1]) * 2

x0_ref = np.concatenate((q0_ref, p0_ref))
xid_ref = np.concatenate((w0_ref, v0_ref))

X = np.eye(4)

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
    quat = R.from_matrix(rot_matrix).as_quat()

    # Store the reference trajectory (quaternion + position)
    X_ref[i + 1] = np.concatenate((quat, position))

    # Store the reference twists
    xi_ref[i + 1] = xid_ref_rt

ErrorStateSE3AutoDiffDynamics( J, X_ref, xi_ref, dt )