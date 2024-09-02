import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.linalg import inv
from traoptlibrary.traopt_utilis import coadjoint, quat2rotm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from traoptlibrary.traopt_utilis import skew, unskew, se3_hat
from scipy.linalg import expm, logm, sqrtm
from traoptlibrary.traopt_dynamics import ErrorStateSE3AutoDiffDynamics

def SE3Dyn(x, u, i, I):
    # Decompose state vector
    q = x[:4]   # quaternion
    p = x[4:7]  # position
    w = x[7:10] # angular velocity
    v = x[10:13]# linear velocity

    # print(w)

    # Compute Omega matrix for quaternion derivative
    Omega = jnp.array([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0]
    ])

    # Quaternion derivative
    dQuat = 0.5 * Omega @ q
    # print(dQuat)

    # Convert quaternion to rotation matrix
    quat = Quaternion(jnp.array(q))  # Extract the quaternion from the X_ref data
    R_matrix = quat.rotation_matrix  # Get the rotation matrix from the quaternion
    # R_matrix = quat2rotm(q)

    # Position and velocity derivative
    # print((R_matrix @ v).shape)
    dx = jnp.concatenate((dQuat, R_matrix @ v))
    # print(dx)

    # Compute coadjoint term
    coadj_term = coadjoint(jnp.concatenate((w, v)))

    # Acceleration calculation
    d2x = inv(I) @ (coadj_term @ I @ jnp.concatenate((w, v)) + u)
    # d2x = inv(I) @ (coadj_term @ (I @ jnp.concatenate((w, v))) + u + 0 * I @ jnp.array([0, 0, 0, *(R_matrix.T @ jnp.array([0, 0, -9.81]))]))

    # Return the full state derivative
    dxdt = jnp.concatenate((dx, d2x))

    return dxdt


if __name__ == "__main__":

    u = jnp.array([ 0, 0, 1, 0, 1, 0 ])
    # u = jnp.array([ 0, 0, 1, 2, 0, 0.2 ])

    q0_ref = np.array([1, 0, 0, 0])
    p0_ref = np.array([0, 0, 0])

    # w0_ref = np.array([0, 0, 1]) * 1 
    # v0_ref = np.array([0.1, 0, 0.1])  * 3

    w0_ref = np.array([0, 0, 1])  
    v0_ref = np.array([0, 1, 0])  

    m = 1
    Ib = np.diag([ 0.5,0.7,0.9 ])
    J = np.block([
        [Ib, np.zeros((3, 3))],
        [np.zeros((3, 3)), m * np.identity(3)]
    ])

    W = np.trace(Ib) * np.identity(3) - Ib
    W = np.block([
        [np.trace(Ib) * np.identity(3) - Ib, np.zeros((3,1))],
        [np.zeros((1,3)), m],
    ])
    W_sqrt = sqrtm(W)
    print(W)
    print(W_sqrt)

# =====================================================
# Quaternion Dynamics Simulation
# =====================================================

    x0 = jnp.array([ 1,0,0,0,0,0,0,
                0,0,0,0,0,0 ])
    Nsim = 400
    dt = 0.01
    x_sim_list_quat = jnp.zeros((Nsim+1, 13))
    x_sim_list_quat = x_sim_list_quat.at[0].set( x0 )

    for i in range(Nsim):
        x_sim_list_quat = x_sim_list_quat.at[i+1].set( x_sim_list_quat[i] + SE3Dyn(x_sim_list_quat[i],u,i,J) * dt )

# =====================================================
# Error State Dynamics Simulation
# =====================================================

    # =====================================================
    # Reference Generation
    # =====================================================

    # x0_ref and X should be kept the same
    x0_ref = np.concatenate((q0_ref, p0_ref))
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

    debug = {"vel_zero": False}
    dyn_se3 = ErrorStateSE3AutoDiffDynamics(J, X_ref, xi_ref, dt, debug=debug)

    x0 = np.zeros((12,))
    u = np.array(u)
    x_sim_list_errstate = np.zeros(( Nsim+1, 12 ))

    x_sim_list_errstate[0] = x0
    for i in range(Nsim):
        x_sim_list_errstate[i+1] = dyn_se3.f( x_sim_list_errstate[i], u, i )

# =====================================================
# Visualization
# =====================================================

    # ----------------  Plotting Trajectory ---------------------------

    interval_plot = int((Nsim + 1) / 40)

    # Initialize the plot
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    # ax2 = fig1.add_subplot(122, projection='3d')
    # ax3 = fig1.add_subplot(133, projection='3d')
    ax1.set_title('Trajectory Comparison')
    # ax2.set_title('Error-State Trajectory')
    # ax3.set_title('Reference Trajectory')

    # Define an initial vector and plot on figure
    initial_vector = np.array([1, 0, 0])  # Example initial vector
    ax1.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], color='g', label='Initial Vector')
    # ax2.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], color='g', label='Initial Vector')

    positions = []
    errorquat_diff_list = []
    dist_errstate_to_ref_list = []

    for i in range(0, Nsim + 1, interval_plot):  

        # =========== 1. Plot the simulated quaternion dynamics configuration trajectory ===========

        quat = Quaternion(x_sim_list_quat[i, :4])  # Extract the quaternion from the X_ref data
        rot_matrix = quat.rotation_matrix  # Get the rotation matrix from the quaternion

        # rot_matrix = quat2rotm(x_sim_list[i, :4])
        rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
        
        # Extract the position 
        position = x_sim_list_quat[i, 4:7]

        # Reconstruct SE3 element
        se3_matrix_quat = np.block([
            [rot_matrix, position.reshape(3, 1)],
            [ np.zeros((1,3)), 1 ],
        ])

        # Plot the rotated vector
        ax1.quiver(position[0], position[1], position[2],
                rotated_vector[0], rotated_vector[1], rotated_vector[2],
                color='c', length=1, arrow_length_ratio=0.05, label='Quaternion' if i == 0 else '')
        
        positions.append(position)
        positions.append(position + rotated_vector)

        # =========== 2. Plot the simulated error-state configuration trajectory ===========

        se3_matrix_errstate = np.block([
            [Quaternion(np.array(X_ref[i, :4])).rotation_matrix, X_ref[i, 4:].reshape(3, 1)],
            [ np.zeros((1,3)), 1 ],
        ]) @ expm( se3_hat( x_sim_list_errstate[i, :6]) )

        rot_matrix = se3_matrix_errstate[:3,:3]  # Get the rotation matrix from the quaternion
        rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector

        position = se3_matrix_errstate[:3, 3]
        
        # Plot the rotated vector
        ax1.quiver(position[0], position[1], position[2],
                rotated_vector[0], rotated_vector[1], rotated_vector[2],
                color='b', length=1,  arrow_length_ratio=0.05, label='Error-State' if i == 0 else '')
        
        # =========== 3. Plot the Reference ===========

        quat = Quaternion(X_ref[i, :4, 0])  # Extract the quaternion from the X_ref data
        rot_matrix = quat.rotation_matrix  # Get the rotation matrix from the quaternion
        rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
        
        # Extract the position 
        position = X_ref[i, 4:, 0]

        # Plot the rotated vector
        ax1.quiver(position[0], position[1], position[2],
                rotated_vector[0], rotated_vector[1], rotated_vector[2],
                color='r', length=1, label='Reference' if i == 0 else '')
        
        # =========== 4. Compare configuration error ===========

        errorquat_diff_list.append( np.linalg.norm( se3_matrix_errstate - se3_matrix_quat ) ) 
        dist_errstate_to_ref_list.append( np.linalg.norm( W_sqrt @ se3_hat( x_sim_list_errstate[i, :6]) @ W_sqrt, 'fro') )

    positions = np.array(positions)

    # Automatically adjust the axis limits to fit the positions data
    if len(positions) > 0:  # Check if positions is not empty
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

        # Add a small margin to the limits to avoid singular transformation
        margin = 1
        ax1.set_xlim([x_min - margin, x_max + margin])
        ax1.set_ylim([y_min - margin, y_max + margin])
        ax1.set_zlim([z_min - margin, z_max + margin])

        # ax2.set_xlim([x_min - margin, x_max + margin])
        # ax2.set_ylim([y_min - margin, y_max + margin])
        # ax2.set_zlim([z_min - margin, z_max + margin])
    else:
        # If positions are empty, set default limits
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])
        # ax2.set_xlim([-1, 1])
        # ax2.set_ylim([-1, 1])
        # ax2.set_zlim([-1, 1])

    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # ax2.legend()
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')

    # ----------------  Plotting Error ---------------------------

    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(121)
    ax2 = fig2.add_subplot(122)

    ax1.plot( errorquat_diff_list )
    ax1.set_title('Dynamics Error w.r.t. Timestep')
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel(r"$\|X_{quat} - X_{errstate})\|_F$")

    plt.rc('text', usetex=True)
    ax2.plot( dist_errstate_to_ref_list, errorquat_diff_list )
    ax2.set_title('Dynamics Error w.r.t. Distance')
    ax2.legend()
    ax2.grid()
    ax2.set_xlabel(r"$d_{SE(3)}(X_{ref,t}, X_{t}) = \|\log(X_{ref,t}^{-1} X_{t})\|_W$")
    ax2.set_ylabel(r"$\|X_{quat} - X_{errstate})\|_F$")

    print("The terminal error is ", errorquat_diff_list[-1])


    plt.show()