import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.linalg import inv
from traopt_utilis import coadjoint, quat2rotm
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

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
    # R_matrix = quat2rotm(q).T  # Transpose for correct orientation
    # quat = Quaternion(jnp.array(q))  # Extract the quaternion from the X_ref data
    # R_matrix = quat.rotation_matrix  # Get the rotation matrix from the quaternion
    R_matrix = quat2rotm(q)

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

    x0 = jnp.array([ 1,0,0,0,0,0,0,
                0,0,0,0,0,0 ])
    Nsim = 400
    dt = 0.01
    x_sim_list = jnp.zeros((Nsim+1, 13))
    x_sim_list = x_sim_list.at[0].set( x0 )

    m = 1
    Ib = np.diag([ 0.5,0.7,0.9 ])
    J = np.block([
        [Ib, np.zeros((3, 3))],
        [np.zeros((3, 3)), m * np.identity(3)]
    ])

    u = jnp.array([ 0, 0, 0, 0, 1, 0 ])

    for i in range(Nsim):
        x_sim_list = x_sim_list.at[i+1].set( x_sim_list[i] + SE3Dyn(x_sim_list[i],u,i,J) * dt )


    interval_plot = int((Nsim + 1) / 40)

    # Initialize the plot
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')

    # Define an initial vector and plot on figure
    initial_vector = np.array([1, 0, 0])  # Example initial vector
    ax1.quiver(0, 0, 0, initial_vector[0], initial_vector[1], initial_vector[2], color='g', label='Initial Vector')

    positions = []

    for i in range(0, Nsim + 1, interval_plot):  

        quat = Quaternion(x_sim_list[i, :4])  # Extract the quaternion from the X_ref data
        rot_matrix = quat.rotation_matrix  # Get the rotation matrix from the quaternion

        # rot_matrix = quat2rotm(x_sim_list[i, :4])
        rotated_vector = rot_matrix @ initial_vector  # Apply the rotation to the initial vector
        
        # Extract the position 
        position = x_sim_list[i, 4:7]

        # Plot the rotated vector
        ax1.quiver(position[0], position[1], position[2],
                rotated_vector[0], rotated_vector[1], rotated_vector[2],
                color='b', length=1, arrow_length_ratio=0.02, label='Rotated Vector' if i == 0 else '')
        
        positions.append(position)
        positions.append(position + rotated_vector)

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
    else:
        # If positions are empty, set default limits
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])

    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    plt.show()