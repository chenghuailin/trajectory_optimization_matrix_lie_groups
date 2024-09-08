import numpy as np
import jax.numpy as jnp
from jax.numpy.linalg import norm
from pyquaternion import Quaternion

def skew( w ):
    """Get the isomorphic element in the Lie algebra for SO3, 
        i.e. the skew symmetric matrix."""
    if isinstance(w, (jnp.ndarray, np.ndarray)) and w.shape == (3,) or w.shape == (3, 1):
        w_flat = w.reshape(-1)
        return jnp.array([
            [0, -w_flat[2], w_flat[1]],
            [w_flat[2], 0, -w_flat[0]],
            [-w_flat[1], w_flat[0], 0]
        ])
    else:
        raise ValueError("Input must be a 3d np or jnp vector")


def unskew(omega_hat):
    """
    Extract the vector from a 3x3 skew-symmetric matrix.
    
    Parameters:
    omega_hat (numpy.ndarray): 3x3 skew-symmetric matrix
    
    Returns:
    numpy.ndarray: 3x1 vector corresponding to the skew-symmetric matrix
    """

    if isinstance(omega_hat, (jnp.ndarray, np.ndarray)) \
        and omega_hat.shape == (3,3):
        return jnp.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])
    else:
        raise ValueError("Input must be a 3x3 np or jnp matrix")

    

def se3_hat( xi ):
    
    """Given the isomorphic cartesian 6d vector [omega, pos], 
        return the Lie algebra se(3) matrix. """
    
    if isinstance(xi, (jnp.ndarray, np.ndarray)) and xi.shape == (6,) or xi.shape == (6, 1):
        xi_flat = xi.reshape(-1)
        return jnp.block([
            [skew(xi_flat[:3]), xi_flat[3:6].reshape(3, 1)],
            [jnp.zeros((1, 3)), 0]
        ])
    else:
        raise ValueError("Input must be a 6d np or jnp array")
    
def se3_vee(se3_mat):

    """Given an Lie algbera se(3) matrix, 
        return the isomorphic cartesian 6d vector [omega, pos]."""
    
    if isinstance(se3_mat, (jnp.ndarray, np.ndarray)) and se3_mat.shape == (4, 4):
        
        # Use the unskew function to get ω from skew(ω)
        omega = unskew(se3_mat[:3, :3])
        
        # Extract the translation vector pos (top-right 3x1 part)
        pos = se3_mat[:3, 3].reshape(3,)
                
        # Combine ω and pos into the 6d vector
        return jnp.concatenate((omega, pos))
    else:
        raise ValueError("Input must be a 4x4 np or jnp array representing an se(3) matrix")

    
def adjoint( xi ):
    """ Get the the adjoint matrix representation of Lie Algebra,
        i.e. the matrix representation of Lie bracket."""
    if xi.shape == (6,) or xi.shape == (6,1):
        xi = xi.reshape(6,)
        w = jnp.array([xi[0], xi[1], xi[2]])
        v = jnp.array([xi[3], xi[4], xi[5]])
        adx = jnp.block([
            [skew(w), jnp.zeros((3, 3))],
            [skew(v), skew(w)]
        ])
        return adx
    else:
        raise ValueError("Input must be a 6-d np or jnp vector")
    
    
def coadjoint( xi ):
    """ Get the the coadjoint matrix representation of Lie Algebra."""
    return adjoint(xi).T

def quat2rotm(quat):
    """ Converts a quaternion to a rotation matrix. """
    q = quat / norm(quat)  # Ensure the quaternion is normalized
    q0, q1, q2, q3 = q

    R = jnp.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)],
        [2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1)],
        [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2)]
    ])
    return R

def quatpos2SE3(x7):
    """ Converts a vector concatenated by quaternion 
        and position to SE(3) matrix. 

        Args: 
            x7: 7-d state vector, numpy array, (7,) or (7,1).
    """
    if x7.shape == (7,) or x7.shape == (7,1):
        x7 = x7.reshape((7,1))
        se3_matrix = jnp.block([
            [ jnp.array(Quaternion(x7[:4, 0]).rotation_matrix), x7[4:].reshape(3, 1)],
            [ jnp.zeros((1,3)), 1 ],
        ])
        return se3_matrix
    else:
        raise ValueError("Input must be a 7-d np or jnp vector")
    
def SE32quatpos(m):
    """ Converts a SE(3) matrix to vector concatenated by quaternion 
        and position. 

        Args: 
            m: SE3 matrix, numpy array, (4, 4) .
    """
    if m.shape == (4,4):
        # rot_matrix = m[:3, :3]
        # position = m[:3, 3]
        # quaternion = Quaternion(matrix=rot_matrix)
        return np.concatenate((
            Quaternion(matrix=m[:3, :3]).elements,
            m[:3, 3]
        )).reshape((7,1))
    else:
        raise ValueError("Input must be a 7-d np or jnp vector")

def is_pos_def(A):
    """ Check if matrix A is PD """
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False