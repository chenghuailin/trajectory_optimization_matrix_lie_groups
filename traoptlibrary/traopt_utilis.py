import numpy as np
from jax import vmap, jit

# import jax.numpy as jnp
import numpy as jnp
from jax.numpy.linalg import norm
# from pyquaternion import Quaternion
from manifpy import SE3, SE3Tangent
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation
from functools import partial

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

def SE32absangle(m) -> np.ndarray:
    """
    Calculate the absolute rotation angle (along geodesic) 
    from a rotation matrix.
    
    Parameters:
    m -- 4x4 SE3 matrix
    
    Returns:
    angle -- Rotation angle (in degrees)
    """
    if m.shape != (4, 4):
        raise ValueError("The input must be a 4x4 SE3 matrix")
    R = m[:3,:3]
    trace = np.trace(R)
    angle_radian = np.arccos((trace - 1) / 2)
    angle_degree = np.rad2deg(angle_radian)

    return angle_degree

def parallel_SE32absangle(m_list):
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Apply SE32absangle conversion in parallel across each SE(3) matrix in m_list
        angle_list = np.array(list(executor.map(SE32absangle, m_list)))
    return angle_list

def rotm2absangle(m) -> np.ndarray:
    """
    Calculate the absolute rotation angle (along geodesic) 
    from a rotation matrix.
    
    Parameters:
    m -- 3x3 rotation matrix
    
    Returns:
    angle -- Rotation angle (in degrees)
    """
    if m.shape != (3, 3):
        raise ValueError("The input must be a 3x3 rotation matrix")
    trace = np.trace(m)
    angle_radian = np.arccos((trace - 1) / 2)
    angle_degree = np.rad2deg(angle_radian)

    return angle_degree

def parallel_rotm2absangle(m_list):
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Apply rotm2absangle conversion in parallel across each rotation matrix in m_list
        angle_list = np.array(list(executor.map(rotm2absangle, m_list)))
    return angle_list

# def quat2rotm(quat):
#     """ Converts a quaternion to a rotation matrix. """
#     q = quat / norm(quat)  # Ensure the quaternion is normalized
#     q0, q1, q2, q3 = q

#     R = jnp.array([
#         [1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)],
#         [2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1)],
#         [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2)]
#     ])
#     return R

def quat2rotm(quat:np.ndarray) -> np.ndarray:
    """ Converts a or a list of scalar_first quaternion to a rotation matrix. """
    return Rotation.from_quat(quat, scalar_first=True).as_matrix()

def rotm2quat(m:np.ndarray) -> np.ndarray:
    """Creates a quaternion from a rotation matrix defining a given orientation.
    
    Parameters
    ----------
    R : [3x3] np.ndarray
        Rotation matrix
            
    Returns
    -------
    q : [4x1] np.ndarray
        quaternion defining the orientation    
    """    
    q1,q2,q3,q0 = Rotation.from_matrix(m).as_quat()
    return np.array([q0,q1,q2,q3])

def rotm2euler(m:np.ndarray, order='zxy') -> np.ndarray:
    """Creates a quaternion from a rotation matrix defining a given orientation.
    
    Parameters
    ----------
    R : [3x3] np.ndarray
        Rotation matrix
            
    Returns
    -------
    q : [4x1] np.ndarray
        quaternion defining the orientation    
    """    
    if order is None:
        order = 'zxy'  # Use a default rotation order
    theta_z,theta_x,theta_y = Rotation.from_matrix(m).as_euler(order,degrees=True)
    return np.array([theta_z,theta_x,theta_y])

def parallel_rotm2euler(m_list, order):
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Apply rotm2absangle conversion in parallel across each rotation matrix in m_list
        rotm2euler_withorder = partial( rotm2euler, order=order )
        angle_list = np.array(list(executor.map(rotm2euler_withorder, m_list)))
    return angle_list

def euler2quat(eulerAngles:jnp.ndarray|list)->jnp.ndarray:
    """
    Convert an Euler angle to a quaternion.
    
    We have used the following definition of Euler angles.

    - Tait-Bryan variant of Euler Angles
    - Yaw-pitch-roll rotation order (ZYX convention), rotating around the z, y and x axes respectively
    - Intrinsic rotation (the axes move with each rotation)
    - Active (otherwise known as alibi) rotation (the point is rotated, not the coordinate system)
    - Right-handed coordinate system with right-handed rotations
    
    Parameters
    ----------
    eulerAngles : 
        [3x1] jnp.ndarray  
        [roll, pitch, yaw] angles in radians 
            
    Returns
    -------
    p : [4x1] jnp.ndarray
        quaternion defining a given orientation
  """
    if isinstance(eulerAngles, list) and len(eulerAngles)==3:
        eulerAngles = np.array(eulerAngles) 
    elif isinstance(eulerAngles, (jnp.ndarray, np.ndarray)) and eulerAngles.size==3:
        pass
    else:
        raise TypeError("The eulerAngles must be given as [3x1] np.ndarray vector or a python list of 3 elements")
    
    roll = eulerAngles[0]
    pitch = eulerAngles[1]
    yaw = eulerAngles[2]
    
    q0 = jnp.cos(roll/2) * jnp.cos(pitch/2) * jnp.cos(yaw/2) + jnp.sin(roll/2) * jnp.sin(pitch/2) * jnp.sin(yaw/2)
    q1 = jnp.sin(roll/2) * jnp.cos(pitch/2) * jnp.cos(yaw/2) - jnp.cos(roll/2) * jnp.sin(pitch/2) * jnp.sin(yaw/2)
    q2 = jnp.cos(roll/2) * jnp.sin(pitch/2) * jnp.cos(yaw/2) + jnp.sin(roll/2) * jnp.cos(pitch/2) * jnp.sin(yaw/2)
    q3 = jnp.cos(roll/2) * jnp.cos(pitch/2) * jnp.sin(yaw/2) - jnp.sin(roll/2) * jnp.sin(pitch/2) * jnp.cos(yaw/2)
    
    p = jnp.r_[q0, q1, q2, q3]
    
    return p

def quatpos2SE3(x7):
    """ Converts a vector concatenated by quaternion 
        and position to SE(3) matrix. 

        Args: 
            x7: 7-d state vector, numpy array, (7,) or (7,1).
    """
    if x7.shape == (7,) or x7.shape == (7,1):
        x7 = x7.reshape((7,1))
        # se3_matrix = jnp.block([
        #     [ jnp.array(Quaternion(x7[:4, 0]).rotation_matrix), x7[4:].reshape(3, 1)],
        #     [ jnp.zeros((1,3)), 1 ],
        # ])
        se3_matrix = jnp.block([
            [ quat2rotm( x7[:4, 0] ), x7[4:].reshape(3, 1)],
            [ jnp.zeros((1,3)), 1 ],
        ])
        return se3_matrix
    else:
        raise ValueError("Input must be a 7-d np or jnp vector")
    
def rotmpos2SE3(m, x):
    """ Converts a vector concatenated by quaternion 
        and position to SE(3) matrix. 

        Args: 
            m: rotation matrix, numpy array, (3,3).
            x: position vector, numpy array, (3,) or (3,1).
    """
    if x.shape == (3,) or x.shape == (3,1):
        x = x.reshape((3,1))
        se3_matrix = jnp.block([
            [ m, x],
            [ jnp.zeros((1,3)), 1 ],
        ])
        return se3_matrix
    else:
        raise ValueError("Input dimension incorrect")
    
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
        return jnp.concatenate((
            # Quaternion(matrix=m[:3, :3]).elements,
            rotm2quat(m[:3, :3]),
            m[:3, 3]
        )).reshape((7,1))
    else:
        raise ValueError("Input must be a 7-d np or jnp vector")
    
vec_SE32quatpos = jit(vmap(SE32quatpos))

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

def SE32manifSE3( x ):
    """ Converts a SE(3) matrix to a manif SE(3) object

        Args: 
            x: SE3 matrix, numpy array, (4, 4) .
    """
    quatpos = SE32quatpos(x)
    pos = quatpos[4:].reshape(3,1)
    q0,q1,q2,q3 = quatpos[:4]
    quat = np.array([q1,q2,q3,q0])
    # return pos, quat
    return SE3(position=pos, quaternion=quat)

def manifSE32SE3( x ):
    """ Converts a manif SE(3) object to a SE(3) matrix.

        Args: 
            x: SE3 matrix, numpy array, (4, 4) .
    """
    # return np.block([
    #     [x.rotation(), x.translation().reshape(-1,1)],
    #     [np.zeros((1,3)),1]
    # ])
    return x.transform()

def se32manifse3( x ):
    """ Converts a se(3) tangent vector, i.e. twist vector,
        to a manif SE3Tangent object

        Args: 
            x: manif SE3Tangent object.
    """
    w = x[:3]
    v = x[3:]
    twist_mnf = np.concatenate((v,w))

    return SE3Tangent(twist_mnf)

def manifse32se3( x ):
    """ Converts a manif SE3Tangent object to
        a se(3) tangent vector, i.e. twist vector.

        Args: 
            x: se(3) tangent vector.
    """
    if type(x) == SE3Tangent:
        x = x.coeffs()

    v = x[:3]
    w = x[3:]
    twist = np.concatenate((w,v))

    return twist

# vec_SE32manifSE3 = jit(vmap(SE32manifSE3))

def Jmnf2J( J ):
    """ Reorder manif Jacobian on SE(3) to
        conventional twist order Jacobian
    """

    J_v_v = J[:3,:3]
    J_v_w = J[:3,3:]
    J_w_v = J[3:,:3]
    J_w_w = J[3:,3:]
    return np.block([
        [J_w_w, J_w_v],
        [J_v_w, J_v_v]
    ])

def parallel_SE32manifSE3(q_ref):
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        # Apply SE32manifSE3 conversion in parallel across each SE(3) matrix in q_ref
        q_ref_manif = list(executor.map(SE32manifSE3, q_ref))
    return q_ref_manif