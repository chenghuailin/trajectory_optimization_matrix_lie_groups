import numpy as np
import jax.numpy as jnp

def skew( w ):
    """Get the isomorphic element in the Lie algebra for SO3, 
        i.e. the skew symmetric matrix."""
    if isinstance(w, np.ndarray) or isinstance(w, jnp.ndarray) and w.shape == (3,) or w.shape == (3, 1):
        w_flat = w.reshape(-1)
        return np.array([
            [0, -w_flat[2], w_flat[1]],
            [w_flat[2], 0, -w_flat[0]],
            [-w_flat[1], w_flat[0], 0]
        ])
    else:
        raise ValueError("Input must be a 3d vector")


def unskew(omega_hat):
    """
    Extract the vector from a 3x3 skew-symmetric matrix.
    
    Parameters:
    omega_hat (numpy.ndarray): 3x3 skew-symmetric matrix
    
    Returns:
    numpy.ndarray: 3x1 vector corresponding to the skew-symmetric matrix
    """
    return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])
    

def se3hat( w ):
    
    """Given the isomorphic cartesian 6d vector, 
        return the Lie algebra se(3) matrix. """
    
    if isinstance(w, np.ndarray) or isinstance(w, jnp.ndarray) and w.shape == (6,) or w.shape == (6, 1):
        w_flat = w.reshape(-1)
        return np.array([
            [0, -w_flat[2], w_flat[1]],
            [w_flat[2], 0, -w_flat[0]],
            [-w_flat[1], w_flat[0], 0]
        ])
    else:
        raise ValueError("Input must be a 6d np or jnp array")