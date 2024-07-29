from traopt_controller import iLQR
from traopt_dynamics import AutoDiffDynamics

from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

seed = 24234156
key = random.key(seed)

def f(x, u):
    g = 9.8 # m/s^2
    m = 2 # kg
    l = 0.5 # m
    
    x1, x2 = x
    dx1 = x2
    dx2 = - (l / g) * jnp.cos(x1) + (1 / (m * l)) * u # (u**2)

    # x1, x2 = x
    # dx1 = x2
    # dx2 = u

    return jnp.array([dx1, dx2])


def fd_rk4(x, u, i, dt):
    g = 9.8 # m/s^2
    m = 2 # kg
    l = 0.5 # m

    s1 = f(x,u)
    s2 = f( x+ dt/2*s1, u )
    s3 = f( x+ dt/2*s2, u )
    s4 = f( x+ dt*s3, u )
    x_next = x + dt/6 * ( s1 + 2 * s2 + 2 * s3 + s4 )
    
    return x_next


if __name__ == "__main__":

    dynamics = AutoDiffDynamics( fd_rk4, 2,1 ) 

    

