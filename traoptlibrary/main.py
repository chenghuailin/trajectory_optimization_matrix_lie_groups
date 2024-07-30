from traopt_controller import iLQR
from traopt_dynamics import AutoDiffDynamics
from traopt_cost import AutoDiffCost

from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from  functools import partial

seed = 24234156
key = random.key(seed)

dt = 0.01

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


fd_rk4_dt = partial( fd_rk4, dt = dt )


def l(x,u,i):
    R = 1
    return 0.5 * u * R * u

def l_terminal(x,u,i):
    Q_terminal = jnp.diag( jnp.array([1000,50]) )
    


if __name__ == "__main__":

    dynamics = AutoDiffDynamics( fd_rk4_dt , 2, 1, hessians=True ) 

    x = random.normal(key, (2,))
    u = random.normal(key, ())
    i = 1
    
    print("x = ",x)
    print("u = ",x)

    print("f = ",   dynamics.f(x,u,i))
    print("fx = ",  dynamics.f_x(x,u,i))
    print("fu = ",  dynamics.f_u(x,u,i))
    print("fxx = ", dynamics.f_xx(x,u,i))
    print("fux = ", dynamics.f_ux(x,u,i))
    print("fuu = ", dynamics.f_uu(x,u,i))

    cost = AutoDiffCost( l, l_terminal, 2, 1 )

    print("l = ",   cost.l(x,u,i))
    print("lx = ",  cost.l_x(x,u,i))
    print("lu = ",  cost.l_u(x,u,i))
    print("lxx = ", cost.l_xx(x,u,i))
    print("luu = ", cost.l_uu(x,u,i))




    

