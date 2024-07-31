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
    l = 0.1 # m
    
    x1, x2 = x

    if isinstance(u, jnp.ndarray) and (u.shape == (1,)):
        if test_flag == 1:
            print("In the controller code, u is ", u)
        u, = u

    dx1 = x2
    dx2 = - (g / l) * jnp.cos(x1) + (1 / (m * (l**2) )) * u # (u**2)

    # print("u = ", u)

    # print("dx1 = ", dx1)
    # print("dx2 = ", dx2)

    # x1, x2 = x
    # u1, u2 = u
    # dx1 = x2
    # dx2 = - (g / l) * jnp.cos(x1) + (1 / (m * (l**2) )) * u1 + u2

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

x_goal = jnp.array([-jnp.pi/2, 0])

def l(x,u,i):
    R = 1
    x_diff = x - x_goal
    Q = jnp.diag( jnp.array([1000,50]) )
    return 0.5 * u * R * u + 0.5 * x_diff.T @ Q @ x_diff

def l_terminal(x,i):
    x_diff = x - x_goal
    Q_terminal = jnp.diag( jnp.array([1000,50]) )
    return 0.5 * x_diff.T @ Q_terminal @ x_diff
    


if __name__ == "__main__":

    state_size = 2
    action_size = 1

    dynamics = AutoDiffDynamics( fd_rk4_dt , state_size, action_size, hessians=True ) 

    x = random.normal(key, (state_size,))
    u = random.normal(key, (action_size,))
    i = 1

    test_flag = 0

    print("x = ",x)
    print("u = ",x)

    # print("f = ",   dynamics.f(x,u,i))
    print("fx = ",  dynamics.f_x(x,u,i))
    print("fu = ",  dynamics.f_u(x,u,i))
    print("fxx = ", dynamics.f_xx(x,u,i))
    print("fux = ", dynamics.f_ux(x,u,i))
    print("fuu = ", dynamics.f_uu(x,u,i))

    cost = AutoDiffCost( l, l_terminal, state_size, action_size )

    print("l = ",   cost.l(x,u,i))
    print("lx = ",  cost.l_x(x,u,i))
    print("lu = ",  cost.l_u(x,u,i))
    print("lxx = ", cost.l_xx(x,u,i))
    print("luu = ", cost.l_uu(x,u,i))

    test_flag = 1

    N = 300
    ilqr = iLQR(dynamics, cost, N)

    x0 = np.zeros(state_size)
    us_init = np.zeros(N)

    xs, us = ilqr.fit(x0, us_init, n_iterations=200)




    

