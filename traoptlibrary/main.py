from traopt_controller import iLQR
from traopt_dynamics import AutoDiffDynamics
from traopt_cost import AutoDiffCost

from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from  functools import partial
import time


seed = 24234156
key = random.key(seed)

dt = 0.01

def f(x, u):
    g = 9.8 # m/s^2
    m = 2 # kg
    l = 0.1 # m
    
    x1, x2 = x

    # print("x = ", x)
    # print("u = ", u)

    if (isinstance(u, jnp.ndarray) or isinstance(u, np.ndarray)) and (u.shape == (1,)):
        u, = u

    # print("u = ", u)

    dx1 = x2
    # print("dx1 = ", dx1)
    dx2 = - (g / l) * jnp.cos(x1) + (1 / (m * (l**2) )) * u # (u**2)
    # print("dx2 = ", dx2)

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

x_goal = jnp.array([jnp.pi/2, 0])

def l(x,u,i):

    if (isinstance(u, jnp.ndarray) or isinstance(u, np.ndarray)) and (u.shape == (1,)):
        u, = u

    R = 100
    x_diff = x - x_goal
    Q = jnp.diag( jnp.array([1000,50]) )
    return 0.5 * u * R * u + 0.5 * x_diff.T @ Q @ x_diff

def l_terminal(x,i):
    x_diff = x - x_goal
    Q_terminal = jnp.diag( jnp.array([1000,50]) )
    return 0.5 * x_diff.T @ Q_terminal @ x_diff

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged, J_hist):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)
    


if __name__ == "__main__":

    state_size = 2
    action_size = 1

    cost = AutoDiffCost( l, l_terminal, state_size, action_size )
    N = 200

    """
    ===========================
    1. Dynamics Operation Test
    ===========================
    """

    # x = random.normal(key, (state_size,))
    # u = random.normal(key, (action_size,))
    # i = 1

    # dynamics = AutoDiffDynamics( fd_rk4_dt , state_size, action_size, hessians=True ) 

    # print("x = ",x)
    # print("u = ",x)

    # print("f = ",   dynamics.f(x,u,i))
    # print("fx = ",  dynamics.f_x(x,u,i))
    # print("fu = ",  dynamics.f_u(x,u,i))
    # print("fxx = ", dynamics.f_xx(x,u,i))
    # print("fux = ", dynamics.f_ux(x,u,i))
    # print("fuu = ", dynamics.f_uu(x,u,i))

    """
    ===========================
    2. ILQR
    ===========================
    """

    print("=========== ilqr iteration start ===========")

    HESSIANS = False

    dynamics = AutoDiffDynamics( fd_rk4_dt , state_size, action_size, hessians=HESSIANS ) 
    ilqr = iLQR(dynamics, cost, N, hessians=HESSIANS)

    x0 = np.zeros(state_size)
    us_init = np.zeros((N, action_size))

    start_time_ilqr = time.perf_counter()
    xs_ilqr, us_ilqr, J_hist_ilqr = ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)
    end_time_ilqr = time.perf_counter()
    time_ilqr = end_time_ilqr - start_time_ilqr

    print("=========== ilqr converged, running time: ", time_ilqr ," ===========")

    """
    ===========================
    3. DDP
    ===========================
    """

    print("=========== ddp iteration start ===========")

    HESSIANS = True

    dynamics = AutoDiffDynamics( fd_rk4_dt , state_size, action_size, hessians=HESSIANS ) 
    ilqr = iLQR(dynamics, cost, N, hessians=HESSIANS)

    x0 = np.zeros(state_size)
    us_init = np.zeros((N, action_size))

    start_time_ddp = time.perf_counter()
    xs_ddp, us_ddp, J_hist_ddp = ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)
    end_time_ddp = time.perf_counter()
    time_ddp = end_time_ddp - start_time_ddp

    print("=========== ddp converged, running time: ", time_ddp ,"s ===========")

    """
    ===========================
    Result Visualization
    ===========================
    """

    plt.figure(1)
    plt.plot( xs_ddp, label=['theta','theta_dot'] )
    plt.title('DDP')
    plt.xlabel('TimeStep')
    plt.ylabel('State')
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.plot( xs_ilqr, label=['theta','theta_dot'] )
    plt.title('ILQR')
    plt.xlabel('TimeStep')
    plt.ylabel('State')
    plt.legend()
    plt.grid()

    plt.figure(3)
    plt.plot(J_hist_ilqr, label='ilqr')
    plt.plot(J_hist_ddp, label='ddp')
    plt.title('Cost Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid()

    plt.show()


    
    






    

