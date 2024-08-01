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
        u, = u

    dx1 = x2
    dx2 = - (g / l) * jnp.cos(x1) + (1 / (m * (l**2) )) * u # (u**2)

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
    R = 1
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
    DDP
    ===========================
    """

    print("=========== ddp iteration start ===========")

    dynamics = AutoDiffDynamics( fd_rk4_dt , state_size, action_size, hessians=True ) 

    ilqr = iLQR(dynamics, cost, N)
    x0 = np.zeros(state_size)
    us_init = np.zeros(N)

    xs_ddp, us_ddp, J_ddp = ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)

    """
    ===========================
    ILQR
    ===========================
    """

    print("=========== ilqr iteration start ===========")

    dynamics = AutoDiffDynamics( fd_rk4_dt , state_size, action_size, hessians=False ) 

    ilqr = iLQR(dynamics, cost, N)
    x0 = np.zeros(state_size)
    us_init = np.zeros(N)

    J_hist_ilqr = []
    xs_ilqr, us_ilqr, J_ilqr = ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)


    """
    ===========================
    Result Visualization
    ===========================
    """

    plt.figure()
    plt.plot( xs_ddp, label=['theta','theta_dot'] )
    plt.title('DDP')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot( xs_ilqr, label=['theta','theta_dot'] )
    plt.title('ILQR')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend()
    plt.grid()
    plt.show()


    
    






    

