from traopt_controller import iLQR
from traopt_dynamics import AutoDiffDynamics
from traopt_cost import AutoDiffCost

from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from  functools import partial
import time
import math


seed = 24234156
key = random.key(seed)

dt = 0.05

def f(x, u):
    ''' Fixed Origin Pendulum Dynamics (Full Actuated System) '''

    g = 9.8 # m/s^2
    m = 2 # kg
    l = 0.1 # m
    
    x1, x2 = x

    if (isinstance(u, jnp.ndarray) or isinstance(u, np.ndarray)) and (u.shape == (1,)):
        u, = u

    dx1 = x2
    dx2 = - (g / l) * jnp.cos(x1) + (1 / (m * (l**2) )) * u # (u**2)

    return jnp.array([dx1, dx2])


# def f(x, u):
#     ''' Car Pole System Dynamics (Underactuated System) '''

#     mc = 1
#     mp = 1
#     l = 1
#     g = 9.8
    
#     x1, x2, x3, x4 = x
#     if (isinstance(u, jnp.ndarray) or isinstance(u, np.ndarray)) and (u.shape == (1,)):
#         u, = u

#     dx1 = x2
#     dx2 = 1/( mc + mp * (math.sin(x3)**2) ) * ( u + mp * math.sin(x3) * ( l * (x4**2) + g * math.cos(x3) ) )
#     dx3 = x4
#     dx4 = 1/( l*mc + l*mp*(math.sin(x3)**2) ) * ( - u*math.cos(x3) - mp * l * (x4**2) * math.cos(x3) * math.sin(x3) - (mc+mp)*g*math.sin(x3) )

#     return jnp.array([dx1, dx2, dx3, dx4])



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

    if (isinstance(u, jnp.ndarray) or isinstance(u, np.ndarray)) and (u.shape == (1,)):
        u, = u

    R = 20
    x_diff = x - x_goal
    Q = jnp.diag( jnp.array([1000,50]) )
    return 0.5 * u * R * u + 0.5 * x_diff.T @ Q @ x_diff

def l_terminal(x,i):
    x_diff = x - x_goal
    Q_terminal = jnp.diag( jnp.array([1000,50]) )
    return 0.5 * x_diff.T @ Q_terminal @ x_diff

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged, J_hist, xs_hist, us_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state)
    


if __name__ == "__main__":

    state_size = 2
    action_size = 1

    cost = AutoDiffCost( l, l_terminal, state_size, action_size )
    N = 20

    x0 = jnp.array([jnp.pi/2+0.3, 0])
    us_init = np.zeros((N, action_size))
    x_goal = jnp.array([jnp.pi/2, 0])

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

    start_time_ilqr = time.perf_counter()
    xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr = \
        ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)
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

    start_time_ddp = time.perf_counter()
    xs_ddp, us_ddp, J_hist_ddp, xs_hist_ddp, us_hist_ddp = \
        ilqr.fit(x0, us_init, n_iterations=200, on_iteration=on_iteration)
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
    plt.title('DDP Final Trajectory')
    plt.xlabel('TimeStep')
    plt.ylabel('State')
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.plot( xs_ilqr, label=['theta','theta_dot'] )
    plt.title('ILQR Final Trajectory')
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

    xs_hist_ilqr = np.array(xs_hist_ilqr)
    us_hist_ilqr = np.array(us_hist_ilqr)
    xs_hist_ddp = np.array(xs_hist_ddp)
    us_hist_ddp = np.array(us_hist_ddp)


    fig, axs = plt.subplots(2, num=4)
    fig.suptitle('iLQR Trajectory Evolution')
    for i in range( xs_hist_ilqr.shape[0] ):
        axs[0].plot( xs_hist_ilqr[i,:,0], label = i )
        axs[1].plot( xs_hist_ilqr[i,:,1], label = i )

    axs[0].set_xlabel('TimeStep')
    axs[0].set_ylabel('Theta')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_xlabel('TimeStep')
    axs[1].set_ylabel('Theta Dot')
    axs[1].legend()
    axs[1].grid()


    fig, axs = plt.subplots(2, num=5)
    fig.suptitle('DDP Trajectory Evolution')
    for i in range( xs_hist_ddp.shape[0] ):
        axs[0].plot( xs_hist_ddp[i,:,0], label = i )
        axs[1].plot( xs_hist_ddp[i,:,1], label = i )

    axs[0].set_xlabel('TimeStep')
    axs[0].set_ylabel('Theta')
    axs[0].legend()
    axs[0].grid()

    axs[1].set_xlabel('TimeStep')
    axs[1].set_ylabel('Theta Dot')
    axs[1].legend()
    axs[1].grid()



    fig_ilqr, axs_ilqr = plt.subplots(2, num=6)
    fig.suptitle('iLQR Trajectory Evolution')
    axs_ilqr[0].set_ylim( np.min(xs_hist_ilqr[:,:,0])-0.2, np.max(xs_hist_ilqr[:,:,0])+0.2 ) 
    axs_ilqr[1].set_ylim( np.min(xs_hist_ilqr[:,:,1])-0.2, np.max(xs_hist_ilqr[:,:,1])+0.2 )

    axs_ilqr[0].grid()
    # axs_ilqr[0].set_xlabel('TimeStep')
    axs_ilqr[0].set_ylabel('Theta')
    axs_ilqr[1].grid()
    axs_ilqr[1].set_xlabel('TimeStep')
    axs_ilqr[1].set_ylabel('Theta Dot')

    line0_ilqr, = axs_ilqr[0].plot(xs_hist_ilqr[0,:,0],lw=2)
    line1_ilqr, = axs_ilqr[1].plot(xs_hist_ilqr[0,:,1],lw=2)

    def func_animation_ilqr(i):
        line0_ilqr.set_ydata(xs_hist_ilqr[i,:,0])
        line1_ilqr.set_ydata(xs_hist_ilqr[i,:,1])
        return line0_ilqr, line1_ilqr

    animation_ilqr = FuncAnimation(fig_ilqr,
                        func = func_animation_ilqr,
                        frames = np.arange(0, xs_hist_ilqr.shape[0]), 
                        interval = 500)
    


    fig_ddp, axs_ddp = plt.subplots(2, num=7)
    fig.suptitle('DDP Trajectory Evolution')
    axs_ddp[0].set_ylim( np.min(xs_hist_ddp[:,:,0])-0.2, np.max(xs_hist_ddp[:,:,0])+0.2 ) 
    axs_ddp[1].set_ylim( np.min(xs_hist_ddp[:,:,1])-0.2, np.max(xs_hist_ddp[:,:,1])+0.2 )

    axs_ddp[0].grid()
    # axs_ddp[0].set_xlabel('TimeStep')
    axs_ddp[0].set_ylabel('Theta')
    axs_ddp[1].grid()
    axs_ddp[1].set_xlabel('TimeStep')
    axs_ddp[1].set_ylabel('Theta Dot')

    line0_ddp, = axs_ddp[0].plot(xs_hist_ddp[0,:,0],lw=2)
    line1_ddp, = axs_ddp[1].plot(xs_hist_ddp[0,:,1],lw=2)

    def func_animation_ddp(i):
        line0_ddp.set_ydata(xs_hist_ddp[i,:,0])
        line1_ddp.set_ydata(xs_hist_ddp[i,:,1])
        return line0_ddp, line1_ddp

    animation_ddp = FuncAnimation(fig_ddp,
                        func = func_animation_ddp,
                        frames = np.arange(0, xs_hist_ddp.shape[0]), 
                        interval = 500)
     

    plt.show()


    
    






    

