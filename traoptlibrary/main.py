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


seed = 24234156
key = random.key(seed)

dt = 0.01

# def f(x, u):
#     ''' Fixed Origin Pendulum Dynamics (Full Actuated System) '''

#     g = 9.8 # m/s^2
#     m = 2 # kg
#     l = 0.1 # m
    
#     x1, x2 = x

#     if (isinstance(u, jnp.ndarray) or isinstance(u, np.ndarray)) and (u.shape == (1,)):
#         u, = u

#     dx1 = x2
#     dx2 = - (g / l) * jnp.cos(x1) + (1 / (m * (l**2) )) * u # (u**2)

#     return jnp.array([dx1, dx2])


def f(x, u):
    ''' Car Pole System Dynamics (Underactuated System) '''

    mc = 1
    mp = 1
    l = 1
    g = 9.8
    
    x1, x2, x3, x4 = x
    if (isinstance(u, jnp.ndarray) or isinstance(u, np.ndarray)) and (u.shape == (1,)):
        u, = u

    dx1 = x2
    dx2 = 1/( mc + mp * (jnp.sin(x3)**2) ) * ( u + mp * jnp.sin(x3) * ( l * (x4**2) + g * jnp.cos(x3) ) )
    dx3 = x4
    dx4 = 1/( l*mc + l*mp*(jnp.sin(x3)**2) ) * ( - u*jnp.cos(x3) - mp * l * (x4**2) * jnp.cos(x3) * jnp.sin(x3) - (mc+mp)*g*jnp.sin(x3) )

    return jnp.array([dx1, dx2, dx3, dx4])



def fd_rk4(x, u, i, dt):

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

    R = 80
    x_diff = x - x_goal
    # Q = jnp.diag( jnp.array([1000,50]) )
    Q = jnp.diag( jnp.array([100,100,10000,100]) )
    return 0.5 * u * R * u + 0.5 * x_diff.T @ Q @ x_diff

def l_terminal(x,i):
    x_diff = x - x_goal
    # Q_terminal = jnp.diag( jnp.array([1000,50]) )
    Q_terminal = jnp.diag( jnp.array([100,100,10000,100]) )
    return 0.5 * x_diff.T @ Q_terminal @ x_diff

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged, alpha, mu, J_hist, xs_hist, us_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = xs[-1]
    print("iteration", iteration_count, info, J_opt, final_state, alpha, mu)
    


if __name__ == "__main__":

    # state_size = 2
    state_size = 4
    action_size = 1

    cost = AutoDiffCost( l, l_terminal, state_size, action_size )
    N = 800

    us_init = np.zeros((N, action_size))

    # x0 = jnp.array([jnp.pi/2+0.3, 0])
    # x_goal = jnp.array([jnp.pi/2, 0])

    x0 = jnp.array([ 9., 0., 0., 0.])
    x_goal = jnp.array([ 10., 0., jnp.pi, 0])

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
    for j in range( state_size ):
        plt.plot( xs_ddp[:,j], label = 'State '+str(j) )
    plt.title('DDP Final Trajectory')
    plt.xlabel('TimeStep')
    plt.ylabel('State')
    plt.legend()
    plt.grid()

    plt.figure(2)
    for j in range( state_size ):
        plt.plot( xs_ilqr[:,j], label = 'State '+str(j) )
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


    fig, axs = plt.subplots(state_size, num=4)
    fig.suptitle('iLQR Trajectory Evolution')
    for i in range( xs_hist_ilqr.shape[0] ):
        for j in range( state_size ):
            axs[j].plot( xs_hist_ilqr[i,:,j], label = str(i) )

            axs[j].set_ylabel('State '+str(j) )
            axs[j].legend()
            axs[j].grid(True)
        
        axs[j].set_xlabel('TimeStep')


    fig, axs = plt.subplots(state_size, num=5)
    fig.suptitle('DDP Trajectory Evolution')
    for i in range( xs_hist_ddp.shape[0] ):
        for j in range( state_size ):
            axs[j].plot( xs_hist_ddp[i,:,j], label = str(i) )

            axs[j].set_ylabel('State '+str(j) )
            axs[j].legend()
            axs[j].grid(True)

        axs[j].set_xlabel('TimeStep')


    # fig_ilqr, axs_ilqr = plt.subplots(state_size, num=6)
    # fig.suptitle('iLQR Trajectory Evolution')
    # for j in range( state_size ):
    #     axs_ilqr[j].set_ylim( np.min(xs_hist_ilqr[:,:,j])-0.2, np.max(xs_hist_ilqr[:,:,j])+0.2 ) 

    #     axs_ilqr[j].grid()
    #     axs_ilqr[j].set_ylabel('State '+str(j))
    
    # axs_ilqr[j].set_xlabel('TimeStep')

    # line_ilqr_list = []
    # for j in range( state_size ):
    #     line_ilqr, = axs_ilqr[j].plot(xs_hist_ilqr[0,:,j],lw=2)
    #     line_ilqr_list.append(line_ilqr)

    # def func_animation_ilqr(i):
    #     for j in range( state_size ):
    #         line_ilqr_list[j].set_ydata(xs_hist_ilqr[i,:,j])
    #         return line_ilqr_list

    # animation_ilqr = FuncAnimation(fig_ilqr,
    #                     func = func_animation_ilqr,
    #                     frames = np.arange(0, xs_hist_ilqr.shape[0]), 
    #                     interval = 500)
    


    # fig_ddp, axs_ddp = plt.subplots(state_size, num=7)
    # fig.suptitle('DDP Trajectory Evolution')
    # for j in range( state_size ):
    #     axs_ddp[j].set_ylim( np.min(xs_hist_ddp[:,:,j])-0.2, np.max(xs_hist_ddp[:,:,j])+0.2 ) 

    #     axs_ddp[j].grid()
    #     axs_ddp[j].set_ylabel('State '+str(j))

    # axs_ddp[j].set_xlabel('TimeStep')

    # line_ddp_list = []
    # for j in range( state_size ):
    #     line_ddp, = axs_ddp[j].plot(xs_hist_ddp[0,:,j],lw=2)
    #     line_ddp_list.append(line_ddp)

    # def func_animation_ddp(i):
    #     for j in range( state_size ):
    #         line_ddp_list[j].set_ydata(xs_hist_ddp[i,:,j])
    #         return line_ilqr_list

    # animation_ddp = FuncAnimation(fig_ddp,
    #                     func = func_animation_ddp,
    #                     frames = np.arange(0, xs_hist_ddp.shape[0]), 
    #                     interval = 500)
     

    plt.show()


    
    






    

