import pickle
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from traoptlibrary.traopt_utilis import rotm2euler, SE32manifSE3, se32manifse3, manifSE32SE3
from traoptlibrary.traopt_dynamics import SE3Dynamics
from traoptlibrary.traopt_cost import SE3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_controller import iLQR_Tracking_SE3_MS, iLQR_Tracking_SE3
from traoptlibrary.traopt_baseline import EmbeddedEuclideanSE3, ConstraintStabilizationSE3, \
                            EmbeddedEuclideanSE3_MatrixNorm, ConstraintStabilizationSE3_MatrixNorm
from scipy.spatial.transform import Rotation
from manifpy import SE3, SE3Tangent

SAVE_RESULTS = False
SAVE_RESULTS_DIR = 'visualization/results_benchmark/results_se3_tracking_benchmark.pkl'

# =====================================================
# Iteration Function
# =====================================================

def on_iteration_ms_se3(iteration_count, xs, us, J_opt, accepted, 
                converged, defect_norm, grad_wrt_input_norm,
                alpha, mu, J_hist, xs_hist, us_hist, grad_hist, defect_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    grad_hist.append(grad_wrt_input_norm.copy())
    defect_hist.append( defect_norm )

    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("Iteration", iteration_count, \
          info, J_opt, defect_norm, \
          grad_wrt_input_norm, alpha, mu)
    
def on_iteration_ss_se3(iteration_count, xs, us, J_opt, accepted, converged,grad_wrt_input_norm,
                  alpha, mu, J_hist, xs_hist, us_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("Iteration", iteration_count, info, J_opt, grad_wrt_input_norm, alpha, mu)


# =====================================================
# Problem Import: Reference, Initial State, Solver Options
# =====================================================

dt = 0.004

path_to_reference_file = \
    'visualization/optimized_trajectories/path_dense_random_columns_4obj.npy'

with open( path_to_reference_file, 'rb' ) as f:
    q_ref = np.load(f)
    xi_ref = np.load(f)

Nsim = q_ref.shape[0] - 1
# Nsim = 300
# q_ref = q_ref[:Nsim+1]
# xi_ref = xi_ref[:Nsim+1]



q0 = SE3(
    position = -1 * np.ones((3,)) + q_ref[0][:3,3],
    quaternion = Rotation.from_euler('zxy', [90.,10.,45.], degrees=True).as_quat() 
).transform()
xi0 = np.ones((6,)) * 1e-1
x0 = [ q0, xi0 ]

m = 1
Ib = np.diag([ 0.5,0.7,0.9 ])
J = np.block([
    [Ib, np.zeros((3, 3))],
    [np.zeros((3, 3)), m * np.identity(3)]
])

# Nsim = 200
# q_ref = q_ref[:Nsim+1]
# xi_ref = xi_ref[:Nsim+1]

max_iterations = 200

tol_gradiant_converge = 1e-12
tol_converge = tol_gradiant_converge

print("Horizon of dataset is", Nsim)
# =====================================================
# Helper Function
# =====================================================

def err_dyn(xk, xk1, dt=dt):
    Xk, wk = xk
    Xk1, _ = xk1
    
    Xk_mnf = SE32manifSE3(Xk)
    wk_mnf = se32manifse3(wk)  
    Xk_sim = Xk_mnf.rplus( wk_mnf * dt ).transform()

    return np.linalg.norm(Xk_sim - Xk1)

# =====================================================
# Algorithms on Matrix Lie Groups: Dynamics, Cost, Controller
# =====================================================

N = Nsim
HESSIANS = False
action_size = 6
state_size = 12

dynamics = SE3Dynamics( J, dt, hessians=HESSIANS )

Q = np.diag([ 
    25., 25., 25., 10., 10., 10.,
    1., 1., 1., 1., 1., 1. 
]) 
P = Q * 10
R = np.identity(6) * 1e-4
cost = SE3TrackingQuadraticGaussNewtonCost( Q, R, P, q_ref, xi_ref )

us_init = np.zeros((N, action_size,))

ilqr_ms_se3 = iLQR_Tracking_SE3_MS( dynamics, cost, N, 
                                    q_ref, xi_ref,
                                    hessians=HESSIANS,
                                    line_search=False,   
                                    rollout='nonlinear' )

ilqr_ss_se3 = iLQR_Tracking_SE3(    dynamics, cost, N, 
                                    hessians=HESSIANS,
                                    rollout='nonlinear' )

# xs_ms_se3, us_ms_se3, J_hist_ms_se3, _, _, \
#     grad_hist_ms_se3, defect_hist_ms_se3= \
#         ilqr_ms_se3.fit(x0, us_init, 
#                         n_iterations=max_iterations, 
#                         tol_grad_norm=tol_gradiant_converge,
#                         on_iteration=on_iteration_ms_se3)

# xs_ss_se3, us_ss_se3, J_hist_ss_se3, _, _, grad_hist_ss_se3 = \
#         ilqr_ss_se3.fit(x0, us_init, 
#                         n_iterations=max_iterations, 
#                         tol_grad_norm=tol_gradiant_converge,
#                         on_iteration=on_iteration_ss_se3)

# =====================================================
# Embedded Euclidean Unconstrained Method
# =====================================================
eps_init = 1e-3
# kappa = 1e-2
# when eps_init kept as 1e-3, neither kappa=1e0 or 1e-5 works for mumps solver

# # intialize the embedded method
ipopt_logcost_euc = EmbeddedEuclideanSE3(   q_ref, xi_ref, dt, J, Q, R, 
                                            eps_init=eps_init )

# ipopt_constr_euc = ConstraintStabilizationSE3( q_ref, xi_ref, dt, J, Q, R, 
#                                               eps_init=eps_init, 
#                                               kappa=kappa )
# ipopt_constr_euc = ConstraintStabilizationSE3_MatrixNorm( q_ref, xi_ref, dt, J, Q, R, 
#                                               eps_init=eps_init, 
#                                               kappa=kappa )

# # get the solution
# xs_logcost_euc, us_logcost_euc, J_hist_logcost_euc, \
#     grad_hist_logcost_euc, defect_hist_logcost_euc = \
#         ipopt_logcost_euc.fit(   x0, us_init, 
#                                 n_iterations=max_iterations,
#                                 tol_norm=tol_converge )

# =====================================================
# Embedded Euclidean Unconstrained Method with Manifold Cost
# =====================================================
# intialize the embedded method
ipopt_unconstr_euc = EmbeddedEuclideanSE3_MatrixNorm(   q_ref, xi_ref, dt, J, Q, R, 
                                                        eps_init=eps_init )

# # get the solution
# xs_unconstr_euc, us_unconstr_euc, J_hist_unconstr_euc, \
#     grad_hist_unconstr_euc, defect_hist_unconstr_euc = \
#         ipopt_unconstr_euc.fit( x0, us_init, 
#                                 n_iterations=max_iterations,
#                                 tol_norm=tol_converge )


# =====================================================
# Save Results
# =====================================================

def save_results_pickle(filename,
                       xs_ms_se3, us_ms_se3, J_hist_ms_se3, grad_hist_ms_se3, defect_hist_ms_se3,
                       xs_ss_se3, us_ss_se3, J_hist_ss_se3, grad_hist_ss_se3,
                       xs_logcost_euc, us_logcost_euc, J_hist_logcost_euc, grad_hist_logcost_euc, defect_hist_logcost_euc,
                       xs_unconstr_euc, us_unconstr_euc, J_hist_unconstr_euc, grad_hist_unconstr_euc, defect_hist_unconstr_euc):
    data = {
        'prob':{
            'J': J,
            'dt': dt,
            'q_ref': q_ref,
            'xi_ref': xi_ref,
            'x0': x0,
            'Q' : Q,
            'P' : P,
            'R' : R
        },
        'ms_se3': {
            'xs': xs_ms_se3,
            'us': us_ms_se3,
            'J_hist': J_hist_ms_se3,
            'grad_hist': grad_hist_ms_se3,
            'defect_hist': defect_hist_ms_se3
        },
        'ss_se3': {
            'xs': xs_ss_se3,
            'us': us_ss_se3,
            'J_hist': J_hist_ss_se3,
            'grad_hist': grad_hist_ss_se3
        },
        'logcost_euc': {
            'xs': xs_logcost_euc,
            'us': us_logcost_euc,
            'J_hist': J_hist_logcost_euc,
            'grad_hist': grad_hist_logcost_euc,
            'defect_hist': defect_hist_logcost_euc
        },
        'unconstr_euc': {
            'xs': xs_unconstr_euc,
            'us': us_unconstr_euc,
            'J_hist': J_hist_unconstr_euc,
            'grad_hist': grad_hist_unconstr_euc,
            'defect_hist': defect_hist_unconstr_euc
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Results saved to {filename}")

# if SAVE_RESULTS:
#     save_results_pickle(SAVE_RESULTS_DIR,
#                        xs_ms_se3, us_ms_se3, J_hist_ms_se3, grad_hist_ms_se3, defect_hist_ms_se3,
#                        xs_ss_se3, us_ss_se3, J_hist_ss_se3, grad_hist_ss_se3,
#                        xs_logcost_euc, us_logcost_euc, J_hist_logcost_euc, grad_hist_logcost_euc, defect_hist_logcost_euc,
#                        xs_unconstr_euc, us_unconstr_euc, J_hist_unconstr_euc, grad_hist_unconstr_euc, defect_hist_unconstr_euc)


# # =====================================================
# # Load Results
# # =====================================================

def load_results_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

results = load_results_pickle(SAVE_RESULTS_DIR)

ms_se3_data = results['ms_se3']
xs_ms_se3 = ms_se3_data['xs']            # 状态序列 (列表，包含 manifpy 的 SO3 对象和 SO3Tangent 对象)
us_ms_se3 = ms_se3_data['us']            # 控制序列 (numpy 数组)
J_hist_ms_se3 = ms_se3_data['J_hist']    # 目标函数历史 (列表)
grad_hist_ms_se3 = ms_se3_data['grad_hist']  # 梯度范数历史 (列表)
defect_hist_ms_se3 = ms_se3_data['defect_hist']  # 缺陷范数历史 (列表)

ss_se3_data = results['ss_se3']
xs_ss_se3 = ss_se3_data['xs']            # 状态序列 (列表，包含 manifpy 的 SO3 对象和 SO3Tangent 对象)
us_ss_se3 = ss_se3_data['us']            # 控制序列 (numpy 数组)
J_hist_ss_se3 = ss_se3_data['J_hist']    # 目标函数历史 (列表)
grad_hist_ss_se3 = ss_se3_data['grad_hist']  # 梯度范数历史 (列表)

logcost_euc_data = results['logcost_euc']
xs_logcost_euc = logcost_euc_data['xs']                  # 状态序列 (numpy 数组)
us_logcost_euc = logcost_euc_data['us']                  # 控制序列 (numpy 数组)
J_hist_logcost_euc = logcost_euc_data['J_hist']          # 目标函数历史 (numpy 数组)
grad_hist_logcost_euc = logcost_euc_data['grad_hist']    # 梯度范数历史 (numpy 数组)
defect_hist_logcost_euc = logcost_euc_data['defect_hist']# 缺陷范数历史 (numpy 数组)

unconstr_euc_data = results['unconstr_euc']
xs_unconstr_euc = unconstr_euc_data['xs']                  # 状态序列 (numpy 数组)
us_unconstr_euc = unconstr_euc_data['us']                  # 控制序列 (numpy 数组)
J_hist_unconstr_euc = unconstr_euc_data['J_hist']          # 目标函数历史 (numpy 数组)
grad_hist_unconstr_euc = unconstr_euc_data['grad_hist']    # 梯度范数历史 (numpy 数组)
defect_hist_unconstr_euc = unconstr_euc_data['defect_hist']# 缺陷范数历史 (numpy 数组)


# # =====================================================
# # Data Type Conversion
# # =====================================================

J_hist_ms_se3 = np.array(J_hist_ms_se3)
grad_hist_ms_se3 = np.array(grad_hist_ms_se3)
defect_hist_ms_se3 = np.array(defect_hist_ms_se3)

J_hist_ss_se3 = np.array(J_hist_ss_se3)
grad_hist_ss_se3 = np.array(grad_hist_ss_se3)

J_hist_logcost_euc = np.array(J_hist_logcost_euc)
grad_hist_logcost_euc = np.array(grad_hist_logcost_euc)
defect_hist_logcost_euc = np.array(defect_hist_logcost_euc)

J_hist_unconstr_euc = np.array(J_hist_unconstr_euc)
grad_hist_unconstr_euc = np.array(grad_hist_unconstr_euc)
defect_hist_unconstr_euc = np.array(defect_hist_unconstr_euc)


# # =====================================================
# # Plotting
# # =====================================================

# 1. Lie constraint violation comparison

violation_orth_ms_se3 = [ np.linalg.norm(x[0][:3,:3].T @ x[0][:3,:3] - np.identity(3)) for x in xs_ms_se3 ]
violation_orth_ss_se3 = [ np.linalg.norm(x[0][:3,:3].T @ x[0][:3,:3] - np.identity(3)) for x in xs_ss_se3 ]
violation_orth_logcost_euc = [ np.linalg.norm(x[0][:3,:3].T @ x[0][:3,:3] - np.identity(3)) for x in xs_logcost_euc ]
violation_orth_unconstr_euc = [ np.linalg.norm(x[0][:3,:3].T @ x[0][:3,:3] - np.identity(3)) for x in xs_unconstr_euc ]

violation_det_ms_se3 = [ 1 - np.linalg.det(x[0][:3,:3]) for x in xs_ms_se3 ]
violation_det_ss_se3 = [ 1 - np.linalg.det(x[0][:3,:3]) for x in xs_ss_se3 ]
violation_det_logcost_euc = [ 1 - np.linalg.det(x[0][:3,:3]) for x in xs_logcost_euc ]
violation_det_unconstr_euc = [ 1 - np.linalg.det(x[0][:3,:3]) for x in xs_unconstr_euc ]

plt.figure()
ax = plt.subplot(131)
plt.plot( violation_orth_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( violation_orth_logcost_euc, label=r'Embedded w. $\mathcal{M}$ Cost' )
plt.plot( violation_orth_ms_se3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( violation_orth_ss_se3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel(r'$||R^T R - I_3||$')
plt.xlabel('Stage')
plt.legend()
plt.grid()

ax = plt.subplot(132)
plt.plot( violation_det_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( violation_det_logcost_euc, label=r'Embedded w. $\mathcal{M}$ Cost' )
plt.plot( violation_det_ms_se3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( violation_det_ss_se3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel(r'$|\text{det}(R)-1|$')
plt.xlabel('Stage')
plt.grid()

# 2. Dynamics violation comparison:
#       By forward simulating the SO3 dynamics and then 
#       compare the simulated trajectory with the solved state

dyn_error_ms_se3 = [ err_dyn( xs_ms_se3[k], xs_ms_se3[k+1] )  for k in range(Nsim) ]
dyn_error_ss_se3 = [ err_dyn( xs_ss_se3[k], xs_ss_se3[k+1] )  for k in range(Nsim) ]
dyn_error_unconstr_euc = [ err_dyn( xs_unconstr_euc[k], xs_unconstr_euc[k+1] )  for k in range(Nsim) ]
dyn_error_logcost_euc = [ err_dyn( xs_logcost_euc[k], xs_logcost_euc[k+1] )  for k in range(Nsim) ]

ax = plt.subplot(133)
plt.plot( dyn_error_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( dyn_error_logcost_euc, label=r'Embedded w. $\mathcal{M}$ Cost' )
plt.plot( dyn_error_ms_se3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( dyn_error_ss_se3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel(r'$||\mathcal{X}_{k+1} - F_\mathcal{X}( \mathcal{X}_k, \xi_k )||$')
plt.xlabel('Stage')
# plt.legend()
plt.grid()

# # 3. cost comparison

plt.figure()
plt.plot( J_hist_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( J_hist_logcost_euc, label=r'Embedded w. $\mathcal{M}$ Cost' )
plt.plot( J_hist_ms_se3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( J_hist_ss_se3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel(r'$J(\mathbf{x},\mathbf{u})$')
plt.xlabel('Iteration')
plt.legend()
plt.grid()

plt.figure()
plt.plot( np.abs(J_hist_unconstr_euc[1:]-J_hist_unconstr_euc[:-1]), label='Embedded Unconstrained' )
plt.plot( np.abs(J_hist_logcost_euc[1:]-J_hist_logcost_euc[:-1]), label=r'Embedded w. $\mathcal{M}$ Cost'  )
plt.plot( np.abs(J_hist_ms_se3[1:]-J_hist_ms_se3[:-1]), label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( np.abs(J_hist_ss_se3[1:]-J_hist_ss_se3[:-1]), label=r'SS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel(r'$|\Delta J(\mathbf{x},\mathbf{u})$|')
plt.xlabel('Iteration')
# plt.legend()
plt.grid()

plt.figure()
plt.plot( grad_hist_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( grad_hist_logcost_euc, label=r'Embedded w. $\mathcal{M}$ Cost' )
plt.plot( grad_hist_ms_se3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( grad_hist_ss_se3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel('Gradient')
plt.xlabel('Iteration')
# plt.legend()
plt.grid()

plt.figure()
plt.plot( defect_hist_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( defect_hist_logcost_euc, label=r'Embedded w. $\mathcal{M}$ Cost' )
plt.plot( defect_hist_ms_se3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel(r'$||d||$')
plt.xlabel('Iteration')
# plt.legend()
plt.grid()

# 4. Big Plotting: State, Input

euler_ms_se3 = np.array([ rotm2euler(x[0][:3,:3]) for x in xs_ms_se3 ] )
euler_ss_se3 = np.array([ rotm2euler(x[0][:3,:3]) for x in xs_ss_se3 ] )
euler_unconstr_euc = np.array([ rotm2euler(x[0][:3,:3]) for x in xs_unconstr_euc ] )
euler_logcost_euc = np.array([ rotm2euler(x[0][:3,:3]) for x in xs_logcost_euc ] )

omega_ms_se3 = np.array([ x[1][:3] for x in xs_ms_se3 ])
omega_ss_se3 = np.array([ x[1][:3] for x in xs_ss_se3 ])
omega_unconstr_euc = np.array([ x[1][:3] for x in xs_unconstr_euc ])
omega_logcost_euc = np.array([ x[1][:3] for x in xs_logcost_euc ])

pos_ms_se3 = np.array([ x[0][:3,3] for x in xs_ms_se3 ] )
pos_ss_se3 = np.array([ x[0][:3,3] for x in xs_ss_se3 ] )
pos_unconstr_euc = np.array([ x[0][:3,3] for x in xs_unconstr_euc ] )
pos_logcost_euc = np.array([ x[0][:3,3] for x in xs_logcost_euc ] )

vel_ms_se3 = np.array([ x[1][3:] for x in xs_ms_se3 ])
vel_ss_se3 = np.array([ x[1][3:] for x in xs_ss_se3 ])
vel_unconstr_euc = np.array([ x[1][3:] for x in xs_unconstr_euc ])
vel_logcost_euc = np.array([ x[1][3:] for x in xs_logcost_euc ])


plt.figure()

plt.subplot(641)
for i in range(3):
    plt.plot( euler_ms_se3[:,i] )
plt.title(r'MS-iLQR on $\mathcal{M}$')
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('degree')
plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
plt.grid()

plt.subplot(642)
for i in range(3):
    plt.plot( euler_ss_se3[:,i] )
plt.title(r'SS-iLQR on $\mathcal{M}$')
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(643)
for i in range(3):
    plt.plot( euler_unconstr_euc[:,i] )
plt.title('Embedded Unconstrained')
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(644)
for i in range(3):
    plt.plot( euler_logcost_euc[:,i] )
plt.title(r'Embedded w. $\mathcal{M}$ Cost')
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(645)
for i in range(3):
    plt.plot( omega_ms_se3[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('rad/s')
plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
plt.grid()

plt.subplot(646)
for i in range(3):
    plt.plot( omega_ss_se3[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(647)
for i in range(3):
    plt.plot( omega_unconstr_euc[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(648)
for i in range(3):
    plt.plot( omega_logcost_euc[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(649)
for i in range(3):
    plt.plot( pos_ms_se3[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('m')
plt.legend([r'$p_x$',r'$p_y$',r'$p_z$'])
plt.grid()

plt.subplot(6,4,10)
for i in range(3):
    plt.plot( pos_ss_se3[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('m')
plt.grid()

plt.subplot(6,4,11)
for i in range(3):
    plt.plot( pos_unconstr_euc[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('m')
plt.grid()

plt.subplot(6,4,12)
for i in range(3):
    plt.plot( pos_logcost_euc[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('m')
plt.grid()

plt.subplot(6,4,13)
for i in range(3):
    plt.plot( vel_ms_se3[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('m/s')
plt.legend([r'$v_x$',r'$v_y$',r'$v_z$'])
plt.grid()

plt.subplot(6,4,14)
for i in range(3):
    plt.plot( vel_ss_se3[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(6,4,15)
for i in range(3):
    plt.plot( vel_unconstr_euc[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(6,4,16)
for i in range(3):
    plt.plot( vel_logcost_euc[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(6,4,17)
for i in range(3):
    plt.plot( us_ms_se3[:,i])
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('angular input')
plt.legend([r'$u_{\theta_x}$',r'$u_{\theta_y}$',r'$u_{\theta_z}$'])
plt.grid()

plt.subplot(6,4,18)
for i in range(3):
    plt.plot( us_ss_se3[:,i])
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(6,4,19)
for i in range(3):
    plt.plot( us_unconstr_euc[:,i])
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(6,4,20)
for i in range(3):
    plt.plot( us_logcost_euc[:,i])
plt.tick_params(axis='x', labelbottom=False)
plt.grid()

plt.subplot(6,4,21)
for i in range(3):
    plt.plot( us_ms_se3[:,3+i])
plt.ylabel('translational input')
plt.xlabel('Stage')
plt.legend([r'$u_{p_x}$',r'$u_{p_y}$',r'$u_{p_z}$'])
plt.grid()

plt.subplot(6,4,22)
for i in range(3):
    plt.plot( us_ss_se3[:,3+i])
plt.xlabel('Stage')
plt.grid()

plt.subplot(6,4,23)
for i in range(3):
    plt.plot( us_unconstr_euc[:,3+i])
plt.xlabel('Stage')
plt.grid()

plt.subplot(6,4,24)
for i in range(3):
    plt.plot( us_logcost_euc[:,3+i])
plt.xlabel('Stage')
plt.grid()


# 5. 3D plotting, plotting all 4 solutions in one figure for comparison

pos_ms_se3 = np.array([x[0][:3,3] for x in xs_ms_se3]).reshape(N+1, 3)
pos_ss_se3 = np.array([x[0][:3,3] for x in xs_ss_se3]).reshape(N+1, 3)
pos_unconstr_euc = np.array([x[0][:3,3] for x in xs_unconstr_euc]).reshape(N+1, 3)
pos_logcost_euc = np.array([x[0][:3,3] for x in xs_logcost_euc]).reshape(N+1, 3)

pos_ref = np.array([x[:3,3] for x in q_ref]).reshape(N+1, 3)

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot(pos_rod_ms_se3[:, 0], pos_rod_ms_se3[:, 1], pos_rod_ms_se3[:, 2],
# #             label=r'MS-iLQR on $\mathcal{M}$')
# # ax.plot(pos_rod_ss_se3[:, 0], pos_rod_ss_se3[:, 1], pos_rod_ss_se3[:, 2],
# #             label=r'SS-iLQR on $\mathcal{M}$')
# # ax.plot(pos_rod_unconstr_euc[:, 0], pos_rod_unconstr_euc[:, 1], pos_rod_unconstr_euc[:, 2],
# #             label='Embedded Unconstrained')
# # ax.plot(pos_rod_constr_euc[:, 0], pos_rod_constr_euc[:, 1], pos_rod_constr_euc[:, 2],
# #             label='Embedded Stabilization')
# # ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2],
# #             label='Reference')
# # plt.legend()

fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.set_title(r'MS-iLQR on $\mathcal{M}$')
ax.plot(pos_ms_se3[:, 0], pos_ms_se3[:, 1], pos_ms_se3[:, 2])
ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend([r'$\mathcal{X}$',r'$\mathcal{X}_\text{ref}$'])
ax.set_zlim(-0.5, 2.5)

ax = fig.add_subplot(222, projection='3d')
ax.set_title(r'SS-iLQR on $\mathcal{M}$')
ax.plot(pos_ss_se3[:, 0], pos_ss_se3[:, 1], pos_ss_se3[:, 2])
ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-0.5, 2.5)

ax = fig.add_subplot(223, projection='3d')
ax.set_title('Embedded Unconstrained')
ax.plot(pos_unconstr_euc[:, 0], pos_unconstr_euc[:, 1], pos_unconstr_euc[:, 2])
ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-0.5, 2.5)

ax = fig.add_subplot(224, projection='3d')
ax.set_title(r'Embedded w. $\mathcal{M}$ Cost')
ax.plot(pos_logcost_euc[:, 0], pos_logcost_euc[:, 1], pos_logcost_euc[:, 2])
ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-0.5, 2.5)

# # =====================================================
# # Special Problem
# # =====================================================

# # For Single Shooting, initial rollout and geodesic angle relation


plt.show()
