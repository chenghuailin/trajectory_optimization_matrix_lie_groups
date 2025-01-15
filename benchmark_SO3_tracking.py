import pickle
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from traoptlibrary.traopt_utilis import rotm2euler
from traoptlibrary.traopt_dynamics import SO3Dynamics
from traoptlibrary.traopt_cost import SO3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_controller import iLQR_Tracking_SO3_MS, iLQR_Tracking_SO3
from traoptlibrary.traopt_baseline import EmbeddedEuclideanSO3, ConstraintStabilizationSO3
from scipy.spatial.transform import Rotation
from manifpy import SO3, SO3Tangent

SAVE_RESULTS = True
SAVE_RESULTS_DIR = 'visualization/results_benchmark/results_so3_tracking_benchmark.pkl'

# =====================================================
# Iteration Function
# =====================================================

def on_iteration_ms_so3(iteration_count, xs, us, J_opt, accepted, 
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
    
def on_iteration_ss_so3(iteration_count, xs, us, J_opt, accepted, converged,grad_wrt_input_norm,
                  alpha, mu, J_hist, xs_hist, us_hist):
    J_hist.append(J_opt)
    xs_hist.append(xs.copy())
    us_hist.append(us.copy())
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("Iteration", iteration_count, info, J_opt, grad_wrt_input_norm, alpha, mu)


# =====================================================
# Problem Import: Reference, Initial State, Solver Options
# =====================================================

path_to_reference_file = \
    'visualization/optimized_trajectories/path_3dpendulum_8shape_tryout.npy'
    # 'visualization/optimized_trajectories/path_3dpendulum_8shape.npy'
    
with open( path_to_reference_file, 'rb' ) as f:
    q_ref = np.load(f)
    xi_ref = np.load(f)
    dt = np.load(f)

Nsim = q_ref.shape[0] - 1
print("Horizon of dataset is", Nsim)

# q0 = SO3( Rotation.from_matrix(q_ref[0]).as_quat() ) 
# xi0 = SO3Tangent( xi_ref[0] )

q0 = SO3( 
    Rotation.from_euler('zxy', [90.,10.,45.], degrees=True).as_quat() 
) 
omega0 = 1e-1
xi0 = SO3Tangent( np.ones((3,1)) * omega0 )
x0_mnf = [ q0, xi0 ]
x0_np = [ q0.rotation(), xi0.coeffs() ]

J = np.diag([ 0.5,0.7,0.9 ])

max_iterations = 100

tol_gradiant_converge = 1e-12
tol_converge = tol_gradiant_converge

# =====================================================
# Helper Function
# =====================================================

def err_dyn(xk, xk1, dt=dt):
    Xk, wk = xk
    Xk1, _ = xk1
    
    Xk_mnf = SO3( Rotation.from_matrix(Xk).as_quat() )
    wk_mnf = SO3Tangent( wk )
    Xk_sim = Xk_mnf.rplus( wk_mnf * dt ).rotation()

    return np.linalg.norm(Xk_sim - Xk1)

# =====================================================
# Algorithms on Matrix Lie Groups: Dynamics, Cost, Controller
# =====================================================

N = Nsim
HESSIANS = False
action_size = 3
state_size = 6

dynamics = SO3Dynamics( J, dt, hessians=HESSIANS )

Q = np.diag([10., 10., 10., 1., 1., 1.,])
P = Q * 10
R = np.identity(3) * 1e-5
cost = SO3TrackingQuadraticGaussNewtonCost( Q, R, P, q_ref, xi_ref )

us_init = np.zeros((N, action_size,))

ilqr_ms_so3 = iLQR_Tracking_SO3_MS( dynamics, cost, N, 
                                    q_ref, xi_ref,
                                    hessians=HESSIANS,
                                    line_search=True,
                                    rollout='nonlinear' )

ilqr_ss_so3 = iLQR_Tracking_SO3(    dynamics, cost, N, 
                                    hessians=HESSIANS,
                                    rollout='nonlinear' )

# xs_ms_so3, us_ms_so3, J_hist_ms_so3, _, _, \
#     grad_hist_ms_so3, defect_hist_ms_so3= \
#         ilqr_ms_so3.fit(x0_mnf, us_init, 
#                         n_iterations=max_iterations, 
#                         tol_grad_norm=tol_gradiant_converge,
#                         on_iteration=on_iteration_ms_so3)
# xs_ms_so3 = [ [x[0].rotation(), x[1].coeffs() ] for x in xs_ms_so3 ]

# xs_ss_so3, us_ss_so3, J_hist_ss_so3, _, _, grad_hist_ss_so3 = \
#         ilqr_ss_so3.fit(x0_mnf, us_init, 
#                         n_iterations=max_iterations, 
#                         tol_grad_norm=tol_gradiant_converge,
#                         on_iteration=on_iteration_ss_so3)
# xs_ss_so3 = [ [x[0].rotation(), x[1].coeffs() ] for x in xs_ss_so3 ]

# =====================================================
# Embedded Space Method
# =====================================================

# intialize the embedded method
ipopt_unconstr_euc = EmbeddedEuclideanSO3( q_ref, xi_ref, dt, J, Q, R )

# # get the solution
# xs_unconstr_euc, us_unconstr_euc, J_hist_unconstr_euc, \
#     grad_hist_unconstr_euc, defect_hist_unconstr_euc = \
#         ipopt_unconstr_euc.fit( x0_np, us_init, 
#                                 n_iterations=max_iterations,
#                                 tol_norm=tol_converge )

# =====================================================
# Constraint Stabilization Method
# =====================================================

# intialize the embedded method
ipopt_constr_euc = ConstraintStabilizationSO3( q_ref, xi_ref, dt, J, Q, R )

# # get the solution
# xs_constr_euc, us_constr_euc, J_hist_constr_euc, \
#     grad_hist_constr_euc, defect_hist_constr_euc = \
#         ipopt_constr_euc.fit(   x0_np, us_init, 
#                                 n_iterations=max_iterations,
#                                 tol_norm=tol_converge )

# =====================================================
# Save Results
# =====================================================

# def save_results_pickle(filename,
#                        xs_ms_so3, us_ms_so3, J_hist_ms_so3, grad_hist_ms_so3, defect_hist_ms_so3,
#                        xs_ss_so3, us_ss_so3, J_hist_ss_so3, grad_hist_ss_so3,
#                        xs_unconstr_euc, us_unconstr_euc, J_hist_unconstr_euc, grad_hist_unconstr_euc, defect_hist_unconstr_euc,
#                        xs_constr_euc, us_constr_euc, J_hist_constr_euc, grad_hist_constr_euc, defect_hist_constr_euc):
#     data = {
#         'ms_so3': {
#             'xs': xs_ms_so3,
#             'us': us_ms_so3,
#             'J_hist': J_hist_ms_so3,
#             'grad_hist': grad_hist_ms_so3,
#             'defect_hist': defect_hist_ms_so3
#         },
#         'ss_so3': {
#             'xs': xs_ss_so3,
#             'us': us_ss_so3,
#             'J_hist': J_hist_ss_so3,
#             'grad_hist': grad_hist_ss_so3
#         },
#         'unconstr_euc': {
#             'xs': xs_unconstr_euc,
#             'us': us_unconstr_euc,
#             'J_hist': J_hist_unconstr_euc,
#             'grad_hist': grad_hist_unconstr_euc,
#             'defect_hist': defect_hist_unconstr_euc
#         },
#         'constr_euc': {
#             'xs': xs_constr_euc,
#             'us': us_constr_euc,
#             'J_hist': J_hist_constr_euc,
#             'grad_hist': grad_hist_constr_euc,
#             'defect_hist': defect_hist_constr_euc
#         }
#     }
    
#     with open(filename, 'wb') as f:
#         pickle.dump(data, f)
#     print(f"Results saved to {filename}")

# if SAVE_RESULTS:
#     save_results_pickle(SAVE_RESULTS_DIR,
#                        xs_ms_so3, us_ms_so3, J_hist_ms_so3, grad_hist_ms_so3, defect_hist_ms_so3,
#                        xs_ss_so3, us_ss_so3, J_hist_ss_so3, grad_hist_ss_so3,
#                        xs_unconstr_euc, us_unconstr_euc, J_hist_unconstr_euc, grad_hist_unconstr_euc, defect_hist_unconstr_euc,
#                        xs_constr_euc, us_constr_euc, J_hist_constr_euc, grad_hist_constr_euc, defect_hist_constr_euc)


# =====================================================
# Load Results
# =====================================================

def load_results_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

results = load_results_pickle(SAVE_RESULTS_DIR)

ms_so3_data = results['ms_so3']
xs_ms_so3 = ms_so3_data['xs']            # 状态序列 (列表，包含 manifpy 的 SO3 对象和 SO3Tangent 对象)
us_ms_so3 = ms_so3_data['us']            # 控制序列 (numpy 数组)
J_hist_ms_so3 = ms_so3_data['J_hist']    # 目标函数历史 (列表)
grad_hist_ms_so3 = ms_so3_data['grad_hist']  # 梯度范数历史 (列表)
defect_hist_ms_so3 = ms_so3_data['defect_hist']  # 缺陷范数历史 (列表)

ss_so3_data = results['ss_so3']
xs_ss_so3 = ss_so3_data['xs']            # 状态序列 (列表，包含 manifpy 的 SO3 对象和 SO3Tangent 对象)
us_ss_so3 = ss_so3_data['us']            # 控制序列 (numpy 数组)
J_hist_ss_so3 = ss_so3_data['J_hist']    # 目标函数历史 (列表)
grad_hist_ss_so3 = ss_so3_data['grad_hist']  # 梯度范数历史 (列表)

unconstr_euc_data = results['unconstr_euc']
xs_unconstr_euc = unconstr_euc_data['xs']                  # 状态序列 (numpy 数组)
us_unconstr_euc = unconstr_euc_data['us']                  # 控制序列 (numpy 数组)
J_hist_unconstr_euc = unconstr_euc_data['J_hist']          # 目标函数历史 (numpy 数组)
grad_hist_unconstr_euc = unconstr_euc_data['grad_hist']    # 梯度范数历史 (numpy 数组)
defect_hist_unconstr_euc = unconstr_euc_data['defect_hist']# 缺陷范数历史 (numpy 数组)

constr_euc_data = results['constr_euc']
xs_constr_euc = constr_euc_data['xs']                      # 状态序列 (numpy 数组)
us_constr_euc = constr_euc_data['us']                      # 控制序列 (numpy 数组)
J_hist_constr_euc = constr_euc_data['J_hist']              # 目标函数历史 (numpy 数组)
grad_hist_constr_euc = constr_euc_data['grad_hist']        # 梯度范数历史 (numpy 数组)
defect_hist_constr_euc = constr_euc_data['defect_hist']    # 缺陷范数历史 (numpy 数组)


# =====================================================
# Data Type Conversion
# =====================================================

J_hist_ms_so3 = np.array(J_hist_ms_so3)
grad_hist_ms_so3 = np.array(grad_hist_ms_so3)
defect_hist_ms_so3 = np.array(defect_hist_ms_so3)

J_hist_ss_so3 = np.array(J_hist_ss_so3)
grad_hist_ss_so3 = np.array(grad_hist_ss_so3)

J_hist_unconstr_euc = np.array(J_hist_unconstr_euc)
grad_hist_unconstr_euc = np.array(grad_hist_unconstr_euc)
defect_hist_unconstr_euc = np.array(defect_hist_unconstr_euc)

J_hist_constr_euc = np.array(J_hist_constr_euc)
grad_hist_constr_euc = np.array(grad_hist_constr_euc)
defect_hist_constr_euc = np.array(defect_hist_constr_euc)

# =====================================================
# Plotting
# =====================================================

# 1. Lie constraint violation comparison

violation_orth_ms_so3 = [ np.linalg.norm(x[0].T @ x[0] - np.identity(3)) for x in xs_ms_so3 ]
violation_orth_ss_so3 = [ np.linalg.norm(x[0].T @ x[0] - np.identity(3)) for x in xs_ss_so3 ]
violation_orth_unconstr_euc = [ np.linalg.norm(x[0].T @ x[0] - np.identity(3)) for x in xs_unconstr_euc ]
violation_orth_constr_euc = [ np.linalg.norm(x[0].T @ x[0] - np.identity(3)) for x in xs_constr_euc ]

violation_det_ms_so3 = [ 1 - np.linalg.det(x[0]) for x in xs_ms_so3 ]
violation_det_ss_so3 = [ 1 - np.linalg.det(x[0]) for x in xs_ss_so3 ]
violation_det_unconstr_euc = [ 1 - np.linalg.det(x[0]) for x in xs_unconstr_euc ]
violation_det_constr_euc  = [ 1 - np.linalg.det(x[0]) for x in xs_constr_euc ]

plt.figure()
ax = plt.subplot(121)
plt.plot( violation_orth_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( violation_orth_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.plot( violation_orth_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( violation_orth_constr_euc, label='Embedded Stabilization' )
plt.yscale('log')
plt.ylabel(r'$||R^T R - I_3||$')
plt.xlabel('Stage')
plt.legend()
plt.grid()

ax = plt.subplot(122)
plt.plot( violation_det_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( violation_det_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.plot( violation_det_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( violation_det_constr_euc, label='Embedded Stabilization' )
plt.yscale('log')
plt.ylabel(r'$||\text{det}(R)-1||$')
plt.xlabel('Stage')
plt.legend()
plt.grid()

# 2. Dynamics violation comparison:
#       By forward simulating the SO3 dynamics and then 
#       compare the simulated trajectory with the solved state

dyn_error_ms_so3 = [ err_dyn( xs_ms_so3[k], xs_ms_so3[k+1] )  for k in range(Nsim) ]
dyn_error_ss_so3 = [ err_dyn( xs_ss_so3[k], xs_ss_so3[k+1] )  for k in range(Nsim) ]
dyn_error_unconstr_euc = [ err_dyn( xs_unconstr_euc[k], xs_unconstr_euc[k+1] )  for k in range(Nsim) ]
dyn_error_constr_euc = [ err_dyn( xs_constr_euc[k], xs_constr_euc[k+1] )  for k in range(Nsim) ]

plt.figure()
plt.plot( dyn_error_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( dyn_error_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.plot( dyn_error_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( dyn_error_constr_euc, label='Embedded Stabilization' )
plt.yscale('log')
plt.ylabel(r'$||\mathcal{X}_{k+1} - F_\mathcal{X}( \mathcal{X}_k, \xi_k )||$')
plt.xlabel('Iteration')
plt.legend()
plt.grid()

# 3. cost comparison

plt.figure()
plt.plot( J_hist_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( J_hist_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.plot( J_hist_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( J_hist_constr_euc, label='Embedded Stabilization' )
plt.yscale('log')
plt.ylabel(r'$J(\mathbf{x},\mathbf{u})$')
plt.xlabel('Iteration')
plt.legend()
plt.grid()

plt.figure()
plt.plot( np.abs(J_hist_ms_so3[1:]-J_hist_ms_so3[:-1]), label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( np.abs(J_hist_ss_so3[1:]-J_hist_ss_so3[:-1]), label=r'SS-iLQR on $\mathcal{M}$' )
plt.plot( np.abs(J_hist_unconstr_euc[1:]-J_hist_unconstr_euc[:-1]), label='Embedded Unconstrained' )
plt.plot( np.abs(J_hist_constr_euc[1:]-J_hist_constr_euc[:-1]), label='Embedded Stabilization' )
plt.yscale('log')
plt.ylabel(r'$|\Delta J(\mathbf{x},\mathbf{u})$|')
plt.xlabel('Iteration')
plt.legend()
plt.grid()

plt.figure()
plt.plot( grad_hist_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( grad_hist_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.plot( grad_hist_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( grad_hist_constr_euc, label='Embedded Stabilization' )
plt.yscale('log')
plt.ylabel('Convergence')
plt.xlabel('Iteration')
plt.legend()
plt.grid()

plt.figure()
plt.plot( defect_hist_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( defect_hist_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( defect_hist_constr_euc, label='Embedded Stabilization' )
plt.yscale('log')
plt.ylabel('Dynamics Defect')
plt.xlabel('Iteration')
plt.legend()
plt.grid()

# 4. Big Plotting: State, Input

euler_ms_so3 = np.array([ rotm2euler(x[0]) for x in xs_ms_so3 ] )
euler_ss_so3 = np.array([ rotm2euler(x[0]) for x in xs_ss_so3 ] )
euler_unconstr_euc = np.array([ rotm2euler(x[0]) for x in xs_unconstr_euc ] )
euler_constr_euc = np.array([ rotm2euler(x[0]) for x in xs_constr_euc ] )

omega_ms_so3 = np.array([ x[1] for x in xs_ms_so3 ])
omega_ss_so3 = np.array([ x[1] for x in xs_ss_so3 ])
omega_unconstr_euc = np.array([ x[1] for x in xs_unconstr_euc ])
omega_constr_euc = np.array([ x[1] for x in xs_constr_euc ])

plt.figure()

plt.subplot(431)
for i in range(3):
    plt.plot( euler_ms_so3[:,i] )
# plt.title('Euler Angle')
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('degree')
plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
plt.grid()

plt.subplot(432)
for i in range(3):
    plt.plot( omega_ms_so3[:,i] )
plt.title(r'MS-iLQR on $\mathcal{M}$')
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('rad/s')
plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
plt.grid()

plt.subplot(433)
for j in range( action_size ):
    plt.plot( us_ms_so3[:,j])
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('input')
plt.legend([r'$u_x$',r'$u_y$',r'$u_z$'])
plt.grid()


plt.subplot(434)
for i in range(3):
    plt.plot( euler_ss_so3[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('degree')
# plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
plt.grid()

plt.subplot(435)
for i in range(3):
    plt.plot( omega_ss_so3[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('rad/s')
# plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
plt.title(r'SS-iLQR on $\mathcal{M}$')
plt.grid()

plt.subplot(436)
for j in range( action_size ):
    plt.plot( us_ss_so3[:,j])
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('input')
# plt.legend([r'$u_x$',r'$u_y$',r'$u_z$'])
plt.grid()

plt.subplot(437)
for i in range(3):
    plt.plot( euler_unconstr_euc[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('degree')
# plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
plt.grid()

plt.subplot(438)
for i in range(3):
    plt.plot( omega_unconstr_euc[:,i] )
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('rad/s')
# plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
plt.title(r'Embedded Unconstrained')
plt.grid()

plt.subplot(439)
for j in range( action_size ):
    plt.plot( us_unconstr_euc[:,j])
plt.tick_params(axis='x', labelbottom=False)
plt.ylabel('input')
# plt.legend([r'$u_x$',r'$u_y$',r'$u_z$'])
plt.grid()

plt.subplot(4,3,10)
for i in range(3):
    plt.plot( euler_constr_euc[:,i] )
plt.ylabel('degree')
plt.xlabel('Stage')
# plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
plt.grid()

plt.subplot(4,3,11)
for i in range(3):
    plt.plot( omega_constr_euc[:,i] )
plt.ylabel('rad/s')
plt.xlabel('Stage')
# plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
plt.title(r'Embedded Stabilization')
plt.grid()

plt.subplot(4,3,12)
for j in range( action_size ):
    plt.plot( us_constr_euc[:,j])
plt.ylabel('input')
plt.xlabel('Stage')
# plt.legend([r'$u_x$',r'$u_y$',r'$u_z$'])
plt.grid()


# 5. 3D plotting, plotting all 4 solutions in one figure for comparison

pendulum_length = 1.2
updown_vector = np.array([0., 0., -pendulum_length]).reshape(3,1)

pos_rod_ms_so3 = np.array([x[0] @ updown_vector for x in xs_ms_so3]).reshape(N+1, 3)
pos_rod_ss_so3 = np.array([x[0] @ updown_vector for x in xs_ss_so3]).reshape(N+1, 3)
pos_rod_unconstr_euc = np.array([x[0] @ updown_vector for x in xs_unconstr_euc]).reshape(N+1, 3)
pos_rod_constr_euc = np.array([x[0] @ updown_vector for x in xs_constr_euc]).reshape(N+1, 3)

pos_rod_ref = np.array([x @ updown_vector for x in q_ref]).reshape(N+1, 3)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(pos_rod_ms_so3[:, 0], pos_rod_ms_so3[:, 1], pos_rod_ms_so3[:, 2],
#             label=r'MS-iLQR on $\mathcal{M}$')
# ax.plot(pos_rod_ss_so3[:, 0], pos_rod_ss_so3[:, 1], pos_rod_ss_so3[:, 2],
#             label=r'SS-iLQR on $\mathcal{M}$')
# ax.plot(pos_rod_unconstr_euc[:, 0], pos_rod_unconstr_euc[:, 1], pos_rod_unconstr_euc[:, 2],
#             label='Embedded Unconstrained')
# ax.plot(pos_rod_constr_euc[:, 0], pos_rod_constr_euc[:, 1], pos_rod_constr_euc[:, 2],
#             label='Embedded Stabilization')
# ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2],
#             label='Reference')
# plt.legend()

fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.set_title(r'MS-iLQR on $\mathcal{M}$')
ax.plot(pos_rod_ms_so3[:, 0], pos_rod_ms_so3[:, 1], pos_rod_ms_so3[:, 2])
ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2])
ax.set_zlim(-1, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend([r'$\mathcal{X}$',r'$\mathcal{X}_\text{ref}$'])

ax = fig.add_subplot(222, projection='3d')
ax.set_title(r'SS-iLQR on $\mathcal{M}$')
ax.plot(pos_rod_ss_so3[:, 0], pos_rod_ss_so3[:, 1], pos_rod_ss_so3[:, 2])
ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-1, 1)

ax = fig.add_subplot(223, projection='3d')
ax.set_title('Embedded Unconstrained')
ax.plot(pos_rod_unconstr_euc[:, 0], pos_rod_unconstr_euc[:, 1], pos_rod_unconstr_euc[:, 2])
ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-1, 1)

ax = fig.add_subplot(224, projection='3d')
ax.set_title('Embedded Stabilization')
ax.plot(pos_rod_constr_euc[:, 0], pos_rod_constr_euc[:, 1], pos_rod_constr_euc[:, 2])
ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-1, 1)

# =====================================================
# Special Problem
# =====================================================

# For Single Shooting, initial rollout and geodesic angle relation





plt.show()
