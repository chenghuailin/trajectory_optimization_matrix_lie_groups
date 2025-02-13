import pickle
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from traoptlibrary.traopt_utilis import rotm2euler, quat2euler
from traoptlibrary.traopt_dynamics import Pendulum3dDyanmics
from traoptlibrary.traopt_cost import SO3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_controller import iLQR_Tracking_SO3_MS, iLQR_Tracking_SO3
from traoptlibrary.traopt_baseline import EmbeddedEuclideanSU2_Pendulum3D, \
    EmbeddedEuclideanSO3_DynamicsConstr_Pendulum3D 
from scipy.spatial.transform import Rotation
from manifpy import SO3, SO3Tangent

np.seterr(all='raise')  # 避免 NumPy 静默忽略数值问题
# np.set_printoptions(precision=16)

SAVE_RESULTS = True
SAVE_RESULTS_DIR = 'visualization/results_benchmark_2nd_draft/results_pendulum_swingup_benchmark.pkl'

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
    'visualization/optimized_trajectories/path_3dpendulum_swingup.npy'
    # 'visualization/optimized_trajectories/path_3dpendulum_8shape.npy'
    
with open( path_to_reference_file, 'rb' ) as f:
    q_ref = np.load(f)
    xi_ref = np.load(f)
    dt = np.load(f)

Nsim = q_ref.shape[0] - 1
print("Horizon of dataset is", Nsim)

q0 = SO3( Rotation.from_euler('xy',[10., 45.], degrees=True).as_quat() ) 
# xi0 = SO3Tangent( np.array([1.,1.,0.]) * 5 )
xi0 = SO3Tangent( np.array([1.,1.,0.]) * 1 )
x0_mnf = [ q0, xi0 ]
x0_np = [ q0.rotation(), xi0.coeffs() ]
x0_np_quat = [Rotation.from_euler('xy',[10., 45.], degrees=True).as_quat(scalar_first=True), xi0.coeffs()]

J = np.diag([ 0.5,0.7,0.9 ])
m = 1
length = 0.5

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

def Omega(wk):
    w1, w2, w3 = wk[0], wk[1], wk[2]
    Omega = np.array([
        [0,   -w1, -w2, -w3],
        [w1,   0,  -w3,  w2],
        [w2,  w3,   0,  -w1],
        [w3, -w2,  w1,   0]
    ])
    return Omega

def err_dyn_quat(xk, xk1, dt=dt):

    Xk, wk = xk
    Xk1, _ = xk1
    Xk_sim = Xk - dt * 0.5 * ( Omega(wk) @ Xk )

    return np.linalg.norm(Xk_sim - Xk1)

def E(qk):
    qk = qk.flatten()
    w0, w1, w2, w3 = qk[0], qk[1], qk[2], qk[3]
    
    # Construct each row of the Omega matrix
    row1 = np.array([w0,    -w1, -w2, -w3])
    row2 = np.array([w1,    w0,  -w3,  w2])
    row3 = np.array([w2,  w3,    w0,  -w1])
    row4 = np.array([w3,  -w2,  w1,    w0])
    
    # Vertically concatenate the rows to form the Omega matrix
    Omega = np.array([ row1, row2, row3, row4 ])
    
    return Omega

def inv_quat(q):
    q = q.flatten()
    qw, qx, qy, qz = q
    return np.array([qw,-qx,-qy,-qz])

def act_quat(q, x):
    x = x.flatten()
    x = np.concatenate(([0], x))
    xyz_4d = np.array([
        E(   E(q.reshape(4,1)) @ x ) 
        @ inv_quat(q).reshape(4,1)
    ]).reshape(4,)
    return xyz_4d[1:]

# =====================================================
# Algorithms on Matrix Lie Groups: Dynamics, Cost, Controller
# =====================================================

N = Nsim # horizon, note the state length = horizon + 1
HESSIANS = False
action_size = 3
state_size = 6

dynamics = Pendulum3dDyanmics(J, m, length, dt, hessians=HESSIANS)

Q = np.diag([ 
    10., 10., 10., 1., 1., 1.,
])
P = Q * 1.5
R = np.identity(3) * 1e-2
cost = SO3TrackingQuadraticGaussNewtonCost( Q, R, P, q_ref, xi_ref )

us_init = np.zeros((N, action_size,))

ilqr_ms_so3 = iLQR_Tracking_SO3_MS( dynamics, cost, N, 
                                    q_ref, xi_ref,
                                    hessians=HESSIANS,
                                    line_search=False,
                                    rollout='nonlinear' )

ilqr_ss_so3 = iLQR_Tracking_SO3(    dynamics, cost, N, 
                                    hessians=HESSIANS,
                                    rollout='nonlinear' )

xs_ms_so3, us_ms_so3, J_hist_ms_so3, _, _, \
    grad_hist_ms_so3, defect_hist_ms_so3= \
        ilqr_ms_so3.fit(x0_mnf, us_init, 
                        n_iterations=max_iterations, 
                        tol_grad_norm=tol_gradiant_converge,
                        on_iteration=on_iteration_ms_so3)
xs_ms_so3 = [ [x[0].rotation(), x[1].coeffs() ] for x in xs_ms_so3 ]

xs_ss_so3, us_ss_so3, J_hist_ss_so3, _, _, grad_hist_ss_so3 = \
        ilqr_ss_so3.fit(x0_mnf, us_init, 
                        n_iterations=max_iterations, 
                        tol_grad_norm=tol_gradiant_converge,
                        on_iteration=on_iteration_ss_so3)
xs_ss_so3 = [ [x[0].rotation(), x[1].coeffs() ] for x in xs_ss_so3 ]


# =====================================================
# Embedded Unconstrained Method
# =====================================================
# intialize the embedded method
ipopt_unconstr_euc = EmbeddedEuclideanSU2_Pendulum3D(   q_ref, xi_ref, dt, 
                                                        J, m, length, Q, R, P)

# get the solution
xs_unconstr_euc, us_unconstr_euc, J_hist_unconstr_euc, \
    grad_hist_unconstr_euc, defect_hist_unconstr_euc = \
        ipopt_unconstr_euc.fit( x0_np_quat, us_init, 
                                n_iterations=max_iterations,
                                tol_norm=tol_converge )


# =====================================================
# Embedded Space Method with M Dynamics Constraint
# =====================================================
eps_init = 5*1e0

# intialize the embedded method
ipopt_dynconstr_euc = EmbeddedEuclideanSO3_DynamicsConstr_Pendulum3D(   q_ref, xi_ref, dt, J, m, 
                                                                        length, Q, R, P, eps_init )

# get the solution
xs_dynconstr_euc, us_dynconstr_euc, J_hist_dynconstr_euc, \
    grad_hist_dynconstr_euc, defect_hist_dynconstr_euc = \
        ipopt_dynconstr_euc.fit(    x0_np, us_init, 
                                    n_iterations=max_iterations,
                                    tol_norm=tol_converge )


# =====================================================
# Save Results
# =====================================================

def save_results_pickle(filename,
                       xs_ms_so3, us_ms_so3, J_hist_ms_so3, grad_hist_ms_so3, defect_hist_ms_so3,
                       xs_ss_so3, us_ss_so3, J_hist_ss_so3, grad_hist_ss_so3,
                       xs_unconstr_euc, us_unconstr_euc, J_hist_unconstr_euc, grad_hist_unconstr_euc, defect_hist_unconstr_euc,
                       xs_dynconstr_euc, us_dynconstr_euc, J_hist_dynconstr_euc, grad_hist_dynconstr_euc, defect_hist_dynconstr_euc):
    data = {
        'prob':{
            'J': J,
            'dt': dt,
            'm': m,
            'length': length,
            'q_ref': q_ref,
            'xi_ref': xi_ref,
            'x0': x0_np,
            'Q' : Q,
            'P' : P,
            'R' : R
        },
        'ms_so3': {
            'xs': xs_ms_so3,
            'us': us_ms_so3,
            'J_hist': J_hist_ms_so3,
            'grad_hist': grad_hist_ms_so3,
            'defect_hist': defect_hist_ms_so3
        },
        'ss_so3': {
            'xs': xs_ss_so3,
            'us': us_ss_so3,
            'J_hist': J_hist_ss_so3,
            'grad_hist': grad_hist_ss_so3
        },
        'unconstr_euc': {
            'xs': xs_unconstr_euc,
            'us': us_unconstr_euc,
            'J_hist': J_hist_unconstr_euc,
            'grad_hist': grad_hist_unconstr_euc,
            'defect_hist': defect_hist_unconstr_euc
        },
        'dynconstr_euc': {
            'xs': xs_dynconstr_euc,
            'us': us_dynconstr_euc,
            'J_hist': J_hist_dynconstr_euc,
            'grad_hist': grad_hist_dynconstr_euc,
            'defect_hist': defect_hist_dynconstr_euc
        }
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Results saved to {filename}")

# if SAVE_RESULTS:
#     save_results_pickle(SAVE_RESULTS_DIR,
#                        xs_ms_so3, us_ms_so3, J_hist_ms_so3, grad_hist_ms_so3, defect_hist_ms_so3,
#                        xs_ss_so3, us_ss_so3, J_hist_ss_so3, grad_hist_ss_so3,
#                        xs_unconstr_euc, us_unconstr_euc, J_hist_unconstr_euc, grad_hist_unconstr_euc, defect_hist_unconstr_euc,
#                        xs_dynconstr_euc, us_dynconstr_euc, J_hist_dynconstr_euc, grad_hist_dynconstr_euc, defect_hist_dynconstr_euc)


# =====================================================
# Load Results
# =====================================================

def load_results_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

results = load_results_pickle(SAVE_RESULTS_DIR)

# ms_so3_data = results['ms_so3']
# xs_ms_so3 = ms_so3_data['xs']            # 状态序列 (列表，包含 manifpy 的 SO3 对象和 SO3Tangent 对象)
# us_ms_so3 = ms_so3_data['us']            # 控制序列 (numpy 数组)
# J_hist_ms_so3 = ms_so3_data['J_hist']    # 目标函数历史 (列表)
# grad_hist_ms_so3 = ms_so3_data['grad_hist']  # 梯度范数历史 (列表)
# defect_hist_ms_so3 = ms_so3_data['defect_hist']  # 缺陷范数历史 (列表)

# ss_so3_data = results['ss_so3']
# xs_ss_so3 = ss_so3_data['xs']            # 状态序列 (列表，包含 manifpy 的 SO3 对象和 SO3Tangent 对象)
# us_ss_so3 = ss_so3_data['us']            # 控制序列 (numpy 数组)
# J_hist_ss_so3 = ss_so3_data['J_hist']    # 目标函数历史 (列表)
# grad_hist_ss_so3 = ss_so3_data['grad_hist']  # 梯度范数历史 (列表)

# unconstr_euc_data = results['unconstr_euc']
# xs_unconstr_euc = unconstr_euc_data['xs']                  # 状态序列 (numpy 数组)
# us_unconstr_euc = unconstr_euc_data['us']                  # 控制序列 (numpy 数组)
# J_hist_unconstr_euc = unconstr_euc_data['J_hist']          # 目标函数历史 (numpy 数组)
# grad_hist_unconstr_euc = unconstr_euc_data['grad_hist']    # 梯度范数历史 (numpy 数组)
# defect_hist_unconstr_euc = unconstr_euc_data['defect_hist']# 缺陷范数历史 (numpy 数组)

# dynconstr_euc_data = results['dynconstr_euc']
# xs_dynconstr_euc = dynconstr_euc_data['xs']                      # 状态序列 (numpy 数组)
# us_dynconstr_euc = dynconstr_euc_data['us']                      # 控制序列 (numpy 数组)
# J_hist_dynconstr_euc = dynconstr_euc_data['J_hist']              # 目标函数历史 (numpy 数组)
# grad_hist_dynconstr_euc = dynconstr_euc_data['grad_hist']        # 梯度范数历史 (numpy 数组)
# defect_hist_dynconstr_euc = dynconstr_euc_data['defect_hist']    # 缺陷范数历史 (numpy 数组)


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

J_hist_dynconstr_euc = np.array(J_hist_dynconstr_euc)
grad_hist_dynconstr_euc = np.array(grad_hist_dynconstr_euc)
defect_hist_dynconstr_euc = np.array(defect_hist_dynconstr_euc)

# =====================================================
# Plotting
# =====================================================

# 1. Lie constraint violation comparison

violation_orth_ms_so3 = [ np.linalg.norm(x[0].T @ x[0] - np.identity(3)) for x in xs_ms_so3 ]
violation_orth_ss_so3 = [ np.linalg.norm(x[0].T @ x[0] - np.identity(3)) for x in xs_ss_so3 ]
violation_norm_unconstr_euc = [ np.linalg.norm( x[0] ) for x in xs_unconstr_euc ]
violation_orth_dynconstr_euc = [ np.linalg.norm(x[0].T @ x[0] - np.identity(3)) for x in xs_dynconstr_euc ]

# violation_det_ms_so3 = [ 1 - np.linalg.det(x[0]) for x in xs_ms_so3 ]
# violation_det_ss_so3 = [ 1 - np.linalg.det(x[0]) for x in xs_ss_so3 ]
# violation_det_dynconstr_euc  = [ 1 - np.linalg.det(x[0]) for x in xs_dynconstr_euc ]

plt.figure()
ax = plt.subplot(131)
plt.plot( violation_orth_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( violation_orth_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
# plt.plot( violation_orth_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( violation_orth_dynconstr_euc, label=r'Embedded w. $\mathcal{M}$ Dynamics', color='red' )
plt.yscale('log')
plt.ylabel(r'$||R^T R - I_3||$')
plt.xlabel('Stage')
# plt.legend()
plt.grid()

# ax = plt.subplot(132)
# plt.plot( violation_det_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
# plt.plot( violation_det_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
# plt.plot( violation_det_dynconstr_euc, label=r'Embedded w. $\mathcal{M}$ Dynamics' )
# plt.yscale('log')
# plt.ylabel(r'$|\text{det}(R)-1|$')
# plt.xlabel('Stage')
# plt.grid()

ax = plt.subplot(132)
plt.plot( 100 * (np.array(violation_norm_unconstr_euc) - 1),   label='Error', color='green'  )
plt.plot( 100 * (np.array(violation_norm_unconstr_euc)**2 - 1),   label='Squared Error', color='green', linestyle='-.')
plt.ylabel(r'Norm Error of $q$ (\%)')
plt.xlabel('Stage')
plt.legend()
plt.grid()

# 2. Dynamics violation comparison:
#       By forward simulating the SO3 dynamics and then 
#       compare the simulated trajectory with the solved state

dyn_error_ms_so3 = [ err_dyn( xs_ms_so3[k], xs_ms_so3[k+1] )  for k in range(Nsim) ]
dyn_error_ss_so3 = [ err_dyn( xs_ss_so3[k], xs_ss_so3[k+1] )  for k in range(Nsim) ]
dyn_error_unconstr_euc = [ err_dyn_quat( xs_unconstr_euc[k], xs_unconstr_euc[k+1] )  for k in range(Nsim) ]
dyn_error_dynconstr_euc = [ err_dyn( xs_dynconstr_euc[k], xs_dynconstr_euc[k+1] )  for k in range(Nsim) ]

ax = plt.subplot(133)
plt.plot( dyn_error_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( dyn_error_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.plot( dyn_error_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( dyn_error_dynconstr_euc, label=r'Embedded w. $\mathcal{M}$ Dynamics' )
plt.yscale('log')
plt.ylabel(r'$||\mathcal{X}_{k+1} - F_\mathcal{X}( \mathcal{X}_k, \xi_k )||$')
plt.xlabel('Stage')
plt.legend()
plt.grid()

# 3. cost comparison

plt.figure()
plt.plot( J_hist_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( J_hist_dynconstr_euc, label=r'Embedded w. $\mathcal{M}$ Dynamics' )
plt.plot( J_hist_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( J_hist_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel(r'$J(\mathbf{x},\mathbf{u})$')
plt.xlabel('Iteration')
plt.legend()
plt.grid()

plt.figure()
plt.plot( np.abs(J_hist_unconstr_euc[1:]-J_hist_unconstr_euc[:-1]), label='Embedded Unconstrained' )
plt.plot( np.abs(J_hist_dynconstr_euc[1:]-J_hist_dynconstr_euc[:-1]), label=r'Embedded w. $\mathcal{M}$ Dynamics' )
plt.plot( np.abs(J_hist_ms_so3[1:]-J_hist_ms_so3[:-1]), label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( np.abs(J_hist_ss_so3[1:]-J_hist_ss_so3[:-1]), label=r'SS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel(r'$|\Delta J(\mathbf{x},\mathbf{u})$|')
plt.xlabel('Iteration')
# plt.legend()
plt.grid()

plt.figure()
plt.plot( grad_hist_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( grad_hist_dynconstr_euc, label=r'Embedded w. $\mathcal{M}$ Dynamics' )
plt.plot( grad_hist_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.plot( grad_hist_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel('Gradiant')
plt.xlabel('Iteration')
# plt.legend()
plt.grid()

plt.figure()
plt.plot( defect_hist_unconstr_euc, label='Embedded Unconstrained' )
plt.plot( defect_hist_dynconstr_euc, label=r'Embedded w. $\mathcal{M}$ Dynamics' )
plt.plot( defect_hist_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
plt.yscale('log')
plt.ylabel(r'$||d||$')
plt.xlabel('Iteration')
# plt.legend()
plt.grid()

# 4. Big Plotting: State, Input

euler_ms_so3 = np.array([ rotm2euler(x[0]) for x in xs_ms_so3 ] )
euler_ss_so3 = np.array([ rotm2euler(x[0]) for x in xs_ss_so3 ] )
euler_unconstr_euc = np.array([ quat2euler(x[0]) for x in xs_unconstr_euc ] )
euler_dynconstr_euc = np.array([ rotm2euler(x[0]) for x in xs_dynconstr_euc ] )

omega_ms_so3 = np.array([ x[1] for x in xs_ms_so3 ])
omega_ss_so3 = np.array([ x[1] for x in xs_ss_so3 ])
omega_unconstr_euc = np.array([ x[1] for x in xs_unconstr_euc ])
omega_dynconstr_euc = np.array([ x[1] for x in xs_dynconstr_euc ])

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
    plt.plot( euler_dynconstr_euc[:,i] )
plt.ylabel('degree')
plt.xlabel('Stage')
# plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
plt.grid()

plt.subplot(4,3,11)
for i in range(3):
    plt.plot( omega_dynconstr_euc[:,i] )
plt.ylabel('rad/s')
plt.xlabel('Stage')
# plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
plt.title(r'Embedded w. $\mathcal{M}$ Dynamics')
plt.grid()

plt.subplot(4,3,12)
for j in range( action_size ):
    plt.plot( us_dynconstr_euc[:,j])
plt.ylabel('input')
plt.xlabel('Stage')
# plt.legend([r'$u_x$',r'$u_y$',r'$u_z$'])
plt.grid()


# 5. 3D plotting, plotting all 4 solutions in one figure for comparison

pendulum_length = 1.2
updown_vector = np.array([0., 0., -pendulum_length]).reshape(3,1)

pos_rod_ms_so3 = np.array([x[0] @ updown_vector for x in xs_ms_so3]).reshape(N+1, 3)
pos_rod_ss_so3 = np.array([x[0] @ updown_vector for x in xs_ss_so3]).reshape(N+1, 3)
pos_rod_dynconstr_euc = np.array([x[0] @ updown_vector for x in xs_dynconstr_euc]).reshape(N+1, 3)

pos_rod_unconstr_euc = np.array([act_quat(x[0],updown_vector) for x in xs_unconstr_euc]).reshape(N+1, 3)

pos_rod_ref = np.array([q_ref[0] @ updown_vector]).reshape(3,)
pos_rod_init = pos_rod_ms_so3[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos_rod_ms_so3[:, 0], pos_rod_ms_so3[:, 1], pos_rod_ms_so3[:, 2],
            label=r'MS-iLQR on $\mathcal{M}$')
ax.plot(pos_rod_ss_so3[:, 0], pos_rod_ss_so3[:, 1], pos_rod_ss_so3[:, 2],
            label=r'SS-iLQR on $\mathcal{M}$')
ax.plot(pos_rod_unconstr_euc[:, 0], pos_rod_unconstr_euc[:, 1], pos_rod_unconstr_euc[:, 2],
            label='Embedded Unconstrained')
ax.plot(pos_rod_dynconstr_euc[:, 0], pos_rod_dynconstr_euc[:, 1], pos_rod_dynconstr_euc[:, 2],
            label=r'Embedded w. $\mathcal{M}$ Dynamics')
# ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2],
#             label='Reference')
ax.quiver( 0, 0, 0, pos_rod_init[0], pos_rod_init[1], pos_rod_init[2], color='blue', label=r'$R_0$' )
ax.quiver( 0, 0, 0, pos_rod_ref[0], pos_rod_ref[1], pos_rod_ref[2], color='red', label=r'$R_\text{ref}$' )
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()

fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.set_title(r'MS-iLQR on $\mathcal{M}$')
ax.plot(pos_rod_ms_so3[:, 0], pos_rod_ms_so3[:, 1], pos_rod_ms_so3[:, 2], label=r'$R$')
ax.quiver( 0, 0, 0, pos_rod_init[0], pos_rod_init[1], pos_rod_init[2], color='blue', label=r'$R_0$' )
ax.quiver( 0, 0, 0, pos_rod_ref[0], pos_rod_ref[1], pos_rod_ref[2], color='red', label=r'$R_\text{ref}$' )
ax.set_xlim(-pendulum_length, pendulum_length)
ax.set_ylim(-pendulum_length, pendulum_length)
ax.set_zlim(-pendulum_length, pendulum_length)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.legend()

ax = fig.add_subplot(222, projection='3d')
ax.set_title(r'SS-iLQR on $\mathcal{M}$')
ax.plot(pos_rod_ss_so3[:, 0], pos_rod_ss_so3[:, 1], pos_rod_ss_so3[:, 2])
ax.quiver( 0, 0, 0, pos_rod_init[0], pos_rod_init[1], pos_rod_init[2], color='blue', label=r'$R_0$' )
ax.quiver( 0, 0, 0, pos_rod_ref[0], pos_rod_ref[1], pos_rod_ref[2], color='red', label=r'$R_\text{ref}$' )
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-pendulum_length, pendulum_length)
ax.set_ylim(-pendulum_length, pendulum_length)
ax.set_zlim(-pendulum_length, pendulum_length)

ax = fig.add_subplot(223, projection='3d')
ax.set_title('Embedded Unconstrained')
ax.plot(pos_rod_unconstr_euc[:, 0], pos_rod_unconstr_euc[:, 1], pos_rod_unconstr_euc[:, 2])
ax.quiver( 0, 0, 0, pos_rod_init[0], pos_rod_init[1], pos_rod_init[2], color='blue', label=r'$R_0$' )
ax.quiver( 0, 0, 0, pos_rod_ref[0], pos_rod_ref[1], pos_rod_ref[2], color='red', label=r'$R_\text{ref}$' )
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-pendulum_length, pendulum_length)
ax.set_ylim(-pendulum_length, pendulum_length)
ax.set_zlim(-pendulum_length, pendulum_length)

ax = fig.add_subplot(224, projection='3d')
ax.set_title(r'Embedded w. $\mathcal{M}$ Dynamics')
ax.plot(pos_rod_dynconstr_euc[:, 0], pos_rod_dynconstr_euc[:, 1], pos_rod_dynconstr_euc[:, 2])
ax.quiver( 0, 0, 0, pos_rod_init[0], pos_rod_init[1], pos_rod_init[2], color='blue', label=r'$R_0$' )
ax.quiver( 0, 0, 0, pos_rod_ref[0], pos_rod_ref[1], pos_rod_ref[2], color='red', label=r'$R_\text{ref}$' )
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-pendulum_length, pendulum_length)
ax.set_ylim(-pendulum_length, pendulum_length)
ax.set_zlim(-pendulum_length, pendulum_length)

# 6. Error Plotting


# manif
xs_mnf_ms_so3 = [[ SO3( Rotation.from_matrix(x[0]).as_quat() ), SO3Tangent(x[1])] for x in xs_ms_so3]
xs_mnf_ss_so3 = [[ SO3( Rotation.from_matrix(x[0]).as_quat() ), SO3Tangent(x[1])] for x in xs_ss_so3]
xs_mnf_unconstr_euc = [[ SO3( Rotation.from_quat(x[0], scalar_first=True).as_quat() ), SO3Tangent(x[1])] for x in xs_unconstr_euc]
xs_mnf_dynconstr_euc = [[ SO3( Rotation.from_matrix(x[0]).as_quat() ), SO3Tangent(x[1])] for x in xs_dynconstr_euc]

err_ms_so3 = [cost._err(x, i) for i, x in enumerate(xs_mnf_ms_so3)]
err_ss_so3 = [cost._err(x, i) for i, x in enumerate(xs_mnf_ss_so3)]
err_unconstr_euc = [cost._err( x, i) for i, x in enumerate(xs_mnf_unconstr_euc)]
err_dynconstr_euc = [cost._err(x, i) for i, x in enumerate(xs_mnf_dynconstr_euc)]

norm_err_q_ms_so3 = [np.linalg.norm( x[0] ) for x in err_ms_so3]
norm_err_q_ss_so3 = [np.linalg.norm( x[0] ) for x in err_ss_so3]
norm_err_q_unconstr_euc  = [np.linalg.norm( x[0] ) for x in err_unconstr_euc]
norm_err_q_dynconstr_euc = [np.linalg.norm( x[0] ) for x in err_dynconstr_euc]

norm_err_v_ms_so3 = [np.linalg.norm( x[1] ) for x in err_ms_so3]
norm_err_v_ss_so3 = [np.linalg.norm( x[1] ) for x in err_ss_so3]
norm_err_v_unconstr_euc  = [np.linalg.norm( x[1] ) for x in err_unconstr_euc]
norm_err_v_dynconstr_euc = [np.linalg.norm( x[1] ) for x in err_dynconstr_euc]

plt.figure()
ax = plt.subplot(121)
plt.plot(norm_err_q_ms_so3, label=r'MS-iLQR on $\mathcal{M}$', linestyle='-', linewidth=2)
plt.plot(norm_err_q_ss_so3, label=r'SS-iLQR on $\mathcal{M}$', linestyle='--', linewidth=2)
plt.plot(norm_err_q_unconstr_euc, label='Embedded Unconstrained', linestyle='-.', linewidth=2)
plt.plot(norm_err_q_dynconstr_euc, label=r'Embedded w. $\mathcal{M}$ Dynamics', linestyle=':', linewidth=3.2)
# plt.yscale('log')
plt.ylabel(r'$||\mathcal{X}_k \ominus \mathcal{X}_{\text{ref},k}||$')
plt.xlabel('Stage')
plt.legend()
plt.grid()

ax = plt.subplot(122)
plt.plot(norm_err_v_ms_so3, label=r'MS-iLQR on $\mathcal{M}$', linestyle='-', linewidth=2)
plt.plot(norm_err_v_ss_so3, label=r'SS-iLQR on $\mathcal{M}$', linestyle='--', linewidth=2)
plt.plot(norm_err_v_unconstr_euc, label='Embedded Unconstrained', linestyle='-.', linewidth=2)
plt.plot(norm_err_v_dynconstr_euc, label=r'Embedded w. $\mathcal{M}$ Dynamics', linestyle=':', linewidth=3.2)
# plt.yscale('log')
plt.ylabel(r'$||\xi_k - \xi_{\text{ref},k}||$')
plt.xlabel('Stage')
# plt.legend()
plt.grid()

# =====================================================
# Special Problem
# =====================================================

# For Single Shooting, initial rollout and geodesic angle relation

plt.show()
