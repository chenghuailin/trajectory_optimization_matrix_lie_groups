import pickle
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from traoptlibrary.traopt_utilis import rotm2euler, SE32manifSE3, se32manifse3, quat2euler
from traoptlibrary.traopt_dynamics import SE3Dynamics
from traoptlibrary.traopt_cost import SE3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_controller import iLQR_Tracking_SE3_MS, iLQR_Tracking_SE3
from traoptlibrary.traopt_baseline import EmbeddedEuclideanSU2_SE3
from scipy.spatial.transform import Rotation
from manifpy import SE3, SE3Tangent

SAVE_RESULTS = False
SAVE_RESULTS_DIR = 'visualization/results_benchmark_2nd_draft/results_se3_tracking_benchmark.pkl'

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

# Nsim = 150
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

N = Nsim
HESSIANS = False
action_size = 6
state_size = 12

dynamics = SE3Dynamics( J, dt, hessians=HESSIANS )

Q = np.diag([ 
    25, 25, 25, 1000., 1000., 1000.,
    1., 1., 1., 1., 1., 1. 
]) 
P = Q * 1
R = np.identity(6) * 1e-4
cost = SE3TrackingQuadraticGaussNewtonCost( Q, R, P, q_ref, xi_ref )

us_init = np.zeros((N, action_size,))

# =====================================================

# intialize the embedded method
ipopt_unconstr_euc = EmbeddedEuclideanSU2_SE3( q_ref, xi_ref, dt, J, Q, R, P )

# get the solution
xs_unconstr_euc, us_unconstr_euc, J_hist_unconstr_euc, \
    grad_hist_unconstr_euc, defect_hist_unconstr_euc = \
        ipopt_unconstr_euc.fit( x0, us_init, 
                                n_iterations=max_iterations,
                                tol_norm=tol_converge )


# # =====================================================
# # Data Type Conversion
# # =====================================================

J_hist_unconstr_euc = np.array(J_hist_unconstr_euc)
grad_hist_unconstr_euc = np.array(grad_hist_unconstr_euc)
defect_hist_unconstr_euc = np.array(defect_hist_unconstr_euc)


# # =====================================================
# # Data Type Conversion
# # =====================================================

quat_unconstr_euc = np.array([ x[0][:4] for x in xs_unconstr_euc ])
pos_unconstr_euc = np.array([ x[0][4:] for x in xs_unconstr_euc ])
pos_ref = np.array([ x[:3,3] for x in q_ref ])

fig = plt.figure()
euler_unconstr_euc = np.array([ quat2euler(x) for x in quat_unconstr_euc ] )
plt.plot(euler_unconstr_euc)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot( pos_unconstr_euc[:, 0], pos_unconstr_euc[:, 1], pos_unconstr_euc[:, 2],label='Embedded Unconstrained')
ax.plot( pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2],label='Reference')
 

plt.show()