import pickle
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from traoptlibrary.traopt_utilis import rotm2euler
from traoptlibrary.traopt_dynamics import SO3Dynamics
from traoptlibrary.traopt_cost import SO3TrackingQuadraticGaussNewtonCost
from traoptlibrary.traopt_controller import iLQR_Tracking_SO3_MS, iLQR_Tracking_SO3
from traoptlibrary.traopt_baseline import EmbeddedEuclideanSU2_Pendulum3D
from scipy.spatial.transform import Rotation
from manifpy import SO3, SO3Tangent

# =====================================================
# SwingUp Problem Import: Reference, Initial State, Solver Options
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

q_ref_quat = [ Rotation.from_matrix(x).as_quat(scalar_first=True) for x in q_ref ]

q0 = SO3( Rotation.from_euler('xy',[10., 45.], degrees=True).as_quat() ) 
# xi0 = SO3Tangent( np.array([1.,1.,0.]) * 5 )
xi0 = SO3Tangent( np.array([1.,1.,0.]) * 1 )
x0_mnf = [ q0, xi0 ]
x0_np = [ Rotation.from_euler('xy',[10., 45.], degrees=True).as_quat(scalar_first=True), xi0.coeffs() ]

J = np.diag([ 0.5,0.7,0.9 ])
m = 1
length = 0.5

max_iterations = 2000

tol_gradiant_converge = 1e-12
tol_converge = tol_gradiant_converge


# =====================================================
# Setup
# =====================================================

N = Nsim
action_size = 3
state_size = 6

Q = np.diag([ 
    250., 10., 10., 1., 1., 1.,
])
P = Q * 1.
R = np.identity(3) * 1e-2

us_init = np.zeros((N, action_size,))

# =====================================================
# Unconstrained Embedded Space Method with Log Cost
# =====================================================

# intialize the embedded method
ipopt_su2_unconstr_euc = EmbeddedEuclideanSU2_Pendulum3D(  q_ref, xi_ref, dt, J, m, length, Q, R, P)

# get the solution
xs_su2_unconstr_euc , us_su2_unconstr_euc , J_hist_su2_unconstr_euc , \
    grad_hist_su2_unconstr_euc , defect_hist_su2_unconstr_euc = \
        ipopt_su2_unconstr_euc.fit(  x0_np, us_init, 
                                    n_iterations=max_iterations,
                                    tol_norm=tol_converge )



# # =====================================================
# # Data Type Conversion
# # =====================================================

J_hist_su2_unconstr_euc = np.array(J_hist_su2_unconstr_euc)
grad_hist_su2_unconstr_euc = np.array(grad_hist_su2_unconstr_euc)
defect_hist_su2_unconstr_euc = np.array(defect_hist_su2_unconstr_euc)

# # =====================================================
# # Plotting
# # =====================================================

# 1. norm constraint violation comparison

violation_norm_su2_unconstr_euc = [ np.linalg.norm(x[0]) for x in xs_su2_unconstr_euc ]

plt.figure()
plt.plot( violation_norm_su2_unconstr_euc, label=r'SU(2)' )
plt.ylabel(r'$||q||$')
plt.xlabel('Stage')
plt.legend()
plt.grid()


# 2. Pendulum rod visualization

def inv_quat(q):
    q = q.flatten()
    qw, qx, qy, qz = q
    return np.array([qw,-qx,-qy,-qz])

def E(qk):
    """
    Constructs the Omega matrix for quaternion dynamics.

    Args:
        wk: CasADi symbolic variable of shape (3, 1) representing angular velocity.

    Returns:
        A 4x4 CasADi DM matrix representing Omega(wk).
    """
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

pendulum_length = 1.2
updown_vector_for_quat = np.array([0., 0., 0., -pendulum_length]).reshape(4,1)
pos_rod_su2_unconstr_euc = np.array([
    E( E(x[0].reshape(4,1)) @ updown_vector_for_quat  ) 
    @ inv_quat(x[0]).reshape(4,1) for x in xs_su2_unconstr_euc
]).reshape(N+1, 4)
pos_rod_ref = np.array([
    E( E(x.reshape(4,1)) @ updown_vector_for_quat  ) 
    @ inv_quat(x).reshape(4,1) for x in q_ref_quat
]).reshape(N+1, 4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_title(r'SS-iLQR on $\mathcal{M}$')
ax.plot(pos_rod_su2_unconstr_euc[:, 1], pos_rod_su2_unconstr_euc[:, 2], pos_rod_su2_unconstr_euc[:, 3])
ax.plot(pos_rod_ref[:, 1], pos_rod_ref[:, 2], pos_rod_ref[:, 3])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-1, 1)


# pos_rod_ms_so3 = np.array([x[0] @ updown_vector for x in xs_ms_so3]).reshape(N+1, 3)
# pos_rod_ss_so3 = np.array([x[0] @ updown_vector for x in xs_ss_so3]).reshape(N+1, 3)
# pos_rod_unconstr_euc = np.array([x[0] @ updown_vector for x in xs_unconstr_euc]).reshape(N+1, 3)
# pos_rod_constr_euc = np.array([x[0] @ updown_vector for x in xs_constr_euc]).reshape(N+1, 3)
# pos_rod_logcost_euc = np.array([x[0] @ updown_vector for x in xs_logcost_euc]).reshape(N+1, 3)


# # 2. Dynamics violation comparison:
# #       By forward simulating the SO3 dynamics and then 
# #       compare the simulated trajectory with the solved state

# dyn_error_ms_so3 = [ err_dyn( xs_ms_so3[k], xs_ms_so3[k+1] )  for k in range(Nsim) ]
# dyn_error_ss_so3 = [ err_dyn( xs_ss_so3[k], xs_ss_so3[k+1] )  for k in range(Nsim) ]
# dyn_error_unconstr_euc = [ err_dyn( xs_unconstr_euc[k], xs_unconstr_euc[k+1] )  for k in range(Nsim) ]
# dyn_error_constr_euc = [ err_dyn( xs_constr_euc[k], xs_constr_euc[k+1] )  for k in range(Nsim) ]
# dyn_error_logcost_euc = [ err_dyn( xs_logcost_euc[k], xs_logcost_euc[k+1] )  for k in range(Nsim) ]

# ax = plt.subplot(133)
# plt.plot( dyn_error_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
# plt.plot( dyn_error_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
# plt.plot( dyn_error_unconstr_euc, label='Embedded Unconstrained' )
# plt.plot( dyn_error_constr_euc, label='Embedded Stabilization' )
# plt.plot( dyn_error_logcost_euc, label=r'Embedded w. $\mathcal{M}$ Cost' )
# plt.yscale('log')
# plt.ylabel(r'$||\mathcal{X}_{k+1} - F_\mathcal{X}( \mathcal{X}_k, \xi_k )||$')
# plt.xlabel('Stage')
# # plt.legend()
# plt.grid()

# # 3. cost comparison

# plt.figure()
# plt.plot( J_hist_unconstr_euc, label='Embedded Unconstrained' )
# plt.plot( J_hist_constr_euc, label='Embedded Stabilization' )
# plt.plot( J_hist_logcost_euc, label=r'Embedded w. $\mathcal{M}$ Cost' )
# plt.plot( J_hist_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
# plt.plot( J_hist_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
# plt.yscale('log')
# plt.ylabel(r'$J(\mathbf{x},\mathbf{u})$')
# plt.xlabel('Iteration')
# plt.legend()
# plt.grid()

# plt.figure()
# plt.plot( np.abs(J_hist_unconstr_euc[1:]-J_hist_unconstr_euc[:-1]), label='Embedded Unconstrained' )
# plt.plot( np.abs(J_hist_constr_euc[1:]-J_hist_constr_euc[:-1]), label='Embedded Stabilization' )
# plt.plot( np.abs(J_hist_logcost_euc[1:]-J_hist_logcost_euc[:-1]), label=r'Embedded w. $\mathcal{M}$ Cost'  )
# plt.plot( np.abs(J_hist_ms_so3[1:]-J_hist_ms_so3[:-1]), label=r'MS-iLQR on $\mathcal{M}$' )
# plt.plot( np.abs(J_hist_ss_so3[1:]-J_hist_ss_so3[:-1]), label=r'SS-iLQR on $\mathcal{M}$' )
# plt.yscale('log')
# plt.ylabel(r'$|\Delta J(\mathbf{x},\mathbf{u})$|')
# plt.xlabel('Iteration')
# # plt.legend()
# plt.grid()

# plt.figure()
# plt.plot( grad_hist_unconstr_euc, label='Embedded Unconstrained' )
# plt.plot( grad_hist_constr_euc, label='Embedded Stabilization' )
# plt.plot( grad_hist_logcost_euc, label=r'Embedded w. $\mathcal{M}$ Cost' )
# plt.plot( grad_hist_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
# plt.plot( grad_hist_ss_so3, label=r'SS-iLQR on $\mathcal{M}$' )
# plt.yscale('log')
# plt.ylabel('Gradiant')
# plt.xlabel('Iteration')
# # plt.legend()
# plt.grid()

# plt.figure()
# plt.plot( defect_hist_unconstr_euc, label='Embedded Unconstrained' )
# plt.plot( defect_hist_constr_euc, label='Embedded Stabilization' )
# plt.plot( defect_hist_logcost_euc, label=r'Embedded w. $\mathcal{M}$ Cost' )
# plt.plot( defect_hist_ms_so3, label=r'MS-iLQR on $\mathcal{M}$' )
# plt.yscale('log')
# plt.ylabel(r'$||d||$')
# plt.xlabel('Iteration')
# # plt.legend()
# plt.grid()

# # 4. Big Plotting: State, Input

# euler_ms_so3 = np.array([ rotm2euler(x[0]) for x in xs_ms_so3 ] )
# euler_ss_so3 = np.array([ rotm2euler(x[0]) for x in xs_ss_so3 ] )
# euler_unconstr_euc = np.array([ rotm2euler(x[0]) for x in xs_unconstr_euc ] )
# euler_constr_euc = np.array([ rotm2euler(x[0]) for x in xs_constr_euc ] )
# euler_logcost_euc = np.array([ rotm2euler(x[0]) for x in xs_logcost_euc ] )

# omega_ms_so3 = np.array([ x[1] for x in xs_ms_so3 ])
# omega_ss_so3 = np.array([ x[1] for x in xs_ss_so3 ])
# omega_unconstr_euc = np.array([ x[1] for x in xs_unconstr_euc ])
# omega_constr_euc = np.array([ x[1] for x in xs_constr_euc ])
# omega_logcost_euc = np.array([ x[1] for x in xs_logcost_euc ])

# plt.figure()

# plt.subplot(531)
# for i in range(3):
#     plt.plot( euler_ms_so3[:,i] )
# # plt.title('Euler Angle')
# plt.tick_params(axis='x', labelbottom=False)
# plt.ylabel('degree')
# plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
# plt.grid()

# plt.subplot(532)
# for i in range(3):
#     plt.plot( omega_ms_so3[:,i] )
# plt.title(r'MS-iLQR on $\mathcal{M}$')
# plt.tick_params(axis='x', labelbottom=False)
# plt.ylabel('rad/s')
# plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
# plt.grid()

# plt.subplot(533)
# for j in range( action_size ):
#     plt.plot( us_ms_so3[:,j])
# plt.tick_params(axis='x', labelbottom=False)
# plt.ylabel('input')
# # plt.ylim([-30,15])
# plt.legend([r'$u_x$',r'$u_y$',r'$u_z$'])
# plt.grid()


# plt.subplot(534)
# for i in range(3):
#     plt.plot( euler_ss_so3[:,i] )
# plt.tick_params(axis='x', labelbottom=False)
# plt.ylabel('degree')
# # plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
# plt.grid()

# plt.subplot(535)
# for i in range(3):
#     plt.plot( omega_ss_so3[:,i] )
# plt.tick_params(axis='x', labelbottom=False)
# plt.ylabel('rad/s')
# # plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
# plt.title(r'SS-iLQR on $\mathcal{M}$')
# plt.grid()

# plt.subplot(536)
# for j in range( action_size ):
#     plt.plot( us_ss_so3[:,j])
# plt.tick_params(axis='x', labelbottom=False)
# plt.ylabel('input')
# # plt.ylim([-30,15])
# # plt.legend([r'$u_x$',r'$u_y$',r'$u_z$'])
# plt.grid()

# plt.subplot(537)
# for i in range(3):
#     plt.plot( euler_unconstr_euc[:,i] )
# plt.tick_params(axis='x', labelbottom=False)
# plt.ylabel('degree')
# # plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
# plt.grid()

# plt.subplot(538)
# for i in range(3):
#     plt.plot( omega_unconstr_euc[:,i] )
# plt.tick_params(axis='x', labelbottom=False)
# plt.ylabel('rad/s')
# # plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
# plt.title(r'Embedded Unconstrained')
# plt.grid()

# plt.subplot(539)
# for j in range( action_size ):
#     plt.plot( us_unconstr_euc[:,j])
# plt.tick_params(axis='x', labelbottom=False)
# plt.ylabel('input')
# # plt.ylim([-30,15])
# # plt.legend([r'$u_x$',r'$u_y$',r'$u_z$'])
# plt.grid()

# plt.subplot(5,3,10)
# for i in range(3):
#     plt.plot( euler_constr_euc[:,i] )
# plt.ylabel('degree')
# plt.tick_params(axis='x', labelbottom=False)
# # plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
# plt.grid()

# plt.subplot(5,3,11)
# for i in range(3):
#     plt.plot( omega_constr_euc[:,i] )
# plt.ylabel('rad/s')
# plt.tick_params(axis='x', labelbottom=False)
# # plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
# plt.title(r'Embedded Stabilization')
# plt.grid()

# plt.subplot(5,3,12)
# for j in range( action_size ):
#     plt.plot( us_constr_euc[:,j])
# plt.ylabel('input')
# # plt.ylim([-30,15])
# plt.tick_params(axis='x', labelbottom=False)
# # plt.legend([r'$u_x$',r'$u_y$',r'$u_z$'])
# plt.grid()

# plt.subplot(5,3,13)
# for i in range(3):
#     plt.plot( euler_logcost_euc[:,i] )
# plt.ylabel('degree')
# plt.xlabel('Stage')
# # plt.legend([r'$\theta_z$',r'$\theta_x$',r'$\theta_y$'])
# plt.grid()

# plt.subplot(5,3,14)
# for i in range(3):
#     plt.plot( omega_logcost_euc[:,i] )
# plt.ylabel('rad/s')
# plt.xlabel('Stage')
# # plt.legend([r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'])
# plt.title(r'Embedded w. $\mathcal{M}$ Cost')
# plt.grid()

# plt.subplot(5,3,15)
# for j in range( action_size ):
#     plt.plot( us_logcost_euc[:,j])
# plt.ylabel('input')
# plt.xlabel('Stage')
# # plt.legend([r'$u_x$',r'$u_y$',r'$u_z$'])
# # plt.ylim([-30,15])
# plt.grid()


# # 5. 3D plotting, plotting all 4 solutions in one figure for comparison

# pendulum_length = 1.2
# updown_vector = np.array([0., 0., -pendulum_length]).reshape(3,1)

# pos_rod_ms_so3 = np.array([x[0] @ updown_vector for x in xs_ms_so3]).reshape(N+1, 3)
# pos_rod_ss_so3 = np.array([x[0] @ updown_vector for x in xs_ss_so3]).reshape(N+1, 3)
# pos_rod_unconstr_euc = np.array([x[0] @ updown_vector for x in xs_unconstr_euc]).reshape(N+1, 3)
# pos_rod_constr_euc = np.array([x[0] @ updown_vector for x in xs_constr_euc]).reshape(N+1, 3)
# pos_rod_logcost_euc = np.array([x[0] @ updown_vector for x in xs_logcost_euc]).reshape(N+1, 3)

# pos_rod_ref = np.array([x @ updown_vector for x in q_ref]).reshape(N+1, 3)

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot(pos_rod_ms_so3[:, 0], pos_rod_ms_so3[:, 1], pos_rod_ms_so3[:, 2],
# #             label=r'MS-iLQR on $\mathcal{M}$')
# # ax.plot(pos_rod_ss_so3[:, 0], pos_rod_ss_so3[:, 1], pos_rod_ss_so3[:, 2],
# #             label=r'SS-iLQR on $\mathcal{M}$')
# # ax.plot(pos_rod_unconstr_euc[:, 0], pos_rod_unconstr_euc[:, 1], pos_rod_unconstr_euc[:, 2],
# #             label='Embedded Unconstrained')
# # ax.plot(pos_rod_constr_euc[:, 0], pos_rod_constr_euc[:, 1], pos_rod_constr_euc[:, 2],
# #             label='Embedded Stabilization')
# # ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2],
# #             label='Reference')
# # plt.legend()

# fig = plt.figure()
# ax = fig.add_subplot(221, projection='3d')
# ax.set_title(r'MS-iLQR on $\mathcal{M}$')
# ax.plot(pos_rod_ms_so3[:, 0], pos_rod_ms_so3[:, 1], pos_rod_ms_so3[:, 2])
# ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2])
# ax.set_zlim(-1, 1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# plt.legend([r'$\mathcal{X}$',r'$\mathcal{X}_\text{ref}$'])

# ax = fig.add_subplot(222, projection='3d')
# ax.set_title(r'SS-iLQR on $\mathcal{M}$')
# ax.plot(pos_rod_ss_so3[:, 0], pos_rod_ss_so3[:, 1], pos_rod_ss_so3[:, 2])
# ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_zlim(-1, 1)

# ax = fig.add_subplot(234, projection='3d')
# ax.set_title('Embedded Unconstrained')
# ax.plot(pos_rod_unconstr_euc[:, 0], pos_rod_unconstr_euc[:, 1], pos_rod_unconstr_euc[:, 2])
# ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_zlim(-1, 1)

# ax = fig.add_subplot(235, projection='3d')
# ax.set_title('Embedded Stabilization')
# ax.plot(pos_rod_constr_euc[:, 0], pos_rod_constr_euc[:, 1], pos_rod_constr_euc[:, 2])
# ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_zlim(-1, 1)

# ax = fig.add_subplot(236, projection='3d')
# ax.set_title(r'Embedded w. $\mathcal{M}$ Cost')
# ax.plot(pos_rod_logcost_euc[:, 0], pos_rod_logcost_euc[:, 1], pos_rod_logcost_euc[:, 2])
# ax.plot(pos_rod_ref[:, 0], pos_rod_ref[:, 1], pos_rod_ref[:, 2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_zlim(-1, 1)

# # =====================================================
# # Special Problem
# # =====================================================

# For Single Shooting, initial rollout and geodesic angle relation

plt.show()
