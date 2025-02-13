import warnings
import casadi as ca
import numpy as np
from manifpy import SE3, SE3Tangent, SO3, SO3Tangent
from scipy.linalg import expm 
from scipy.spatial.transform import Rotation
from traoptlibrary.traopt_controller import BaseController
from traoptlibrary.traopt_utilis import SE32manifSE3
import spatial_casadi as sc

### ================================
### SU(2) Baselines 
### ================================

class EmbeddedEuclideanSU2(BaseController):
    """
    Baseline: Embedded Euclidean Space method.
    使用CasADi + IPOPT在 R^4 上直接进行优化，
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        Q, R, P,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Qx, Qw, QxN, QwN: weight matrices for cost
            R_mat: control cost weight
            max_ipopt_iter: ipopt最大迭代次数
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(EmbeddedEuclideanSU2, self).__init__()

        # self.q_ref = q_ref
        self.q_ref = [Rotation.from_matrix(q).as_quat(scalar_first=True) for q in q_ref ]

        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

        # Weight Matrics
        Qx = Q[:3,:3]
        Qw = Q[3:,3:]
        Px = P[:3,:3]
        Pw = P[3:,3:]
        self.Qx = Qx
        self.Qw = Qw
        self.QxN = Px
        self.QwN = Pw
        self.R_ = R

        # Matrix Norm Weight
        self.alpha = Qx[0,0]
        self.alphaN = Px[0,0]

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            xs_hist: [若干次迭代] 状态的历史，这里只存 初值 和 最终解
            us_hist: [同上]
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        q0, w0 = x0[0], x0[1]   # R0 shape (3,3), w0 shape (3,)
        dt = self.dt

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        q_vars = []
        w_vars = []
        u_vars = []

        for k in range(Nsim+1):
            # 优化变量: R_k是3x3, w_k是3x1
            qk = opti.variable(4,1)
            wk = opti.variable(3,1)
            q_vars.append(qk)
            w_vars.append(wk)

            if k < Nsim:
                uk = opti.variable(3,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(q_vars[0], q0)
        opti.set_initial(w_vars[0], w0)
        for k in range(1, Nsim+1):
            opti.set_initial(q_vars[k], self.q_ref[k])
            opti.set_initial(w_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        J_inv = self.J_inv

        def Omega(wk):
            """
            Constructs the Omega matrix for quaternion dynamics.

            Args:
                wk: CasADi symbolic variable of shape (3, 1) representing angular velocity.

            Returns:
                A 4x4 CasADi DM matrix representing Omega(wk).
            """
            w1, w2, w3 = wk[0], wk[1], wk[2]
            
            # Construct each row of the Omega matrix
            row1 = ca.horzcat(0,    -w1, -w2, -w3)
            row2 = ca.horzcat(w1,    0,  -w3,  w2)
            row3 = ca.horzcat(w2,  w3,    0,  -w1)
            row4 = ca.horzcat(w3,  -w2,  w1,    0)
            
            # Vertically concatenate the rows to form the Omega matrix
            Omega = ca.vertcat(row1, row2, row3, row4)
            
            return Omega
        
        def E(qk):
            """
            Constructs the Omega matrix for quaternion dynamics.

            Args:
                wk: CasADi symbolic variable of shape (3, 1) representing angular velocity.

            Returns:
                A 4x4 CasADi DM matrix representing Omega(wk).
            """
            w0, w1, w2, w3 = qk[0], qk[1], qk[2], qk[3]
            
            # Construct each row of the Omega matrix
            row1 = ca.horzcat(w0,    -w1, -w2, -w3)
            row2 = ca.horzcat(w1,    w0,  -w3,  w2)
            row3 = ca.horzcat(w2,  w3,    w0,  -w1)
            row4 = ca.horzcat(w3,  -w2,  w1,    w0)
            
            # Vertically concatenate the rows to form the Omega matrix
            Omega = ca.vertcat(row1, row2, row3, row4)
            
            return Omega


        def inv_quat(quat):
            qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
            return ca.vertcat(qw, -qx, -qy, -qz)

        # initial constraint
        opti.subject_to(q_vars[0] - q0 == 0)
        opti.subject_to(w_vars[0] - w0 == 0)

        # 动力学
        for k in range(Nsim):
            qk     = q_vars[k]
            qk_next= q_vars[k+1]
            wk     = w_vars[k]
            wk_next= w_vars[k+1]
            uk     = u_vars[k]

            #  manifold dynamics
            qk_prop = qk - dt * 0.5 * ca.mtimes( Omega(wk), qk )
            opti.subject_to( qk_next - qk_prop == 0)

            # w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )
            cross_term = ca.cross( self.J@wk, wk )
            wk_prop = wk + dt * ( J_inv @ (cross_term + uk) )
            opti.subject_to( wk_next - wk_prop == 0)

            # # norm == 1
            # opti.subject_to( ca.sumsqr(qk) - 1 < 1e-12 )
            # opti.subject_to(  1 - ca.sumsqr(qk) < 1e-12 )
            # opti.subject_to(  ca.sumsqr(qk) == 1 )

        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            qk = q_vars[k]
            wk = w_vars[k]
            uk = u_vars[k]

            # 参考值
            q_ref_k = self.q_ref[k]
            w_ref_k = self.xi_ref[k]

            q_diff = qk - ca.DM(q_ref_k)
            cost_att = self.alpha * ca.sumsqr(q_diff)

            # q_diff = E( qk ) @ inv_quat( q_ref_k )
            # q_diff = E(inv_quat( q_ref_k )) @ qk
            # cost_att = self.alpha * ca.sumsqr(q_diff[1:])

            # q_diff = E( q_ref_k ) @ inv_quat( qk )
            # q_diff = E( qk ) @ inv_quat( q_ref_k )
            # cost_att = q_diff[1:].T @ self.Qx @ q_diff[1:]

            w_diff = wk - ca.DM(w_ref_k)
            cost_w = w_diff.T @ self.Qw @ w_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_w + cost_u

        # 终端项
        q_N = q_vars[Nsim]
        w_N = w_vars[Nsim]
        q_refN = self.q_ref[Nsim]
        w_refN = self.xi_ref[Nsim]

        q_diff = q_N - ca.DM(q_refN)
        cost_attN = self.alphaN * ca.sumsqr(q_diff)

        # q_diff = E( q_N ) @ inv_quat( q_refN )
        # q_diff = E(inv_quat( q_refN )) @ q_N
        # cost_attN = self.alphaN * ca.sumsqr(q_diff[1:])

        # # q_diff = E( q_refN ) @ inv_quat( q_N )
        # q_diff = E( q_N ) @ inv_quat( q_refN )
        # cost_attN = q_diff[1:].T @ self.Qx @ q_diff[1:]

        w_diffN = w_N - ca.DM(w_refN)
        cost_wN = w_diffN.T @ self.QwN @ w_diffN

        cost_expr += cost_attN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        q_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            q_sol_k = sol.value(q_vars[k]).ravel()
            w_sol_k = sol.value(w_vars[k]).ravel()
            q_sol.append(q_sol_k)
            w_sol.append(w_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([q_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 3)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist
    

class EmbeddedEuclideanSU2_Pendulum3D(BaseController):
    """
    Baseline: Embedded Euclidean Space method.
    使用CasADi + IPOPT在 R^4 上直接进行优化，
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        m,
        length, 
        Q, R, P,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Qx, Qw, QxN, QwN: weight matrices for cost
            R_mat: control cost weight
            max_ipopt_iter: ipopt最大迭代次数
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(EmbeddedEuclideanSU2_Pendulum3D, self).__init__()

        # self.q_ref = q_ref
        self.q_ref = [Rotation.from_matrix(q).as_quat(scalar_first=True) for q in q_ref ]

        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)
        self.m = m
        self.g = 9.8
        self.l = length


        # Weight Matrics
        Qx = Q[:3,:3]
        Qw = Q[3:,3:]
        Px = P[:3,:3]
        Pw = P[3:,3:]
        self.Qx = Qx
        self.Qw = Qw
        self.QxN = Px
        self.QwN = Pw
        self.R_ = R

        # Matrix Norm Weight
        self.alpha = Qx[0,0]
        self.alphaN = Px[0,0]

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            xs_hist: [若干次迭代] 状态的历史，这里只存 初值 和 最终解
            us_hist: [同上]
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        q0, w0 = x0[0], x0[1]   # R0 shape (3,3), w0 shape (3,)
        dt = self.dt

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        q_vars = []
        w_vars = []
        u_vars = []

        for k in range(Nsim+1):
            # 优化变量: R_k是3x3, w_k是3x1
            qk = opti.variable(4,1)
            wk = opti.variable(3,1)
            q_vars.append(qk)
            w_vars.append(wk)

            if k < Nsim:
                uk = opti.variable(3,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(q_vars[0], q0)
        opti.set_initial(w_vars[0], w0)
        for k in range(1, Nsim+1):
            opti.set_initial(q_vars[k], self.q_ref[k])
            opti.set_initial(w_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        J_inv = self.J_inv

        def Omega(wk):
            """
            Constructs the Omega matrix for quaternion dynamics.

            Args:
                wk: CasADi symbolic variable of shape (3, 1) representing angular velocity.

            Returns:
                A 4x4 CasADi DM matrix representing Omega(wk).
            """
            w1, w2, w3 = wk[0], wk[1], wk[2]
            
            # Construct each row of the Omega matrix
            row1 = ca.horzcat(0,    -w1, -w2, -w3)
            row2 = ca.horzcat(w1,    0,  -w3,  w2)
            row3 = ca.horzcat(w2,  w3,    0,  -w1)
            row4 = ca.horzcat(w3,  -w2,  w1,    0)
            
            # Vertically concatenate the rows to form the Omega matrix
            Omega = ca.vertcat(row1, row2, row3, row4)
            
            return Omega
        
        def E(qk):
            w0, w1, w2, w3 = qk[0], qk[1], qk[2], qk[3]
            
            # Construct each row of the Omega matrix
            row1 = ca.horzcat(w0,    -w1, -w2, -w3)
            row2 = ca.horzcat(w1,    w0,  -w3,  w2)
            row3 = ca.horzcat(w2,  w3,    w0,  -w1)
            row4 = ca.horzcat(w3,  -w2,  w1,    w0)
            
            # Vertically concatenate the rows to form the Omega matrix
            Omega = ca.vertcat(row1, row2, row3, row4)
            
            return Omega

        def inv_quat(quat):
            qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
            return ca.vertcat(qw, -qx, -qy, -qz)
        
        def act_quat(q, x):
            x = ca.vertcat(0, x)
            q_inv = inv_quat(q)
            x_rotated = E( E(q) @ x ) @ q_inv
            return x_rotated[1:]

        # initial constraint
        opti.subject_to(q_vars[0] - q0 == 0)
        opti.subject_to(w_vars[0] - w0 == 0)

        # 动力学
        for k in range(Nsim):
            qk     = q_vars[k]
            qk_next= q_vars[k+1]
            wk     = w_vars[k]
            wk_next= w_vars[k+1]
            uk     = u_vars[k]

            #  manifold dynamics
            qk_prop = qk - dt * 0.5 * ca.mtimes( Omega(wk), qk )
            opti.subject_to( qk_next - qk_prop == 0)

            # w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )
            down_vec = ca.DM(  np.array([[0],[0],[-1]])  )
            rho = self.l / 2 * down_vec

            Rk = sc.Rotation.from_quat( qk/(ca.sqrt( ca.sumsqr(qk) ) ), seq='wxyz').as_matrix()
            g_term = ca.cross( self.m * self.g * rho,  Rk.T @ down_vec)

            M = ca.cross( self.m * rho, Rk.T @ uk )

            # g_term = ca.cross( self.m * self.g * rho,  act_quat( inv_quat(qk), down_vec))

            # M = ca.cross( self.m * rho, act_quat( inv_quat(qk), uk) )
            
            cross_term = ca.cross( self.J@wk, wk )

            wk_prop = wk + dt * ( J_inv @ ( 
                cross_term +  g_term  +  M
            ))
            opti.subject_to( wk_next - wk_prop == 0)

            # # norm == 1
            # opti.subject_to( ca.sumsqr(qk) - 1 < 1e-12 )
            # opti.subject_to(  1 - ca.sumsqr(qk) < 1e-12 )
            # opti.subject_to(  ca.sumsqr(qk) == 1 )

        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            qk = q_vars[k]
            wk = w_vars[k]
            uk = u_vars[k]

            # 参考值
            q_ref_k = self.q_ref[k]
            w_ref_k = self.xi_ref[k]

            q_diff = qk - ca.DM(q_ref_k)
            cost_att = self.alpha * ca.sumsqr(q_diff)

            # q_diff = E( qk ) @ inv_quat( q_ref_k )
            # q_diff = E(inv_quat( q_ref_k )) @ qk
            # cost_att = self.alpha * ca.sumsqr(q_diff[1:])

            # q_diff = E( q_ref_k ) @ inv_quat( qk )
            # q_diff = E( qk ) @ inv_quat( q_ref_k )
            # cost_att = q_diff[1:].T @ self.Qx @ q_diff[1:]

            w_diff = wk - ca.DM(w_ref_k)
            cost_w = w_diff.T @ self.Qw @ w_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_w + cost_u

        # 终端项
        q_N = q_vars[Nsim]
        w_N = w_vars[Nsim]
        q_refN = self.q_ref[Nsim]
        w_refN = self.xi_ref[Nsim]

        q_diff = q_N - ca.DM(q_refN)
        cost_attN = self.alphaN * ca.sumsqr(q_diff)

        # q_diff = E( q_N ) @ inv_quat( q_refN )
        # q_diff = E(inv_quat( q_refN )) @ q_N
        # cost_attN = self.alphaN * ca.sumsqr(q_diff[1:])

        # # q_diff = E( q_refN ) @ inv_quat( q_N )
        # q_diff = E( q_N ) @ inv_quat( q_refN )
        # cost_attN = q_diff[1:].T @ self.Qx @ q_diff[1:]

        w_diffN = w_N - ca.DM(w_refN)
        cost_wN = w_diffN.T @ self.QwN @ w_diffN

        cost_expr += cost_attN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        q_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            q_sol_k = sol.value(q_vars[k]).ravel()
            w_sol_k = sol.value(w_vars[k]).ravel()
            q_sol.append(q_sol_k)
            w_sol.append(w_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([q_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 3)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist
    

class EmbeddedEuclideanSU2_SE3(BaseController):
    """
    Baseline: Embedded Euclidean Space method.
    使用CasADi + IPOPT在 R^4 上直接进行优化，
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        Q, R, P,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Qx, Qw, QxN, QwN: weight matrices for cost
            R_mat: control cost weight
            max_ipopt_iter: ipopt最大迭代次数
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(EmbeddedEuclideanSU2_SE3, self).__init__()

        # self.q_ref = q_ref
        self.q_ref = [Rotation.from_matrix(q[:3,:3]).as_quat(scalar_first=True) for q in q_ref ]
        self.p_ref = [q[:3,3] for q in q_ref ]

        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)
        self.I = J[:3,:3]
        self.I_inv = np.linalg.inv(self.I)
        self.m = J[4,4]

        # Weight Matrics
        Qx = Q[:6,:6]
        Qw = Q[6:,6:]
        Px = P[:6,:6]
        Pw = P[6:,6:]
        self.Qx = Qx
        self.Qp = Qx[3:,3:]
        self.Qw = Qw
        self.QxN = Px
        self.QpN = Px[3:,3:]
        self.QwN = Pw
        self.R_ = R

        # Matrix Norm Weight
        self.alpha = Qx[0,0]
        self.alphaN = Px[0,0]

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            xs_hist: [若干次迭代] 状态的历史，这里只存 初值 和 最终解
            us_hist: [同上]
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        q0, xi0 = x0[0], x0[1]   # q0 shape (4,1), w0 shape (3,)
        p0  =  q0[:3,3]
        q0  =  Rotation.from_matrix(q0[:3,:3]).as_quat(scalar_first=True) 
        dt = self.dt

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        q_vars = []
        p_vars = []
        xi_vars = []
        u_vars = []

        for k in range(Nsim+1):
            # 优化变量: R_k是3x3, w_k是3x1
            qk = opti.variable(4,1)
            pk = opti.variable(3,1)
            xik = opti.variable(6,1)
            q_vars.append(qk)
            p_vars.append(pk)
            xi_vars.append(xik)

            if k < Nsim:
                uk = opti.variable(6,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(q_vars[0], q0)
        opti.set_initial(p_vars[0], p0)
        opti.set_initial(xi_vars[0], xi0)
        for k in range(1, Nsim+1):
            opti.set_initial(q_vars[k], self.q_ref[k])
            opti.set_initial(p_vars[k], self.p_ref[k])
            opti.set_initial(xi_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        I_inv = self.I_inv
        J_inv = self.J_inv
        m = self.m

        def Omega(wk):
            """
            Constructs the Omega matrix for quaternion dynamics.

            Args:
                wk: CasADi symbolic variable of shape (3, 1) representing angular velocity.

            Returns:
                A 4x4 CasADi DM matrix representing Omega(wk).
            """
            w1, w2, w3 = wk[0], wk[1], wk[2]
            
            # Construct each row of the Omega matrix
            row1 = ca.horzcat(0,    -w1, -w2, -w3)
            row2 = ca.horzcat(w1,    0,  -w3,  w2)
            row3 = ca.horzcat(w2,  w3,    0,  -w1)
            row4 = ca.horzcat(w3,  -w2,  w1,    0)
            
            # Vertically concatenate the rows to form the Omega matrix
            Omega = ca.vertcat(row1, row2, row3, row4)
            
            return Omega
        
        def E(qk):
            w0, w1, w2, w3 = qk[0], qk[1], qk[2], qk[3]
            
            # Construct each row of the Omega matrix
            row1 = ca.horzcat(w0,    -w1, -w2, -w3)
            row2 = ca.horzcat(w1,    w0,  -w3,  w2)
            row3 = ca.horzcat(w2,  w3,    w0,  -w1)
            row4 = ca.horzcat(w3,  -w2,  w1,    w0)
            
            # Vertically concatenate the rows to form the Omega matrix
            Omega = ca.vertcat(row1, row2, row3, row4)
            
            return Omega

        def inv_quat(quat):
            qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
            return ca.vertcat(qw, -qx, -qy, -qz)
        
        def act_quat(q, x):
            x = ca.vertcat(0, x)
            q_inv = inv_quat(q)
            x_rotated = E( E(q) @ x ) @ q_inv
            return x_rotated[1:]

        def ad_se3(xi):
            """
            返回 ad_{xi} (6x6)，其中 xi=[omega, v].
            ad_{[ω,v]} = [[ω^∧, 0],
                        [ v^∧, ω^∧]].
            """
            omega = xi[0:3]
            v     = xi[3:6]
            O = ca.skew(omega)
            V = ca.skew(v)
            top = ca.horzcat(O, ca.MX.zeros(3,3))
            bot = ca.horzcat(V, O)
            return ca.vertcat(top, bot)

        def adT_se3(xi):
            """
            返回 ad^*_{xi} = (ad_{xi})^T (6x6).
            常在惯性力学方程中出现: ad^*_{xi} (J xi).
            """
            return ad_se3(xi).T

        # initial constraint
        opti.subject_to(q_vars[0] - q0 == 0)
        opti.subject_to(p_vars[0] - p0 == 0)
        opti.subject_to(xi_vars[0] - xi0 == 0)

        # 动力学
        for k in range(Nsim):
            qk      = q_vars[k]
            qk_next = q_vars[k+1]
            pk      = p_vars[k]
            pk_next = p_vars[k+1]
            xik     = xi_vars[k]
            xik_next= xi_vars[k+1]
            wk      = xi_vars[k][:3]
            vk      = xi_vars[k][3:]
            uk      = u_vars[k]

            #  manifold dynamics
            qk_prop = qk - dt * 0.5 * ca.mtimes( Omega(wk), qk )
            opti.subject_to( qk_next - qk_prop == 0)

            pk_prop = pk +  dt * act_quat( qk , vk )
            opti.subject_to( pk_next - pk_prop == 0)

            # w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )            
            xik_prop = dt*ca.mtimes(J_inv, (adT_se3(xik) @ (self.J @ xik) + uk))
            opti.subject_to( xik_next - xik_prop == 0)

            # # norm == 1
            # opti.subject_to( ca.sumsqr(qk) - 1 < 1e-12 )
            # opti.subject_to(  1 - ca.sumsqr(qk) < 1e-12 )
            # opti.subject_to(  ca.sumsqr(qk) == 1 )

        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            qk = q_vars[k]
            pk = p_vars[k]
            xik = xi_vars[k]
            uk = u_vars[k]

            # 参考值
            q_ref_k = self.q_ref[k]
            p_ref_k = self.p_ref[k]
            xi_ref_k = self.xi_ref[k]

            q_diff = qk - ca.DM(q_ref_k)
            cost_att = self.alpha * ca.sumsqr(q_diff)

            # q_diff = 1 - ca.norm_2( qk.T @ q_ref_k )
            # cost_att = self.alpha * ca.sumsqr(q_diff)

            # q_diff = E( qk ) @ inv_quat( q_ref_k )
            # q_diff = E(inv_quat( q_ref_k )) @ qk
            # cost_att = self.alpha * ca.sumsqr(q_diff[1:])

            # q_diff = E( q_ref_k ) @ inv_quat( qk )
            # q_diff = E( qk ) @ inv_quat( q_ref_k )
            # cost_att = q_diff[1:].T @ self.Qxq @ q_diff[1:]

            p_diff = pk - ca.DM(p_ref_k)
            cost_p = p_diff.T @ self.Qp @ p_diff

            xi_diff = xik - ca.DM(xi_ref_k)
            cost_w = xi_diff.T @ self.Qw @ xi_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_p + cost_w + cost_u

        # 终端项
        q_N = q_vars[Nsim]
        p_N = p_vars[Nsim]   
        xi_N = xi_vars[Nsim]
        q_refN = self.q_ref[Nsim]
        p_refN = self.p_ref[Nsim]
        xi_refN = self.xi_ref[Nsim]

        q_diffN = q_N - ca.DM(q_refN)
        cost_attN = self.alphaN * ca.sumsqr(q_diffN)

        # q_diffN = 1 - ca.norm_2( q_N.T @ q_refN )
        # cost_attN = self.alphaN * ca.sumsqr(q_diffN)

        # q_diff = E( q_N ) @ inv_quat( q_refN )
        # q_diff = E(inv_quat( q_refN )) @ q_N
        # cost_attN = self.alphaN * ca.sumsqr(q_diff[1:])

        # q_diff = E( q_refN ) @ inv_quat( q_N )
        # q_diff = E( q_N ) @ inv_quat( q_refN )
        # cost_attN = q_diff[1:].T @ self.QxqN @ q_diff[1:]

        p_diffN = p_N - ca.DM(p_refN)
        cost_pN = p_diffN.T @ self.QpN @ p_diffN

        xi_diffN = xi_N - ca.DM(xi_refN)
        cost_wN = xi_diffN.T @ self.QwN @ xi_diffN

        cost_expr += cost_attN + cost_pN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        q_sol = []
        p_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            q_sol_k = sol.value(q_vars[k]).ravel()
            p_sol_k = sol.value(p_vars[k]).ravel()
            xi_sol_k = sol.value(xi_vars[k]).ravel()
            q_sol.append(q_sol_k)
            p_sol.append(p_sol_k)
            w_sol.append(xi_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        quatpos_sol = [ np.concatenate((q_sol[i], p_sol[i])) for i in range(Nsim+1) ] 

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([quatpos_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 6)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist


class EmbeddedEuclideanSU2_Drone(BaseController):
    """
    Baseline: Embedded Euclidean Space method.
    使用CasADi + IPOPT在 R^4 上直接进行优化，
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        Q, R, P,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Qx, Qw, QxN, QwN: weight matrices for cost
            R_mat: control cost weight
            max_ipopt_iter: ipopt最大迭代次数
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(EmbeddedEuclideanSU2_Drone, self).__init__()

        # self.q_ref = q_ref
        self.q_ref = [Rotation.from_matrix(q[:3,:3]).as_quat(scalar_first=True) for q in q_ref ]
        self.p_ref = [q[:3,3] for q in q_ref ]

        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)
        self.I = J[:3,:3]
        self.I_inv = np.linalg.inv(self.I)
        self.m = J[4,4]
        self.g = 9.8

        # Weight Matrics
        Qx = Q[:6,:6]
        Qw = Q[6:,6:]
        Px = P[:6,:6]
        Pw = P[6:,6:]
        self.Qx = Qx
        self.Qxq = Qx[:3,:3]
        self.Qp = Qx[3:,3:]
        self.Qw = Qw

        self.QxN = Px
        self.QxqN = Px[:3,:3]
        self.QpN = Px[3:,3:]
        self.QwN = Pw

        self.R_ = R

        # Matrix Norm Weight
        self.alpha = Qx[0,0]
        self.alphaN = Px[0,0]

        self.Pu = np.zeros((6,4))
        self.Pu[0,0] = 1.
        self.Pu[1,1] = 1.
        self.Pu[2,2] = 1.
        self.Pu[5,3] = 1.      

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            xs_hist: [若干次迭代] 状态的历史，这里只存 初值 和 最终解
            us_hist: [同上]
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        q0, xi0 = x0[0], x0[1]   # q0 shape (4,1), w0 shape (3,)
        p0  =  q0[:3,3]
        q0  =  Rotation.from_matrix(q0[:3,:3]).as_quat(scalar_first=True) 
        dt = self.dt

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        q_vars = []
        p_vars = []
        xi_vars = []
        u_vars = []

        for k in range(Nsim+1):
            # 优化变量: R_k是3x3, w_k是3x1
            qk = opti.variable(4,1)
            pk = opti.variable(3,1)
            xik = opti.variable(6,1)
            q_vars.append(qk)
            p_vars.append(pk)
            xi_vars.append(xik)

            if k < Nsim:
                uk = opti.variable(4,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(q_vars[0], q0)
        opti.set_initial(p_vars[0], p0)
        opti.set_initial(xi_vars[0], xi0)
        for k in range(1, Nsim+1):
            opti.set_initial(q_vars[k], self.q_ref[k])
            opti.set_initial(p_vars[k], self.p_ref[k])
            opti.set_initial(xi_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        I_inv = self.I_inv
        J_inv = self.J_inv
        m = self.m

        def Omega(wk):
            """
            Constructs the Omega matrix for quaternion dynamics.

            Args:
                wk: CasADi symbolic variable of shape (3, 1) representing angular velocity.

            Returns:
                A 4x4 CasADi DM matrix representing Omega(wk).
            """
            w1, w2, w3 = wk[0], wk[1], wk[2]
            
            # Construct each row of the Omega matrix
            row1 = ca.horzcat(0,    -w1, -w2, -w3)
            row2 = ca.horzcat(w1,    0,  -w3,  w2)
            row3 = ca.horzcat(w2,  w3,    0,  -w1)
            row4 = ca.horzcat(w3,  -w2,  w1,    0)
            
            # Vertically concatenate the rows to form the Omega matrix
            Omega = ca.vertcat(row1, row2, row3, row4)
            
            return Omega
        
        def E(qk):
            w0, w1, w2, w3 = qk[0], qk[1], qk[2], qk[3]
            
            # Construct each row of the Omega matrix
            row1 = ca.horzcat(w0,    -w1, -w2, -w3)
            row2 = ca.horzcat(w1,    w0,  -w3,  w2)
            row3 = ca.horzcat(w2,  w3,    w0,  -w1)
            row4 = ca.horzcat(w3,  -w2,  w1,    w0)
            
            # Vertically concatenate the rows to form the Omega matrix
            Omega = ca.vertcat(row1, row2, row3, row4)
            
            return Omega

        def inv_quat(quat):
            qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
            return ca.vertcat(qw, -qx, -qy, -qz)
        
        def act_quat(q, x):
            x = ca.vertcat(0, x)
            q_inv = inv_quat(q)
            x_rotated = E( E(q) @ x ) @ q_inv
            return x_rotated[1:]

        def ad_se3(xi):
            """
            返回 ad_{xi} (6x6)，其中 xi=[omega, v].
            ad_{[ω,v]} = [[ω^∧, 0],
                        [ v^∧, ω^∧]].
            """
            omega = xi[0:3]
            v     = xi[3:6]
            O = ca.skew(omega)
            V = ca.skew(v)
            top = ca.horzcat(O, ca.MX.zeros(3,3))
            bot = ca.horzcat(V, O)
            return ca.vertcat(top, bot)

        def adT_se3(xi):
            """
            返回 ad^*_{xi} = (ad_{xi})^T (6x6).
            常在惯性力学方程中出现: ad^*_{xi} (J xi).
            """
            return ad_se3(xi).T


        # initial constraint
        opti.subject_to(q_vars[0] - q0 == 0)
        opti.subject_to(p_vars[0] - p0 == 0)
        opti.subject_to(xi_vars[0] - xi0 == 0)

        # 动力学
        for k in range(Nsim):
            qk      = q_vars[k]
            qk_next = q_vars[k+1]
            pk      = p_vars[k]
            pk_next = p_vars[k+1]
            xik     = xi_vars[k]
            xik_next= xi_vars[k+1]
            wk      = xi_vars[k][:3]
            vk      = xi_vars[k][3:]
            uk      = u_vars[k]

            Rk = sc.Rotation.from_quat( qk/(ca.sqrt( ca.sumsqr(qk) ) ), seq='wxyz').as_matrix()

            #  manifold dynamics
            qk_prop = qk - dt * 0.5 * ca.mtimes( Omega(wk), qk )
            opti.subject_to( qk_next - qk_prop == 0)

            pk_prop = pk +  dt * ( Rk @ vk )
            opti.subject_to( pk_next - pk_prop == 0)

            # w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )            
            down_vec = np.array([[0],[0],[-1]])
            
            # g_acc = self.m * self.g * act_quat( inv_quat(qk), down_vec)  

            g_acc = self.m * self.g * Rk.T @ down_vec
            g_acc = ca.veccat( np.zeros((3,1)), g_acc )
            
            left  = xik_next
            right = xik + dt*ca.mtimes(J_inv, (adT_se3(xik) @ (self.J @ xik) + g_acc + self.Pu @ uk))
            # right = xik + dt*ca.mtimes(J_inv, (adT_se3(xik) @ (self.J @ xik) + self.Pu @ uk))
            opti.subject_to(left - right == 0)

            # # norm == 1
            # opti.subject_to( ca.sumsqr(qk) - 1 < 1e-12 )
            # opti.subject_to(  1 - ca.sumsqr(qk) < 1e-12 )
            # opti.subject_to(  ca.sumsqr(qk) == 1 )

        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            qk = q_vars[k]
            pk = p_vars[k]
            xik = xi_vars[k]
            uk = u_vars[k]

            # 参考值
            q_ref_k = self.q_ref[k]
            p_ref_k = self.p_ref[k]
            xi_ref_k = self.xi_ref[k]

            q_diff = qk - ca.DM(q_ref_k)
            cost_att = self.alpha * ca.sumsqr(q_diff)

            # q_diff = 1 - ca.norm_2( qk.T @ q_ref_k )
            # cost_att = self.alpha * ca.sumsqr(q_diff)

            # q_diff = E( qk ) @ inv_quat( q_ref_k )
            # q_diff = E(inv_quat( q_ref_k )) @ qk
            # cost_att = self.alpha * ca.sumsqr(q_diff[1:])

            # q_diff = E( q_ref_k ) @ inv_quat( qk )
            # q_diff = E( qk ) @ inv_quat( q_ref_k )
            # cost_att = q_diff[1:].T @ self.Qxq @ q_diff[1:]

            p_diff = pk - ca.DM(p_ref_k)
            cost_p = p_diff.T @ self.Qp @ p_diff

            xi_diff = xik - ca.DM(xi_ref_k)
            cost_w = xi_diff.T @ self.Qw @ xi_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_p + cost_w + cost_u
            # cost_expr += cost_p + cost_w + cost_u

        # 终端项
        q_N = q_vars[Nsim]
        p_N = p_vars[Nsim]   
        xi_N = xi_vars[Nsim]
        q_refN = self.q_ref[Nsim]
        p_refN = self.p_ref[Nsim]
        xi_refN = self.xi_ref[Nsim]

        q_diffN = q_N - ca.DM(q_refN)
        cost_attN = self.alphaN * ca.sumsqr(q_diffN)

        # q_diffN = 1 - ca.norm_2( q_N.T @ q_refN )
        # cost_attN = self.alphaN * ca.sumsqr(q_diffN)

        # q_diff = E( q_N ) @ inv_quat( q_refN )
        # q_diff = E(inv_quat( q_refN )) @ q_N
        # cost_attN = self.alphaN * ca.sumsqr(q_diff[1:])

        # q_diff = E( q_refN ) @ inv_quat( q_N )
        # q_diff = E( q_N ) @ inv_quat( q_refN )
        # cost_attN = q_diff[1:].T @ self.QxqN @ q_diff[1:]

        p_diffN = p_N - ca.DM(p_refN)
        cost_pN = p_diffN.T @ self.QpN @ p_diffN

        xi_diffN = xi_N - ca.DM(xi_refN)
        cost_wN = xi_diffN.T @ self.QwN @ xi_diffN

        cost_expr += cost_attN + cost_pN + cost_wN
        # cost_expr += cost_pN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        q_sol = []
        p_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            q_sol_k = sol.value(q_vars[k]).ravel()
            p_sol_k = sol.value(p_vars[k]).ravel()
            xi_sol_k = sol.value(xi_vars[k]).ravel()
            q_sol.append(q_sol_k)
            p_sol.append(p_sol_k)
            w_sol.append(xi_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        quatpos_sol = [ np.concatenate((q_sol[i], p_sol[i])) for i in range(Nsim+1) ] 

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([quatpos_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 4)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist
   


### ================================
### SO(3) Baselines 
### ================================

class EmbeddedEuclideanSO3_DynamicsConstr_LogCost(BaseController):
    """
    Baseline: Embedded Euclidean Space method.
    使用CasADi + IPOPT在R^{3x3} (flatten)上直接进行优化，
    对SO(3)只用软约束或后续修正，而不在矩阵李群上显式地保持约束。
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        Q, R, P,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Qx, Qw, QxN, QwN: weight matrices for cost
            R_mat: control cost weight
            max_ipopt_iter: ipopt最大迭代次数
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(EmbeddedEuclideanSO3_DynamicsConstr_LogCost, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

        # Weight Matrics
        Qx = Q[:3,:3]
        Qw = Q[3:,3:]
        Px = P[:3,:3]
        Pw = P[3:,3:]
        self.Qx = Qx
        self.Qw = Qw
        self.QxN = Px
        self.QwN = Pw
        self.R_ = R

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            xs_hist: [若干次迭代] 状态的历史，这里只存 初值 和 最终解
            us_hist: [同上]
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        R0, w0 = x0[0], x0[1]   # R0 shape (3,3), w0 shape (3,)
        dt = self.dt

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        R_vars = []
        w_vars = []
        u_vars = []

        for k in range(Nsim+1):
            # 优化变量: R_k是3x3, w_k是3x1
            Rk = opti.variable(3,3)
            wk = opti.variable(3,1)
            R_vars.append(Rk)
            w_vars.append(wk)

            if k < Nsim:
                uk = opti.variable(3,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(R_vars[0], R0)
        opti.set_initial(w_vars[0], w0)
        for k in range(1, Nsim+1):
            opti.set_initial(R_vars[k], self.q_ref[k])
            opti.set_initial(w_vars[k], self.xi_ref[k])

            # delta_q = SO3Tangent( np.random.rand(3,) * self.eps_init )
            # q_refk_mnf = SO3( Rotation.from_matrix( self.q_ref[k] ).as_quat() )
            # opti.set_initial(R_vars[k], q_refk_mnf.rplus(delta_q).rotation())
             
            # opti.set_initial(w_vars[k], self.xi_ref[k] + np.random.rand(3,) * self.eps_init)

        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        J_inv = self.J_inv

        def exp_so3(omega, h):
            """
            SO(3) 指数映射的Rodrigues公式(仅演示,见你已有的更完整实现).
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2) * h
            I3 = ca.DM.eye(3)
            skew_omega = ca.skew(omega)
            sin_t_t = ca.if_else(theta<1e-9, 1 - theta**2/6.0, ca.sin(theta)/theta)
            one_m_cos_t_t2 = ca.if_else(theta<1e-9, 0.5 - theta**2/24., (1-ca.cos(theta))/theta**2)
            A = I3 + sin_t_t*skew_omega*h + one_m_cos_t_t2*(skew_omega@skew_omega)*(h*h)
            return A

        def log_so3(R):
            """
            SO(3) 的对数映射(仅示例).
            """
            trR = R[0,0] + R[1,1] + R[2,2]
            cos_theta = 0.5*(trR - 1)
            cos_theta_clamped = ca.fmax(ca.fmin(cos_theta,1), -1)
            theta = ca.acos(cos_theta_clamped)

            skew_part = 0.5*ca.vertcat(R[2,1]-R[1,2],
                                       R[0,2]-R[2,0],
                                       R[1,0]-R[0,1])
            sin_t = ca.sin(theta)
            factor = ca.if_else(ca.fabs(sin_t)<1e-9, 1.0, theta/(sin_t))
            return factor * skew_part

        # 初始条件
        opti.subject_to(R_vars[0] - R0 == 0)
        opti.subject_to(w_vars[0] - w0 == 0)

        # 动力学
        for k in range(Nsim):
            Rk     = R_vars[k]
            Rk_next= R_vars[k+1]
            wk     = w_vars[k]
            wk_next= w_vars[k+1]
            uk     = u_vars[k]

            # 离散化: R_{k+1} = R_k * exp(w_k^∧ * dt)
            Rk_prop = ca.mtimes(Rk, exp_so3(wk, dt))
            opti.subject_to( Rk_next - Rk_prop == 0)

            # w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )
            cross_term = ca.cross(self.J@wk, wk)
            wk_prop = wk + dt*(J_inv@(cross_term + uk))
            opti.subject_to( wk_next - wk_prop == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）
        
        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            Rk = R_vars[k]
            wk = w_vars[k]
            uk = u_vars[k]

            # 参考值
            R_ref_k = self.q_ref[k]
            w_ref_k = self.xi_ref[k]

            R_err = ca.mtimes(Rk, ca.DM(R_ref_k).T)   # R_k * R_ref_k^T
            log_R_err = log_so3(R_err)
            cost_att = log_R_err.T @ self.Qx @ log_R_err

            w_diff = wk - ca.DM(w_ref_k)
            cost_w = w_diff.T @ self.Qw @ w_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_w + cost_u

        # 终端项
        R_N = R_vars[Nsim]
        w_N = w_vars[Nsim]
        R_refN = self.q_ref[Nsim]
        w_refN = self.xi_ref[Nsim]

        R_errN = ca.mtimes(R_N, ca.DM(R_refN).T)
        log_R_errN = log_so3(R_errN)
        cost_attN = log_R_errN.T @ self.QxN @ log_R_errN

        w_diffN = w_N - ca.DM(w_refN)
        cost_wN = w_diffN.T @ self.QwN @ w_diffN

        cost_expr += cost_attN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm,
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        R_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            R_sol_k = sol.value(R_vars[k])
            w_sol_k = sol.value(w_vars[k]).ravel()
            R_sol.append(R_sol_k)
            w_sol.append(w_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([R_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 3)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist
    

# Can't solve
class EmbeddedEuclideanSO3_DynamicsConstr_LogCost_Pendulum3D(BaseController):
    """
    Baseline: Embedded Euclidean Space method.
    使用CasADi + IPOPT在R^{3x3} (flatten)上直接进行优化，
    对SO(3)只用软约束或后续修正，而不在矩阵李群上显式地保持约束。
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        m,
        length, 
        Q, R,
        eps_init=1e-2,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Qx, Qw, QxN, QwN: weight matrices for cost
            R_mat: control cost weight
            max_ipopt_iter: ipopt最大迭代次数
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(EmbeddedEuclideanSO3_DynamicsConstr_LogCost_Pendulum3D, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)
        self.m = m
        self.g = 9.8
        self.l = length

        self.eps_init = eps_init

        # Weight Matrics
        Qx = Q[:3,:3]
        Qw = Q[3:,3:]
        self.Qx = Qx
        self.Qw = Qw
        self.QxN = 10 * Qx
        self.QwN = 10 * Qw
        self.R_ = R

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            xs_hist: [若干次迭代] 状态的历史，这里只存 初值 和 最终解
            us_hist: [同上]
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        R0, w0 = x0[0], x0[1]   # R0 shape (3,3), w0 shape (3,)
        dt = self.dt

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        R_vars = []
        w_vars = []
        u_vars = []

        for k in range(Nsim+1):
            # 优化变量: R_k是3x3, w_k是3x1
            Rk = opti.variable(3,3)
            wk = opti.variable(3,1)
            R_vars.append(Rk)
            w_vars.append(wk)

            if k < Nsim:
                uk = opti.variable(3,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(R_vars[0], R0)
        opti.set_initial(w_vars[0], w0)
               
        for k in range(1, Nsim+1):
            # opti.set_initial(R_vars[k], self.q_ref[k])
            # opti.set_initial(w_vars[k], self.xi_ref[k])
            opti.set_initial(R_vars[k], ca.DM.eye(3))
            opti.set_initial(w_vars[k], self.xi_ref[k] + np.random.rand(3,) * self.eps_init)
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        J_inv = self.J_inv

        def exp_so3(omega, h):
            """
            SO(3) 指数映射的Rodrigues公式(仅演示,见你已有的更完整实现).
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2) * h
            I3 = ca.DM.eye(3)
            skew_omega = ca.skew(omega)
            sin_t_t = ca.if_else(theta<1e-9, 1 - theta**2/6.0, ca.sin(theta)/theta)
            one_m_cos_t_t2 = ca.if_else(theta<1e-9, 0.5 - theta**2/24., (1-ca.cos(theta))/theta**2)
            A = I3 + sin_t_t*skew_omega*h + one_m_cos_t_t2*(skew_omega@skew_omega)*(h*h)
            return A

        def log_so3(R):
            """
            SO(3) 的对数映射(仅示例).
            """
            trR = R[0,0] + R[1,1] + R[2,2]
            cos_theta = 0.5*(trR - 1)
            cos_theta_clamped = ca.fmax(ca.fmin(cos_theta,1), -1)
            theta = ca.acos(cos_theta_clamped)

            skew_part = 0.5*ca.vertcat(R[2,1]-R[1,2],
                                       R[0,2]-R[2,0],
                                       R[1,0]-R[0,1])
            sin_t = ca.sin(theta)
            factor = ca.if_else(ca.fabs(sin_t)<1e-9, 1.0, theta/(sin_t))
            return factor * skew_part

        # 初始条件
        opti.subject_to(R_vars[0] - R0 == 0)
        opti.subject_to(w_vars[0] - w0 == 0)

        # 动力学
        for k in range(Nsim):
            Rk     = R_vars[k]
            Rk_next= R_vars[k+1]
            wk     = w_vars[k]
            wk_next= w_vars[k+1]
            uk     = u_vars[k]

            # 1. R_{k+1} = R_k * exp(w_k^∧ * dt)
            Rk_prop = ca.mtimes(Rk, exp_so3(wk, dt))
            opti.subject_to( Rk_next - Rk_prop == 0)

            # 2. w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )
            down_vec = ca.DM(  np.array([[0],[0],[-1]])  )
            rho = self.l / 2 * down_vec
            g_term = ca.cross( self.m * self.g * rho, ca.mtimes(Rk.T, down_vec) )

            M = ca.cross( self.m * rho, ca.mtimes(Rk.T, uk) )

            cross_term = ca.cross(self.J@wk, wk)

            wk_prop = wk + dt * ( J_inv @ ( 
                cross_term +  g_term  +  M
            ))
            opti.subject_to( wk_next - wk_prop == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）
        
        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            Rk = R_vars[k]
            wk = w_vars[k]
            uk = u_vars[k]

            # 参考值
            R_ref_k = self.q_ref[k]
            w_ref_k = self.xi_ref[k]

            R_err = ca.mtimes(Rk, ca.DM(R_ref_k).T)   # R_k * R_ref_k^T
            log_R_err = log_so3(R_err)
            cost_att = log_R_err.T @ self.Qx @ log_R_err

            w_diff = wk - ca.DM(w_ref_k)
            cost_w = w_diff.T @ self.Qw @ w_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_w + cost_u

        # 终端项
        R_N = R_vars[Nsim]
        w_N = w_vars[Nsim]
        R_refN = self.q_ref[Nsim]
        w_refN = self.xi_ref[Nsim]

        R_errN = ca.mtimes(R_N, ca.DM(R_refN).T)
        log_R_errN = log_so3(R_errN)
        cost_attN = log_R_errN.T @ self.QxN @ log_R_errN

        w_diffN = w_N - ca.DM(w_refN)
        cost_wN = w_diffN.T @ self.QwN @ w_diffN

        cost_expr += cost_attN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm,
             "linear_solver": "ma57",
            "hsllib": "libcoinhsl.so"
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        R_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            R_sol_k = sol.value(R_vars[k])
            w_sol_k = sol.value(w_vars[k]).ravel()
            R_sol.append(R_sol_k)
            w_sol.append(w_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([R_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 3)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist
    


class EmbeddedEuclideanSO3_DynamicsConstr(BaseController):
    """
    Baseline: Embedded Euclidean Space method.
    使用CasADi + IPOPT在R^{3x3} (flatten)上直接进行优化，
    对SO(3)只用软约束或后续修正，而不在矩阵李群上显式地保持约束。
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        Q, R, P,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Qx, Qw, QxN, QwN: weight matrices for cost
            R_mat: control cost weight
            max_ipopt_iter: ipopt最大迭代次数
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(EmbeddedEuclideanSO3_DynamicsConstr, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

        # Weight Matrics
        Qx = Q[:3,:3]
        Qw = Q[3:,3:]
        Px = P[:3,:3]
        Pw = P[3:,3:]
        self.Qx = Qx
        self.Qw = Qw
        self.QxN = Px
        self.QwN = Pw
        self.R_ = R

        # Matrix Norm Weight
        self.alpha = Qx[0,0]
        self.alphaN = Px[0,0]

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            xs_hist: [若干次迭代] 状态的历史，这里只存 初值 和 最终解
            us_hist: [同上]
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        R0, w0 = x0[0], x0[1]   # R0 shape (3,3), w0 shape (3,)
        dt = self.dt

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        R_vars = []
        w_vars = []
        u_vars = []

        for k in range(Nsim+1):
            # 优化变量: R_k是3x3, w_k是3x1
            Rk = opti.variable(3,3)
            wk = opti.variable(3,1)
            R_vars.append(Rk)
            w_vars.append(wk)

            if k < Nsim:
                uk = opti.variable(3,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(R_vars[0], R0)
        opti.set_initial(w_vars[0], w0)
        for k in range(1, Nsim+1):
            opti.set_initial(R_vars[k], self.q_ref[k])
            opti.set_initial(w_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        J_inv = self.J_inv

        def exp_so3(omega, h):
            """
            SO(3) 指数映射的Rodrigues公式(仅演示,见你已有的更完整实现).
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2) * h
            I3 = ca.DM.eye(3)
            skew_omega = ca.skew(omega)
            sin_t_t = ca.if_else(theta<1e-9, 1 - theta**2/6.0, ca.sin(theta)/theta)
            one_m_cos_t_t2 = ca.if_else(theta<1e-9, 0.5 - theta**2/24., (1-ca.cos(theta))/theta**2)
            A = I3 + sin_t_t*skew_omega*h + one_m_cos_t_t2*(skew_omega@skew_omega)*(h*h)
            return A

        # def exp_so3(omega, h):
        #     """
        #     SO(3) 指数映射的Rodrigues公式(仅演示,见你已有的更完整实现).
        #     """
        #     return expm( ca.skew(omega) * h )

        # initial constraint
        opti.subject_to(R_vars[0] - R0 == 0)
        opti.subject_to(w_vars[0] - w0 == 0)

        # 动力学
        for k in range(Nsim):
            Rk     = R_vars[k]
            Rk_next= R_vars[k+1]
            wk     = w_vars[k]
            wk_next= w_vars[k+1]
            uk     = u_vars[k]

            #  R_{k+1} = R_k * exp(w_k^∧ * dt)
            Rk_prop = ca.mtimes(Rk, exp_so3(wk, dt))
            opti.subject_to( Rk_next - Rk_prop == 0)

            # w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )
            cross_term = ca.cross(self.J@wk, wk)
            wk_prop = wk + dt*(J_inv@(cross_term + uk))
            opti.subject_to( wk_next - wk_prop == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）
        
        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            Rk = R_vars[k]
            wk = w_vars[k]
            uk = u_vars[k]

            # 参考值
            R_ref_k = self.q_ref[k]
            w_ref_k = self.xi_ref[k]

            R_diff = Rk - ca.DM(R_ref_k)
            cost_att = self.alpha * ca.sumsqr(R_diff)

            w_diff = wk - ca.DM(w_ref_k)
            cost_w = w_diff.T @ self.Qw @ w_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_w + cost_u

        # 终端项
        R_N = R_vars[Nsim]
        w_N = w_vars[Nsim]
        R_refN = self.q_ref[Nsim]
        w_refN = self.xi_ref[Nsim]

        R_diff = R_N - ca.DM(R_refN)
        cost_attN = self.alphaN * ca.sumsqr(R_diff)

        w_diffN = w_N - ca.DM(w_refN)
        cost_wN = w_diffN.T @ self.QwN @ w_diffN

        cost_expr += cost_attN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        R_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            R_sol_k = sol.value(R_vars[k])
            w_sol_k = sol.value(w_vars[k]).ravel()
            R_sol.append(R_sol_k)
            w_sol.append(w_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([R_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 3)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist
    


class EmbeddedEuclideanSO3_DynamicsConstr_Pendulum3D(BaseController):
    """
    Baseline: Embedded Euclidean Space method.
    使用CasADi + IPOPT在R^{3x3} (flatten)上直接进行优化，
    对SO(3)只用软约束或后续修正，而不在矩阵李群上显式地保持约束。
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        m,
        length, 
        Q, R, P,
        eps_init=1e-2,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Qx, Qw, QxN, QwN: weight matrices for cost
            R_mat: control cost weight
            max_ipopt_iter: ipopt最大迭代次数
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(EmbeddedEuclideanSO3_DynamicsConstr_Pendulum3D, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)
        self.m = m
        self.g = 9.8
        self.l = length

        self.eps_init = eps_init

       # Weight Matrics
        Qx = Q[:3,:3]
        Qw = Q[3:,3:]
        Px = P[:3,:3]
        Pw = P[3:,3:]
        self.Qx = Qx
        self.Qw = Qw
        self.QxN = Px
        self.QwN = Pw
        self.R_ = R

        # Matrix Norm Weight
        self.alpha = Qx[0,0]
        self.alphaN = Px[0,0]

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            xs_hist: [若干次迭代] 状态的历史，这里只存 初值 和 最终解
            us_hist: [同上]
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        R0, w0 = x0[0], x0[1]   # R0 shape (3,3), w0 shape (3,)
        dt = self.dt

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        R_vars = []
        w_vars = []
        u_vars = []

        for k in range(Nsim+1):
            # 优化变量: R_k是3x3, w_k是3x1
            Rk = opti.variable(3,3)
            wk = opti.variable(3,1)
            R_vars.append(Rk)
            w_vars.append(wk)

            if k < Nsim:
                uk = opti.variable(3,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(R_vars[0], R0)
        opti.set_initial(w_vars[0], w0)
        for k in range(1, Nsim+1):
            opti.set_initial(R_vars[k], self.q_ref[k])
            opti.set_initial(w_vars[k], self.xi_ref[k]+ self.eps_init * np.random.rand(3,))
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        J_inv = self.J_inv

        def exp_so3(omega, h):
            """
            SO(3) 指数映射的Rodrigues公式(仅演示,见你已有的更完整实现).
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2) * h
            I3 = ca.DM.eye(3)
            skew_omega = ca.skew(omega)
            sin_t_t = ca.if_else(theta<1e-9, 1 - theta**2/6.0, ca.sin(theta)/theta)
            one_m_cos_t_t2 = ca.if_else(theta<1e-9, 0.5 - theta**2/24., (1-ca.cos(theta))/theta**2)
            A = I3 + sin_t_t*skew_omega*h + one_m_cos_t_t2*(skew_omega@skew_omega)*(h*h)
            return A

        # def exp_so3(omega, h):
        #     """
        #     SO(3) 指数映射的Rodrigues公式(仅演示,见你已有的更完整实现).
        #     """
        #     return expm( ca.skew(omega) * h )

        # 初始条件
        opti.subject_to(R_vars[0] - R0 == 0)
        opti.subject_to(w_vars[0] - w0 == 0)

        # 动力学
        for k in range(Nsim):
            Rk     = R_vars[k]
            Rk_next= R_vars[k+1]
            wk     = w_vars[k]
            wk_next= w_vars[k+1]
            uk     = u_vars[k]

            # 1. R_{k+1} = R_k * exp(w_k^∧ * dt)
            Rk_prop = ca.mtimes(Rk, exp_so3(wk, dt))
            opti.subject_to( Rk_next - Rk_prop == 0)

            # 2. w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )

            # cross_term = ca.cross(self.J@wk, wk)
            # wk_prop = wk + dt*(J_inv@(cross_term + uk))
            # opti.subject_to( wk_next - wk_prop == 0)

            down_vec = ca.DM(  np.array([[0],[0],[-1]])  )
            rho = self.l / 2 * down_vec
            g_term = ca.cross( self.m * self.g * rho, ca.mtimes(Rk.T, down_vec) )

            M = ca.cross( self.m * rho, ca.mtimes(Rk.T, uk) )

            cross_term = ca.cross(self.J@wk, wk)

            wk_prop = wk + dt * ( J_inv @ ( 
                cross_term +  g_term  +  M
            ))
            opti.subject_to( wk_next - wk_prop == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）
        
        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            Rk = R_vars[k]
            wk = w_vars[k]
            uk = u_vars[k]

            # 参考值
            R_ref_k = self.q_ref[k]
            w_ref_k = self.xi_ref[k]

            R_diff = Rk - ca.DM(R_ref_k)
            cost_att = self.alpha * ca.sumsqr(R_diff)

            w_diff = wk - ca.DM(w_ref_k)
            cost_w = w_diff.T @ self.Qw @ w_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_w + cost_u

        # 终端项
        R_N = R_vars[Nsim]
        w_N = w_vars[Nsim]
        R_refN = self.q_ref[Nsim]
        w_refN = self.xi_ref[Nsim]

        R_diff = R_N - ca.DM(R_refN)
        cost_attN = self.alphaN * ca.sumsqr(R_diff)

        w_diffN = w_N - ca.DM(w_refN)
        cost_wN = w_diffN.T @ self.QwN @ w_diffN

        cost_expr += cost_attN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        R_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            R_sol_k = sol.value(R_vars[k])
            w_sol_k = sol.value(w_vars[k]).ravel()
            R_sol.append(R_sol_k)
            w_sol.append(w_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([R_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 3)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist
    


class ConstraintStabilizationSO3(BaseController):
    """
    Baseline: Constraint Stabilization method.
    在离散化R_{k+1}时加一个额外的kappa项以强制趋近正交.
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        Q, R,
        kappa = 1e0,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Q: weight matrices for cost
            R: control cost weight
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(ConstraintStabilizationSO3, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

        self.kappa = kappa

        # Weight Matrics
        Qx = Q[:3,:3]
        Qw = Q[3:,3:]
        self.Qx = Qx
        self.Qw = Qw
        self.QxN = 10 * Qx
        self.QwN = 10 * Qw
        self.R_ = R

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       # 忽略, 只是为了接口统一; 我们在NLP中会把 u 作为未知量
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        R0, w0 = x0[0], x0[1]   # R0 shape (3,3), w0 shape (3,)
        dt = self.dt

        # 先存一些迭代历史(仅作示例)
        J_hist = []
        xs_hist = []
        us_hist = []
        grad_hist = []
        defect_hist = []

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        R_vars = []
        w_vars = []
        u_vars = []

        for k in range(Nsim+1):
            Rk = opti.variable(3,3)
            wk = opti.variable(3,1)
            R_vars.append(Rk)
            w_vars.append(wk)

            if k < Nsim:
                uk = opti.variable(3,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(R_vars[0], R0)
        opti.set_initial(w_vars[0], w0)
        for k in range(1, Nsim+1):
            opti.set_initial(R_vars[k], self.q_ref[k])
            opti.set_initial(w_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        J_inv = self.J_inv

        def exp_so3(omega, h):
            """
            SO(3) 指数映射的Rodrigues公式(仅演示,见你已有的更完整实现).
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2) * h
            I3 = ca.DM.eye(3)
            skew_omega = ca.skew(omega)
            sin_t_t = ca.if_else(theta<1e-9, 1 - theta**2/6.0, ca.sin(theta)/theta)
            one_m_cos_t_t2 = ca.if_else(theta<1e-9, 0.5 - theta**2/24., (1-ca.cos(theta))/theta**2)
            A = I3 + sin_t_t*skew_omega*h + one_m_cos_t_t2*(skew_omega@skew_omega)*(h*h)
            return A

        def log_so3(R):
            """
            SO(3) 的对数映射(仅示例).
            """
            trR = R[0,0] + R[1,1] + R[2,2]
            cos_theta = 0.5*(trR - 1)
            cos_theta_clamped = ca.fmax(ca.fmin(cos_theta,1), -1)
            theta = ca.acos(cos_theta_clamped)

            skew_part = 0.5*ca.vertcat(R[2,1]-R[1,2],
                                       R[0,2]-R[2,0],
                                       R[1,0]-R[0,1])
            sin_t = ca.sin(theta)
            factor = ca.if_else(ca.fabs(sin_t)<1e-9, 1.0, theta/(sin_t))
            return factor * skew_part

        # 初始条件
        opti.subject_to(R_vars[0] - R0 == 0)
        opti.subject_to(w_vars[0] - w0 == 0)

        # 动力学
        for k in range(Nsim):
            Rk     = R_vars[k]
            Rk_next= R_vars[k+1]
            wk     = w_vars[k]
            wk_next= w_vars[k+1]
            uk     = u_vars[k]

            # 离散化: R_{k+1} = R_k * exp(w_k^∧ * dt)
            Rk_prop = ca.mtimes(Rk, 
                                exp_so3(wk, dt) + self.kappa/2 * ( ca.inv(ca.mtimes(ca.transpose(Rk), Rk)) - ca.DM.eye(3)))
            opti.subject_to( Rk_next - Rk_prop == 0)

            # w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )
            cross_term = ca.cross(self.J@wk, wk)
            wk_prop = wk + dt*(J_inv@(cross_term + uk))
            opti.subject_to( wk_next - wk_prop == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）
        
        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            Rk = R_vars[k]
            wk = w_vars[k]
            uk = u_vars[k]

            # 参考值
            R_ref_k = self.q_ref[k]
            w_ref_k = self.xi_ref[k]

            R_err = ca.mtimes(Rk, ca.DM(R_ref_k).T)   # R_k * R_ref_k^T
            log_R_err = log_so3(R_err)
            cost_att = log_R_err.T @ self.Qx @ log_R_err

            w_diff = wk - ca.DM(w_ref_k)
            cost_w = w_diff.T @ self.Qw @ w_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_w + cost_u

        # 终端项
        R_N = R_vars[Nsim]
        w_N = w_vars[Nsim]
        R_refN = self.q_ref[Nsim]
        w_refN = self.xi_ref[Nsim]

        R_errN = ca.mtimes(R_N, ca.DM(R_refN).T)
        log_R_errN = log_so3(R_errN)
        cost_attN = log_R_errN.T @ self.QxN @ log_R_errN

        w_diffN = w_N - ca.DM(w_refN)
        cost_wN = w_diffN.T @ self.QwN @ w_diffN

        cost_expr += cost_attN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        R_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            R_sol_k = sol.value(R_vars[k])
            w_sol_k = sol.value(w_vars[k]).ravel()
            R_sol.append(R_sol_k)
            w_sol.append(w_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([R_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 3)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist



class ConstraintStabilizationSO3_Pendulum3D(BaseController):
    """
    Baseline: Constraint Stabilization method.
    在离散化R_{k+1}时加一个额外的kappa项以强制趋近正交.
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        Q, R,
        eps_init = 1e-2,
        kappa = 1e0,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Q: weight matrices for cost
            R: control cost weight
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(ConstraintStabilizationSO3, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

        self.eps_init = eps_init
        self.kappa = kappa

        # Weight Matrics
        Qx = Q[:3,:3]
        Qw = Q[3:,3:]
        self.Qx = Qx
        self.Qw = Qw
        self.QxN = 10 * Qx
        self.QwN = 10 * Qw
        self.R_ = R

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       # 忽略, 只是为了接口统一; 我们在NLP中会把 u 作为未知量
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        R0, w0 = x0[0], x0[1]   # R0 shape (3,3), w0 shape (3,)
        dt = self.dt

        # 先存一些迭代历史(仅作示例)
        J_hist = []
        xs_hist = []
        us_hist = []
        grad_hist = []
        defect_hist = []

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        R_vars = []
        w_vars = []
        u_vars = []

        for k in range(Nsim+1):
            Rk = opti.variable(3,3)
            wk = opti.variable(3,1)
            R_vars.append(Rk)
            w_vars.append(wk)

            if k < Nsim:
                uk = opti.variable(3,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(R_vars[0], R0)
        opti.set_initial(w_vars[0], w0)
        for k in range(1, Nsim+1):
            opti.set_initial(R_vars[k], self.q_ref[k])
            opti.set_initial(w_vars[k], self.xi_ref[k] + np.random.rand(3,) * self.eps_init )
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        J_inv = self.J_inv

        def exp_so3(omega, h):
            """
            SO(3) 指数映射的Rodrigues公式(仅演示,见你已有的更完整实现).
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2) * h
            I3 = ca.DM.eye(3)
            skew_omega = ca.skew(omega)
            sin_t_t = ca.if_else(theta<1e-9, 1 - theta**2/6.0, ca.sin(theta)/theta)
            one_m_cos_t_t2 = ca.if_else(theta<1e-9, 0.5 - theta**2/24., (1-ca.cos(theta))/theta**2)
            A = I3 + sin_t_t*skew_omega*h + one_m_cos_t_t2*(skew_omega@skew_omega)*(h*h)
            return A

        def log_so3(R):
            """
            SO(3) 的对数映射(仅示例).
            """
            trR = R[0,0] + R[1,1] + R[2,2]
            cos_theta = 0.5*(trR - 1)
            cos_theta_clamped = ca.fmax(ca.fmin(cos_theta,1), -1)
            theta = ca.acos(cos_theta_clamped)

            skew_part = 0.5*ca.vertcat(R[2,1]-R[1,2],
                                       R[0,2]-R[2,0],
                                       R[1,0]-R[0,1])
            sin_t = ca.sin(theta)
            factor = ca.if_else(ca.fabs(sin_t)<1e-9, 1.0, theta/(sin_t))
            return factor * skew_part

        # 初始条件
        opti.subject_to(R_vars[0] - R0 == 0)
        opti.subject_to(w_vars[0] - w0 == 0)

        # 动力学
        for k in range(Nsim):
            Rk     = R_vars[k]
            Rk_next= R_vars[k+1]
            wk     = w_vars[k]
            wk_next= w_vars[k+1]
            uk     = u_vars[k]

            # 离散化: R_{k+1} = R_k * exp(w_k^∧ * dt)
            Rk_prop = ca.mtimes(Rk, 
                                exp_so3(wk, dt) + self.kappa/2 * ( ca.inv(ca.mtimes(ca.transpose(Rk), Rk)) - ca.DM.eye(3)))
            opti.subject_to( Rk_next - Rk_prop == 0)

            # w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )
            cross_term = ca.cross(self.J@wk, wk)
            wk_prop = wk + dt*(J_inv@(cross_term + uk))
            opti.subject_to( wk_next - wk_prop == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）
        
        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            Rk = R_vars[k]
            wk = w_vars[k]
            uk = u_vars[k]

            # 参考值
            R_ref_k = self.q_ref[k]
            w_ref_k = self.xi_ref[k]

            R_err = ca.mtimes(Rk, ca.DM(R_ref_k).T)   # R_k * R_ref_k^T
            log_R_err = log_so3(R_err)
            cost_att = log_R_err.T @ self.Qx @ log_R_err

            w_diff = wk - ca.DM(w_ref_k)
            cost_w = w_diff.T @ self.Qw @ w_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_w + cost_u

        # 终端项
        R_N = R_vars[Nsim]
        w_N = w_vars[Nsim]
        R_refN = self.q_ref[Nsim]
        w_refN = self.xi_ref[Nsim]

        R_errN = ca.mtimes(R_N, ca.DM(R_refN).T)
        log_R_errN = log_so3(R_errN)
        cost_attN = log_R_errN.T @ self.QxN @ log_R_errN

        w_diffN = w_N - ca.DM(w_refN)
        cost_wN = w_diffN.T @ self.QwN @ w_diffN

        cost_expr += cost_attN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        R_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            R_sol_k = sol.value(R_vars[k])
            w_sol_k = sol.value(w_vars[k]).ravel()
            R_sol.append(R_sol_k)
            w_sol.append(w_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([R_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 3)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist



class ConstraintStabilizationSO3_MatrixNorm(BaseController):
    """
    Baseline: Constraint Stabilization method.
    在离散化R_{k+1}时加一个额外的kappa项以强制趋近正交.
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        Q, R,
        kappa = 1e-1,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Q: weight matrices for cost
            R: control cost weight
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(ConstraintStabilizationSO3_MatrixNorm, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

        self.kappa = kappa

        # Weight Matrics
        Qx = Q[:3,:3]
        Qw = Q[3:,3:]
        self.Qx = Qx
        self.Qw = Qw
        self.QxN = 10 * Qx
        self.QwN = 10 * Qw
        self.R_ = R

        # Matrix Norm Weight
        self.alpha = Qx[0,0]
        self.alphaN = self.alpha * 10

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       # 忽略, 只是为了接口统一; 我们在NLP中会把 u 作为未知量
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        R0, w0 = x0[0], x0[1]   # R0 shape (3,3), w0 shape (3,)
        dt = self.dt

        # 先存一些迭代历史(仅作示例)
        J_hist = []
        xs_hist = []
        us_hist = []
        grad_hist = []
        defect_hist = []

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        R_vars = []
        w_vars = []
        u_vars = []

        for k in range(Nsim+1):
            Rk = opti.variable(3,3)
            wk = opti.variable(3,1)
            R_vars.append(Rk)
            w_vars.append(wk)

            if k < Nsim:
                uk = opti.variable(3,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(R_vars[0], R0)
        opti.set_initial(w_vars[0], w0)
        for k in range(1, Nsim+1):
            opti.set_initial(R_vars[k], self.q_ref[k])
            opti.set_initial(w_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        J_inv = self.J_inv

        def exp_so3(omega, h):
            """
            SO(3) 指数映射的Rodrigues公式(仅演示,见你已有的更完整实现).
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
            I3 = ca.DM.eye(3)
            skew_omega = ca.skew(omega)
            sin_t_t = ca.if_else(theta<1e-9, 1 - theta**2/6.0, ca.sin(theta)/theta)
            one_m_cos_t_t2 = ca.if_else(theta<1e-9, 0.5 - theta**2/24., (1-ca.cos(theta))/theta**2)
            A = I3 + sin_t_t*skew_omega*h + one_m_cos_t_t2*(skew_omega@skew_omega)*(h*h)
            return A

        # 初始条件
        opti.subject_to(R_vars[0] - R0 == 0)
        opti.subject_to(w_vars[0] - w0 == 0)

        # 动力学
        for k in range(Nsim):
            Rk     = R_vars[k]
            Rk_next= R_vars[k+1]
            wk     = w_vars[k]
            wk_next= w_vars[k+1]
            uk     = u_vars[k]

            # 离散化: R_{k+1} = R_k * exp(w_k^∧ * dt)
            Rk_prop = ca.mtimes(Rk, 
                                exp_so3(wk, dt) + self.kappa/2 * (    ca.inv( ca.mtimes(ca.transpose(Rk), Rk) ) - ca.DM.eye(3)    ))
            opti.subject_to( Rk_next - Rk_prop == 0)

            # w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )
            cross_term = ca.cross(self.J@wk, wk)
            wk_prop = wk + dt*(J_inv@(cross_term + uk))
            opti.subject_to( wk_next - wk_prop == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）
        
        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            Rk = R_vars[k]
            wk = w_vars[k]
            uk = u_vars[k]

            # 参考值
            R_ref_k = self.q_ref[k]
            w_ref_k = self.xi_ref[k]

            R_diff = Rk - ca.DM(R_ref_k)
            cost_att = self.alpha * ca.sumsqr(R_diff)

            w_diff = wk - ca.DM(w_ref_k)
            cost_w = w_diff.T @ self.Qw @ w_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_w + cost_u

        # 终端项
        R_N = R_vars[Nsim]
        w_N = w_vars[Nsim]
        R_refN = self.q_ref[Nsim]
        w_refN = self.xi_ref[Nsim]

        R_diff = R_N - ca.DM(R_refN)
        cost_attN = self.alphaN * ca.sumsqr(R_diff)

        w_diffN = w_N - ca.DM(w_refN)
        cost_wN = w_diffN.T @ self.QwN @ w_diffN

        cost_expr += cost_attN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        R_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            R_sol_k = sol.value(R_vars[k])
            w_sol_k = sol.value(w_vars[k]).ravel()
            R_sol.append(R_sol_k)
            w_sol.append(w_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([R_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 3)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist



class ConstraintStabilizationSO3_Pendulum3D_MatrixNorm(BaseController):
    """
    Baseline: Constraint Stabilization method.
    在离散化R_{k+1}时加一个额外的kappa项以强制趋近正交.
    """

    def __init__(
        self,
        q_ref,
        xi_ref,
        dt,
        J,
        m,
        length, 
        Q, R,
        eps_init=1e-2,
        kappa = 1e0,
        verbose=False
    ):
        """
        构造函数，传入参考轨迹、模型参数、权重矩阵等。

        Args:
            q_ref: list/ndarray of shape (N+1, 3, 3) or something representing the desired rotation matrix
            xi_ref: list/ndarray of shape (N+1, 3) for angular velocity references
            dt: float or ndarray, time step
            J: inertia matrix
            Q: weight matrices for cost
            R: control cost weight
            verbose: 是否在求解过程中输出 ipopt 的详细信息
        """
        super(ConstraintStabilizationSO3_Pendulum3D_MatrixNorm, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)
        self.m = m
        self.g = 9.8
        self.l = length

        self.eps_init = eps_init
        self.kappa = kappa

        # Weight Matrics
        Qx = Q[:3,:3]
        Qw = Q[3:,3:]
        self.Qx = Qx
        self.Qw = Qw
        self.QxN = 10 * Qx
        self.QwN = 10 * Qw
        self.R_ = R

        # Matrix Norm Weight
        self.alpha = Qx[0,0]
        self.alphaN = self.alpha * 10

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N = len(self.q_ref) - 1

    def fit(
        self, 
        x0,            # [R0, w0], 其中 R0是3x3, w0是3x1
        us_init,       # 忽略, 只是为了接口统一; 我们在NLP中会把 u 作为未知量
        n_iterations=200,  
        tol_norm=1e-6,
    ):
        """
        构建并调用 IPOPT 优化，得到最优 (R, w, u) 序列。

        Returns:
            xs: [N+1, [R_k, w_k]] 最优解的状态轨迹
            us: [N, 3] 最优解的控制输入序列
            J_hist: [若干] 各迭代的cost
            grad_hist: [同上] 这里可自行约定如何衡量
            defect_hist: [同上] 用来存放动力学违背量
        """
        # ----------------------------
        #  0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        R0, w0 = x0[0], x0[1]   # R0 shape (3,3), w0 shape (3,)
        dt = self.dt

        # 先存一些迭代历史(仅作示例)
        J_hist = []
        xs_hist = []
        us_hist = []
        grad_hist = []
        defect_hist = []

        # ----------------------------
        #  1) 构建CasADi优化变量
        # ----------------------------
        opti = ca.Opti()
        R_vars = []
        w_vars = []
        u_vars = []

        for k in range(Nsim+1):
            Rk = opti.variable(3,3)
            wk = opti.variable(3,1)
            R_vars.append(Rk)
            w_vars.append(wk)

            if k < Nsim:
                uk = opti.variable(3,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(R_vars[0], R0)
        opti.set_initial(w_vars[0], w0)
        for k in range(1, Nsim+1):
            opti.set_initial(R_vars[k], self.q_ref[k])
            opti.set_initial(w_vars[k], self.xi_ref[k]+ self.eps_init * np.random.rand(3,))
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])

        # ----------------------------
        #  2) 动力学约束 + 初始条件 + 正交约束(可选)
        # ----------------------------
        J_inv = self.J_inv

        def exp_so3(omega, h):
            """
            SO(3) 指数映射的Rodrigues公式(仅演示,见你已有的更完整实现).
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
            I3 = ca.DM.eye(3)
            skew_omega = ca.skew(omega)
            sin_t_t = ca.if_else(theta<1e-9, 1 - theta**2/6.0, ca.sin(theta)/theta)
            one_m_cos_t_t2 = ca.if_else(theta<1e-9, 0.5 - theta**2/24., (1-ca.cos(theta))/theta**2)
            A = I3 + sin_t_t*skew_omega*h + one_m_cos_t_t2*(skew_omega@skew_omega)*(h*h)
            return A

        # 初始条件
        opti.subject_to(R_vars[0] - R0 == 0)
        opti.subject_to(w_vars[0] - w0 == 0)

        # 动力学
        for k in range(Nsim):
            Rk     = R_vars[k]
            Rk_next= R_vars[k+1]
            wk     = w_vars[k]
            wk_next= w_vars[k+1]
            uk     = u_vars[k]

            # 离散化: R_{k+1} = R_k * exp(w_k^∧ * dt)
            Rk_prop = ca.mtimes(Rk, 
                                exp_so3(wk, dt) + self.kappa/2 * ( ca.inv(ca.mtimes(ca.transpose(Rk), Rk)) - ca.DM.eye(3)))
            opti.subject_to( Rk_next - Rk_prop == 0)

            # 2. w_{k+1} = w_k + dt*J_inv( J w_k x w_k + u_k )

            # cross_term = ca.cross(self.J@wk, wk)
            # wk_prop = wk + dt*(J_inv@(cross_term + uk))
            # opti.subject_to( wk_next - wk_prop == 0)

            down_vec = ca.DM(  np.array([[0],[0],[-1]])  )
            rho = self.l / 2 * down_vec
            g_term = ca.cross( self.m * self.g * rho, ca.mtimes(Rk.T, down_vec) )

            M = ca.cross( self.m * rho, ca.mtimes(Rk.T, uk) )

            cross_term = ca.cross(self.J@wk, wk)

            wk_prop = wk + dt * ( J_inv @ ( 
                cross_term +  g_term  +  M
            ))
            opti.subject_to( wk_next - wk_prop == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）
        
        # ----------------------------
        #  3) 构建目标函数
        # ----------------------------
        cost_expr = 0

        for k in range(Nsim):
            Rk = R_vars[k]
            wk = w_vars[k]
            uk = u_vars[k]

            # 参考值
            R_ref_k = self.q_ref[k]
            w_ref_k = self.xi_ref[k]

            R_diff = Rk - ca.DM(R_ref_k)
            cost_att = self.alpha * ca.sumsqr(R_diff)

            w_diff = wk - ca.DM(w_ref_k)
            cost_w = w_diff.T @ self.Qw @ w_diff

            cost_u = uk.T @ self.R_ @ uk

            cost_expr += cost_att + cost_w + cost_u

        # 终端项
        R_N = R_vars[Nsim]
        w_N = w_vars[Nsim]
        R_refN = self.q_ref[Nsim]
        w_refN = self.xi_ref[Nsim]

        R_diff = R_N - ca.DM(R_refN)
        cost_attN = self.alphaN * ca.sumsqr(R_diff)

        w_diffN = w_N - ca.DM(w_refN)
        cost_wN = w_diffN.T @ self.QwN @ w_diffN

        cost_expr += cost_attN + cost_wN

        # 设置目标
        opti.minimize(cost_expr)

        # ----------------------------
        #  4) 配置 IPOPT 并求解
        # ----------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------        
        R_sol = []
        w_sol = []
        u_sol = []

        for k in range(Nsim+1):
            R_sol_k = sol.value(R_vars[k])
            w_sol_k = sol.value(w_vars[k]).ravel()
            R_sol.append(R_sol_k)
            w_sol.append(w_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # （1）把 R_sol, w_sol 转化为 xs 的形式
        xs = []
        for k in range(Nsim+1):
            xs.append([R_sol[k], w_sol[k]])  # 只是一个简单的列表结构

        # （2）reshape u_sol into us
        us = np.array(u_sol).reshape(Nsim, 3)

        # （3）Cost
        J_hist = sol.stats()['iterations']['obj']

        # （4）动力学违背(Defect)
        defect_hist = sol.stats()['iterations']['inf_pr']

        # （5）gradient
        grad_hist = sol.stats()['iterations']['inf_du']

        return xs, us, J_hist, grad_hist, defect_hist


### ================================
### SE(3) Baselines 
### ================================

class EmbeddedEuclideanSE3_DynamicsConstr_LogCost(BaseController):
    """
    Baseline (SE3): Embedded Euclidean method.
    在 R^(4x4) 上直接进行优化, 不对 SE(3) 施加硬约束, 
    允许通过 cost 拉近到参考姿态 + 动力学方程等式约束.
    """

    def __init__(self, 
                 q_ref,             # ndarray/list of (N+1, 4,4) 参考位姿(齐次变换)
                 xi_ref,            # ndarray/list of (N+1, 6)   参考 twist (omega+v)
                 dt,                # 时间步长(或可是 list/array)
                 J,                 # 惯性/质量矩阵(6x6)
                 Q, R, P,             # 代价权重: 这里 Q 是 12x12 的块形式(或你项目中的结构)
                 eps_init=1e-2,     # 初始化时对reference的扰动
                 thold_approx=1e-9,
                 verbose=False):
        """
        Args:
            q_ref:  (N+1, 4,4), reference SE(3).
            xi_ref: (N+1, 6), reference twist.
            dt:     float or array.
            J:      (6,6).
            Q:      (12,12) or含有[姿态/位置,twist]的对角
            R:      (6,6) control cost
        """
        super(EmbeddedEuclideanSE3_DynamicsConstr_LogCost, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

        self.eps_init = eps_init
        self.thold_approx = thold_approx
        
        self.QX  = Q[:6,:6]
        self.QXi = Q[6:,6:]
        self.QXn  = P[:6,:6]
        self.QXin = P[6:,6:]
        self.R_   = R

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N    = len(q_ref) - 1 

    def fit(self, 
            x0,            # 初始状态 [X0(4x4), xi0(6,)]
            us_init,       # 初始控制 guess, shape (N, 6)
            n_iterations=200, 
            tol_norm=1e-6):
        """
        返回: 
            xs:  [N+1, [X_k, xi_k]]
            us:  [N, 6]
            J_hist, grad_hist, defect_hist
        """
        Nsim = self.N
        X0, xi0 = x0[0], x0[1]
        dt   = self.dt


        # ---------------------------
        # 1) 定义优化变量
        # ---------------------------
        opti = ca.Opti()
        X_vars = []
        xi_vars= []
        u_vars = []

        for k in range(Nsim+1):
            Xk = opti.variable(4,4)
            xik= opti.variable(6,1)
            X_vars.append(Xk)
            xi_vars.append(xik)

            if k < Nsim:
                uk = opti.variable(6,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(X_vars[0], X0)
        opti.set_initial(xi_vars[0], xi0)
        for k in range(1, Nsim+1):

            q_refk_mnf = SE32manifSE3(self.q_ref[k])
            delta_q = SE3Tangent(np.ones((6,)) * self.eps_init)
            opti.set_initial(X_vars[k], q_refk_mnf.rplus(delta_q).transform())   

            opti.set_initial(xi_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])


        # ---------------------------
        # 2) 定义辅助函数
        # ---------------------------
        def ad_se3(xi):
            """
            返回 ad_{xi} (6x6)，其中 xi=[omega, v].
            ad_{[ω,v]} = [[ω^∧, 0],
                        [ v^∧, ω^∧]].
            """
            omega = xi[0:3]
            v     = xi[3:6]
            O = ca.skew(omega)
            V = ca.skew(v)
            top = ca.horzcat(O, ca.MX.zeros(3,3))
            bot = ca.horzcat(V, O)
            return ca.vertcat(top, bot)

        def adT_se3(xi):
            """
            返回 ad^*_{xi} = (ad_{xi})^T (6x6).
            常在惯性力学方程中出现: ad^*_{xi} (J xi).
            """
            return ad_se3(xi).T

        def V_so3(omega):
            """
            辅助积分矩阵 V(omega) = I + (1-cosθ)/θ^2 ω^∧ + (θ - sinθ)/θ^3 (ω^∧)^2,
            其中 θ = ||omega||.
            当 θ 非常小时，用 Taylor 展开近似.
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
            O = ca.skew(omega)
            I3 = ca.MX.eye(3)

            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))
            theta_minus_sin_over_t3 = ca.if_else(theta < self.thold_approx,
                                                1/6.0 - theta**2/120.0,
                                                (theta - ca.sin(theta))/(theta**3))
            # Todo: non-smooth?

            return I3 \
                + one_minus_cos_over_t2 * O \
                + theta_minus_sin_over_t3 * (O @ O)

        def invV_so3(omega):
            """
            V(omega) 的逆矩阵。通常可用级数或 closed-form 近似。
            这里简单地直接用 casadi 的 inv() 反演,
            如果对大规模或频繁调用有性能顾虑, 可使用解析公式或插值近似.
            """
            return ca.inv(V_so3(omega))

        def log_se3(X):
            """
            SE(3) 的对数映射: 返回 6x1 向量 [ω; v].
            完整实现: 
            1) 先从 R= X[0:3,0:3] 得到 omega (SO(3)对数)
            2) 再从 p= X[0:3,3] 得到 v = V(omega)^{-1} * p
            """
            R = X[0:3,0:3]
            p = X[0:3,3]

            # 1) 先做 SO(3) 的对数
            trace_R = R[0,0] + R[1,1] + R[2,2]
            cos_theta = 0.5*(trace_R - 1)
            cos_theta_clamped = ca.fmax(ca.fmin(cos_theta, 1), -1)
            theta = ca.acos(cos_theta_clamped)

            # skew_part = [ R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1) ]
            skew_part = ca.vertcat(
                R[2,1] - R[1,2],
                R[0,2] - R[2,0],
                R[1,0] - R[0,1]
            )
            sin_theta = ca.sin(theta)
            factor = ca.if_else(
                ca.fabs(sin_theta) < self.thold_approx,
                0.5,  # 退化近似 => ω=0
                theta/(2*sin_theta)
            )
            omega = factor * skew_part  # 3x1

            # 2) 再对平移:  p => v = V(omega)^(-1) * p
            #    V(omega)见 V_so3(omega). 这里需要先算:
            v = invV_so3(omega) @ p
            # v = V_so3(omega) @ p
            # v = p

            return ca.vertcat(omega, v)

        def exp_se3(xi, h=1.0):
            """
            SE(3) 上的指数映射, 带步长 h:
            exp_se3(xi, h) = exp( (h * xi)^\wedge ).
            其中 xi=[ω; v], 并用罗德里格斯公式+V(ω) 计算 4x4 齐次变换矩阵.
            """
            # 1) 拆出角速度和线速度, 乘以 h
            omega = xi[0:3]
            v     = xi[3:6]
            h_omega = h * omega
            h_v     = h * v

            # 2) 先算 R_exp = exp(h_omega^∧) by Rodrigues
            theta = ca.sqrt(h_omega[0]**2 + h_omega[1]**2 + h_omega[2]**2)
            Omega = ca.skew(h_omega)
            I3 = ca.MX.eye(3)

            sin_t_over_t = ca.if_else(theta < self.thold_approx,
                                    1 - theta**2/6.0,
                                    ca.sin(theta)/theta)
            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))

            R_exp = I3 + sin_t_over_t*Omega + one_minus_cos_over_t2*(Omega @ Omega)

            # 3) 平移部分 p_exp = V(h_omega) * (h_v)
            V_exp = V_so3(h_omega)
            p_exp = V_exp @ h_v

            # 4) 拼回 4x4
            X_exp = ca.vertcat(
                ca.horzcat(R_exp, p_exp),
                ca.horzcat(ca.MX.zeros(1,3), ca.MX.ones(1,1))
            )
            return X_exp

        def inv_se3(X):
            """
            给定一个 4x4 的 SE(3) 齐次变换矩阵 X = [[R, p],
                                                    [0, 1]],
            直接返回它的逆:  [[R^T, -R^T p],
                            [ 0,       1   ]].
            注意: 如果 X 不是合法的 SE(3) 矩阵（R 非正交、det != 1 等），则此公式不成立。
            """
            R = X[0:3, 0:3]
            p = X[0:3, 3]

            # R^T
            R_T = R.T

            # -R^T pd
            minus_RTp = -ca.mtimes(R_T, p)

            # 拼回 4x4
            X_inv = ca.vertcat(
                ca.horzcat(R_T, minus_RTp),
                ca.horzcat(ca.DM.zeros(1,3), ca.DM.ones(1,1))
            )
            return X_inv
        
        # ---------------------------
        # 3) 动力学约束 + 初始条件
        # ---------------------------
        J_inv = self.J_inv

        # 初始
        opti.subject_to(X_vars[0] - X0 == 0)
        opti.subject_to(xi_vars[0] - xi0 == 0)

        # 每个时刻
        for k in range(Nsim):
            Xk    = X_vars[k]
            Xk1   = X_vars[k+1]
            xik   = xi_vars[k]
            xik1  = xi_vars[k+1]
            uk    = u_vars[k]

            # X{k+1} = Xk * exp_se3(xi_k, dt)
            Xk_prop = ca.mtimes(Xk, exp_se3(xik, dt))
            opti.subject_to(Xk1 - Xk_prop == 0)

            # xi{k+1} = xi_k + dt*J^-1( ad^*_{xi_k}(J xi_k) + u_k )
            left  = xik1
            right = xik + dt*ca.mtimes(J_inv, (adT_se3(xik) @ (self.J @ xik) + uk))
            opti.subject_to(left - right == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）

        # ---------------------------
        # 4) 构建目标函数
        # ---------------------------
        cost_expr = 0

        for k in range(Nsim):
            Xk = X_vars[k]
            xik= xi_vars[k]
            uk = u_vars[k]
            Xref_k  = self.q_ref[k]
            xiref_k = self.xi_ref[k]

            Xref_k_inv = inv_se3(Xref_k)
            X_err = ca.mtimes(Xk, Xref_k_inv)    # 4x4
            log_Xerr = log_se3(X_err)           # 6x1
            
            xi_diff = xik - ca.DM(xiref_k.reshape((6,1)))
            
            # [X_err(6x1)]^T QX [X_err(6x1)] + [xi_diff(6x1)]^T QXi [xi_diff(6x1)] + ...
            cost_att = log_Xerr.T @ ca.DM(self.QX) @ log_Xerr
            cost_xi  = xi_diff.T  @ ca.DM(self.QXi) @ xi_diff
            cost_u   = uk.T @ ca.DM(self.R_) @ uk

            cost_expr += cost_att + cost_xi + cost_u
        
        # 末端项
        Xk    = X_vars[Nsim]
        xik   = xi_vars[Nsim]
        Xref_k = self.q_ref[Nsim]
        xiref_k= self.xi_ref[Nsim]

        Xref_k_inv = inv_se3(Xref_k)
        X_err = ca.mtimes(Xk, Xref_k_inv)
        log_Xerr = log_se3(X_err)
        cost_att = log_Xerr.T @ ca.DM(self.QXn) @ log_Xerr

        xi_diff  = xik - ca.DM(xiref_k.reshape((6,1)))
        cost_xi  = xi_diff.T  @ ca.DM(self.QXin) @ xi_diff
        
        cost_expr += cost_att + cost_xi

        opti.minimize(cost_expr)

        # ---------------------------
        # 5) 配置IPOPT, 求解
        # ---------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm,
            # "linear_solver": "ma86",
            # "hsllib": "libcoinhsl.so"
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ---------------------------
        # 6) 解析解 & 返回
        # ---------------------------
        X_sol = []
        xi_sol= []
        u_sol = []

        for k in range(Nsim+1):
            X_sol_k  = sol.value(X_vars[k])
            xi_sol_k = sol.value(xi_vars[k]).ravel()   # 6,
            X_sol.append(X_sol_k)
            xi_sol.append(xi_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # xs: [ [X_sol[0], xi_sol[0]], ..., [X_sol[N], xi_sol[N]] ]
        xs = []
        for k in range(Nsim+1):
            xs.append([ X_sol[k], xi_sol[k] ])

        us = np.array(u_sol).reshape(Nsim, 6)

        # 若需要从 ipopt 读取迭代历史(与 iLQR 的 iteration 不同), 
        # 直接从 stats() 提取:
        stats = sol.stats()
        iter_stats = stats.get('iterations', {})
        J_hist     = iter_stats.get('obj', [])
        grad_hist  = iter_stats.get('inf_du', [])
        defect_hist= iter_stats.get('inf_pr', [])

        return xs, us, J_hist, grad_hist, defect_hist



class EmbeddedEuclideanSE3_DynamicsConstr_LogCost_Drone(BaseController):
    """
    Baseline (SE3): Embedded Euclidean method.
    在 R^(4x4) 上直接进行优化, 不对 SE(3) 施加硬约束, 
    允许通过 cost 拉近到参考姿态 + 动力学方程等式约束.
    """

    def __init__(self, 
                 q_ref,             # ndarray/list of (N+1, 4,4) 参考位姿(齐次变换)
                 xi_ref,            # ndarray/list of (N+1, 6)   参考 twist (omega+v)
                 dt,                # 时间步长(或可是 list/array)
                 J,                 # 惯性/质量矩阵(6x6)
                 Q, R, P,           # 代价权重: 这里 Q 是 12x12 的块形式(或你项目中的结构)
                 eps_init=1e-2,     # 初始化时对reference的扰动
                 thold_approx=1e-9,
                 verbose=False):
        """
        Args:
            q_ref:  (N+1, 4,4), reference SE(3).
            xi_ref: (N+1, 6), reference twist.
            dt:     float or array.
            J:      (6,6).
            Q:      (12,12) or含有[姿态/位置,twist]的对角
            R:      (6,6) control cost
        """
        super(EmbeddedEuclideanSE3_DynamicsConstr_LogCost_Drone, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)
        self.Ib = J[0:3, 0:3] 
        self.m = J[4,4]
        self.g = 9.8

        self.eps_init = eps_init
        self.thold_approx = thold_approx
        
        self.QX  = Q[:6,:6]
        self.QXi = Q[6:,6:]
        self.QXn  = P[:6,:6]
        self.QXin = P[6:,6:]
        self.R_   = R

        self.Pu = np.zeros((6,4))
        self.Pu[0,0] = 1.
        self.Pu[1,1] = 1.
        self.Pu[2,2] = 1.
        self.Pu[5,3] = 1.        

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N    = len(q_ref) - 1 

    def fit(self, 
            x0,            # 初始状态 [X0(4x4), xi0(6,)]
            us_init,       # 初始控制 guess, shape (N, 6)
            n_iterations=200, 
            tol_norm=1e-6):
        """
        返回: 
            xs:  [N+1, [X_k, xi_k]]
            us:  [N, 6]
            J_hist, grad_hist, defect_hist
        """
        Nsim = self.N
        X0, xi0 = x0[0], x0[1]
        dt   = self.dt


        # ---------------------------
        # 1) 定义优化变量
        # ---------------------------
        opti = ca.Opti()
        X_vars = []
        xi_vars= []
        u_vars = []

        for k in range(Nsim+1):
            Xk = opti.variable(4,4)
            xik= opti.variable(6,1)
            X_vars.append(Xk)
            xi_vars.append(xik)

            if k < Nsim:
                uk = opti.variable(4,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(X_vars[0], X0)
        opti.set_initial(xi_vars[0], xi0)
        for k in range(1, Nsim+1):

            q_refk_mnf = SE32manifSE3(self.q_ref[k])
            delta_q = SE3Tangent(np.ones((6,)) * self.eps_init)
            opti.set_initial(X_vars[k], q_refk_mnf.rplus(delta_q).transform())   

            opti.set_initial(xi_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])


        # ---------------------------
        # 2) 定义辅助函数
        # ---------------------------
        def ad_se3(xi):
            """
            返回 ad_{xi} (6x6)，其中 xi=[omega, v].
            ad_{[ω,v]} = [[ω^∧, 0],
                        [ v^∧, ω^∧]].
            """
            omega = xi[0:3]
            v     = xi[3:6]
            O = ca.skew(omega)
            V = ca.skew(v)
            top = ca.horzcat(O, ca.MX.zeros(3,3))
            bot = ca.horzcat(V, O)
            return ca.vertcat(top, bot)

        def adT_se3(xi):
            """
            返回 ad^*_{xi} = (ad_{xi})^T (6x6).
            常在惯性力学方程中出现: ad^*_{xi} (J xi).
            """
            return ad_se3(xi).T

        def V_so3(omega):
            """
            辅助积分矩阵 V(omega) = I + (1-cosθ)/θ^2 ω^∧ + (θ - sinθ)/θ^3 (ω^∧)^2,
            其中 θ = ||omega||.
            当 θ 非常小时，用 Taylor 展开近似.
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
            O = ca.skew(omega)
            I3 = ca.MX.eye(3)

            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))
            theta_minus_sin_over_t3 = ca.if_else(theta < self.thold_approx,
                                                1/6.0 - theta**2/120.0,
                                                (theta - ca.sin(theta))/(theta**3))
            # Todo: non-smooth?

            return I3 \
                + one_minus_cos_over_t2 * O \
                + theta_minus_sin_over_t3 * (O @ O)

        def invV_so3(omega):
            """
            V(omega) 的逆矩阵。通常可用级数或 closed-form 近似。
            这里简单地直接用 casadi 的 inv() 反演,
            如果对大规模或频繁调用有性能顾虑, 可使用解析公式或插值近似.
            """
            return ca.inv(V_so3(omega))

        def log_se3(X):
            """
            SE(3) 的对数映射: 返回 6x1 向量 [ω; v].
            完整实现: 
            1) 先从 R= X[0:3,0:3] 得到 omega (SO(3)对数)
            2) 再从 p= X[0:3,3] 得到 v = V(omega)^{-1} * p
            """
            R = X[0:3,0:3]
            p = X[0:3,3]

            # 1) 先做 SO(3) 的对数
            trace_R = R[0,0] + R[1,1] + R[2,2]
            cos_theta = 0.5*(trace_R - 1)
            cos_theta_clamped = ca.fmax(ca.fmin(cos_theta, 1), -1)
            theta = ca.acos(cos_theta_clamped)

            # skew_part = [ R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1) ]
            skew_part = ca.vertcat(
                R[2,1] - R[1,2],
                R[0,2] - R[2,0],
                R[1,0] - R[0,1]
            )
            sin_theta = ca.sin(theta)
            factor = ca.if_else(
                ca.fabs(sin_theta) < self.thold_approx,
                0.5,  # 退化近似 => ω=0
                theta/(2*sin_theta)
            )
            omega = factor * skew_part  # 3x1

            # 2) 再对平移:  p => v = V(omega)^(-1) * p
            #    V(omega)见 V_so3(omega). 这里需要先算:
            v = invV_so3(omega) @ p
            # v = V_so3(omega) @ p
            # v = p

            return ca.vertcat(omega, v)

        def exp_se3(xi, h=1.0):
            """
            SE(3) 上的指数映射, 带步长 h:
            exp_se3(xi, h) = exp( (h * xi)^\wedge ).
            其中 xi=[ω; v], 并用罗德里格斯公式+V(ω) 计算 4x4 齐次变换矩阵.
            """
            # 1) 拆出角速度和线速度, 乘以 h
            omega = xi[0:3]
            v     = xi[3:6]
            h_omega = h * omega
            h_v     = h * v

            # 2) 先算 R_exp = exp(h_omega^∧) by Rodrigues
            theta = ca.sqrt(h_omega[0]**2 + h_omega[1]**2 + h_omega[2]**2)
            Omega = ca.skew(h_omega)
            I3 = ca.MX.eye(3)

            sin_t_over_t = ca.if_else(theta < self.thold_approx,
                                    1 - theta**2/6.0,
                                    ca.sin(theta)/theta)
            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))

            R_exp = I3 + sin_t_over_t*Omega + one_minus_cos_over_t2*(Omega @ Omega)

            # 3) 平移部分 p_exp = V(h_omega) * (h_v)
            V_exp = V_so3(h_omega)
            p_exp = V_exp @ h_v

            # 4) 拼回 4x4
            X_exp = ca.vertcat(
                ca.horzcat(R_exp, p_exp),
                ca.horzcat(ca.MX.zeros(1,3), ca.MX.ones(1,1))
            )
            return X_exp

        def inv_se3(X):
            """
            给定一个 4x4 的 SE(3) 齐次变换矩阵 X = [[R, p],
                                                    [0, 1]],
            直接返回它的逆:  [[R^T, -R^T p],
                            [ 0,       1   ]].
            注意: 如果 X 不是合法的 SE(3) 矩阵（R 非正交、det != 1 等），则此公式不成立。
            """
            R = X[0:3, 0:3]
            p = X[0:3, 3]

            # R^T
            R_T = R.T

            # -R^T pd
            minus_RTp = -ca.mtimes(R_T, p)

            # 拼回 4x4
            X_inv = ca.vertcat(
                ca.horzcat(R_T, minus_RTp),
                ca.horzcat(ca.DM.zeros(1,3), ca.DM.ones(1,1))
            )
            return X_inv
        
        # ---------------------------
        # 3) 动力学约束 + 初始条件
        # ---------------------------
        J_inv = self.J_inv

        # 初始
        opti.subject_to(X_vars[0] - X0 == 0)
        opti.subject_to(xi_vars[0] - xi0 == 0)

        # 每个时刻
        for k in range(Nsim):
            Xk    = X_vars[k]
            Xk1   = X_vars[k+1]
            xik   = xi_vars[k]
            xik1  = xi_vars[k+1]
            uk    = u_vars[k]

            # 1. X{k+1} = Xk * exp_se3(xi_k, dt)
            Xk_prop = ca.mtimes(Xk, exp_se3(xik, dt))
            opti.subject_to(Xk1 - Xk_prop == 0)

            # 2.  xi{k+1} = xi_k + dt*J^-1( ad^*_{xi_k}(J xi_k) + u_k )
            
            down_vec = np.array([[0],[0],[-1]])
            Rk  = Xk[:3,:3]
            g_acc = self.m * self.g * Rk.T @ down_vec
            g_acc = ca.veccat( np.zeros((3,1)), g_acc )
            
            left  = xik1
            right = xik + dt*ca.mtimes(J_inv, (adT_se3(xik) @ (self.J @ xik) + g_acc + self.Pu @ uk))

            opti.subject_to(left - right == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）

        # ---------------------------
        # 4) 构建目标函数
        # ---------------------------
        cost_expr = 0

        for k in range(Nsim):
            Xk = X_vars[k]
            xik= xi_vars[k]
            uk = u_vars[k]
            Xref_k  = self.q_ref[k]
            xiref_k = self.xi_ref[k]

            Xref_k_inv = inv_se3(Xref_k)
            X_err = ca.mtimes(Xk, Xref_k_inv)    # 4x4
            log_Xerr = log_se3(X_err)           # 6x1
            
            xi_diff = xik - ca.DM(xiref_k.reshape((6,1)))
            
            # [X_err(6x1)]^T QX [X_err(6x1)] + [xi_diff(6x1)]^T QXi [xi_diff(6x1)] + ...
            cost_att = log_Xerr.T @ ca.DM(self.QX) @ log_Xerr
            cost_xi  = xi_diff.T  @ ca.DM(self.QXi) @ xi_diff
            cost_u   = uk.T @ ca.DM(self.R_) @ uk

            cost_expr += cost_att + cost_xi + cost_u
        
        # 末端项
        Xk    = X_vars[Nsim]
        xik   = xi_vars[Nsim]
        Xref_k = self.q_ref[Nsim]
        xiref_k= self.xi_ref[Nsim]

        Xref_k_inv = inv_se3(Xref_k)
        X_err = ca.mtimes(Xk, Xref_k_inv)
        log_Xerr = log_se3(X_err)
        cost_att = log_Xerr.T @ ca.DM(self.QXn) @ log_Xerr

        xi_diff  = xik - ca.DM(xiref_k.reshape((6,1)))
        cost_xi  = xi_diff.T  @ ca.DM(self.QXin) @ xi_diff
        
        cost_expr += cost_att + cost_xi

        opti.minimize(cost_expr)

        # ---------------------------
        # 5) 配置IPOPT, 求解
        # ---------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm,
            # "linear_solver": "ma86",
            # "hsllib": "libcoinhsl.so"
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ---------------------------
        # 6) 解析解 & 返回
        # ---------------------------
        X_sol = []
        xi_sol= []
        u_sol = []

        for k in range(Nsim+1):
            X_sol_k  = sol.value(X_vars[k])
            xi_sol_k = sol.value(xi_vars[k]).ravel()   # 6,
            X_sol.append(X_sol_k)
            xi_sol.append(xi_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # xs: [ [X_sol[0], xi_sol[0]], ..., [X_sol[N], xi_sol[N]] ]
        xs = []
        for k in range(Nsim+1):
            xs.append([ X_sol[k], xi_sol[k] ])

        us = np.array(u_sol).reshape(Nsim, 4)

        # 若需要从 ipopt 读取迭代历史(与 iLQR 的 iteration 不同), 
        # 直接从 stats() 提取:
        stats = sol.stats()
        iter_stats = stats.get('iterations', {})
        J_hist     = iter_stats.get('obj', [])
        grad_hist  = iter_stats.get('inf_du', [])
        defect_hist= iter_stats.get('inf_pr', [])

        return xs, us, J_hist, grad_hist, defect_hist



class EmbeddedEuclideanSE3_DynamicsConstr(BaseController):
    """
    Baseline (SE3): Embedded Euclidean method.
    在 R^(4x4) 上直接进行优化, 不对 SE(3) 施加硬约束, 
    允许通过 cost 拉近到参考姿态 + 动力学方程等式约束.
    """

    def __init__(self, 
                 q_ref,             # ndarray/list of (N+1, 4,4) 参考位姿(齐次变换)
                 xi_ref,            # ndarray/list of (N+1, 6)   参考 twist (omega+v)
                 dt,                # 时间步长(或可是 list/array)
                 J,                 # 惯性/质量矩阵(6x6)
                 Q, R, P,             # 代价权重: 这里 Q 是 12x12 的块形式(或你项目中的结构)
                 eps_init=1e-2,     # 初始化时对reference的扰动
                 thold_approx=1e-9,
                 verbose=False):
        """
        Args:
            q_ref:  (N+1, 4,4), reference SE(3).
            xi_ref: (N+1, 6), reference twist.
            dt:     float or array.
            J:      (6,6).
            Q:      (12,12) or含有[姿态/位置,twist]的对角
            R:      (6,6) control cost
        """
        super(EmbeddedEuclideanSE3_DynamicsConstr, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

        self.eps_init = eps_init
        self.thold_approx = thold_approx
        
        self.QX  = Q[:6,:6]
        self.QXi = Q[6:,6:]
        self.QXn  = P[:6,:6]
        self.QXin = P[6:,6:]
        self.R_   = R

        self.alpha = self.QX[0,0]
        self.alphaN = self.QXn[0,0]

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N    = len(q_ref) - 1 

    def fit(self, 
            x0,            # 初始状态 [X0(4x4), xi0(6,)]
            us_init,       # 初始控制 guess, shape (N, 6)
            n_iterations=200, 
            tol_norm=1e-6):
        """
        返回: 
            xs:  [N+1, [X_k, xi_k]]
            us:  [N, 6]
            J_hist, grad_hist, defect_hist
        """
        Nsim = self.N
        X0, xi0 = x0[0], x0[1]
        dt   = self.dt


        # ---------------------------
        # 1) 定义优化变量
        # ---------------------------
        opti = ca.Opti()
        X_vars = []
        xi_vars= []
        u_vars = []

        for k in range(Nsim+1):
            Xk = opti.variable(4,4)
            xik= opti.variable(6,1)
            X_vars.append(Xk)
            xi_vars.append(xik)

            if k < Nsim:
                uk = opti.variable(6,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(X_vars[0], X0)
        opti.set_initial(xi_vars[0], xi0)
        for k in range(1, Nsim+1):

            q_refk_mnf = SE32manifSE3(self.q_ref[k])
            delta_q = SE3Tangent(np.ones((6,)) * self.eps_init)
            opti.set_initial(X_vars[k], q_refk_mnf.rplus(delta_q).transform())   

            opti.set_initial(xi_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])


        # ---------------------------
        # 2) 定义辅助函数
        # ---------------------------
        def ad_se3(xi):
            """
            返回 ad_{xi} (6x6)，其中 xi=[omega, v].
            ad_{[ω,v]} = [[ω^∧, 0],
                        [ v^∧, ω^∧]].
            """
            omega = xi[0:3]
            v     = xi[3:6]
            O = ca.skew(omega)
            V = ca.skew(v)
            top = ca.horzcat(O, ca.MX.zeros(3,3))
            bot = ca.horzcat(V, O)
            return ca.vertcat(top, bot)

        def adT_se3(xi):
            """
            返回 ad^*_{xi} = (ad_{xi})^T (6x6).
            常在惯性力学方程中出现: ad^*_{xi} (J xi).
            """
            return ad_se3(xi).T

        def V_so3(omega):
            """
            辅助积分矩阵 V(omega) = I + (1-cosθ)/θ^2 ω^∧ + (θ - sinθ)/θ^3 (ω^∧)^2,
            其中 θ = ||omega||.
            当 θ 非常小时，用 Taylor 展开近似.
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
            O = ca.skew(omega)
            I3 = ca.MX.eye(3)

            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))
            theta_minus_sin_over_t3 = ca.if_else(theta < self.thold_approx,
                                                1/6.0 - theta**2/120.0,
                                                (theta - ca.sin(theta))/(theta**3))
            # Todo: non-smooth?

            return I3 \
                + one_minus_cos_over_t2 * O \
                + theta_minus_sin_over_t3 * (O @ O)

        def invV_so3(omega):
            """
            V(omega) 的逆矩阵。通常可用级数或 closed-form 近似。
            这里简单地直接用 casadi 的 inv() 反演,
            如果对大规模或频繁调用有性能顾虑, 可使用解析公式或插值近似.
            """
            return ca.inv(V_so3(omega))

        def log_se3(X):
            """
            SE(3) 的对数映射: 返回 6x1 向量 [ω; v].
            完整实现: 
            1) 先从 R= X[0:3,0:3] 得到 omega (SO(3)对数)
            2) 再从 p= X[0:3,3] 得到 v = V(omega)^{-1} * p
            """
            R = X[0:3,0:3]
            p = X[0:3,3]

            # 1) 先做 SO(3) 的对数
            trace_R = R[0,0] + R[1,1] + R[2,2]
            cos_theta = 0.5*(trace_R - 1)
            cos_theta_clamped = ca.fmax(ca.fmin(cos_theta, 1), -1)
            theta = ca.acos(cos_theta_clamped)

            # skew_part = [ R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1) ]
            skew_part = ca.vertcat(
                R[2,1] - R[1,2],
                R[0,2] - R[2,0],
                R[1,0] - R[0,1]
            )
            sin_theta = ca.sin(theta)
            factor = ca.if_else(
                ca.fabs(sin_theta) < self.thold_approx,
                0.5,  # 退化近似 => ω=0
                theta/(2*sin_theta)
            )
            omega = factor * skew_part  # 3x1

            # 2) 再对平移:  p => v = V(omega)^(-1) * p
            #    V(omega)见 V_so3(omega). 这里需要先算:
            v = invV_so3(omega) @ p
            # v = V_so3(omega) @ p
            # v = p

            return ca.vertcat(omega, v)

        def exp_se3(xi, h=1.0):
            """
            SE(3) 上的指数映射, 带步长 h:
            exp_se3(xi, h) = exp( (h * xi)^\wedge ).
            其中 xi=[ω; v], 并用罗德里格斯公式+V(ω) 计算 4x4 齐次变换矩阵.
            """
            # 1) 拆出角速度和线速度, 乘以 h
            omega = xi[0:3]
            v     = xi[3:6]
            h_omega = h * omega
            h_v     = h * v

            # 2) 先算 R_exp = exp(h_omega^∧) by Rodrigues
            theta = ca.sqrt(h_omega[0]**2 + h_omega[1]**2 + h_omega[2]**2)
            Omega = ca.skew(h_omega)
            I3 = ca.MX.eye(3)

            sin_t_over_t = ca.if_else(theta < self.thold_approx,
                                    1 - theta**2/6.0,
                                    ca.sin(theta)/theta)
            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))

            R_exp = I3 + sin_t_over_t*Omega + one_minus_cos_over_t2*(Omega @ Omega)

            # 3) 平移部分 p_exp = V(h_omega) * (h_v)
            V_exp = V_so3(h_omega)
            p_exp = V_exp @ h_v

            # 4) 拼回 4x4
            X_exp = ca.vertcat(
                ca.horzcat(R_exp, p_exp),
                ca.horzcat(ca.MX.zeros(1,3), ca.MX.ones(1,1))
            )
            return X_exp

        def inv_se3(X):
            """
            给定一个 4x4 的 SE(3) 齐次变换矩阵 X = [[R, p],
                                                    [0, 1]],
            直接返回它的逆:  [[R^T, -R^T p],
                            [ 0,       1   ]].
            注意: 如果 X 不是合法的 SE(3) 矩阵（R 非正交、det != 1 等），则此公式不成立。
            """
            R = X[0:3, 0:3]
            p = X[0:3, 3]

            # R^T
            R_T = R.T

            # -R^T pd
            minus_RTp = -ca.mtimes(R_T, p)

            # 拼回 4x4
            X_inv = ca.vertcat(
                ca.horzcat(R_T, minus_RTp),
                ca.horzcat(ca.DM.zeros(1,3), ca.DM.ones(1,1))
            )
            return X_inv
        
        # ---------------------------
        # 3) 动力学约束 + 初始条件
        # ---------------------------
        J_inv = self.J_inv

        # 初始
        opti.subject_to(X_vars[0] - X0 == 0)
        opti.subject_to(xi_vars[0] - xi0 == 0)

        # 每个时刻
        for k in range(Nsim):
            Xk    = X_vars[k]
            Xk1   = X_vars[k+1]
            xik   = xi_vars[k]
            xik1  = xi_vars[k+1]
            uk    = u_vars[k]

            # X{k+1} = Xk * exp_se3(xi_k, dt)
            Xk_prop = ca.mtimes(Xk, exp_se3(xik, dt))
            opti.subject_to(Xk1 - Xk_prop == 0)

            # xi{k+1} = xi_k + dt*J^-1( ad^*_{xi_k}(J xi_k) + u_k )
            left  = xik1
            right = xik + dt*ca.mtimes(J_inv, (adT_se3(xik) @ (self.J @ xik) + uk))
            opti.subject_to(left - right == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）

        # ---------------------------
        # 4) 构建目标函数
        # ---------------------------
        cost_expr = 0

        for k in range(Nsim):
            Xk = X_vars[k]
            xik= xi_vars[k]
            uk = u_vars[k]

            Xref_k  = self.q_ref[k]
            xiref_k = self.xi_ref[k]

            X_diff = Xk - ca.DM(Xref_k)
            
            xi_diff = xik - ca.DM(xiref_k.reshape((6,1)))
            
            # [X_err(6x1)]^T QX [X_err(6x1)] + [xi_diff(6x1)]^T QXi [xi_diff(6x1)] + ...
            cost_att = self.alpha * ca.sumsqr(X_diff)
            cost_xi  = xi_diff.T  @ ca.DM(self.QXi) @ xi_diff
            cost_u   = uk.T @ ca.DM(self.R_) @ uk

            cost_expr += cost_att + cost_xi + cost_u
        
        # 末端项
        Xk    = X_vars[Nsim]
        xik   = xi_vars[Nsim]
        Xref_k = self.q_ref[Nsim]
        xiref_k= self.xi_ref[Nsim]

        X_diff = Xk - ca.DM(Xref_k)
        cost_att = self.alphaN * ca.sumsqr(X_diff)

        xi_diff  = xik - ca.DM(xiref_k.reshape((6,1)))
        cost_xi  = xi_diff.T  @ ca.DM(self.QXin) @ xi_diff
        
        cost_expr += cost_att + cost_xi

        opti.minimize(cost_expr)

        # ---------------------------
        # 5) 配置IPOPT, 求解
        # ---------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm,
            # "linear_solver": "ma86",
            # "hsllib": "libcoinhsl.so"
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ---------------------------
        # 6) 解析解 & 返回
        # ---------------------------
        X_sol = []
        xi_sol= []
        u_sol = []

        for k in range(Nsim+1):
            X_sol_k  = sol.value(X_vars[k])
            xi_sol_k = sol.value(xi_vars[k]).ravel()   # 6,
            X_sol.append(X_sol_k)
            xi_sol.append(xi_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # xs: [ [X_sol[0], xi_sol[0]], ..., [X_sol[N], xi_sol[N]] ]
        xs = []
        for k in range(Nsim+1):
            xs.append([ X_sol[k], xi_sol[k] ])

        us = np.array(u_sol).reshape(Nsim, 6)

        # 若需要从 ipopt 读取迭代历史(与 iLQR 的 iteration 不同), 
        # 直接从 stats() 提取:
        stats = sol.stats()
        iter_stats = stats.get('iterations', {})
        J_hist     = iter_stats.get('obj', [])
        grad_hist  = iter_stats.get('inf_du', [])
        defect_hist= iter_stats.get('inf_pr', [])

        return xs, us, J_hist, grad_hist, defect_hist



class EmbeddedEuclideanSE3_DynamicsConstr_Drone(BaseController):
    """
    Baseline (SE3): Embedded Euclidean method.
    在 R^(4x4) 上直接进行优化, 不对 SE(3) 施加硬约束, 
    允许通过 cost 拉近到参考姿态 + 动力学方程等式约束.
    """

    def __init__(self, 
                 q_ref,             # ndarray/list of (N+1, 4,4) 参考位姿(齐次变换)
                 xi_ref,            # ndarray/list of (N+1, 6)   参考 twist (omega+v)
                 dt,                # 时间步长(或可是 list/array)
                 J,                 # 惯性/质量矩阵(6x6)
                 Q, R, P,           # 代价权重: 这里 Q 是 12x12 的块形式(或你项目中的结构)
                 eps_init=1e-2,     # 初始化时对reference的扰动
                 thold_approx=1e-9,
                 verbose=False):
        """
        Args:
            q_ref:  (N+1, 4,4), reference SE(3).
            xi_ref: (N+1, 6), reference twist.
            dt:     float or array.
            J:      (6,6).
            Q:      (12,12) or含有[姿态/位置,twist]的对角
            R:      (6,6) control cost
        """
        super(EmbeddedEuclideanSE3_DynamicsConstr_Drone, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)
        self.Ib = J[0:3, 0:3] 
        self.m = J[4,4]
        self.g = 9.8

        self.eps_init = eps_init
        self.thold_approx = thold_approx
        
        self.QX  = Q[:6,:6]
        self.QXi = Q[6:,6:]
        self.QXn  = P[:6,:6]
        self.QXin = P[6:,6:]
        self.R_   = R

        self.alpha = self.QX[0,0]
        self.alphaN = self.QXn[0,0]

        self.Pu = np.zeros((6,4))
        self.Pu[0,0] = 1.
        self.Pu[1,1] = 1.
        self.Pu[2,2] = 1.
        self.Pu[5,3] = 1.      

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N    = len(q_ref) - 1 

    def fit(self, 
            x0,            # 初始状态 [X0(4x4), xi0(6,)]
            us_init,       # 初始控制 guess, shape (N, 6)
            n_iterations=200, 
            tol_norm=1e-6):
        """
        返回: 
            xs:  [N+1, [X_k, xi_k]]
            us:  [N, 6]
            J_hist, grad_hist, defect_hist
        """
        Nsim = self.N
        X0, xi0 = x0[0], x0[1]
        dt   = self.dt


        # ---------------------------
        # 1) 定义优化变量
        # ---------------------------
        opti = ca.Opti()
        X_vars = []
        xi_vars= []
        u_vars = []

        for k in range(Nsim+1):
            Xk = opti.variable(4,4)
            xik= opti.variable(6,1)
            X_vars.append(Xk)
            xi_vars.append(xik)

            if k < Nsim:
                uk = opti.variable(4,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(X_vars[0], X0)
        opti.set_initial(xi_vars[0], xi0)
        for k in range(1, Nsim+1):

            q_refk_mnf = SE32manifSE3(self.q_ref[k])
            delta_q = SE3Tangent(np.ones((6,)) * self.eps_init)
            opti.set_initial(X_vars[k], q_refk_mnf.rplus(delta_q).transform())   

            opti.set_initial(xi_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])


        # ---------------------------
        # 2) 定义辅助函数
        # ---------------------------
        def ad_se3(xi):
            """
            返回 ad_{xi} (6x6)，其中 xi=[omega, v].
            ad_{[ω,v]} = [[ω^∧, 0],
                        [ v^∧, ω^∧]].
            """
            omega = xi[0:3]
            v     = xi[3:6]
            O = ca.skew(omega)
            V = ca.skew(v)
            top = ca.horzcat(O, ca.MX.zeros(3,3))
            bot = ca.horzcat(V, O)
            return ca.vertcat(top, bot)

        def adT_se3(xi):
            """
            返回 ad^*_{xi} = (ad_{xi})^T (6x6).
            常在惯性力学方程中出现: ad^*_{xi} (J xi).
            """
            return ad_se3(xi).T

        def V_so3(omega):
            """
            辅助积分矩阵 V(omega) = I + (1-cosθ)/θ^2 ω^∧ + (θ - sinθ)/θ^3 (ω^∧)^2,
            其中 θ = ||omega||.
            当 θ 非常小时，用 Taylor 展开近似.
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
            O = ca.skew(omega)
            I3 = ca.MX.eye(3)

            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))
            theta_minus_sin_over_t3 = ca.if_else(theta < self.thold_approx,
                                                1/6.0 - theta**2/120.0,
                                                (theta - ca.sin(theta))/(theta**3))
            # Todo: non-smooth?

            return I3 \
                + one_minus_cos_over_t2 * O \
                + theta_minus_sin_over_t3 * (O @ O)

        def invV_so3(omega):
            """
            V(omega) 的逆矩阵。通常可用级数或 closed-form 近似。
            这里简单地直接用 casadi 的 inv() 反演,
            如果对大规模或频繁调用有性能顾虑, 可使用解析公式或插值近似.
            """
            return ca.inv(V_so3(omega))

        def exp_se3(xi, h=1.0):
            """
            SE(3) 上的指数映射, 带步长 h:
            exp_se3(xi, h) = exp( (h * xi)^\wedge ).
            其中 xi=[ω; v], 并用罗德里格斯公式+V(ω) 计算 4x4 齐次变换矩阵.
            """
            # 1) 拆出角速度和线速度, 乘以 h
            omega = xi[0:3]
            v     = xi[3:6]
            h_omega = h * omega
            h_v     = h * v

            # 2) 先算 R_exp = exp(h_omega^∧) by Rodrigues
            theta = ca.sqrt(h_omega[0]**2 + h_omega[1]**2 + h_omega[2]**2)
            Omega = ca.skew(h_omega)
            I3 = ca.MX.eye(3)

            sin_t_over_t = ca.if_else(theta < self.thold_approx,
                                    1 - theta**2/6.0,
                                    ca.sin(theta)/theta)
            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))

            R_exp = I3 + sin_t_over_t*Omega + one_minus_cos_over_t2*(Omega @ Omega)

            # 3) 平移部分 p_exp = V(h_omega) * (h_v)
            V_exp = V_so3(h_omega)
            p_exp = V_exp @ h_v

            # 4) 拼回 4x4
            X_exp = ca.vertcat(
                ca.horzcat(R_exp, p_exp),
                ca.horzcat(ca.MX.zeros(1,3), ca.MX.ones(1,1))
            )
            return X_exp
        
        # ---------------------------
        # 3) 动力学约束 + 初始条件
        # ---------------------------
        J_inv = self.J_inv

        # 初始
        opti.subject_to(X_vars[0] - X0 == 0)
        opti.subject_to(xi_vars[0] - xi0 == 0)

        # 每个时刻
        for k in range(Nsim):
            Xk    = X_vars[k]
            Xk1   = X_vars[k+1]
            xik   = xi_vars[k]
            xik1  = xi_vars[k+1]
            uk    = u_vars[k]

            # X{k+1} = Xk * exp_se3(xi_k, dt)
            Xk_prop = ca.mtimes(Xk, exp_se3(xik, dt))
            opti.subject_to(Xk1 - Xk_prop == 0)

            # xi{k+1} = xi_k + dt*J^-1( ad^*_{xi_k}(J xi_k) + u_k )
            
            down_vec = np.array([[0],[0],[-1]])
            Rk  = Xk[:3,:3]
            g_acc = self.m * self.g * Rk.T @ down_vec
            g_acc = ca.veccat( np.zeros((3,1)), g_acc )
            
            left  = xik1
            right = xik + dt*ca.mtimes(J_inv, (adT_se3(xik) @ (self.J @ xik) + g_acc + self.Pu @ uk))

            opti.subject_to(left - right == 0)

            # （如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）

        # ---------------------------
        # 4) 构建目标函数
        # ---------------------------
        cost_expr = 0

        for k in range(Nsim):
            Xk = X_vars[k]
            xik= xi_vars[k]
            uk = u_vars[k]

            Xref_k  = self.q_ref[k]
            xiref_k = self.xi_ref[k]

            X_diff = Xk - ca.DM(Xref_k)
            
            xi_diff = xik - ca.DM(xiref_k.reshape((6,1)))
            
            # [X_err(6x1)]^T QX [X_err(6x1)] + [xi_diff(6x1)]^T QXi [xi_diff(6x1)] + ...
            cost_att = self.alpha * ca.sumsqr(X_diff)
            cost_xi  = xi_diff.T  @ ca.DM(self.QXi) @ xi_diff
            cost_u   = uk.T @ ca.DM(self.R_) @ uk

            cost_expr += cost_att + cost_xi + cost_u
        
        # 末端项
        Xk    = X_vars[Nsim]
        xik   = xi_vars[Nsim]
        Xref_k = self.q_ref[Nsim]
        xiref_k= self.xi_ref[Nsim]

        X_diff = Xk - ca.DM(Xref_k)
        cost_att = self.alphaN * ca.sumsqr(X_diff)

        xi_diff  = xik - ca.DM(xiref_k.reshape((6,1)))
        cost_xi  = xi_diff.T  @ ca.DM(self.QXin) @ xi_diff
        
        cost_expr += cost_att + cost_xi

        opti.minimize(cost_expr)

        # ---------------------------
        # 5) 配置IPOPT, 求解
        # ---------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm,
            # "linear_solver": "ma86",
            # "hsllib": "libcoinhsl.so"
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ---------------------------
        # 6) 解析解 & 返回
        # ---------------------------
        X_sol = []
        xi_sol= []
        u_sol = []

        for k in range(Nsim+1):
            X_sol_k  = sol.value(X_vars[k])
            xi_sol_k = sol.value(xi_vars[k]).ravel()   # 6,
            X_sol.append(X_sol_k)
            xi_sol.append(xi_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # xs: [ [X_sol[0], xi_sol[0]], ..., [X_sol[N], xi_sol[N]] ]
        xs = []
        for k in range(Nsim+1):
            xs.append([ X_sol[k], xi_sol[k] ])

        us = np.array(u_sol).reshape(Nsim, 4)

        # 若需要从 ipopt 读取迭代历史(与 iLQR 的 iteration 不同), 
        # 直接从 stats() 提取:
        stats = sol.stats()
        iter_stats = stats.get('iterations', {})
        J_hist     = iter_stats.get('obj', [])
        grad_hist  = iter_stats.get('inf_du', [])
        defect_hist= iter_stats.get('inf_pr', [])

        return xs, us, J_hist, grad_hist, defect_hist



# Doesn't work
class ConstraintStabilizationSE3(BaseController):
    """
    Baseline (SE3): Embedded Euclidean method.
    在 R^(4x4) 上直接进行优化, 不对 SE(3) 施加硬约束, 
    允许通过 cost 拉近到参考姿态 + 动力学方程等式约束.
    """

    def __init__(self, 
                 q_ref,             # ndarray/list of (N+1, 4,4) 参考位姿(齐次变换)
                 xi_ref,            # ndarray/list of (N+1, 6)   参考 twist (omega+v)
                 dt,                # 时间步长(或可是 list/array)
                 J,                 # 惯性/质量矩阵(6x6)
                 Q, R,              # 代价权重: 这里 Q 是 12x12 的块形式(或你项目中的结构)
                 eps_init=1e-2,     # 初始化时对reference的扰动
                 kappa=1e0,
                 thold_approx=1e-9,
                 verbose=False):
        """
        Args:
            q_ref:  (N+1, 4,4), reference SE(3).
            xi_ref: (N+1, 6), reference twist.
            dt:     float or array.
            J:      (6,6).
            Q:      (12,12) or含有[姿态/位置,twist]的对角
            R:      (6,6) control cost
        """
        super(ConstraintStabilizationSE3, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

        self.eps_init = eps_init
        self.thold_approx = thold_approx
        self.kappa = kappa
        
        # Weight Matrics
        self.QX  = Q[:6,:6]
        self.QXi = Q[6:,6:]
        self.QXn  = 10.0 * self.QX
        self.QXin = 10.0 * self.QXi
        self.R_   = R

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N    = len(q_ref) - 1 

    def fit(self, 
            x0,            # 初始状态 [X0(4x4), xi0(6,)]
            us_init,       # 初始控制 guess, shape (N, 6)
            n_iterations=200, 
            tol_norm=1e-6):
        """
        返回: 
            xs:  [N+1, [X_k, xi_k]]
            us:  [N, 6]
            J_hist, grad_hist, defect_hist
        """
        # ----------------------------
        # 0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        X0, xi0 = x0[0], x0[1]
        dt   = self.dt

        # ---------------------------
        # 1) 定义优化变量
        # ---------------------------
        opti = ca.Opti()
        X_vars = []
        xi_vars= []
        u_vars = []

        for k in range(Nsim+1):
            Xk = opti.variable(4,4)
            xik= opti.variable(6,1)
            X_vars.append(Xk)
            xi_vars.append(xik)

            if k < Nsim:
                uk = opti.variable(6,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(X_vars[0], X0)
        opti.set_initial(xi_vars[0], xi0)
        for k in range(1, Nsim+1):
            q_refk_mnf = SE32manifSE3(self.q_ref[k])
            delta_q = SE3Tangent(np.ones((6,)) * self.eps_init)
            opti.set_initial(X_vars[k], q_refk_mnf.rplus(delta_q).transform())   
            opti.set_initial(xi_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])


        # ---------------------------
        # 2) 定义辅助函数
        # ---------------------------
        def ad_se3(xi):
            """
            返回 ad_{xi} (6x6)，其中 xi=[omega, v].
            ad_{[ω,v]} = [[ω^∧, 0],
                        [ v^∧, ω^∧]].
            """
            omega = xi[0:3]
            v     = xi[3:6]
            O = ca.skew(omega)
            V = ca.skew(v)
            top = ca.horzcat(O, ca.MX.zeros(3,3))
            bot = ca.horzcat(V, O)
            return ca.vertcat(top, bot)

        def adT_se3(xi):
            """
            返回 ad^*_{xi} = (ad_{xi})^T (6x6).
            常在惯性力学方程中出现: ad^*_{xi} (J xi).
            """
            return ad_se3(xi).T

        def V_so3(omega):
            """
            辅助积分矩阵 V(omega) = I + (1-cosθ)/θ^2 ω^∧ + (θ - sinθ)/θ^3 (ω^∧)^2,
            其中 θ = ||omega||.
            当 θ 非常小时，用 Taylor 展开近似.
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
            O = ca.skew(omega)
            I3 = ca.MX.eye(3)

            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))
            theta_minus_sin_over_t3 = ca.if_else(theta < self.thold_approx,
                                                1/6.0 - theta**2/120.0,
                                                (theta - ca.sin(theta))/(theta**3))
            # Todo: non-smooth?

            return I3 \
                + one_minus_cos_over_t2 * O \
                + theta_minus_sin_over_t3 * (O @ O)

        def invV_so3(omega):
            """
            V(omega) 的逆矩阵。通常可用级数或 closed-form 近似。
            这里简单地直接用 casadi 的 inv() 反演,
            如果对大规模或频繁调用有性能顾虑, 可使用解析公式或插值近似.
            """
            return ca.inv(V_so3(omega))

        def log_se3(X):
            """
            SE(3) 的对数映射: 返回 6x1 向量 [ω; v].
            完整实现: 
            1) 先从 R= X[0:3,0:3] 得到 omega (SO(3)对数)
            2) 再从 p= X[0:3,3] 得到 v = V(omega)^{-1} * p
            """
            R = X[0:3,0:3]
            p = X[0:3,3]

            # 1) 先做 SO(3) 的对数
            trace_R = R[0,0] + R[1,1] + R[2,2]
            cos_theta = 0.5*(trace_R - 1)
            cos_theta_clamped = ca.fmax(ca.fmin(cos_theta, 1), -1)
            theta = ca.acos(cos_theta_clamped)

            # skew_part = [ R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1) ]
            skew_part = ca.vertcat(
                R[2,1] - R[1,2],
                R[0,2] - R[2,0],
                R[1,0] - R[0,1]
            )
            sin_theta = ca.sin(theta)
            factor = ca.if_else(
                ca.fabs(sin_theta) < self.thold_approx,
                0.5,  # 退化近似 => ω=0
                theta/(2*sin_theta)
            )
            omega = factor * skew_part  # 3x1

            # 2) 再对平移:  p => v = V(omega)^(-1) * p
            #    V(omega)见 V_so3(omega). 这里需要先算:
            v = invV_so3(omega) @ p
            # v = V_so3(omega) @ p
            # v = p

            return ca.vertcat(omega, v)

        def exp_se3(xi, h=1.0):
            """
            SE(3) 上的指数映射, 带步长 h:
            exp_se3(xi, h) = exp( (h * xi)^\wedge ).
            其中 xi=[ω; v], 并用罗德里格斯公式+V(ω) 计算 4x4 齐次变换矩阵.
            """
            # 1) 拆出角速度和线速度, 乘以 h
            omega = xi[0:3]
            v     = xi[3:6]
            h_omega = h * omega
            h_v     = h * v

            # 2) 先算 R_exp = exp(h_omega^∧) by Rodrigues
            theta = ca.sqrt(h_omega[0]**2 + h_omega[1]**2 + h_omega[2]**2)
            Omega = ca.skew(h_omega)
            I3 = ca.MX.eye(3)

            sin_t_over_t = ca.if_else(theta < self.thold_approx,
                                    1 - theta**2/6.0,
                                    ca.sin(theta)/theta)
            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))

            R_exp = I3 + sin_t_over_t*Omega + one_minus_cos_over_t2*(Omega @ Omega)

            # 3) 平移部分 p_exp = V(h_omega) * (h_v)
            V_exp = V_so3(h_omega)
            p_exp = V_exp @ h_v

            # 4) 拼回 4x4
            X_exp = ca.vertcat(
                ca.horzcat(R_exp, p_exp),
                ca.horzcat(ca.MX.zeros(1,3), ca.MX.ones(1,1))
            )
            return X_exp

        def inv_se3(X):
            """
            给定一个 4x4 的 SE(3) 齐次变换矩阵 X = [[R, p],
                                                    [0, 1]],
            直接返回它的逆:  [[R^T, -R^T p],
                            [ 0,       1   ]].
            注意: 如果 X 不是合法的 SE(3) 矩阵（R 非正交、det != 1 等），则此公式不成立。
            """
            R = X[0:3, 0:3]
            p = X[0:3, 3]

            # R^T
            R_T = R.T

            # -R^T pd
            minus_RTp = -ca.mtimes(R_T, p)

            # 拼回 4x4
            X_inv = ca.vertcat(
                ca.horzcat(R_T, minus_RTp),
                ca.horzcat(ca.DM.zeros(1,3), ca.DM.ones(1,1))
            )
            return X_inv
        
        # ---------------------------
        # 3) 动力学约束 + 初始条件
        # ---------------------------
        J_inv = self.J_inv

        # 初始
        opti.subject_to(X_vars[0] - X0 == 0)
        opti.subject_to(xi_vars[0] - xi0 == 0)

        # 每个时刻
        for k in range(Nsim):
            Xk    = X_vars[k]
            Xk1   = X_vars[k+1]
            xik   = xi_vars[k]
            xik1  = xi_vars[k+1]
            uk    = u_vars[k]

            # 1. X{k+1} = Xk * exp_se3(xi_k, dt)
            Rk = Xk[:3,:3]
            stab_term = self.kappa/2 * ( ca.inv(ca.mtimes(ca.transpose(Rk), Rk)) - ca.DM.eye(3))
            stab_term = ca.vertcat(
                ca.horzcat( stab_term, ca.DM.zeros((3,1)) ),
                ca.reshape([0,0,0,1], 1, 4)
            )
            Xk_prop = ca.mtimes(Xk, 
                                exp_se3(xik, dt) + stab_term )
            # Xk_prop = ca.mtimes(Xk, 
            #                     exp_se3(xik, dt))
            opti.subject_to(Xk1 - Xk_prop == 0)

            # 2. xi{k+1} = xi_k + dt*J^-1( ad^*_{xi_k}(J xi_k) + u_k )
            left  = xik1
            right = xik + dt*ca.mtimes(J_inv, (adT_se3(xik) @ (self.J @ xik) + uk))
            opti.subject_to(left - right == 0)

            # 3.（如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）

            # 4. SE3
            opti.subject_to( Xk[3,:3] == 0 )

        # ---------------------------
        # 4) 构建目标函数
        # ---------------------------
        cost_expr = 0

        for k in range(Nsim):
            Xk = X_vars[k]
            xik= xi_vars[k]
            uk = u_vars[k]

            Xref_k  = self.q_ref[k]
            xiref_k = self.xi_ref[k]

            Xref_k_inv = inv_se3(Xref_k)
            X_err = ca.mtimes(Xk, Xref_k_inv)    # 4x4
            log_Xerr = log_se3(X_err)           # 6x1
            
            xi_diff = xik - ca.DM(xiref_k.reshape((6,1)))
            
            # [X_err(6x1)]^T QX [X_err(6x1)] + [xi_diff(6x1)]^T QXi [xi_diff(6x1)] + ...
            cost_att = log_Xerr.T @ ca.DM(self.QX) @ log_Xerr
            cost_xi  = xi_diff.T  @ ca.DM(self.QXi) @ xi_diff
            cost_u   = uk.T @ ca.DM(self.R_) @ uk

            cost_expr += cost_att + cost_xi + cost_u
        
        # 末端项
        Xk    = X_vars[Nsim]
        xik   = xi_vars[Nsim]
        Xref_k = self.q_ref[Nsim]
        xiref_k= self.xi_ref[Nsim]

        Xref_k_inv = inv_se3(Xref_k)
        X_err = ca.mtimes(Xk, Xref_k_inv)
        log_Xerr = log_se3(X_err)
        cost_att = log_Xerr.T @ ca.DM(self.QXn) @ log_Xerr

        xi_diff  = xik - ca.DM(xiref_k.reshape((6,1)))
        cost_xi  = xi_diff.T  @ ca.DM(self.QXin) @ xi_diff
        
        cost_expr += cost_att + cost_xi

        opti.minimize(cost_expr)

        # ---------------------------
        # 5) 配置IPOPT, 求解
        # ---------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm,
            # "linear_solver": "ma97",
            # "hsllib": "libcoinhsl.so"
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ---------------------------
        # 6) 解析解 & 返回
        # ---------------------------
        X_sol = []
        xi_sol= []
        u_sol = []

        for k in range(Nsim+1):
            X_sol_k  = sol.value(X_vars[k])
            xi_sol_k = sol.value(xi_vars[k]).ravel()   # 6,
            X_sol.append(X_sol_k)
            xi_sol.append(xi_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # xs: [ [X_sol[0], xi_sol[0]], ..., [X_sol[N], xi_sol[N]] ]
        xs = []
        for k in range(Nsim+1):
            xs.append([ X_sol[k], xi_sol[k] ])

        us = np.array(u_sol).reshape(Nsim, 6)

        # 若需要从 ipopt 读取迭代历史(与 iLQR 的 iteration 不同), 
        # 直接从 stats() 提取:
        stats = sol.stats()
        iter_stats = stats.get('iterations', {})
        J_hist     = iter_stats.get('obj', [])
        grad_hist  = iter_stats.get('inf_du', [])
        defect_hist= iter_stats.get('inf_pr', [])

        return xs, us, J_hist, grad_hist, defect_hist
    

# Doesn't work
class ConstraintStabilizationSE3_MatrixNorm(BaseController):
    """
    Baseline (SE3): Embedded Euclidean method.
    在 R^(4x4) 上直接进行优化, 不对 SE(3) 施加硬约束, 
    允许通过 cost 拉近到参考姿态 + 动力学方程等式约束.
    """

    def __init__(self, 
                 q_ref,             # ndarray/list of (N+1, 4,4) 参考位姿(齐次变换)
                 xi_ref,            # ndarray/list of (N+1, 6)   参考 twist (omega+v)
                 dt,                # 时间步长(或可是 list/array)
                 J,                 # 惯性/质量矩阵(6x6)
                 Q, R,              # 代价权重: 这里 Q 是 12x12 的块形式(或你项目中的结构)
                 eps_init=1e-2,     # 初始化时对reference的扰动
                 kappa=1e0,
                 thold_approx=1e-9,
                 verbose=False):
        """
        Args:
            q_ref:  (N+1, 4,4), reference SE(3).
            xi_ref: (N+1, 6), reference twist.
            dt:     float or array.
            J:      (6,6).
            Q:      (12,12) or含有[姿态/位置,twist]的对角
            R:      (6,6) control cost
        """
        super(ConstraintStabilizationSE3_MatrixNorm, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

        self.eps_init = eps_init
        self.thold_approx = thold_approx
        self.kappa = kappa
        
        self.QX  = Q[:6,:6]
        self.QXi = Q[6:,6:]
        self.QXn  = 10.0 * self.QX
        self.QXin = 10.0 * self.QXi
        self.R_   = R

        self.alpha = self.QX[0,0]
        self.alphaN = 10 * self.alpha

        # IPOPT solver setup
        self.verbose = verbose

        # Horizon
        self.N    = len(q_ref) - 1 

    def fit(self, 
            x0,            # 初始状态 [X0(4x4), xi0(6,)]
            us_init,       # 初始控制 guess, shape (N, 6)
            n_iterations=200, 
            tol_norm=1e-6):
        """
        返回: 
            xs:  [N+1, [X_k, xi_k]]
            us:  [N, 6]
            J_hist, grad_hist, defect_hist
        """
        # ----------------------------
        # 0) 一些准备工作
        # ----------------------------
        Nsim = self.N
        X0, xi0 = x0[0], x0[1]
        dt   = self.dt

        # ---------------------------
        # 1) 定义优化变量
        # ---------------------------
        opti = ca.Opti()
        X_vars = []
        xi_vars= []
        u_vars = []

        for k in range(Nsim+1):
            Xk = opti.variable(4,4)
            xik= opti.variable(6,1)
            X_vars.append(Xk)
            xi_vars.append(xik)

            if k < Nsim:
                uk = opti.variable(6,1)
                u_vars.append(uk)

        # 设置初值
        opti.set_initial(X_vars[0], X0)
        opti.set_initial(xi_vars[0], xi0)
        for k in range(1, Nsim+1):
            q_refk_mnf = SE32manifSE3(self.q_ref[k])
            delta_q = SE3Tangent(np.ones((6,)) * self.eps_init)
            opti.set_initial(X_vars[k], q_refk_mnf.rplus(delta_q).transform())   
            opti.set_initial(xi_vars[k], self.xi_ref[k])
        for k in range(Nsim):
            opti.set_initial(u_vars[k], us_init[k])


        # ---------------------------
        # 2) 定义辅助函数
        # ---------------------------
        def ad_se3(xi):
            """
            返回 ad_{xi} (6x6)，其中 xi=[omega, v].
            ad_{[ω,v]} = [[ω^∧, 0],
                        [ v^∧, ω^∧]].
            """
            omega = xi[0:3]
            v     = xi[3:6]
            O = ca.skew(omega)
            V = ca.skew(v)
            top = ca.horzcat(O, ca.MX.zeros(3,3))
            bot = ca.horzcat(V, O)
            return ca.vertcat(top, bot)

        def adT_se3(xi):
            """
            返回 ad^*_{xi} = (ad_{xi})^T (6x6).
            常在惯性力学方程中出现: ad^*_{xi} (J xi).
            """
            return ad_se3(xi).T

        def V_so3(omega):
            """
            辅助积分矩阵 V(omega) = I + (1-cosθ)/θ^2 ω^∧ + (θ - sinθ)/θ^3 (ω^∧)^2,
            其中 θ = ||omega||.
            当 θ 非常小时，用 Taylor 展开近似.
            """
            theta = ca.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
            O = ca.skew(omega)
            I3 = ca.MX.eye(3)

            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))
            theta_minus_sin_over_t3 = ca.if_else(theta < self.thold_approx,
                                                1/6.0 - theta**2/120.0,
                                                (theta - ca.sin(theta))/(theta**3))
            # Todo: non-smooth?

            return I3 \
                + one_minus_cos_over_t2 * O \
                + theta_minus_sin_over_t3 * (O @ O)

        def invV_so3(omega):
            """
            V(omega) 的逆矩阵。通常可用级数或 closed-form 近似。
            这里简单地直接用 casadi 的 inv() 反演,
            如果对大规模或频繁调用有性能顾虑, 可使用解析公式或插值近似.
            """
            return ca.inv(V_so3(omega))

        def log_se3(X):
            """
            SE(3) 的对数映射: 返回 6x1 向量 [ω; v].
            完整实现: 
            1) 先从 R= X[0:3,0:3] 得到 omega (SO(3)对数)
            2) 再从 p= X[0:3,3] 得到 v = V(omega)^{-1} * p
            """
            R = X[0:3,0:3]
            p = X[0:3,3]

            # 1) 先做 SO(3) 的对数
            trace_R = R[0,0] + R[1,1] + R[2,2]
            cos_theta = 0.5*(trace_R - 1)
            cos_theta_clamped = ca.fmax(ca.fmin(cos_theta, 1), -1)
            theta = ca.acos(cos_theta_clamped)

            # skew_part = [ R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1) ]
            skew_part = ca.vertcat(
                R[2,1] - R[1,2],
                R[0,2] - R[2,0],
                R[1,0] - R[0,1]
            )
            sin_theta = ca.sin(theta)
            factor = ca.if_else(
                ca.fabs(sin_theta) < self.thold_approx,
                0.5,  # 退化近似 => ω=0
                theta/(2*sin_theta)
            )
            omega = factor * skew_part  # 3x1

            # 2) 再对平移:  p => v = V(omega)^(-1) * p
            #    V(omega)见 V_so3(omega). 这里需要先算:
            v = invV_so3(omega) @ p
            # v = V_so3(omega) @ p
            # v = p

            return ca.vertcat(omega, v)

        def exp_se3(xi, h=1.0):
            """
            SE(3) 上的指数映射, 带步长 h:
            exp_se3(xi, h) = exp( (h * xi)^\wedge ).
            其中 xi=[ω; v], 并用罗德里格斯公式+V(ω) 计算 4x4 齐次变换矩阵.
            """
            # 1) 拆出角速度和线速度, 乘以 h
            omega = xi[0:3]
            v     = xi[3:6]
            h_omega = h * omega
            h_v     = h * v

            # 2) 先算 R_exp = exp(h_omega^∧) by Rodrigues
            theta = ca.sqrt(h_omega[0]**2 + h_omega[1]**2 + h_omega[2]**2)
            Omega = ca.skew(h_omega)
            I3 = ca.MX.eye(3)

            sin_t_over_t = ca.if_else(theta < self.thold_approx,
                                    1 - theta**2/6.0,
                                    ca.sin(theta)/theta)
            one_minus_cos_over_t2 = ca.if_else(theta < self.thold_approx,
                                            0.5 - theta**2/24.0,
                                            (1 - ca.cos(theta))/(theta**2))

            R_exp = I3 + sin_t_over_t*Omega + one_minus_cos_over_t2*(Omega @ Omega)

            # 3) 平移部分 p_exp = V(h_omega) * (h_v)
            V_exp = V_so3(h_omega)
            p_exp = V_exp @ h_v

            # 4) 拼回 4x4
            X_exp = ca.vertcat(
                ca.horzcat(R_exp, p_exp),
                ca.horzcat(ca.MX.zeros(1,3), ca.MX.ones(1,1))
            )
            return X_exp

        def inv_se3(X):
            """
            给定一个 4x4 的 SE(3) 齐次变换矩阵 X = [[R, p],
                                                    [0, 1]],
            直接返回它的逆:  [[R^T, -R^T p],
                            [ 0,       1   ]].
            注意: 如果 X 不是合法的 SE(3) 矩阵（R 非正交、det != 1 等），则此公式不成立。
            """
            R = X[0:3, 0:3]
            p = X[0:3, 3]

            # R^T
            R_T = R.T

            # -R^T pd
            minus_RTp = -ca.mtimes(R_T, p)

            # 拼回 4x4
            X_inv = ca.vertcat(
                ca.horzcat(R_T, minus_RTp),
                ca.horzcat(ca.DM.zeros(1,3), ca.DM.ones(1,1))
            )
            return X_inv
        
        # ---------------------------
        # 3) 动力学约束 + 初始条件
        # ---------------------------
        J_inv = self.J_inv

        # 初始
        opti.subject_to(X_vars[0] - X0 == 0)
        opti.subject_to(xi_vars[0] - xi0 == 0)

        # 每个时刻
        for k in range(Nsim):
            Xk    = X_vars[k]
            Xk1   = X_vars[k+1]
            xik   = xi_vars[k]
            xik1  = xi_vars[k+1]
            uk    = u_vars[k]

            # 1. X{k+1} = Xk * exp_se3(xi_k, dt)
            Rk = Xk[:3,:3]
            stab_term = self.kappa/2 * ( ca.inv(ca.mtimes(ca.transpose(Rk), Rk)) - ca.DM.eye(3))
            stab_term = ca.vertcat(
                ca.horzcat( stab_term, ca.DM.zeros((3,1)) ),
                ca.reshape([0,0,0,1], 1, 4)
            )
            Xk_prop = ca.mtimes(Xk, 
                                exp_se3(xik, dt) + stab_term )
            # Xk_prop = ca.mtimes(Xk, 
            #                     exp_se3(xik, dt))
            opti.subject_to(Xk1 - Xk_prop == 0)

            # 2. xi{k+1} = xi_k + dt*J^-1( ad^*_{xi_k}(J xi_k) + u_k )
            left  = xik1
            right = xik + dt*ca.mtimes(J_inv, (adT_se3(xik) @ (self.J @ xik) + uk))
            opti.subject_to(left - right == 0)

            # 3.（如果要强行正交R_k^T R_k=I作为硬约束，也可加: 
            #   opti.subject_to(ca.mtimes(Rk.T, Rk) - ca.DM.eye(3) == 0)
            #   这会使问题变得更难）

            # 4. SE3
            opti.subject_to( Xk[3,:3] == 0 )

        # ---------------------------
        # 4) 构建目标函数
        # ---------------------------
        cost_expr = 0

        for k in range(Nsim):
            Xk = X_vars[k]
            xik= xi_vars[k]
            uk = u_vars[k]

            Xref_k  = self.q_ref[k]
            xiref_k = self.xi_ref[k]

            X_diff = Xk - ca.DM(Xref_k)
            
            xi_diff = xik - ca.DM(xiref_k.reshape((6,1)))
            
            # [X_err(6x1)]^T QX [X_err(6x1)] + [xi_diff(6x1)]^T QXi [xi_diff(6x1)] + ...
            cost_att = self.alpha * ca.sumsqr(X_diff)
            cost_xi  = xi_diff.T  @ ca.DM(self.QXi) @ xi_diff
            cost_u   = uk.T @ ca.DM(self.R_) @ uk

            cost_expr += cost_att + cost_xi + cost_u
        
        # 末端项
        Xk    = X_vars[Nsim]
        xik   = xi_vars[Nsim]
        Xref_k = self.q_ref[Nsim]
        xiref_k= self.xi_ref[Nsim]

        X_diff = Xk - ca.DM(Xref_k)
        cost_att = self.alphaN * ca.sumsqr(X_diff)

        xi_diff  = xik - ca.DM(xiref_k.reshape((6,1)))
        cost_xi  = xi_diff.T  @ ca.DM(self.QXin) @ xi_diff
        
        cost_expr += cost_att + cost_xi

        opti.minimize(cost_expr)

        # ---------------------------
        # 5) 配置IPOPT, 求解
        # ---------------------------
        p_opts = {"verbose": self.verbose}
        s_opts = {
            "max_iter": n_iterations,
            "tol": tol_norm,
            "acceptable_tol": tol_norm,
            # "linear_solver": "ma97",
            # "hsllib": "libcoinhsl.so"
        }
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            return None, None, [], [], []

        # ---------------------------
        # 6) 解析解 & 返回
        # ---------------------------
        X_sol = []
        xi_sol= []
        u_sol = []

        for k in range(Nsim+1):
            X_sol_k  = sol.value(X_vars[k])
            xi_sol_k = sol.value(xi_vars[k]).ravel()   # 6,
            X_sol.append(X_sol_k)
            xi_sol.append(xi_sol_k)
            if k < Nsim:
                u_sol_k = sol.value(u_vars[k]).ravel()
                u_sol.append(u_sol_k)

        # xs: [ [X_sol[0], xi_sol[0]], ..., [X_sol[N], xi_sol[N]] ]
        xs = []
        for k in range(Nsim+1):
            xs.append([ X_sol[k], xi_sol[k] ])

        us = np.array(u_sol).reshape(Nsim, 6)

        # 若需要从 ipopt 读取迭代历史(与 iLQR 的 iteration 不同), 
        # 直接从 stats() 提取:
        stats = sol.stats()
        iter_stats = stats.get('iterations', {})
        J_hist     = iter_stats.get('obj', [])
        grad_hist  = iter_stats.get('inf_du', [])
        defect_hist= iter_stats.get('inf_pr', [])

        return xs, us, J_hist, grad_hist, defect_hist
    

