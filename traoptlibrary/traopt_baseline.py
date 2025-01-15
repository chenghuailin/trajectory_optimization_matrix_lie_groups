# file: traopt_baseline.py

import warnings
import casadi as ca
import numpy as np

from scipy.spatial.transform import Rotation
from traoptlibrary.traopt_controller import BaseController

class EmbeddedEuclideanSO3(BaseController):
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
        Q, R,
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
        super(EmbeddedEuclideanSO3, self).__init__()

        self.q_ref = q_ref
        self.xi_ref = xi_ref
        self.dt = dt
        self.J = J
        self.J_inv = np.linalg.inv(J)

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
        cost_expr = 0

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
        s_opts = {"max_iter": n_iterations,
                  "tol": tol_norm,
                  "acceptable_tol": tol_norm}
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            sol = None

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------
        if sol is None:
            # 失败：可根据需要返回None或一些无效值
            return None, None, [], [], [], [], []
        
        # 最优解
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
        cost_expr = 0

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
        s_opts = {"max_iter": n_iterations,
                  "tol": tol_norm,
                  "acceptable_tol": tol_norm}
        opti.solver("ipopt", p_opts, s_opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            warnings.warn(f"IPOPT solver failed: {e}")
            sol = None

        # ----------------------------
        #  5) 取回解并计算一次Cost/Defect
        # ----------------------------
        if sol is None:
            # 失败：可根据需要返回None或一些无效值
            return None, None, [], [], [], [], []
        
        # 最优解
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

