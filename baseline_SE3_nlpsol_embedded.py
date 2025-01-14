import casadi as ca
import numpy as np
from scipy.spatial.transform import Rotation
from manifpy import SE3

# ================
# 1. 读取参考轨迹
# ================
dt = 0.004

path_to_reference_file = 'visualization/optimized_trajectories/path_dense_random_columns_4obj.npy'
with open(path_to_reference_file, 'rb') as f:
    q_ref = np.load(f)      # shape = (N, 4,4)
    xi_ref = np.load(f)     # shape = (N, 6)
# Nsim = q_ref.shape[0] - 1
Nsim = 50
print("Horizon of dataset is", Nsim)

# 初始状态
# q0 = q_ref[0]       # (4,4) 齐次变换矩阵
# xi0 = xi_ref[0]     # (6,)  含 [omega, v]

q0 = Rotation.from_euler('zxy', [30., 30., 30.], degrees=False).as_matrix() 
xi0 = np.ones((6,)) * 1e-10

u0 = [1.,2.,3.,4.,5.,6.]

threshold = 1e-9

# ================
# 2. 助手函数
# ================

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

    one_minus_cos_over_t2 = ca.if_else(theta < threshold,
                                       0.5 - theta**2/24.0,
                                       (1 - ca.cos(theta))/(theta**2))
    theta_minus_sin_over_t3 = ca.if_else(theta < threshold,
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
        ca.fabs(sin_theta) < threshold,
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

    sin_t_over_t = ca.if_else(theta < threshold,
                              1 - theta**2/6.0,
                              ca.sin(theta)/theta)
    one_minus_cos_over_t2 = ca.if_else(theta < threshold,
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


# ================
# 3. Slices
# ================

def gen_indices(Nsim):
    """
    为 X_k, xi_k, u_k 生成在大向量 x 里的切片（start, end）索引。
    返回: X_slices, xi_slices, u_slices
      - X_slices[k] -> (start, end) 表示 X_k 在 x 里的存放区间
      - xi_slices[k] -> ...
      - u_slices[k] -> ...
    """
    X_slices = []
    xi_slices= []
    u_slices = []

    # 当前大向量的起始位置
    offset = 0

    # 先放 X_k: 16个数
    for k in range(Nsim+1):
        X_slices.append( (offset, offset+16) )
        offset += 16

    # 再放 xi_k: 6个数
    for k in range(Nsim+1):
        xi_slices.append( (offset, offset+6) )
        offset += 6

    # 最后放 u_k: 6个数 (只有 0~Nsim-1)
    for k in range(Nsim):
        u_slices.append( (offset, offset+6) )
        offset += 6

    return X_slices, xi_slices, u_slices

def slice_as_SE3(x, s):
    """
    从大向量 x 的 slice s 里取出 16 个元素, 并 reshape 成 (4,4).
    """
    flat16 = x[s[0] : s[1]]  # 这是一维 16
    return ca.reshape(flat16, 4, 4)  # 缺省按列优先/行优先要注意, CasADi默认为列优先

def slice_as_6(x, s):
    """从 x[s[0] : s[1]] 里取出 6 维向量 (6,1)."""
    return ca.reshape(x[s[0] : s[1]], 6, 1)

def build_problem(q_ref, xi_ref, dt, Nsim):
    """
    构造 {'x': x, 'f': cost, 'g': g} 给 nlpsol 用
    """
    # 先建立符号变量 x
    nvar = 16*(Nsim+1) + 6*(Nsim+1) + 6*Nsim
    x = ca.MX.sym('x', nvar, 1)  # nvar x 1

    X_slices, xi_slices, u_slices = gen_indices(Nsim)

    # 惯性/刚体参数(仅示例)
    m = 1
    J_b_np = np.diag([0.5,0.7,0.9, m,m,m])
    J_b_inv_np = np.linalg.inv(J_b_np)
    J_b = ca.DM(J_b_np)
    J_b_inv = ca.DM(J_b_inv_np)

    # 权重(与原来一样)
    Q_X   = np.diag([10.,10.,10., 1.,1.,1.])
    Q_xi  = np.diag([1.,1.,1., 1.,1.,1.])
    Q_XN  = 10*Q_X
    Q_xiN = 10*Q_xi
    R_u   = 1e-3 * np.eye(6)

    # 准备生成 cost
    cost = 0

    # 约束容器
    g_list = []

    # =============== 动力学约束 + 初始条件 ===============
    for k in range(Nsim):
        # 取出 X_k, X_{k+1}, xi_k, xi_{k+1}, u_k
        Xk   = slice_as_SE3(x, X_slices[k])      # 4x4
        Xk1  = slice_as_SE3(x, X_slices[k+1])    # 4x4
        xik  = slice_as_6(x, xi_slices[k])       # (6,1)
        xik1 = slice_as_6(x, xi_slices[k+1])     # (6,1)
        uk   = slice_as_6(x, u_slices[k])        # (6,1)

        # (1) X_{k+1} = X_k * exp_se3(xi_k, dt)
        Xk_prop = ca.mtimes(Xk, exp_se3(xik, dt))  # 4x4
        # 这里要把 Xk1 - Xk_prop = 0 当作 16 个标量约束
        g_list.append( (Xk1 - Xk_prop).reshape((16,1)) )

        # (2) xi_{k+1} = xi_k + J_b^{-1} (ad^*_{xi_k}(J_b xi_k) + u_k) * dt
        left  = xik1
        right = xik + dt * J_b_inv @ ( adT_se3(xik) @ (J_b @ xik) + uk )
        g_list.append( (left - right) )  # (6,1)

    # (3) 初始条件
    X0   = slice_as_SE3(x, X_slices[0])
    xi0  = slice_as_6(x, xi_slices[0])
    q0_dm = ca.DM(q0)   # (4,4)
    xi0_dm= ca.DM(xi0)  # (6,)

    g_list.append( (X0 - q0_dm).reshape((16,1)) )
    g_list.append( (xi0 - xi0_dm.reshape((6,1))) )

    # =============== 构建目标函数 ===============
    # 中间阶段 0 ~ Nsim-1
    for k in range(Nsim):
        Xk  = slice_as_SE3(x, X_slices[k])
        xik = slice_as_6(x, xi_slices[k])
        uk  = slice_as_6(x, u_slices[k])

        Xref_k   = ca.DM(q_ref[k])
        xiref_k  = ca.DM(xi_ref[k])

        Xref_k_inv = inv_se3(Xref_k)
        X_err = ca.mtimes(Xk, Xref_k_inv)
        log_Xerr = log_se3(X_err)   # 6x1
        xi_diff  = xik - xiref_k

        cost_att = log_Xerr.T @ Q_X  @ log_Xerr
        cost_xi  = xi_diff.T  @ Q_xi @ xi_diff
        cost_u   = uk.T       @ R_u  @ uk
        cost += cost_att + cost_xi + cost_u

    # 末端项 Nsim
    XN   = slice_as_SE3(x, X_slices[Nsim])
    xiN  = slice_as_6(x, xi_slices[Nsim])
    Xref_N   = ca.DM(q_ref[Nsim])
    xiref_N  = ca.DM(xi_ref[Nsim])
    Xref_N_inv = inv_se3(Xref_N)

    X_errN = XN @ Xref_N_inv
    log_XerrN = log_se3(X_errN)
    xi_diffN  = xiN - xiref_N

    cost_attN = log_XerrN.T @ Q_XN  @ log_XerrN
    cost_xiN  = xi_diffN.T  @ Q_xiN @ xi_diffN
    cost += cost_attN + cost_xiN

    # 把所有 g 堆叠成一个长向量
    g = ca.vertcat(*g_list)

    # 返回给 nlpsol
    nlp_dict = {
        'x': x,
        'f': cost,
        'g': g
    }
    return nlp_dict

def pack_initial_guess(q_ref, xi_ref, q0, xi0, u0, Nsim):
    """
    根据 q_ref, xi_ref, u0 等信息，打包成 x0 (大向量)
    也可以自己另写各种更灵活的策略
    """
    X_slices, xi_slices, u_slices = gen_indices(Nsim)
    nvar = 16*(Nsim+1) + 6*(Nsim+1) + 6*Nsim
    x0 = np.zeros((nvar,))

    # 先把 X0, xi0
    # ... 也可以在这里改成 user 的任何猜测

    sX = X_slices[0]
    x0[sX[0]:sX[1]] = q0.reshape(16,order='F')
    sXi = xi_slices[0]
    x0[sXi[0]:sXi[1]] = xi0.reshape(6,order='F')

    for k in range(1,Nsim+1):
        # Xk
        sX = X_slices[k]
        x0[sX[0]:sX[1]] = q_ref[k].reshape(16,order='F')  # flatten
        # xi_k
        sXi = xi_slices[k]
        x0[sXi[0]:sXi[1]] = xi_ref[k].reshape(6,order='F')

    # 再把 u_k
    for k in range(Nsim):
        sU = u_slices[k]
        x0[sU[0]:sU[1]] = u0  # 假设都一样

    return x0

def solve_nlpsol(q_ref, xi_ref, dt, Nsim, u0):
    # 1) build nlp
    nlp_dict = build_problem(q_ref, xi_ref, dt, Nsim)

    # 2) solver options
    p_opts = {"verbose": False}
    s_opts = {"max_iter": 1000}
    solver = ca.nlpsol('solver', 'ipopt', nlp_dict, {"print_time":False, 
                                                     "ipopt": s_opts,
                                                     **p_opts})

    # 3) init guess
    x0 = pack_initial_guess(q_ref, xi_ref, u0, Nsim)

    # 4) call solver
    sol = None
    try:
        sol_raw = solver(x0=x0, lbg=0, ubg=0)
        sol = sol_raw['x'].full().flatten()  # ndarray
    except RuntimeError as e:
        print("Solver failed:", e)
        return None

    # 5) 解析解
    if sol is not None:
        # 这里把 sol (大向量) 拆回 X_sol, xi_sol, u_sol
        X_slices, xi_slices, u_slices = gen_indices(Nsim)
        X_sol = []
        xi_sol= []
        u_sol = []

        for k in range(Nsim+1):
            Xk_flat = sol[X_slices[k][0]:X_slices[k][1]]
            X_sol_k = Xk_flat.reshape((4,4), order='F')
            X_sol.append(X_sol_k)

            xi_k_flat = sol[xi_slices[k][0]:xi_slices[k][1]]
            xi_sol_k = xi_k_flat.reshape((6,))
            xi_sol.append(xi_sol_k)

            if k < Nsim:
                u_k_flat = sol[u_slices[k][0]:u_slices[k][1]]
                u_sol_k = u_k_flat.reshape((6,))
                u_sol.append(u_sol_k)

        return X_sol, xi_sol, u_sol
    else:
        return None
    

if __name__ == "__main__":
    dt = 0.004
    path_to_reference_file = 'visualization/optimized_trajectories/path_dense_random_columns_4obj.npy'
    with open(path_to_reference_file, 'rb') as f:
        q_ref = np.load(f)
        xi_ref = np.load(f)

    Nsim = 50
    print("Horizon of dataset is", Nsim)

    # 初始控制猜测
    u0 = np.array([1.,2.,3.,4.,5.,6.])

    # 直接走 solve_nlpsol
    solution = solve_nlpsol(q_ref, xi_ref, dt, Nsim, u0)
    if solution is None:
        print("No solution found.")
    else:
        X_sol, xi_sol, u_sol = solution
        print("First state:\n", X_sol[0])
        print("First twist:\n", xi_sol[0])
        print("First control:\n", u_sol[0] if len(u_sol)>0 else None)