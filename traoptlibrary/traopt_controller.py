import abc
import warnings
import numpy as np
import jax.numpy as jnp
import time
from jax import jit
from traoptlibrary.traopt_utilis import is_pos_def, vec_SE32quatpos, se3_vee,\
        se3_hat, SE32manifSE3, manifSE32SE3, manifse32se3, se32manifse3
from scipy.linalg import logm, inv, expm
from traoptlibrary.traopt_cost import ALConstrainedCost
from manifpy import SO3, SO3Tangent
from scipy.spatial.transform import Rotation 

class BaseController():

    """Base trajectory optimizer controller."""

    @abc.abstractmethod
    def fit(self, x0, us_init, *args, **kwargs):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            *args, **kwargs: Additional positional and key-word arguments.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        raise NotImplementedError
    

class PDViolationError(Exception):
    """Custom exception class for handling positive definite violation errors"""    
    pass

# =================================================================================
# Euclidean Space Controller
# =================================================================================

class iLQR(BaseController):

    """Finite Horizon Iterative Linear Quadratic Regulator."""

    def __init__(self, dynamics, cost, N, max_reg=1e10, hessians=False):
        """Constructs an iLQR solver.

        Args:
            dynamics: Plant dynamics.
            cost: Cost function.
            N: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._action_size = dynamics.action_size
        self._state_size = dynamics.state_size

        self._k = np.zeros((N, self._action_size))
        self._K = np.zeros((N, self._action_size, self._state_size))

        super(iLQR, self).__init__()

    def fit(self, x0, us_init, n_iterations=100, tol_J=1e-6, tol_grad_norm=1e-3,
             on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0
        # self._mu = 0.0
        self._delta = self._delta_0

        # Add time 
        start_time = time.perf_counter()

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-np.arange(10)**2)

        us = us_init.copy()
        k = self._k
        K = self._K

        J_hist = []
        xs_hist = []
        us_hist = []
        # xs_hist = np.array()
        # us_hist = np.array()

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Start Iteration:", iteration, ", Used Time:", time_calc )

            # Forward rollout only if it needs to be recomputed.
            if changed:
                (xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux,
                 F_uu) = self._forward_rollout(x0, us)
                J_opt = L.sum()
                changed = False
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            # print("Iteration:", iteration, "Dynamics Rollout Finished, Used Time:", time_calc )

            try:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                           F_xx, F_ux, F_uu)
                
                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                # print("Iteration:", iteration, "Backward Pass Finished, Used Time:", time_calc )

                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._control(xs, us, k, K, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)

                    _, grad_wrt_input_norm = self._gradient_wrt_control( F_x, F_u, L_x, L_u )
                    if grad_wrt_input_norm < tol_grad_norm:
                        converged = True
                        accepted = True
                        break

                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol_J:
                        # if np.abs(J_opt - J_new) < tol:
                        # if grad_wrt_input_norm < tol:
                            converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        changed = True

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0

                        # Accept this.
                        accepted = True
                        break

            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            end_time = time.perf_counter()
            time_calc = end_time - start_time   
            # print("Iteration:", iteration, "Control Rollout and Line Search Finished, Used Time:", time_calc )

            # accepted = True
            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(iteration, xs, us, J_opt, accepted, converged, grad_wrt_input_norm,
                             alpha, self._mu, J_hist, xs_hist, us_hist)

            if converged:
                break

        # Store fit parameters.
        self._k = k
        self._K = K
        self._nominal_xs = xs
        self._nominal_us = us

        return xs, us, J_hist, xs_hist, us_hist

    def _control(self, xs, us, k, K, alpha=1.0):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
        """
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        xs_new[0] = xs[0].copy()

        for i in range(self.N):
            # Eq (12).
            us_new[i] = us[i] + alpha * k[i] + K[i].dot(xs_new[i] - xs[i])

            # Eq (8c).
            xs_new[i + 1] = self.dynamics.f(xs_new[i], us_new[i], i)

        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
                                                     range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _forward_rollout(self, x0, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            x0: Initial state [state_size].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                xs: State path [N+1, state_size].
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us.shape[0]

        xs = np.empty((N + 1, state_size))
        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))

        if self._use_hessians:
            F_xx = np.empty((N, state_size, state_size, state_size))
            F_ux = np.empty((N, state_size, action_size, state_size))
            F_uu = np.empty((N, state_size, action_size, action_size))
        else:
            F_xx = None
            F_ux = None
            F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))

        xs[0] = x0
        for i in range(N):
            x = xs[i]
            u = us[i]
            # print("In the forward rollout code, u is ", u)

            xs[i + 1] = self.dynamics.f(x, u, i)
            F_x[i] = self.dynamics.f_x(x, u, i)
            F_u[i] = self.dynamics.f_u(x, u, i)
            # F_u[i] = self.dynamics.f_u(x, u, i)[:, np.newaxis]

            L[i] = self.cost.l(x, u, i, terminal=False)
            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x, u, i)

                # print("Shpae of F_ux[i] is ", F_ux[i].shape)
                # print("Shpae of dynamics.f_ux is ", self.dynamics.f_ux(x, u, i).shape)
                # print("x is ",x)
                # print("u is ",u)
                # print("f_ux is", self.dynamics.f_ux(x, u, i))

                F_ux[i] = self.dynamics.f_ux(x, u, i)
                F_uu[i] = self.dynamics.f_uu(x, u, i)

        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        L_x[-1] = self.cost.l_x(x, None, N, terminal=True)
        L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True)

        return xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

        Args:
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):
            if self._use_hessians:
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                     L_u[i], L_xx[i], L_ux[i],
                                                     L_uu[i], V_x, V_xx,
                                                     F_xx[i], F_ux[i], F_uu[i])
            else:
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                     L_u[i], L_xx[i], L_ux[i],
                                                     L_uu[i], V_x, V_xx)
                
            # try:
            #     if np.any( Q_uu <= 0):
            #         raise PDViolationError("Quu is not PD")
            # except PDViolationError as e:
            #     print(f"Positive Definite Assumption Violation: {e}")

            if not is_pos_def(Q_uu + Q_uu.T):
                pass

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self,
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

        Args:
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)
        # Q_uu = 0.5 * (Q_uu + Q_uu.T) 

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    
    def _gradient_wrt_control(self, F_x, F_u, L_x, L_u):
        """Solve adjoint equations using a for loop.

        Args:
            F_x: dynamics Jacobians with respect to state.
            F_u: dynamics Jacobians with respect to control.
            L_x: cost gradients with respect to state.
            L_u: cost gradients with respect to control.

        Returns:
            gradient, adjoints, final adjoint variable.
        """

        # P = np.zeros((self.N + 1, self._state_size))
        g = np.zeros((self.N, self._action_size))

        p = L_x[self.N] # Initialize adjoint variable with terminal cost gradient
        # P[self.N] = p
        g_norm_sum = 0
        
        for t in range(self.N - 1, -1, -1):  # backward recursion of Adjoint equations.
            g[t] = L_u[t] + np.matmul(F_u[t].T, p)
            p = L_x[t] + np.matmul(F_x[t].T, p) 
            g_norm_sum = g_norm_sum + np.linalg.norm(g[t])
            # P[t] = p

        return g, g_norm_sum/self.N

# =================================================================================
# SO3 Controller
# =================================================================================

class iLQR_Tracking_SO3(BaseController):

    """
    Finite Horizon Iterative Linear Quadratic Regulator for Exact SO3 Dynamics.
    """

    def __init__(self, dynamics, cost, N, max_reg=1e10, hessians=False,
                 rollout='nonlinear', debug=None):
        """Constructs an iLQR solver for Error-State Dynamics

        Args:
            dynamics: Plant dynamics.
            cost: Cost function.
            N: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            rollout: Determine the rollout method, 'linear' or 'nonlinear'.
            debug: Indication of debug mode on or not
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._action_size = dynamics.action_size
        self._state_size = dynamics.state_size
        self._error_state_size  = dynamics._error_state_size

        self._rollout_mode = rollout
        self._debug = debug

        self._k = np.zeros((N, self._action_size))
        self._K = np.zeros((N, self._action_size, self._state_size))

        super(iLQR_Tracking_SO3, self).__init__()

    def fit(self, x0, us_init, 
            n_iterations=100, 
            tol_J=1e-6, tol_grad_norm=1e-6,
            on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state, list of configuration and velocity, [q, xi],
                q:  SE3, [4, 4],
                xi: twist velocity, stack by anguler and linear velocity, [w, v]
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration for info update

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0
        # self._mu = 0.0

        # Add time 
        start_time = time.perf_counter()

        # Backtracking line search candidates 0 < alpha <= 1.
        # alphas = 1.1**(-np.arange(10)**2)
        alphas = 1.1**(-np.arange(13)**2)

        us = us_init.copy()
        k = self._k
        K = self._K

        J_hist = []
        xs_hist = []
        us_hist = []
        grad_hist = []

        converged = False

        xs = self._init_rollout( x0, us )
        xs_hist.append(xs.copy())
        us_hist.append(us.copy())

        for iteration in range(n_iterations):
            accepted = False
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Start Iteration:", iteration, ", Used Time:", time_calc )

            (F_x, F_u, L, L_x, L_u, L_xx, L_ux, 
                L_uu, F_xx, F_ux, F_uu) = self._linearization(xs, us)
            J_opt = L.sum()

            _, grad_wrt_input_norm = self._gradient_wrt_control( F_x, F_u, L_x, L_u )
            grad_hist.append(grad_wrt_input_norm)
            if grad_wrt_input_norm < tol_grad_norm:
                converged = True
                print("Iteration", iteration-1, "converged, gradient w.r.t. input:", grad_wrt_input_norm )
                break
            print("Iteration:", iteration, "Gradient w.r.t. input:", grad_wrt_input_norm )
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Iteration:", iteration, "Linearization Finished, Used Time:", time_calc, "Cost:", J_opt )

            if converged == False:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                           F_xx, F_ux, F_uu)
                
                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration:", iteration, "Backward Pass Finished, Used Time:", time_calc )

                for alpha in alphas:
                    xs_new, us_new = self._rollout(xs, us, k, K, F_x, F_u, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)

                    end_time = time.perf_counter()
                    time_calc = end_time - start_time   
                    print("Iteration:", iteration, "Rollout Finished, Used Time:", time_calc, "Alpha:", alpha, "Cost:", J_new )

                    if J_new < J_opt:
                        # if np.abs((J_opt - J_new) / J_opt) < tol_J:
                        #     converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new

                        # Accept this.
                        accepted = True
                        break

            end_time = time.perf_counter()
            time_calc = end_time - start_time   
            print("Iteration:", iteration, "Rollout and Line Search Finished, Used Time:", time_calc )
           
            if on_iteration:
                on_iteration(iteration, xs, us, J_opt,
                            accepted, converged, grad_wrt_input_norm,
                            alpha, self._mu, 
                            J_hist, xs_hist, us_hist)
                        
            if converged:
                break

            if not accepted:
                warnings.warn("Couldn't find descent direction, regularization and line search step exhausted")
                break

        # Store fit parameters.
        self._k = k
        self._K = K

        return xs, us, J_hist, xs_hist, us_hist, grad_hist
    
    def _init_rollout(self, x0, us):
        """ Initially rollout a dynamically feasible trajectory.

        Args:
            x0: initial state, [q0, vel0]

        Returns:
            xs: initial states trajectory
        """
        xs = [None] * (self.N + 1)
        xs[0] = x0
        for i in range(self.N):
            xs[i+1] = self.dynamics.f(xs[i], us[i], i)
        return xs

    def _rollout(self, xs, us, k, K, F_x, F_u, alpha=1.0):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
        """
        xs_new = [None] * (self.N + 1)
        us_new = np.zeros_like(us)

        xs_new[0] = xs[0].copy()

        for i in range(self.N):

            q_new, xi_new = xs_new[i]
            q, xi = xs[i]
            q_next, xi_next = xs[i+1]

            q_err = (q_new - q).coeffs()
            xi_err = (xi_new - xi).coeffs()
            xs_err = np.concatenate((q_err, xi_err))

            us_err = alpha * k[i] + K[i].dot(xs_err)
            us_new[i] = us[i] + us_err

            if self._rollout_mode == 'linear':

                q_next_new = q_next + SO3Tangent( F_x[i,:3,:] @ xs_err + F_u[i,:3,:] @ us_err )
                xi_next_new = xi_next + F_x[i,3:,:] @ xs_err + F_u[i,3:,:] @ us_err
                xs_new[i + 1] = [q_next_new, xi_next_new].copy()

            elif self._rollout_mode == 'nonlinear':

                q_next, xi_next = self.dynamics.f( 
                    [ q_new, xi_new ],
                    us_new[i],
                    i
                )
                xs_new[i + 1] = [q_next, xi_next].copy()
                
        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
                                                     range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _linearization(self, xs, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            xs: Nominal state trajectory [N+1, [q, vel]].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                xs: State path [N+1, state_size].
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us.shape[0]

        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))

        if self._use_hessians:
            F_xx = np.empty((N, state_size, state_size, state_size))
            F_ux = np.empty((N, state_size, action_size, state_size))
            F_uu = np.empty((N, state_size, action_size, action_size))
        else:
            F_xx = None
            F_ux = None
            F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))

        for i in range(N):
            x = xs[i]
            u = us[i]

            F_x[i] = self.dynamics.f_x(x, u, i)
            F_u[i] = self.dynamics.f_u(x, u, i)                     

            L[i] = self.cost.l(x, u, i, terminal=False)
            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x, u, i)
                F_ux[i] = self.dynamics.f_ux(x, u, i)
                F_uu[i] = self.dynamics.f_uu(x, u, i)

        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        L_x[-1] = self.cost.l_x(x, None, N, terminal=True)
        L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True)

        return F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

        Args:
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):

            while True:
                if self._use_hessians:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x, V_xx,
                                                        F_xx[i], F_ux[i], F_uu[i])
                else:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x, V_xx)
                
                if not is_pos_def(Q_uu + Q_uu.T):
                    # Increase regularization term.
                    self._delta = max(1.0, self._delta) * self._delta_0
                    self._mu = max(self._mu_min, self._mu * self._delta)

                    if self._mu_max and self._mu >= self._mu_max:
                        warnings.warn("exceeded max regularization term")
                        break
                else:
                    self._delta = min(1.0, self._delta) / self._delta_0
                    self._mu *= self._delta
                    if self._mu <= self._mu_min:
                            self._mu = 0.0
                    break

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self,
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

        Args:
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)
        # Q_uu = 0.5 * (Q_uu + Q_uu.T) 

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    
    def _gradient_wrt_control(self, F_x, F_u, L_x, L_u):
        """Solve adjoint equations using a for loop.

        Args:
            F_x: dynamics Jacobians with respect to state.
            F_u: dynamics Jacobians with respect to control.
            L_x: cost gradients with respect to state.
            L_u: cost gradients with respect to control.

        Returns:
            gradient, adjoints, final adjoint variable.
        """

        # P = np.zeros((self.N + 1, self._state_size))
        g = np.zeros((self.N, self._action_size))

        p = L_x[self.N] # Initialize adjoint variable with terminal cost gradient
        # P[self.N] = p
        g_norm_sum = 0
        
        for t in range(self.N - 1, -1, -1):  # backward recursion of Adjoint equations.
            g[t] = L_u[t] + np.matmul(F_u[t].T, p)
            p = L_x[t] + np.matmul(F_x[t].T, p) 
            g_norm_sum = g_norm_sum + np.linalg.norm(g[t])
            # P[t] = p

        return g, g_norm_sum/self.N


class iLQR_Tracking_SO3_MS(BaseController):

    """
    Finite Horizon Multiple Shooting 
    Iterative Linear Quadratic Regulator for Exact SE3 Dynamics.
    """

    def __init__(self, dynamics, cost, N, 
                 q_ref, xi_ref, 
                 max_reg=1e10, hessians=False,
                 line_search=False,
                 rollout='linear', debug=None):
        """Constructs an iLQR solver for Error-State Dynamics

        Args:
            dynamics: Plant error-state dynamics.
            cost: Cost function.
            N: Horizon length.
            q_ref: Tracking reference configuration.
            xi_ref: Tracking reference velocity.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            rollout: Determine the rollout method, 'linear' or 'nonlinear'.
            debug: Indication of debug mode on or not
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        # self._mu = 0.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._action_size = dynamics.action_size
        self._state_size = dynamics.state_size
        self._error_state_size = dynamics._error_state_size
        self._pos_state_size = dynamics._pos_state_size
        self._vel_state_size = dynamics._vel_state_size

        self._q_ref = [ SO3(Rotation.from_matrix(q).as_quat()) for q in q_ref]
        self._xi_ref = [ SO3Tangent(xi) for xi in xi_ref ]

        self._rollout_mode = rollout
        self._debug = debug
        self._line_search = line_search

        self._defect_mu0 = 10.
        self._defect_rho = 0.5
        self._defect_gamma = 0.05
        self._defect_mu_min = 10.
        self._defect_kappa = 1e-14

        self._k = np.zeros((N, self._action_size))
        self._K = np.zeros((N, self._action_size, self._state_size))

        super(iLQR_Tracking_SO3_MS, self).__init__()

    @property
    def xi_ref(self):
        return self._xi_ref
    
    @property
    def q_ref(self):
        return self._q_ref

    @property
    def state_size(self):
        return self._state_size

    @property
    def action_size(self):
        return self._action_size
    
    @property
    def error_state_size(self):
        return self._error_state_size

    @property
    def pos_state_size(self):
        return self._pos_state_size
    
    @property
    def vel_state_size(self):
        return self._vel_state_size

    def get_q_ref(self,i):
        return self._q_ref[i]
    
    def get_xi_ref(self,i):
        return self._xi_ref[i]

    def fit(self, x0, us_init, n_iterations=100, 
            tol_J=1e-6, tol_grad_norm=1e-6, tol_d_norm=1e-6,
            on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state, list of configuration and velocity, [q, xi],
                q:  SE3, [4, 4],
                xi: twist velocity, stack by anguler and linear velocity, [w, v]
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration for info update

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0
        # self._mu = 0.0

        # Add time 
        start_time = time.perf_counter()

        # Backtracking line search candidates 0 < alpha <= 1.
        # alphas = 1.1**(-np.arange(10)**2)
        alphas = 1.1**(-np.arange(13)**2)

        us = us_init.copy()
        k = self._k
        K = self._K

        J_hist = []
        xs_hist = []
        us_hist = []
        grad_hist = []
        defect_hist = []

        converged = False

        xs = self._initial_guess( x0 )
        xs_hist.append(xs.copy())
        us_hist.append(us.copy())

        if self._line_search:
            d_weight_prev = self._defect_mu0

        for iteration in range(n_iterations):
            accepted = False
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Start Iteration:", iteration, ", Used Time:", time_calc )

            # Forward rollout only if it needs to be recomputed.
            
            (d, F_x, F_u, L, L_x, L_u, L_xx, L_ux, 
                L_uu, F_xx, F_ux, F_uu) = self._linearization(xs, us)
            d_norm = self._compute_defect_norm( d )
            if iteration == 0:
                defect_hist.append(d_norm)
            J_opt = L.sum()

            # _, grad_wrt_input_norm = self._gradient_wrt_control_ss( F_x, F_u, L_x, L_u )
            # if grad_wrt_input_norm < tol_grad_norm and d_norm < tol_d_norm:
            #     converged = True
            #     accepted = True
            #     print("Iteration", iteration-1, "converged, gradient w.r.t. input:", grad_wrt_input_norm )
            #     break
            # print("Iteration:", iteration, "Gradient w.r.t. input:", grad_wrt_input_norm )
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Iteration:", iteration, "Linearization Finished, Used Time:", 
                  time_calc, "Cost:", J_opt, "DefectNorm:", d_norm )

            if converged == False:
                # Backward pass.
                k, K, V_x, V_xx = self._backward_pass(d, F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                           F_xx, F_ux, F_uu)

                _, grad_wrt_input_norm = self._gradient_wrt_control( d, F_x, F_u, L_x, L_u, V_x, V_xx )
                if grad_wrt_input_norm < tol_grad_norm and d_norm < tol_d_norm:
                    converged = True
                    accepted = True
                    grad_hist.append( grad_wrt_input_norm )
                    print("Iteration", iteration-1, "converged, gradient w.r.t. input:", grad_wrt_input_norm )
                    break
                print("Iteration:", iteration, "Gradient w.r.t. input:", grad_wrt_input_norm )
                
                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration:", iteration, "Backward Pass Finished, Used Time:", time_calc )

                # _, _, xs_errs, us_errs = self._rollout(xs, us, k, K, d, F_x, F_u, rollout="linear")
                # exact_lqr_cost_change = self._expected_cost_change( 
                #     xs_errs, us_errs, L_x, L_u, L_xx, L_ux, L_uu, alpha=1.0 
                # )
                # d_weight = self._update_defect_weight( exact_lqr_cost_change, d_norm)
                # merit = J_opt + d_weight * d_norm

                if self._line_search:
                    _, _, xs_errs, us_errs = self._rollout(xs, us, k, K, d, F_x, F_u, rollout="linear")

                    exact_lqr_cost_change = self._expected_cost_change( 
                        xs_errs, us_errs, L_x, L_u, L_xx, L_ux, L_uu, alpha=1.0 
                    )
                    d_weight = self._update_defect_weight( exact_lqr_cost_change, d_norm, d_weight_prev )
                    d_weight_prev = d_weight
                    merit = J_opt + d_weight * d_norm
                    print("Iteration:", iteration, "Nominal, Cost:", J_opt, "Merit:", merit, "d_weight:", d_weight )

                    # Backtracking line search.
                    for alpha in alphas:
                        xs_new, us_new, xs_errs, us_errs = self._rollout(xs, us, k, K, d, F_x, F_u, alpha)

                        J_new = self._trajectory_cost( xs_new, us_new )
                        d_new = self._compute_defect( xs_new, us_new )
                        d_norm_new = self._compute_defect_norm( d_new )
                        J_expected_change = self._scale_cost_change( exact_lqr_cost_change, alpha )

                        merit_new = J_new + d_weight * d_norm_new

                        end_time = time.perf_counter()
                        time_calc = end_time - start_time   
                        print("Iteration:", iteration, "One Rollout Finished, Used Time:", time_calc, 
                              "\n         Alpha:", alpha, "Cost:", J_new, "Merit:", merit_new )

                        if merit_new - merit < self._defect_gamma * ( J_expected_change - alpha * d_weight * d_norm ):
                            # if np.abs((J_opt - J_new) / J_opt) < tol_J and d_norm <  tol_d_norm:
                            #     converged = True

                            J_opt = J_new
                            xs = xs_new
                            us = us_new

                            # Accept this.
                            accepted = True
                            print("Line Search Finished")
                            break
                            
                        if alpha == alphas[-1]:
                            print("Line Search Finished, Step Size Exhausted")
                
                else:
                    alpha = 1
                    xs_new, us_new, xs_errs, us_errs = self._rollout(xs, us, k, K, d, F_x, F_u, alpha)

                    J_new = self._trajectory_cost( xs_new, us_new )
                    d_new = self._compute_defect( xs_new, us_new )
                    d_norm_new = self._compute_defect_norm( d_new )

                    accepted = True

                    # if np.abs((J_opt - J_new) / (J_opt) ) < tol_J:
                    #     converged = True

                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration:", iteration, "Forward Rollout Finished, Used Time:", time_calc, "Alpha:", alpha, "Cost:", J_new )

                if accepted:    
                    J_opt = J_new
                    xs = xs_new
                    us = us_new

                # # Accept this.
                # accepted = True

            # end_time = time.perf_counter()
            # time_calc = end_time - start_time   
            # print("Iteration:", iteration, "Rollout and Line Search Finished, Used Time:", time_calc )
           
            if on_iteration:
                on_iteration(iteration, xs, us, J_opt,
                            accepted, converged, d_norm_new,
                            grad_wrt_input_norm,
                            alpha, self._mu, 
                            J_hist, xs_hist, us_hist, grad_hist, defect_hist)
                    
            if converged:
                break

            if (not accepted) and self._line_search:
                warnings.warn("Couldn't find descent direction, regularization and line search step exhausted")
                break

        # Store fit parameters.
        self._k = k
        self._K = K

        return xs, us, J_hist, xs_hist, us_hist, grad_hist, defect_hist

    def _rollout(self, xs, us, k, K, d, 
                 F_x, F_u, alpha=1.0,
                 rollout=None):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            F_x: Jacobian of dynamics w.r.t. state, [N+1, state_size, state_size].
            F_u: Jacobian of dynamics w.r.t. input, [N, state_size, input_size].
            alpha: Line search coefficient.
            rollout: Rollout mode, "nonlinear" / "linear" / "hybrid" .

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
                xs_errs: state deviation from nominal [N, action_size].
                us_errs: control deviation from nominal [N, control_size].
        """
        xs_new = [None] * (self.N + 1)
        us_new = np.zeros_like(us)

        xs_new[0] = xs[0].copy()

        xs_errs = np.empty((self.N+1, self.state_size))
        us_errs = np.empty((self.N, self.action_size))

        if rollout is None:
            rollout = self._rollout_mode

        for i in range(self.N):

            q_new, xi_new = xs_new[i]

            q, xi = xs[i]
            q_next, xi_next = xs[i+1]

            # q_new_mnf = SE32manifSE3(q_new)
            # q_mnf = SE32manifSE3(q)
            # q_next_mnf = SE32manifSE3(q_next)

            q_err = ( q_new - q ).coeffs()
            xi_err = ( xi_new - xi ).coeffs()
            xs_err = np.concatenate((q_err, xi_err))
            us_err = alpha * k[i] + K[i].dot(xs_err)
            us_new[i] = us[i] + us_err

            xs_errs[i] = xs_err
            us_errs[i] = us_err

            defect_q_mnf = SO3Tangent(d[i,:3])
            defect_xi = SO3Tangent(d[i,3:])

            if rollout == 'nonlinear':    

                fq_new, fxi_new = self.dynamics.f( 
                    [ q_new, xi_new ],
                    us_new[i],
                    i
                )
                fq, fxi = self.dynamics.f( 
                    [ q, xi ],
                    us[i],
                    i
                )

                q_next_new = q_next * (alpha * defect_q_mnf).exp() * fq.inverse() * fq_new
                xi_next_new = xi_next + fxi_new - fxi + alpha * defect_xi

                xs_new[i+1] = [ q_next_new, xi_next_new ]

            elif rollout == "linear":

                q_next_new = q_next + \
                    ( SO3Tangent(F_x[i,:3,:] @ xs_err + F_u[i,:3,:] @ us_err ) + alpha * defect_q_mnf )
                xi_next_new = xi_next + \
                    ( SO3Tangent(F_x[i,3:,:] @ xs_err + F_u[i,3:,:] @ us_err) + alpha * defect_xi )
                xs_new[i + 1] = [q_next_new, xi_next_new].copy()

        # Compute Terminal Error
        q_new, xi_new = xs_new[self.N]
        q, xi = xs[self.N]

        q_err = (q_new - q).coeffs()
        xi_err = (xi_new - xi).coeffs()

        xs_err = np.concatenate((q_err, xi_err))
        xs_errs[self.N] = xs_err

        return xs_new, us_new, xs_errs, us_errs

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
                                                     range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)
    
    def _expected_cost_change(self, xs_errs, us_errs, l_x, l_u, l_xx, l_ux, l_uu, alpha=1.0):
        """Computes the expected cost change
        """
        cost_first_order = 0.
        cost_second_order = 0.
        for i in range(self.N):
            cost_first_order += l_x[i].T @ xs_errs[i] + l_u[i].T @ us_errs[i]
            cost_second_order += xs_errs[i].T @ l_xx[i] @ xs_errs[i] \
                                +  us_errs[i].T @ l_uu[i] @ us_errs[i] \
                                +  2 * us_errs[i].T @ l_ux[i] @ xs_errs[i] 
        i = self.N
        cost_first_order += l_x[i].T @ xs_errs[i]
        cost_second_order += xs_errs[i].T @ l_xx[i] @ xs_errs[i]
        return cost_first_order, cost_second_order
    
    def _scale_cost_change( self, ecc, alpha ):
        return alpha * ecc[0] + 0.5 * (alpha**2) * ecc[1] 

    def _update_defect_weight( self, expected_cost_change, d_norm, defect_weight_prev ):
        """Computes the expected cost change
        """                   
        if d_norm < self._defect_kappa:
            defect_weight = defect_weight_prev
            return defect_weight

        defect_weight = self._defect_mu0 \
            + np.abs(expected_cost_change[0]+0.5*expected_cost_change[1]) / ( (1 - self._defect_rho) * d_norm )
        
        # Didn't ensure monotonic increasing compared to Li's works

        defect_weight = max( self._defect_mu_min, defect_weight )

        return defect_weight

    def _compute_defect(self, xs, us):
        """Compute the dynamics defect

        Args:
            xs: State path [N+1, [q, xi]].
            us: Control path [N, action_size].

        Returns:
            d: defect vector [N, state_size].
        """
        d = np.empty((self.N, self.state_size))
        for i in range(self.N):
            f_xs = self.dynamics.f(xs[i], us[i], i)
            d_q = (f_xs[0] - xs[i+1][0]).coeffs()
            d_xi = (f_xs[1] - xs[i+1][1]).coeffs()
            d[i] = np.concatenate(
                ( d_q, d_xi )
            )
        return d

    def _compute_defect_norm(self, defect, order=2):
        """Compute norm of defect

        Args:
            d: defect vector [N, state_size].

        Returns:
            dnorm: scalar.
        """
        return np.linalg.norm(defect.reshape(-1,), order)

    def _linearization(self, xs, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            xs: Nominal state trajectory [N+1, [q, vel]].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us.shape[0]

        d = np.empty((N, state_size))
        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))

        if self._use_hessians:
            F_xx = np.empty((N, state_size, state_size, state_size))
            F_ux = np.empty((N, state_size, action_size, state_size))
            F_uu = np.empty((N, state_size, action_size, action_size))
        else:
            F_xx = None
            F_ux = None
            F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))
     
        for i in range(N):
            x = xs[i]
            x_next = xs[i+1]
            u = us[i]

            d_q = (self.dynamics.f(x, u, i)[0] - x_next[0]).coeffs()
            d_xi = (self.dynamics.f(x, u, i)[1] - x_next[1]).coeffs()
            d[i] = np.concatenate(
                ( d_q, d_xi )
            )

            F_x[i] = self.dynamics.f_x(x, u, i)
            F_u[i] = self.dynamics.f_u(x, u, i)                     

            L[i] = self.cost.l(x, u, i, terminal=False)
            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x, u, i)
                F_ux[i] = self.dynamics.f_ux(x, u, i)
                F_uu[i] = self.dynamics.f_uu(x, u, i)

        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        L_x[-1] = self.cost.l_x(x, None, N, terminal=True)
        L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True)

        return d, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass(self,
                       d, 
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

        Args:
            d: Dynamics defect for multiple shooting [N, state_size].
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
                V_x: value function 1st-order derivative w.r.t. state.
                    [N+1, state_size]
                V_xx: value function 2nd-order derivative (hessian) w.r.t. state.
                    [N+1, state_size, state_size]
        """
        V_x = np.empty((self.N + 1, self.state_size))
        V_xx = np.empty((self.N + 1, self.state_size, self.state_size))

        V_x[self.N] = L_x[-1]
        V_xx[self.N] = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):

            while True:
                if self._use_hessians:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(d[i], 
                                                        F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x[i+1], V_xx[i+1],
                                                        F_xx[i], F_ux[i], F_uu[i])
                else:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(d[i], 
                                                         F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x[i+1], V_xx[i+1])
                
                if not is_pos_def(Q_uu + Q_uu.T):
                    # Increase regularization term.
                    self._delta = max(1.0, self._delta) * self._delta_0
                    self._mu = max(self._mu_min, self._mu * self._delta)

                    if self._mu_max and self._mu >= self._mu_max:
                        warnings.warn("exceeded max regularization term")
                        break
                else:
                    self._delta = min(1.0, self._delta) / self._delta_0
                    self._mu *= self._delta
                    if self._mu <= self._mu_min:
                            self._mu = 0.0
                    break

                # if not is_pos_def(Q_uu + Q_uu.T):
                #     # Increase regularization term.
                #     if self._mu == 0.0:
                #         self._mu = 1e-3
                #     else:
                #         self._mu *= 10

                #     if self._mu_max and self._mu >= self._mu_max:
                #         warnings.warn("exceeded max regularization term")
                #         break
                # else:
                #     self._mu = 0.0
                #     break

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x[i] = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x[i] += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx[i] = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx[i] += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx[i] = 0.5 * (V_xx[i] + V_xx[i].T)  # To maintain symmetry.

        return k, K, V_x, V_xx

    def _Q(self,
           d, 
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

        Args:
            d: defect of state dynamics [state_size].
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x + V_xx.dot(d))
        Q_u = l_u + f_u.T.dot(V_x + V_xx.dot(d))
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)
        # Q_uu = 0.5 * (Q_uu + Q_uu.T) 

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    
    def _gradient_wrt_control(self, d, F_x, F_u, L_x, L_u, V_x, V_xx):
        """Solve adjoint equations using a for loop.

        Args:
            d: dynamics defect.
            F_x: dynamics Jacobians with respect to state.
            F_u: dynamics Jacobians with respect to control.
            L_x: cost gradients with respect to state.
            L_u: cost gradients with respect to control.
            V_x: value function 1st-order derivative w.r.t. state.
            V_xx: value function 2nd-order derivative (hessian) w.r.t. state.

        Returns:
            gradient, adjoints, final adjoint variable.
        """

        g = np.zeros((self.N, self._action_size))
        g_norm_sum = 0

        for t in range(self.N - 1, -1, -1):
            g[t] = L_u[t] + np.matmul( F_u[t].T, V_x[t+1] + np.matmul(V_xx[t+1].T, d[t]) )
            g_norm_sum = g_norm_sum + np.linalg.norm(g[t])

        return g, g_norm_sum/self.N
    
    def _gradient_wrt_control_ss(self, F_x, F_u, L_x, L_u):
        """Solve adjoint equations using a for loop.

        Args:
            F_x: dynamics Jacobians with respect to state.
            F_u: dynamics Jacobians with respect to control.
            L_x: cost gradients with respect to state.
            L_u: cost gradients with respect to control.

        Returns:
            gradient, adjoints, final adjoint variable.
        """

        # P = np.zeros((self.N + 1, self._state_size))
        g = np.zeros((self.N, self._action_size))

        p = L_x[self.N] # Initialize adjoint variable with terminal cost gradient
        # P[self.N] = p
        g_norm_sum = 0
        
        for t in range(self.N - 1, -1, -1):  # backward recursion of Adjoint equations.
            g[t] = L_u[t] + np.matmul(F_u[t].T, p)
            p = L_x[t] + np.matmul(F_x[t].T, p) 
            g_norm_sum = g_norm_sum + np.linalg.norm(g[t])
            # P[t] = p

        return g, g_norm_sum/self.N
    
    def _initial_guess(self, x0):
        """Generate initial guess for shooting states based on reference

        Args:
            x0: initial state, [q0, vel0]

        Returns:
            xs: initial states trajectory
        """
        xs = []
        xs.append(x0) 
        for i in range(1, self.N+1):
            xs.append([ self.get_q_ref(i), self.get_xi_ref(i) ])
        return xs


# =================================================================================
# SE3 Controller
# =================================================================================

class iLQR_Tracking_SE3(BaseController):

    """
    Finite Horizon Iterative Linear Quadratic Regulator for Exact SE3 Dynamics.
    """

    def __init__(self, dynamics, cost, N, max_reg=1e10, hessians=False,
                 rollout='linear', debug=None):
        """Constructs an iLQR solver for Error-State Dynamics

        Args:
            dynamics: Plant error-state dynamics.
            cost: Cost function.
            N: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            rollout: Determine the rollout method, 'linear' or 'nonlinear'.
            debug: Indication of debug mode on or not
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._action_size = dynamics.action_size
        self._state_size = dynamics.state_size
        self._error_state_size  = dynamics._error_state_size

        self._rollout_mode = rollout
        self._debug = debug

        self._k = np.zeros((N, self._action_size))
        self._K = np.zeros((N, self._action_size, self._state_size))

        super(iLQR_Tracking_SE3, self).__init__()

    def fit(self, x0, us_init, n_iterations=100, tol_J=1e-6, tol_grad_norm=1e-3,
             on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state, list of configuration and velocity, [q, xi],
                q:  SE3, [4, 4],
                xi: twist velocity, stack by anguler and linear velocity, [w, v]
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration for info update

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0
        # self._mu = 0.0

        # Add time 
        start_time = time.perf_counter()

        # Backtracking line search candidates 0 < alpha <= 1.
        # alphas = 1.1**(-np.arange(10)**2)
        alphas = 1.1**(-np.arange(13)**2)

        us = us_init.copy()
        k = self._k
        K = self._K

        J_hist = []
        xs_hist = []
        us_hist = []
        grad_hist = []

        changed = True
        converged = False

        xs = self._init_rollout( x0, us )
        xs_hist.append(xs.copy())
        us_hist.append(us.copy())

        for iteration in range(n_iterations):
            accepted = False
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Start Iteration:", iteration, ", Used Time:", time_calc )

            (F_x, F_u, L, L_x, L_u, L_xx, L_ux, 
                L_uu, F_xx, F_ux, F_uu) = self._linearization(xs, us)
            J_opt = L.sum()

            _, grad_wrt_input_norm = self._gradient_wrt_control( F_x, F_u, L_x, L_u )
            grad_hist.append(grad_wrt_input_norm)
            if grad_wrt_input_norm < tol_grad_norm:
                converged = True
                print("Iteration", iteration-1, "converged, gradient w.r.t. input:", grad_wrt_input_norm )
                break
            print("Iteration:", iteration, "Gradient w.r.t. input:", grad_wrt_input_norm )
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Iteration:", iteration, "Linearization Finished, Used Time:", time_calc, "Cost:", J_opt )

            if converged == False:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                           F_xx, F_ux, F_uu)
                
                # norm_k_list = np.array([ np.linalg.norm(m) for m in k ])
                # norm_K_list = np.array([ np.linalg.norm(m) for m in K ])
                # import matplotlib.pyplot as plt
                # plt.plot(norm_K_list)
                # plt.plot(norm_k_list)
                # plt.legend(["K","k"])
                # plt.grid()

                # A_cl = [ F_x[i]+F_u[i]@K[i] for i in range(self.N) ]
                # eig_A_cl = [np.linalg.eig(A_cl[i])[0] for i in range(self.N)]
                # if iteration == 21:
                #     pass
                
                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration:", iteration, "Backward Pass Finished, Used Time:", time_calc )

                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._rollout(xs, us, k, K, F_x, F_u, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)

                    end_time = time.perf_counter()
                    time_calc = end_time - start_time   
                    print("Iteration:", iteration, "Rollout Finished, Used Time:", time_calc, "Alpha:", alpha, "Cost:", J_new )

                    if J_new < J_opt:
                        # if np.abs((J_opt - J_new) / J_opt) < tol_J:
                        #     converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new

                        # Accept this.
                        accepted = True
                        break

            end_time = time.perf_counter()
            time_calc = end_time - start_time   
            print("Iteration:", iteration, "Rollout and Line Search Finished, Used Time:", time_calc )
           
            if on_iteration:
                on_iteration(iteration, xs, us, J_opt,
                            accepted, converged, grad_wrt_input_norm,
                            alpha, self._mu, 
                            J_hist, xs_hist, us_hist)
                    
            if converged:
                break

            if not accepted:
                warnings.warn("Couldn't find descent direction, regularization and line search step exhausted")
                break

        # Store fit parameters.
        self._k = k
        self._K = K

        return xs, us, J_hist, xs_hist, us_hist, grad_hist
    
    def _init_rollout(self, x0, us):
        """ Initially rollout a dynamically feasible trajectory.

        Args:
            x0: initial state, [q0, vel0]

        Returns:
            xs: initial states trajectory
        """
        xs = [None] * (self.N + 1)
        xs[0] = x0
        for i in range(self.N):
            xs[i+1] = self.dynamics.f(xs[i], us[i], i)
        return xs

    def _rollout(self, xs, us, k, K, F_x, F_u, alpha=1.0):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
        """
        xs_new = [None] * (self.N + 1)
        us_new = np.zeros_like(us)

        xs_new[0] = xs[0].copy()

        for i in range(self.N):

            q_new, xi_new = xs_new[i]
            q, xi = xs[i]
            q_next, xi_next = xs[i+1]

            q_new_mnf = SE32manifSE3(q_new)
            q_mnf = SE32manifSE3(q)
            q_err = manifse32se3( q_new_mnf - q_mnf )

            xi_err = xi_new - xi

            xs_err = np.concatenate((q_err, xi_err))
            us_err = alpha * k[i] + K[i].dot(xs_err)
            us_new[i] = us[i] + us_err

            if self._rollout_mode == 'linear':

                q_next_mnf = SE32manifSE3(q_next)
                q_next_new = manifSE32SE3( q_next_mnf + se32manifse3(F_x[i,:6,:] @ xs_err + F_u[i,:6,:] @ us_err) )
                xi_next_new = xi_next + F_x[i,6:,:] @ xs_err + F_u[i,6:,:] @ us_err
                xs_new[i + 1] = [q_next_new, xi_next_new].copy()

            elif self._rollout_mode == 'nonlinear':

                q_next, xi_next = self.dynamics.f( 
                    [ q_new, xi_new ],
                    us_new[i],
                    i
                )
                xs_new[i + 1] = [q_next, xi_next].copy()
                
        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
                                                     range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _linearization(self, xs, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            xs: Nominal state trajectory [N+1, [q, vel]].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                xs: State path [N+1, state_size].
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us.shape[0]

        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))

        if self._use_hessians:
            F_xx = np.empty((N, state_size, state_size, state_size))
            F_ux = np.empty((N, state_size, action_size, state_size))
            F_uu = np.empty((N, state_size, action_size, action_size))
        else:
            F_xx = None
            F_ux = None
            F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))

        for i in range(N):
            x = xs[i]
            u = us[i]

            F_x[i] = self.dynamics.f_x(x, u, i)
            F_u[i] = self.dynamics.f_u(x, u, i)                     

            L[i] = self.cost.l(x, u, i, terminal=False)
            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x, u, i)
                F_ux[i] = self.dynamics.f_ux(x, u, i)
                F_uu[i] = self.dynamics.f_uu(x, u, i)

        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        L_x[-1] = self.cost.l_x(x, None, N, terminal=True)
        L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True)

        return F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

        Args:
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):

            while True:
                if self._use_hessians:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x, V_xx,
                                                        F_xx[i], F_ux[i], F_uu[i])
                else:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x, V_xx)
                
                if not is_pos_def(Q_uu + Q_uu.T):
                    # Increase regularization term.
                    self._delta = max(1.0, self._delta) * self._delta_0
                    self._mu = max(self._mu_min, self._mu * self._delta)


                    if self._mu_max and self._mu >= self._mu_max:
                        warnings.warn("exceeded max regularization term")
                        break
                else:
                    self._delta = min(1.0, self._delta) / self._delta_0
                    self._mu *= self._delta
                    if self._mu <= self._mu_min:
                            self._mu = 0.0
                    break

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self,
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

        Args:
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)
        # Q_uu = 0.5 * (Q_uu + Q_uu.T) 

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    
    def _gradient_wrt_control(self, F_x, F_u, L_x, L_u):
        """Solve adjoint equations using a for loop.

        Args:
            F_x: dynamics Jacobians with respect to state.
            F_u: dynamics Jacobians with respect to control.
            L_x: cost gradients with respect to state.
            L_u: cost gradients with respect to control.

        Returns:
            gradient, adjoints, final adjoint variable.
        """

        # P = np.zeros((self.N + 1, self._state_size))
        g = np.zeros((self.N, self._action_size))

        p = L_x[self.N] # Initialize adjoint variable with terminal cost gradient
        # P[self.N] = p
        g_norm_sum = 0
        
        for t in range(self.N - 1, -1, -1):  # backward recursion of Adjoint equations.
            g[t] = L_u[t] + np.matmul(F_u[t].T, p)
            p = L_x[t] + np.matmul(F_x[t].T, p) 
            g_norm_sum = g_norm_sum + np.linalg.norm(g[t])
            # P[t] = p

        return g, g_norm_sum/self.N


class iLQR_Tracking_SE3_MS(BaseController):

    """
    Finite Horizon Multiple Shooting 
    Iterative Linear Quadratic Regulator for Exact SE3 Dynamics.
    """

    def __init__(self, dynamics, cost, N, 
                 q_ref, xi_ref, 
                 max_reg=1e10, hessians=False,
                 line_search=False,
                 rollout='linear', debug=None):
        """Constructs an iLQR solver for Error-State Dynamics

        Args:
            dynamics: Plant error-state dynamics.
            cost: Cost function.
            N: Horizon length.
            q_ref: Tracking reference configuration.
            xi_ref: Tracking reference velocity.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            rollout: Determine the rollout method, 'linear' or 'nonlinear'.
            debug: Indication of debug mode on or not
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._action_size = dynamics.action_size
        self._state_size = dynamics.state_size
        self._error_state_size = dynamics._error_state_size

        self._q_ref = q_ref
        self._xi_ref = xi_ref

        self._rollout_mode = rollout
        self._debug = debug
        self._line_search = line_search

        self._defect_mu0 = 10.
        self._defect_rho = 0.5
        self._defect_gamma = 0.05 # used to be 0.05
        self._defect_mu_min = self._defect_mu0
        self._defect_kappa = 1e-12

        self._k = np.zeros((N, self._action_size))
        self._K = np.zeros((N, self._action_size, self._state_size))

        super(iLQR_Tracking_SE3_MS, self).__init__()

    @property
    def xi_ref(self):
        return self._xi_ref
    
    @property
    def q_ref(self):
        return self._q_ref

    @property
    def state_size(self):
        return self._state_size

    @property
    def action_size(self):
        return self._action_size
    
    @property
    def error_state_size(self):
        return self._error_state_size

    def get_q_ref(self,i):
        return self._q_ref[i]
    
    def get_xi_ref(self,i):
        return self._xi_ref[i]

    def fit(self, x0, us_init, n_iterations=100, 
            tol_J=1e-6, tol_grad_norm=1e-6, tol_d_norm=1e-6,
            on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state, list of configuration and velocity, [q, xi],
                q:  SE3, [4, 4],
                xi: twist velocity, stack by anguler and linear velocity, [w, v]
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration for info update

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0
        # self._mu = 0.0

        # Add time 
        start_time = time.perf_counter()

        # Backtracking line search candidates 0 < alpha <= 1.
        # alphas = 1.1**(-np.arange(10)**2)
        alphas = 1.1**(-np.arange(20)**2)

        us = us_init.copy()
        k = self._k
        K = self._K

        J_hist = []
        xs_hist = []
        us_hist = []
        grad_hist = []
        defect_hist = []

        converged = False

        xs = self._initial_guess( x0 )
        xs_hist.append(xs.copy())
        us_hist.append(us.copy())

        # if self._line_search:
        d_weight_prev = self._defect_mu0

        for iteration in range(n_iterations):
            accepted = False
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Start Iteration:", iteration, ", Used Time:", time_calc )

            # Forward rollout only if it needs to be recomputed.
            
            (d, F_x, F_u, L, L_x, L_u, L_xx, L_ux, 
                L_uu, F_xx, F_ux, F_uu) = self._linearization(xs, us)
            d_norm = self._compute_defect_norm( d )
            if iteration == 0:
                defect_hist.append(d_norm)
            J_opt = L.sum()

            # _, grad_wrt_input_norm = self._gradient_wrt_control_ss( F_x, F_u, L_x, L_u )
            # if grad_wrt_input_norm < tol_grad_norm and d_norm < tol_d_norm:
            #     converged = True
            #     accepted = True
            #     print("Iteration", iteration-1, "converged, gradient w.r.t. input:", grad_wrt_input_norm )
            #     break
            # print("Iteration:", iteration, "Gradient w.r.t. input:", grad_wrt_input_norm )
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Iteration:", iteration, "Linearization Finished, Used Time:", 
                  time_calc, "Cost:", J_opt, "DefectNorm:", d_norm )

            if converged == False:
                # Backward pass.
                k, K, V_x, V_xx = self._backward_pass(d, F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                           F_xx, F_ux, F_uu)

                _, grad_wrt_input_norm = self._gradient_wrt_control( d, F_x, F_u, L_x, L_u, V_x, V_xx )
                if grad_wrt_input_norm < tol_grad_norm and d_norm < tol_d_norm:
                    converged = True
                    accepted = True
                    print("Iteration", iteration-1, "converged, gradient w.r.t. input:", grad_wrt_input_norm )
                    break
                print("Iteration:", iteration, "Gradient w.r.t. input:", grad_wrt_input_norm )
                
                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration:", iteration, "Backward Pass Finished, Used Time:", time_calc )

                # _, _, xs_errs, us_errs = self._rollout(xs, us, k, K, d, F_x, F_u, rollout="linear")
                # exact_lqr_cost_change = self._expected_cost_change( 
                #     xs_errs, us_errs, L_x, L_u, L_xx, L_ux, L_uu, alpha=1.0 
                # )
                # d_weight = self._update_defect_weight( exact_lqr_cost_change, d_norm)
                # merit = J_opt + d_weight * d_norm

                # if iteration > 9:
                #     self._line_search = True

                if self._line_search:
                    _, _, xs_errs, us_errs = self._rollout(xs, us, k, K, d, F_x, F_u, rollout="linear")

                    exact_lqr_cost_change = self._expected_cost_change( 
                        xs_errs, us_errs, L_x, L_u, L_xx, L_ux, L_uu, alpha=1.0 
                    )
                    d_weight = self._update_defect_weight( exact_lqr_cost_change, d_norm, d_weight_prev )
                    d_weight_prev = d_weight
                    merit = J_opt + d_weight * d_norm
                    print("Iteration:", iteration, "Nominal, Cost:", J_opt, "Merit:", merit, "d_weight:", d_weight )

                    # Backtracking line search.
                    for alpha in alphas:
                        xs_new, us_new, xs_errs, us_errs = self._rollout(xs, us, k, K, d, F_x, F_u, alpha)

                        J_new = self._trajectory_cost( xs_new, us_new )
                        d_new = self._compute_defect( xs_new, us_new )
                        d_norm_new = self._compute_defect_norm( d_new )
                        J_expected_change = self._scale_cost_change( exact_lqr_cost_change, alpha )

                        merit_new = J_new + d_weight * d_norm_new

                        end_time = time.perf_counter()
                        time_calc = end_time - start_time   
                        print("Iteration:", iteration, "One Rollout Finished, Used Time:", time_calc, 
                              "\n          Alpha:", alpha, "Cost:", J_new, "Merit:", merit_new,  )

                        if merit_new - merit < self._defect_gamma * ( J_expected_change - alpha * d_weight * d_norm ):
                            # if np.abs((J_opt - J_new) / J_opt) < tol_J and d_norm <  tol_d_norm:
                            #     converged = True

                            J_opt = J_new
                            xs = xs_new
                            us = us_new

                            # Accept this.
                            accepted = True
                            print("Line Search Finished")
                            break

                        if alpha == alphas[-1]:
                            print("Line Search Finished, Step Size Exhausted")
                
                else:
                    alpha = 1
                    xs_new, us_new, xs_errs, us_errs = self._rollout(xs, us, k, K, d, F_x, F_u, alpha)

                    J_new = self._trajectory_cost( xs_new, us_new )
                    d_new = self._compute_defect( xs_new, us_new )
                    d_norm_new = self._compute_defect_norm( d_new )

                    accepted = True

                    # if np.abs((J_opt - J_new) / J_opt) < tol_J:
                    #     converged = True

                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration:", iteration, "Forward Rollout Finished, Used Time:", time_calc, "Alpha:", alpha, "Cost:", J_new )

                if accepted:    
                    J_opt = J_new
                    xs = xs_new
                    us = us_new

                # # Accept this.
                # accepted = True

            # end_time = time.perf_counter()
            # time_calc = end_time - start_time   
            # print("Iteration:", iteration, "Rollout and Line Search Finished, Used Time:", time_calc )
           
            if on_iteration:
                on_iteration(iteration, xs, us, J_opt,
                            accepted, converged, d_norm_new,
                            grad_wrt_input_norm,
                            alpha, self._mu, 
                            J_hist, xs_hist, us_hist, grad_hist, defect_hist)
                    
            if converged:
                break

            if not accepted:
                warnings.warn("Couldn't find descent direction, regularization and line search step exhausted")
                break

        # Store fit parameters.
        self._k = k
        self._K = K

        return xs, us, J_hist, xs_hist, us_hist, grad_hist, defect_hist

    def _rollout(self, xs, us, k, K, d, 
                 F_x, F_u, alpha=1.0,
                 rollout=None):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            F_x: Jacobian of dynamics w.r.t. state, [N+1, state_size, state_size].
            F_u: Jacobian of dynamics w.r.t. input, [N, state_size, input_size].
            alpha: Line search coefficient.
            rollout: Rollout mode, "nonlinear" / "linear" / "hybrid" .

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
                xs_errs: state deviation from nominal [N, action_size].
                us_errs: control deviation from nominal [N, control_size].
        """
        xs_new = [None] * (self.N + 1)
        us_new = np.zeros_like(us)

        xs_new[0] = xs[0].copy()

        xs_errs = np.empty((self.N+1, self.state_size))
        us_errs = np.empty((self.N, self.action_size))

        if rollout is None:
            rollout = self._rollout_mode

        for i in range(self.N):

            q_new, xi_new = xs_new[i]
            q, xi = xs[i]
            q_next, xi_next = xs[i+1]

            q_new_mnf = SE32manifSE3(q_new)
            q_mnf = SE32manifSE3(q)
            q_next_mnf = SE32manifSE3(q_next)
            q_err = manifse32se3( q_new_mnf - q_mnf )

            xi_err = xi_new - xi

            xs_err = np.concatenate((q_err, xi_err))
            us_err = alpha * k[i] + K[i].dot(xs_err)
            us_new[i] = us[i] + us_err

            xs_errs[i] = xs_err
            us_errs[i] = us_err

            defect_q_mnf = se32manifse3(d[i,:6])
            defect_xi = d[i,6:]

            if rollout == 'nonlinear':    

                fq_new, fxi_new = self.dynamics.f( 
                    [ q_new, xi_new ],
                    us_new[i],
                    i
                )
                fq, fxi = self.dynamics.f( 
                    [ q, xi ],
                    us[i],
                    i
                )

                fq_new_mnf = SE32manifSE3( fq_new )
                fq_mnf = SE32manifSE3( fq )

                q_next_new = manifSE32SE3( 
                    q_next_mnf * (alpha * defect_q_mnf).exp() * fq_mnf.inverse() * fq_new_mnf 
                )
                xi_next_new = xi_next + fxi_new - fxi + alpha * defect_xi

                xs_new[i+1] = [ q_next_new, xi_next_new ]

            elif rollout == "linear":

                q_next_new = manifSE32SE3( 
                    q_next_mnf + ( se32manifse3(F_x[i,:6,:] @ xs_err + F_u[i,:6,:] @ us_err ) + alpha * defect_q_mnf )
                )
                xi_next_new = xi_next + F_x[i,6:,:] @ xs_err + F_u[i,6:,:] @ us_err + alpha * defect_xi
                xs_new[i + 1] = [q_next_new, xi_next_new].copy()

        # Compute Terminal Error
        q_new, xi_new = xs_new[self.N]
        q, xi = xs[self.N]

        q_new_mnf = SE32manifSE3(q_new)
        q_mnf = SE32manifSE3(q)
        q_err = manifse32se3( q_new_mnf - q_mnf )
        xi_err = xi_new - xi

        xs_err = np.concatenate((q_err, xi_err))
        xs_errs[self.N] = xs_err

        return xs_new, us_new, xs_errs, us_errs

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
                                                     range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)
    
    def _expected_cost_change(self, xs_errs, us_errs, l_x, l_u, l_xx, l_ux, l_uu, alpha=1.0):
        """Computes the expected cost change
        """
        cost_first_order = 0.
        cost_second_order = 0.
        for i in range(self.N):
            cost_first_order += l_x[i].T @ xs_errs[i] + l_u[i].T @ us_errs[i]
            cost_second_order += xs_errs[i].T @ l_xx[i] @ xs_errs[i] \
                                +  us_errs[i].T @ l_uu[i] @ us_errs[i] \
                                +  2 * us_errs[i].T @ l_ux[i] @ xs_errs[i] 
        i = self.N
        cost_first_order += l_x[i].T @ xs_errs[i]
        cost_second_order += xs_errs[i].T @ l_xx[i] @ xs_errs[i]
        return cost_first_order, cost_second_order
    
    def _scale_cost_change( self, ecc, alpha ):
        return alpha * ecc[0] + 0.5 * (alpha**2) * ecc[1] 

    def _update_defect_weight( self, expected_cost_change, d_norm, defect_weight_prev ):
        """Computes the expected cost change
        """                   
        if d_norm < self._defect_kappa:
            defect_weight = defect_weight_prev
            return defect_weight

        defect_weight = self._defect_mu0 \
            + np.abs(expected_cost_change[0]+0.5*expected_cost_change[1]) / ( (1 - self._defect_rho) * d_norm )
        # defect_weight = self._defect_mu0 \
        #     + (expected_cost_change[0]+0.5*expected_cost_change[1]) / ( (1 - self._defect_rho) * d_norm )
        
        defect_weight = max( self._defect_mu_min, defect_weight )

        return defect_weight

    def _compute_defect(self, xs, us):
        """Compute the dynamics defect

        Args:
            xs: State path [N+1, [q, xi]].
            us: Control path [N, action_size].

        Returns:
            d: defect vector [N, state_size].
        """
        d = np.empty((self.N, self.state_size))
        for i in range(self.N):
            f_xs = self.dynamics.f(xs[i], us[i], i)
            d_q = manifse32se3( 
                SE32manifSE3(f_xs[0]) - SE32manifSE3(xs[i+1][0])
            )
            d_xi = f_xs[1] - xs[i+1][1]
            d[i] = np.concatenate(
                ( d_q, d_xi )
            )
        return d

    def _compute_defect_norm(self, defect, order=2):
        """Compute norm of defect

        Args:
            d: defect vector [N, state_size].

        Returns:
            dnorm: scalar.
        """
        return np.linalg.norm(defect.reshape(-1,), order)

    def _linearization(self, xs, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            xs: Nominal state trajectory [N+1, [q, vel]].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us.shape[0]

        d = np.empty((N, state_size))
        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))

        if self._use_hessians:
            F_xx = np.empty((N, state_size, state_size, state_size))
            F_ux = np.empty((N, state_size, action_size, state_size))
            F_uu = np.empty((N, state_size, action_size, action_size))
        else:
            F_xx = None
            F_ux = None
            F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))
     
        for i in range(N):
            x = xs[i]
            x_next = xs[i+1]
            u = us[i]

            d_q = manifse32se3( 
                SE32manifSE3(self.dynamics.f(x, u, i)[0]) - SE32manifSE3(x_next[0])
            )
            d_xi = self.dynamics.f(x, u, i)[1] - x_next[1]
            d[i] = np.concatenate(
                ( d_q, d_xi )
            )

            F_x[i] = self.dynamics.f_x(x, u, i)
            F_u[i] = self.dynamics.f_u(x, u, i)                     

            L[i] = self.cost.l(x, u, i, terminal=False)
            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x, u, i)
                F_ux[i] = self.dynamics.f_ux(x, u, i)
                F_uu[i] = self.dynamics.f_uu(x, u, i)

        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        L_x[-1] = self.cost.l_x(x, None, N, terminal=True)
        L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True)

        return d, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass(self,
                       d, 
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

        Args:
            d: Dynamics defect for multiple shooting [N, state_size].
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
                V_x: value function 1st-order derivative w.r.t. state.
                    [N+1, state_size]
                V_xx: value function 2nd-order derivative (hessian) w.r.t. state.
                    [N+1, state_size, state_size]
        """
        V_x = np.empty((self.N + 1, self.state_size))
        V_xx = np.empty((self.N + 1, self.state_size, self.state_size))

        V_x[self.N] = L_x[-1]
        V_xx[self.N] = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):

            while True:
                if self._use_hessians:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(d[i], 
                                                        F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x[i+1], V_xx[i+1],
                                                        F_xx[i], F_ux[i], F_uu[i])
                else:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(d[i], 
                                                         F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x[i+1], V_xx[i+1])
                
                if not is_pos_def(Q_uu + Q_uu.T):
                    # Increase regularization term.
                    self._delta = max(1.0, self._delta) * self._delta_0
                    self._mu = max(self._mu_min, self._mu * self._delta)


                    if self._mu_max and self._mu >= self._mu_max:
                        warnings.warn("exceeded max regularization term")
                        break
                else:
                    self._delta = min(1.0, self._delta) / self._delta_0
                    self._mu *= self._delta
                    if self._mu <= self._mu_min:
                            self._mu = 0.0
                    break

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x[i] = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x[i] += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx[i] = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx[i] += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx[i] = 0.5 * (V_xx[i] + V_xx[i].T)  # To maintain symmetry.

        return k, K, V_x, V_xx

    def _Q(self,
           d, 
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

        Args:
            d: defect of state dynamics [state_size].
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x + V_xx.dot(d))
        Q_u = l_u + f_u.T.dot(V_x + V_xx.dot(d))
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)
        # Q_uu = 0.5 * (Q_uu + Q_uu.T) 

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    
    def _gradient_wrt_control(self, d, F_x, F_u, L_x, L_u, V_x, V_xx):
        """Solve adjoint equations using a for loop.

        Args:
            d: dynamics defect.
            F_x: dynamics Jacobians with respect to state.
            F_u: dynamics Jacobians with respect to control.
            L_x: cost gradients with respect to state.
            L_u: cost gradients with respect to control.
            V_x: value function 1st-order derivative w.r.t. state.
            V_xx: value function 2nd-order derivative (hessian) w.r.t. state.

        Returns:
            gradient, adjoints, final adjoint variable.
        """

        g = np.zeros((self.N, self._action_size))
        g_norm_sum = 0

        for t in range(self.N - 1, -1, -1):
            g[t] = L_u[t] + np.matmul( F_u[t].T, V_x[t+1] + np.matmul(V_xx[t+1].T, d[t]) )
            g_norm_sum = g_norm_sum + np.linalg.norm(g[t])

        return g, g_norm_sum/self.N

    def _gradient_wrt_control_ss(self, F_x, F_u, L_x, L_u):
        """Solve adjoint equations using a for loop.

        Args:
            F_x: dynamics Jacobians with respect to state.
            F_u: dynamics Jacobians with respect to control.
            L_x: cost gradients with respect to state.
            L_u: cost gradients with respect to control.

        Returns:
            gradient, adjoints, final adjoint variable.
        """

        # P = np.zeros((self.N + 1, self._state_size))
        g = np.zeros((self.N, self._action_size))

        p = L_x[self.N] # Initialize adjoint variable with terminal cost gradient
        # P[self.N] = p
        g_norm_sum = 0
        
        for t in range(self.N - 1, -1, -1):  # backward recursion of Adjoint equations.
            g[t] = L_u[t] + np.matmul(F_u[t].T, p)
            p = L_x[t] + np.matmul(F_x[t].T, p) 
            g_norm_sum = g_norm_sum + np.linalg.norm(g[t])
            # P[t] = p

        return g, g_norm_sum/self.N
    
    def _initial_guess(self, x0):
        """Generate initial guess for shooting states based on reference

        Args:
            x0: initial state, [q0, vel0]

        Returns:
            xs: initial states trajectory
        """
        xs = []
        xs.append(x0) 
        for i in range(1, self.N+1):
            xs.append([ self.get_q_ref(i), self.get_xi_ref(i) ])
        return xs


class AL_iLQR_Tracking_SE3_MS(BaseController):
    """
    Finite Horizon Multiple Shooting with Constraints based on Augmented Lagrangian 
    Iterative Linear Quadratic Regulator for Exact SE3 Dynamics.
    """

    def __init__(self, dynamics, cost, constraints, N, 
                 q_ref, xi_ref,
                 mu_scale=10.,
                 max_reg=1e10, hessians=False,
                 line_search=False,
                 rollout='nonlinear', debug=None):
        """Constructs an iLQR solver for Error-State Dynamics

        Args:
            dynamics: Plant error-state dynamics.
            cost: Cost function.
            constraints: Inequality constraints.
            N: Horizon length.
            q_ref: Tracking reference configuration.
            xi_ref: Tracking reference velocity.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            line_search: indicator if line search is used or not in forward rollout
            rollout: Determine the rollout method, 'linear' or 'nonlinear'.
            debug: Indication of debug mode on or not
        """
        self.dynamics = dynamics
        self.cost = cost
        self.constr = constraints
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        self._action_size = dynamics.action_size
        self._state_size = dynamics.state_size
        self._error_state_size = dynamics._error_state_size
        self._constr_size = constraints.constr_size

        self._mu0 = 1e-2
        self._mu_scale = mu_scale
        self._mu_max = 1e8
        self._lmbd0 = np.zeros((self.N+1, self._constr_size))
        self._Imu0 = np.tile(
            self._mu0 * np.eye(self._constr_size), 
            (self.N + 1, 1, 1)
        )
        
        self.al = ALConstrainedCost( cost, constraints, N )
        self.ilqr_solver = iLQR_Tracking_SE3_MS(dynamics, self.al, N, 
                                                q_ref, xi_ref, 
                                                max_reg=max_reg,
                                                hessians=hessians,
                                                line_search=line_search,
                                                rollout='nonlinear',
                                                debug=debug)

        super(AL_iLQR_Tracking_SE3_MS, self).__init__()

    @property
    def xi_ref(self):
        return self._xi_ref
    
    @property
    def q_ref(self):
        return self._q_ref

    @property
    def state_size(self):
        return self._state_size

    @property
    def action_size(self):
        return self._action_size

    def fit(self, x0, us_init, 
            n_al_iters=100, n_ilqr_iters=200, 
            tol_J=1e-6, tol_grad_norm=1e-6, tol_constr=1e-2,
            on_iteration_al=None, on_iteration_ilqr=None):

        self.al.lmbd, self.al.Imu, self.al.mu = self._al_inital_param()

        constr_converged = False
        lmbd_hist = []
        mu_hist = []
        violation_hist = []
        nactive_hist = []

        for iteration in range( n_al_iters ):

            print("--------------------------------------------------------")
            print("AL Outer-Loop Iteration Start:", iteration)

            xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr, grad_hist_ilqr = \
                self.ilqr_solver.fit(   x0, us_init,
                                        n_iterations=n_ilqr_iters, 
                                        tol_J=1e-6, tol_grad_norm=1e-6,
                                        on_iteration=on_iteration_ilqr  )
            
            constr_eval = [ 
                self.constr.g( xs_ilqr[i], us_ilqr[i], i ) for i in range(self.N) 
            ]
            constr_eval.append( 
                self.constr.g( xs_ilqr[self.N], None, self.N, terminal=True ) 
            )
            constr_eval = np.array( constr_eval )
            
            if np.max( constr_eval ) < tol_constr:
                constr_converged = True

            if on_iteration_al:
                on_iteration_al( 
                    iteration, constr_converged,
                    self.al.lmbd, self.al.Imu, self.al.mu,
                    constr_eval, 
                    lmbd_hist, mu_hist, violation_hist, nactive_hist
                )

            if constr_converged:
                break

            self.al.lmbd, self.al.Imu, self.al.mu = self._al_update_param(constr_eval)

        return xs_ilqr, us_ilqr, J_hist_ilqr, xs_hist_ilqr, us_hist_ilqr, grad_hist_ilqr, \
                lmbd_hist, mu_hist, violation_hist, nactive_hist
                

    def _al_update_param( self, constr_eval ):

        lmbd = self.al.lmbd
        Imu = self.al.Imu
        mu = self.al.mu

        lmbd_new = lmbd.copy()
        mu_new = min( mu * self._mu_scale, self._mu_max )
        Imu_new = Imu.copy()

        for i in range( self.N+1 ):
            lmbd_new[i] = np.clip(
                lmbd[i] + Imu[i] @ constr_eval[i],
                a_min = 0., a_max=None
            )
            Imu_new[i] = np.diag( 
                # lmbd[i] or lmbd_new[i] ?
                np.where( (constr_eval[i] < 0.) & ( lmbd_new[i] == 0. ), 0., mu_new )
            )
        
        return lmbd_new, Imu_new, mu_new

    def _al_inital_param(self,):
        return self._lmbd0, self._Imu0, self._mu0


# =================================================================================
# Error Statte Controller 
# =================================================================================

class iLQR_Tracking_ErrorState_Approx(BaseController):

    """Finite Horizon Iterative Linear Quadratic Regulator for ErrorState Dynamics.
        Rollout implemented based on the linearized dynamics based on error-state."""

    def __init__(self, dynamics, cost, N, max_reg=1e10, hessians=False,
                 rollout='linear', debug=None):
        """Constructs an iLQR solver for Error-State Dynamics

        Args:
            dynamics: Plant error-state dynamics.
            cost: Cost function.
            N: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            rollout: Determine the rollout method, 'linear' or 'nonlinear'.
            debug: Indication of debug mode on or not
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._action_size = dynamics.action_size
        self._state_size = dynamics.state_size
        self._error_state_size  = dynamics._error_state_size

        self._rollout_mode = rollout
        self._debug = debug

        self._k = np.zeros((N, self._action_size))
        self._K = np.zeros((N, self._action_size, self._state_size))

        self.f_nonlinear = jit(self.dynamics._fd_euler_fc_group)

        super(iLQR_Tracking_ErrorState_Approx, self).__init__()

    def fit(self, x0, us_init, n_iterations=100, tol_J=1e-6, tol_grad_norm=1e-3,
             on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0
        # self._mu = 0.0

        # Add time 
        start_time = time.perf_counter()

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-np.arange(10)**2)

        us = us_init.copy()
        k = self._k
        K = self._K

        J_hist = []
        xs_hist = []
        us_hist = []
        qs_hist = []
        xis_hist = []

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Start Iteration:", iteration, ", Used Time:", time_calc )

            # Forward rollout only if it needs to be recomputed.
            if changed:
                (xs, qs, xis, F_x, F_u, L, L_x, L_u, 
                L_xx, L_ux, L_uu, F_xx, F_ux, F_uu) = self._linearization(x0, us)
                J_opt = L.sum()
                if len(xs_hist) == 0 and len(us_hist) == 0 :
                    xs_hist.append(xs.copy())
                    us_hist.append(us.copy())
                    qs_hist.append(qs.copy())
                    xis_hist.append(xis.copy())
                changed = False

                _, grad_wrt_input_norm = self._gradient_wrt_control( F_x, F_u, L_x, L_u )
                if grad_wrt_input_norm < tol_grad_norm:
                    converged = True
                    # accepted = True
                    changed = False
                    print("Iteration", iteration-1, "converged, gradient w.r.t. input:", grad_wrt_input_norm )
                    break
                print("Iteration:", iteration, "Gradient w.r.t. input:", grad_wrt_input_norm )
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Iteration:", iteration, "Linearization Finished, Used Time:", time_calc, "Cost:", J_opt )

            if converged == False:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                           F_xx, F_ux, F_uu)
                
                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration:", iteration, "Backward Pass Finished, Used Time:", time_calc )

                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._rollout(xs, us, k, K, F_x, F_u, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)

                    end_time = time.perf_counter()
                    time_calc = end_time - start_time   
                    print("Iteration:", iteration, "Rollout Finished, Used Time:", time_calc, "Alpha:", alpha, "Cost:", J_new )

                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol_J:
                            converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        changed = True

                        # Accept this.
                        accepted = True
                        break

            end_time = time.perf_counter()
            time_calc = end_time - start_time   
            print("Iteration:", iteration, "Rollout and Line Search Finished, Used Time:", time_calc )
           
            if on_iteration:
                on_iteration(iteration, xs, us, J_opt,
                            accepted, converged, grad_wrt_input_norm,
                            alpha, self._mu, 
                            J_hist, xs_hist, us_hist)
                    
            if converged:
                break

            if not accepted:
                warnings.warn("Problem infeasible, regularization and line search step exhausted")
                break

        # Store fit parameters.
        self._k = k
        self._K = K

        return xs, us, J_hist, \
            jnp.array(xs_hist), jnp.array(us_hist)

    def _rollout(self, xs, us, k, K, F_x, F_u, alpha=1.0):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
        """
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)

        xs_new[0] = xs[0].copy()

        for i in range(self.N):

            xs_err = xs_new[i] - xs[i]

            us_err = alpha * k[i] + K[i].dot(xs_err)
            
            us_new[i] = us[i] + us_err

            if self._rollout_mode == 'linear':

                xs_new[i + 1] = xs[i + 1] + F_x[i] @ xs_err + F_u[i] @ us_err

            elif self._rollout_mode == 'nonlinear':

                q_next, xi_next = self.f_nonlinear( 
                    self.dynamics._q_ref[i] @ expm(se3_hat(xs_new[i, :self._error_state_size])),
                    xs_new[i, self._error_state_size:],
                    us_new[i],
                    i
                )
                xs_new[i+1] = np.concatenate(
                    ( se3_vee( logm(self.dynamics._q_ref_inv[i+1] @ q_next) ) , xi_next ) 
                )
                
        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
                                                     range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _linearization(self, x0, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            x0: Initial state [state_size].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                xs: State path [N+1, state_size].
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us.shape[0]

        xs = np.empty((N + 1, state_size))
        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))

        if self.dynamics._debug.get('derivative_compare'):
            F_x_autodiff = np.empty((N, state_size, state_size))
            F_u_autodiff = np.empty((N, state_size, action_size))

        if self._use_hessians:
            F_xx = np.empty((N, state_size, state_size, state_size))
            F_ux = np.empty((N, state_size, action_size, state_size))
            F_uu = np.empty((N, state_size, action_size, action_size))
        else:
            F_xx = None
            F_ux = None
            F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))

        qs = self.dynamics.q_ref.copy()
        xis = self.dynamics.xi_ref.copy()   
     
        xs[0] = x0

        for i in range(N):
            x = xs[i]
            u = us[i]

            xs[i + 1] = self.dynamics.f(xs[i], u, i)

            if self.dynamics._debug.get('derivative_compare'):
                F_x[i], F_x_autodiff[i] = self.dynamics.f_x(x, u, i)
                if np.max( np.abs(F_x[i] - F_x_autodiff[i]) ) > 1e-6:
                    # print( np.abs(F_x[i] - F_x_autodiff[i]) )
                    pass
                F_u[i], F_u_autodiff[i] = self.dynamics.f_u(x, u, i)
                if np.max( np.abs(F_u[i] - F_u_autodiff[i]) ) > 1e-6:
                    # print( np.abs(F_u[i] - F_u_autodiff[i]) )
                    pass
            else:                  
                F_x[i] = self.dynamics.f_x(x, u, i)
                F_u[i] = self.dynamics.f_u(x, u, i)                     

            L[i] = self.cost.l(x, u, i, terminal=False)
            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x, u, i)
                F_ux[i] = self.dynamics.f_ux(x, u, i)
                F_uu[i] = self.dynamics.f_uu(x, u, i)

        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        L_x[-1] = self.cost.l_x(x, None, N, terminal=True)
        L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True)

        return xs, qs, xis, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

        Args:
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):

            while True:
                if self._use_hessians:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x, V_xx,
                                                        F_xx[i], F_ux[i], F_uu[i])
                else:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x, V_xx)
                
                if not is_pos_def(Q_uu + Q_uu.T):
                    # Increase regularization term.
                    self._delta = max(1.0, self._delta) * self._delta_0
                    self._mu = max(self._mu_min, self._mu * self._delta)


                    if self._mu_max and self._mu >= self._mu_max:
                        warnings.warn("exceeded max regularization term")
                        break
                else:
                    self._delta = min(1.0, self._delta) / self._delta_0
                    self._mu *= self._delta
                    if self._mu <= self._mu_min:
                            self._mu = 0.0
                    break

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self,
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

        Args:
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)
        # Q_uu = 0.5 * (Q_uu + Q_uu.T) 

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    
    def _gradient_wrt_control(self, F_x, F_u, L_x, L_u):
        """Solve adjoint equations using a for loop.

        Args:
            F_x: dynamics Jacobians with respect to state.
            F_u: dynamics Jacobians with respect to control.
            L_x: cost gradients with respect to state.
            L_u: cost gradients with respect to control.

        Returns:
            gradient, adjoints, final adjoint variable.
        """

        # P = np.zeros((self.N + 1, self._state_size))
        g = np.zeros((self.N, self._action_size))

        p = L_x[self.N] # Initialize adjoint variable with terminal cost gradient
        # P[self.N] = p
        g_norm_sum = 0
        
        for t in range(self.N - 1, -1, -1):  # backward recursion of Adjoint equations.
            g[t] = L_u[t] + np.matmul(F_u[t].T, p)
            p = L_x[t] + np.matmul(F_x[t].T, p) 
            g_norm_sum = g_norm_sum + np.linalg.norm(g[t])
            # P[t] = p

        return g, g_norm_sum/self.N


class iLQR_Generation_ErrorState_Approx_LinearRollout(BaseController):

    """Finite Horizon Iterative Linear Quadratic Regulator for ErrorState Dynamics.
        Rollout implemented based on the linearized dynamics based on error-state."""

    def __init__(self, dynamics, cost, N, max_reg=1e10, hessians=False,
                 debug=None):
        """Constructs an iLQR solver for Error-State Dynamics

        Args:
            dynamics: Plant error-state dynamics.
            cost: Cost function.
            N: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            debug: Indication of debug mode on or not
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._action_size = dynamics.action_size
        self._state_size = dynamics.state_size

        self._debug = debug

        self._k = np.zeros((N, self._action_size))
        self._K = np.zeros((N, self._action_size, self._state_size))

        super(iLQR_Generation_ErrorState_Approx_LinearRollout, self).__init__()

    def fit(self, x0, us_init, n_iterations=100, tol_J=1e-6, tol_grad_norm=1e-3,
             on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0
        # self._mu = 0.0

        # Add time 
        start_time = time.perf_counter()

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-np.arange(10)**2)

        us = us_init.copy()
        k = self._k
        K = self._K

        J_hist = []
        xs_hist = []
        us_hist = []
        qs_hist = []
        xis_hist = []

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Start Iteration:", iteration, ", Used Time:", time_calc )

            # Forward rollout only if it needs to be recomputed.
            if changed:
                (xs, qs, xis, F_x, F_u, L, L_x, L_u, 
                L_xx, L_ux, L_uu, F_xx, F_ux, F_uu) = self._linearization(x0, us)
                # (xs, qs, xis, F_x, F_x_autodiff, F_u, L, L_x, L_u, 
                #  L_xx, L_ux, L_uu, F_xx, F_ux, F_uu) = self._linearization(x0, us)
                J_opt = L.sum()
                if len(xs_hist) == 0 and len(us_hist) == 0 :
                    xs_hist.append(xs.copy())
                    us_hist.append(us.copy())
                    qs_hist.append(qs.copy())
                    xis_hist.append(xis.copy())
                changed = False

                _, grad_wrt_input_norm = self._gradient_wrt_control( F_x, F_u, L_x, L_u )
                if grad_wrt_input_norm < tol_grad_norm:
                    converged = True
                    # accepted = True
                    changed = False
                    break
                print("Iteration:", iteration, "Linearization gradient:", grad_wrt_input_norm )
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Iteration:", iteration, "Linearization Finished, Used Time:", time_calc, "Cost:", J_opt )

            if converged == False:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                           F_xx, F_ux, F_uu)
                
                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration:", iteration, "Backward Pass Finished, Used Time:", time_calc )

                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._rollout(xs, us, k, K, F_x, F_u, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)

                    end_time = time.perf_counter()
                    time_calc = end_time - start_time   
                    print("Iteration:", iteration, "Rollout Finished, Used Time:", time_calc, "Alpha:", alpha, "Cost:", J_new )

                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol_J:
                            converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        changed = True

                        # Accept this.
                        accepted = True
                        break

            end_time = time.perf_counter()
            time_calc = end_time - start_time   
            print("Iteration:", iteration, "Rollout and Line Search Finished, Used Time:", time_calc )

            if accepted:

                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print(f"Iteration {iteration} start reference update, Used Time: {time_calc}")

                qs_new, xis_new = self.dynamics.ref_reinitialize( xs )
                qs = qs_new
                xis = xis_new

                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration", iteration, "dynamics reinitialization finished, Used Time:", time_calc )

                new_X_ref = vec_SE32quatpos( qs_new )
                _ = self.cost.ref_reinitialize( new_X_ref )

                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration", iteration, "cost reinitialization finished, Used Time:", time_calc )

            
            if on_iteration:
                on_iteration(iteration, xs, us, qs, xis, J_opt,
                            accepted, converged, changed, 
                            grad_wrt_input_norm,
                            alpha, self._mu, 
                            J_hist, xs_hist, us_hist, qs_hist, xis_hist)
                    
            if converged:
                break

            if not accepted:
                warnings.warn("Problem infeasible, regularization and line search step exhausted")
                break

        # Store fit parameters.
        self._k = k
        self._K = K

        return xs, us, qs, J_hist, \
            np.array(xs_hist), np.array(us_hist), \
            np.array(qs_hist), np.array(xis_hist) 

    def _rollout(self, xs, us, k, K, F_x, F_u, alpha=1.0):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
        """
        xs_new = np.zeros_like(xs)
        # xs_new = xs.copy()
        us_new = np.zeros_like(us)
        xs_new[0] = xs[0].copy()

        xs_new2 = np.zeros_like(xs)
        us_new2 = np.zeros_like(us)
        xs_new2[0] = xs[0].copy()

        for i in range(self.N):
            # # Eq (12).
            # us_new[i] = us[i] + alpha * k[i] + K[i].dot(xs_new[i] - xs[i])

            # # Eq (8c).
            # xs_new[i + 1] = self.dynamics.f(xs_new[i], us_new[i], i)

            # --------------------------------------------------------- #

            xs_err = xs_new2[i] - xs[i]

            us_err = alpha * k[i] + K[i].dot(xs_err)
            us_new2[i] = us[i] + us_err

            xs_new2[i + 1] = xs[i + 1] + F_x[i] @ xs_err + F_u[i] @ us_err

        # return xs_new, us_new
        return xs_new2, us_new2

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
                                                     range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _linearization(self, x0, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            x0: Initial state [state_size].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                xs: State path [N+1, state_size].
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us.shape[0]

        xs = np.empty((N + 1, state_size))
        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))

        if self.dynamics._debug.get('derivative_compare'):
            F_x_autodiff = np.empty((N, state_size, state_size))
            F_u_autodiff = np.empty((N, state_size, action_size))

        if self._use_hessians:
            F_xx = np.empty((N, state_size, state_size, state_size))
            F_ux = np.empty((N, state_size, action_size, state_size))
            F_uu = np.empty((N, state_size, action_size, action_size))
        else:
            F_xx = None
            F_ux = None
            F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))

        xs = [ np.concatenate(( np.zeros((6,)), xirefi.reshape(6,) )) 
              for xirefi in self.dynamics._xi_ref ]
        xs = np.array(xs)   
        qs = self.dynamics.q_ref.copy()
        xis = self.dynamics.xi_ref.copy()   
     
        # xs[0] = x0

        xs_debug = np.empty((N + 1, state_size))
        xs_debug[0] = xs[0]

        for i in range(N):
            x = xs[i]
            u = us[i]

            xs_debug[i + 1] = self.dynamics.f(xs_debug[i], u, i)

            if self.dynamics._debug.get('derivative_compare'):
                F_x[i], F_x_autodiff[i] = self.dynamics.f_x(x, u, i)
                if np.max( np.abs(F_x[i] - F_x_autodiff[i]) ) > 1e-6:
                    # print( np.abs(F_x[i] - F_x_autodiff[i]) )
                    pass
                F_u[i], F_u_autodiff[i] = self.dynamics.f_u(x, u, i)
                if np.max( np.abs(F_u[i] - F_u_autodiff[i]) ) > 1e-6:
                    # print( np.abs(F_u[i] - F_u_autodiff[i]) )
                    pass
            else:                  
                F_x[i] = self.dynamics.f_x(x, u, i)
                F_u[i] = self.dynamics.f_u(x, u, i)                     

            L[i] = self.cost.l(x, u, i, terminal=False)
            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x, u, i)
                F_ux[i] = self.dynamics.f_ux(x, u, i)
                F_uu[i] = self.dynamics.f_uu(x, u, i)

        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        L_x[-1] = self.cost.l_x(x, None, N, terminal=True)
        L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True)

        return xs, qs, xis, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

        Args:
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):

            while True:
                if self._use_hessians:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x, V_xx,
                                                        F_xx[i], F_ux[i], F_uu[i])
                else:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x, V_xx)
                
                if not is_pos_def(Q_uu + Q_uu.T):
                    # Increase regularization term.
                    self._delta = max(1.0, self._delta) * self._delta_0
                    self._mu = max(self._mu_min, self._mu * self._delta)


                    if self._mu_max and self._mu >= self._mu_max:
                        warnings.warn("exceeded max regularization term")
                        break
                else:
                    self._delta = min(1.0, self._delta) / self._delta_0
                    self._mu *= self._delta
                    if self._mu <= self._mu_min:
                            self._mu = 0.0
                    break

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self,
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

        Args:
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)
        # Q_uu = 0.5 * (Q_uu + Q_uu.T) 

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    
    def _gradient_wrt_control(self, F_x, F_u, L_x, L_u):
        """Solve adjoint equations using a for loop.

        Args:
            F_x: dynamics Jacobians with respect to state.
            F_u: dynamics Jacobians with respect to control.
            L_x: cost gradients with respect to state.
            L_u: cost gradients with respect to control.

        Returns:
            gradient, adjoints, final adjoint variable.
        """

        # P = np.zeros((self.N + 1, self._state_size))
        g = np.zeros((self.N, self._action_size))

        p = L_x[self.N] # Initialize adjoint variable with terminal cost gradient
        # P[self.N] = p
        g_norm_sum = 0
        
        for t in range(self.N - 1, -1, -1):  # backward recursion of Adjoint equations.
            g[t] = L_u[t] + np.matmul(F_u[t].T, p)
            p = L_x[t] + np.matmul(F_x[t].T, p) 
            g_norm_sum = g_norm_sum + np.linalg.norm(g[t])
            # P[t] = p

        return g, g_norm_sum/self.N


class iLQR_Generation_ErrorState_Approx_NonlinearRollout(BaseController):

    """Finite Horizon Iterative Linear Quadratic Regulator for ErrorState Dynamics.
        Rollout implemented based on the nonlinear dynamics with exp map."""

    def __init__(self, dynamics, cost, N, max_reg=1e10, 
                 hessians=False, autodiff_dyn=True):
        """Constructs an iLQR solver for Error-State Dynamics

        Args:
            dynamics: Plant error-state dynamics.
            cost: Cost function.
            N: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            tracking: Indication of either controller tracks the 
                reference trajectory, or generates a trajecotory towards the 
                given goal configuration, when using the error-state dynamics.
            autodiff_dyn: Indication of either use autodiff to obtain the
                derivative of dynamics or use the analytical jacobian
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        # self._mu_min = 1e-3
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._action_size = dynamics.action_size
        self._state_size = dynamics.state_size
        self._error_state_size = dynamics.error_state_size
        self._vel_state_size = dynamics.vel_state_size

        self._k = np.zeros((N, self._action_size))
        self._K = np.zeros((N, self._action_size, self._state_size))

        super(iLQR_Generation_ErrorState_Approx_NonlinearRollout, self).__init__()

    def fit(self, x0, us_init, n_iterations=100, tol_J=1e-6, tol_grad_norm=1e-3,
             on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0
        # self._delta = self._delta_0

        # Add time 
        start_time = time.perf_counter()

        # Backtracking line search candidates 0 < alpha <= 1.
        # alphas = 1.1**(-np.arange(10)**2)
        alphas = 1.1**(-np.arange(15)**2)

        us = us_init.copy()
        k = self._k
        K = self._K

        J_hist = []
        xs_hist = []
        us_hist = []
        qs_hist = []
        xis_hist = []

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Start Iteration:", iteration, ", Used Time:", time_calc )

            # Forward rollout only if it needs to be recomputed.
            if changed:
                (xs, qs, xis, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux,
                 F_uu) = self._linearization(x0, us)
                J_opt = L.sum()
                if len(xs_hist) == 0 and len(us_hist) == 0 :
                    xs_hist.append(xs.copy())
                    us_hist.append(us.copy())
                    qs_hist.append(qs.copy())
                    xis_hist.append(xis.copy())
                changed = False
            
            end_time = time.perf_counter()
            time_calc = end_time - start_time
            print("Iteration:", iteration, "Linearization Finished, Used Time:", time_calc, "Cost:", J_opt )

            try:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                           F_xx, F_ux, F_uu)
                
                end_time = time.perf_counter()
                time_calc = end_time - start_time   
                print("Iteration:", iteration, "Backward Pass Finished, Used Time:", time_calc )

                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new, qs_new, xis_new = self._rollout(xs, us, qs, xis, k, K, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)
                    
                    end_time = time.perf_counter()
                    time_calc = end_time - start_time   
                    print("Iteration:", iteration, "Rollout Finished, Used Time:", time_calc, "Cost:", J_new)

                    _, grad_wrt_input_norm = self._gradient_wrt_control( F_x, F_u, L_x, L_u )
                    if grad_wrt_input_norm < tol_grad_norm:

                        converged = True
                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        qs = qs_new
                        xis = xis_new
                        changed = True
                        accepted = True
                        break

                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol_J:
                            converged = True

                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        qs = qs_new
                        xis = xis_new
                        changed = True

                        # Accept this.
                        accepted = True
                        break

            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            end_time = time.perf_counter()
            time_calc = end_time - start_time   
            print("Iteration:", iteration, "Rollout and Line Search Finished, Used Time:", time_calc )

            # TODO: not sure if reinitialization should be here 
            # or after converged checkpoint ??
            if accepted :

                self.dynamics._q_ref = qs_new
                self.dynamics._xi_ref = xis_new

                new_X_ref = vec_SE32quatpos( qs_new )
                _ = self.cost.ref_reinitialize( new_X_ref )

            if converged:
                break

            if on_iteration:
                on_iteration(iteration, xs, us, qs, xis, J_opt,
                             accepted, converged, grad_wrt_input_norm,
                             alpha, self._mu, 
                             J_hist, xs_hist, us_hist, qs_hist, xis_hist)
            
            if not accepted:
                warnings.warn("Problem infeasible, regularization and line search step exhausted")
                break


        # Store fit parameters.
        self._k = k
        self._K = K

        return xs, us, qs, J_hist, \
            np.array(xs_hist), np.array(us_hist), \
            np.array(qs_hist), np.array(xis_hist) 

    def _rollout(self, xs, us, qs, xis, k, K, alpha=1.0):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
        """
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        qs_new = np.zeros_like(qs)
        xis_new = np.zeros_like(xis)

        xs_new[0] = xs[0].copy()
        qs_new[0] = qs[0].copy()
        xis_new[0] = xis[0].copy()

        for i in range(self.N):
            us_new[i] = us[i] + alpha * k[i] + K[i].dot(xs_new[i] - xs[i])

            qs_new[i + 1], xis_new[i + 1] = \
                self.dynamics.f_rollout(qs_new[i], xis_new[i], us_new[i], i)
            
            xs_new[i + 1, :self._error_state_size] = se3_vee(logm(
                inv( qs[i+1] ) @ qs_new[i + 1]
            ).real )

            xs_new[i+1, self._error_state_size:] = xis_new[i+1]

        return xs_new, us_new, qs_new, xis_new 

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us,
                                                     range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _linearization(self, x0, us):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            x0: Initial state [state_size].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                xs: State path [N+1, state_size].
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us.shape[0]

        xs = np.empty((N + 1, state_size))
        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))

        if self._use_hessians:
            F_xx = np.empty((N, state_size, state_size, state_size))
            F_ux = np.empty((N, state_size, action_size, state_size))
            F_uu = np.empty((N, state_size, action_size, action_size))
        else:
            F_xx = None
            F_ux = None
            F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))

        xs = [ np.concatenate(( np.zeros((6,)), xirefi.reshape(6,) )) 
              for xirefi in self.dynamics.xi_ref ]
        xs = np.array(xs)
        qs = self.dynamics.q_ref.copy()
        xis = self.dynamics.xi_ref.copy()

        for i in range(N):
            x = xs[i]
            u = us[i]

            # xs[i + 1] = self.dynamics.f(x, u, i)
            F_x[i] = self.dynamics.f_x(x, u, i)
            F_u[i] = self.dynamics.f_u(x, u, i)

            L[i] = self.cost.l(x, u, i, terminal=False)
            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x, u, i)
                F_ux[i] = self.dynamics.f_ux(x, u, i)
                F_uu[i] = self.dynamics.f_uu(x, u, i)

        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        L_x[-1] = self.cost.l_x(x, None, N, terminal=True)
        L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True)

        return xs, qs, xis, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

        Args:
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):

            while True:
                if self._use_hessians:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x, V_xx,
                                                        F_xx[i], F_ux[i], F_uu[i])
                else:
                    Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                        L_u[i], L_xx[i], L_ux[i],
                                                        L_uu[i], V_x, V_xx)

                if not is_pos_def( Q_uu + Q_uu.T ):
                    # 如果不是正定的，增大正则化参数 mu，并重新计算
                    self._delta = max(1.0, self._delta) * self._delta_0
                    self._mu = max(self._mu_min, self._mu * self._delta)
                    if self._mu_max and self._mu >= self._mu_max:
                        warnings.warn("exceeded max regularization term during backward pass")
                        break

                    # self._mu = self._mu * 4
                    # if self._mu_max and self._mu >= self._mu_max:
                    #     warnings.warn("exceeded max regularization term during backward pass")
                    #     break
                else:
                    self._delta = min(1.0, self._delta) / self._delta_0
                    self._mu *= self._delta
                    if self._mu <= self._mu_min:
                            self._mu = 0.0
                    break
                    # self._mu = max(self._mu / 4, self._mu_min)
                    # break


            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self,
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

        Args:
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

        # Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x) +  self._mu * np.eye(self.dynamics.state_size)
        # Q_ux = l_ux + f_u.T.dot(V_xx).dot(f_x)
        # Q_uu = l_uu + f_u.T.dot(V_xx).dot(f_u) +  self._mu * np.eye(self.dynamics.action_size)

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu
    
    def _gradient_wrt_control(self, F_x, F_u, L_x, L_u):
        """Solve adjoint equations using a for loop.

        Args:
            F_x: dynamics Jacobians with respect to state.
            F_u: dynamics Jacobians with respect to control.
            L_x: cost gradients with respect to state.
            L_u: cost gradients with respect to control.

        Returns:
            gradient, adjoints, final adjoint variable.
        """

        # P = np.zeros((self.N + 1, self._state_size))
        g = np.zeros((self.N, self._action_size))

        p = L_x[self.N] # Initialize adjoint variable with terminal cost gradient
        # P[self.N] = p
        g_norm_sum = 0
        
        for t in range(self.N - 1, -1, -1):  # backward recursion of Adjoint equations.
            g[t] = L_u[t] + np.matmul(F_u[t].T, p)
            p = L_x[t] + np.matmul(F_x[t].T, p) 
            g_norm_sum = g_norm_sum + np.linalg.norm(g[t])
            # P[t] = p

        return g, g_norm_sum/self.N
