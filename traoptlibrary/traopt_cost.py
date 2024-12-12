import abc
import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd, hessian, jit
from traoptlibrary.traopt_utilis import adjoint, quatpos2SE3, se3_vee, SE32manifSE3, \
                                        parallel_SE32manifSE3, Jmnf2J, manifse32se3
from jax.scipy.linalg import expm
from scipy.linalg import logm
from functools import partial
from manifpy import SE3, SE3Tangent, SO3, SO3Tangent
from scipy.spatial.transform import Rotation 

class BaseCost():

    """Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    @abc.abstractmethod
    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        raise NotImplementedError
    

class AutoDiffCost(BaseCost):

    """Auto-differentiated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

    def __init__(self, l, l_terminal, state_size, action_size, **kwargs):
        """Constructs an AutoDiffCost.

        Args:
            l: Function for instantaneous stage cost.
                This needs to be a function of x and u and must return a scalar.
                Args:
                    x: Current state [state_size].
                    u: Current control [action_size].
                    i: Current time step.
            l_terminal: Function for terminal cost.
                This needs to be a function of x only and must retunr a scalar.
                Args:
                    x: Current state [state_size].
                    i: Current time step.
            state_size: State variable dimension.
            action_size: Action variable dimension.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._state_size = state_size
        self._action_size = action_size

        self._l = jit(l)
        self._l_x = jit(jacfwd(l))
        self._l_u = jit(jacfwd(l, argnums=1))

        self._l_xx = jit(hessian(l, argnums=0))
        self._l_ux = jit(jacfwd( jacfwd(l, argnums=1) ))
        self._l_uu = jit(hessian(l, argnums=1))

        # Terminal cost only depends on x, so we only need to evaluate the x
        # partial derivatives.

        self._l_terminal = jit(l_terminal)
        self._l_x_terminal = jit(jacfwd(l_terminal))
        self._l_xx_terminal = jit(hessian(l_terminal))

        super(AutoDiffCost, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size
    
    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self._l_terminal(x,i)

        return self._l(x,u,i)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return self._l_x_terminal(x,i)

        return self._l_x(x,u,i)

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        return self._l_u(x,u,i)

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        if terminal:
            return self._l_xx_terminal(x,i)

        return self._l_xx(x,u,i)

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        return self._l_ux(x,u,i)

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        return self._l_uu(x,u,i)

# =================================================================================
# SO3 Cost
# =================================================================================

class SO3TrackingQuadraticGaussNewtonCost(BaseCost):

    """
        Stage Cost defined on Lie Algebra for SE(3) ErrorDynamics.
        - Implemented with Jax for autodiff.
        - 2nd order quadratic cost, penalizing both position deviation and velocity deviation.
        - Used for tracking the given reference trajectory
        - Used for exact dynamics, i.e. state is \Psi
        - Stage cost 
                l( (X_k, xi_k), u_k, k ) = ||Log(Xbar_k^{-1} X_k)||^2_Q1 
                                            + ||xi_k - xibar_k||^2_Q2
                                            + ||u||^2_R
        - Terminal cost
                l( (X_k, xi_k), u_k, k ) = ||Log(Xbar_N^{-1} X_N)||^2_P1 
                                            + ||xi_N - xibar_N||^2_P2
    """

    def __init__(self, Q, R, P, q_ref, xi_ref,
                 state_size=(3,3), action_size=3, **kwargs):
        """Constructor.

        Args:
            Q: State weighting matrix for the stage cost. Shape: [state_size, state_size].
                This matrix penalizes the state deviation (pos & vel) from the reference at each time step.
            R: Control weighting matrix for the stage cost. Shape: [action_size, action_size].
                This matrix penalizes the magnitude of control inputs at each time step.
            P: State weighting matrix for the terminal cost. Shape: [state_size, state_size].
                This matrix penalizes the state deviation (pos & vel) from the reference at the final time step.
            q_ref: configuration reference, list of SO(3) matrix,
                 (N, 3, 3)
            xi_ref: velocity reference along the trajectory, described in Lie Algebra,
                 (N, velocity_size,)
            state_size: Tuple of State variable dimension, 
                ( error state size, velocity state size ).
            action_size: Input variable dimension.
            **kwargs: Additional keyword-arguments for backup usage.
        """
        self._state_size = state_size[0] + state_size[1]
        self._pos_state_size = state_size[0] 
        self._vel_state_size = state_size[1] 
        self._action_size = action_size

        self._q_ref = [ SO3( Rotation.from_matrix(q).as_quat()) for q in q_ref ] 
        self._xi_ref = [ SO3Tangent(xi) for xi in xi_ref ]

        self._Q = Q
        self._R = R
        self._P = P

        super(SO3TrackingQuadraticGaussNewtonCost, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size
    
    @property
    def pos_state_size(self):
        """Position state size."""
        return self._pos_state_size

    @property
    def vel_state_size(self):
        """Velocity state size."""
        return self._vel_state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size
    
    @property
    def Q(self):
        """State cost coefficient matrix in the quadratic stage cost."""
        return self._Q
    
    @property
    def R(self):
        """Input cost coefficient matrix in the quadratic stage cost."""
        return self._R
    
    @property
    def P(self):
        """State cost coefficient matrix in the quadratic terminal cost."""
        return self._P
    
    def _err(self, x, i):
        """Return the error with the reference
        """
        q, xi = x 

        q_ref = self._q_ref[i] 
        xi_ref = self._xi_ref[i]

        q_err = q.lminus( q_ref ).coeffs()
        # q_err = q.rminus( q_ref ).coeffs()

        vel_err = (xi - xi_ref).coeffs()

        return q_err, vel_err
    
    def _l(self, x, u, i ):
        """Stage cost function.

        Args:
            x: Current state, tuple of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.

        Returns:
            Instantaneous cost (scalar).
        """
        q, xi = x 
        u = u.reshape(self.action_size, 1)

        q_ref = self._q_ref[i] 
        xi_ref = self._xi_ref[i]

        # Compute the logarithmic map of the pose error.
        # q_err = q.rminus( q_ref ).coeffs()
        q_err = q.lminus( q_ref ).coeffs()
        q_err = q_err.reshape(self.pos_state_size,1)
        q_cost = q_err.T @ self._Q[:self.pos_state_size, :self.pos_state_size] @ q_err

        # Compute velocity error.
        vel_err = (xi - xi_ref).coeffs()
        vel_err = vel_err.reshape(self.vel_state_size,1)
        v_cost = vel_err.T @ self._Q[self.vel_state_size:, self.vel_state_size:] @ vel_err

        # Compute control cost.
        u_cost = u.T @ self._R @ u

        return (q_cost + v_cost + u_cost).reshape(-1)[0]

    def _l_terminal(self, x, i):
        """Terminal cost function.

        Args:
            x: Current state, list of pose and twist, [q, xi]:
                q:  SE3 matrix, 4x4
                xi: twist velocity
            i: Current time step.

        Returns:
            Instantaneous cost (scalar).
        """
        q, xi = x 

        q_ref = self._q_ref[i] 
        xi_ref = self._xi_ref[i]

        # Compute the logarithmic map of the pose error.
        # q_err = q.rminus( q_ref ).coeffs().reshape(self.pos_state_size,1)
        q_err = q.lminus( q_ref ).coeffs().reshape(self.pos_state_size,1)
        q_cost = q_err.T @ self._Q[:self.pos_state_size, :self.pos_state_size] @ q_err

        # Compute velocity error.
        vel_err = (xi - xi_ref).coeffs().reshape(self.vel_state_size,1)
        v_cost = vel_err.T @ self._Q[self.vel_state_size:, self.vel_state_size:] @ vel_err

        return (q_cost + v_cost).reshape(-1)[0]

    
    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self._l_terminal(x,i)

        return self._l(x,u,i)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        q, xi = x 

        q_ref = self._q_ref[i] 
        xi_ref = self._xi_ref[i]

        J_e_x = np.empty((3,3))
        # err = q.rminus(q_ref, J_e_x).reshape(6,1)
        q_err = q.lminus(q_ref, J_e_x).coeffs().reshape(3,1)
        J_q = (J_e_x.T * 2) @ self._Q[:self.pos_state_size, :self.pos_state_size] @ q_err

        J_xi = 2 * self._Q[self.pos_state_size:, self.pos_state_size:] \
            @ ( xi - xi_ref ).coeffs().reshape(3,1)
        
        return np.vstack(
            (J_q, J_xi)
        ).reshape((self.state_size,))

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        return 2 * self._R @ u

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.
            Derived using Gauss-Newton.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        q, _ = x 

        q_ref = self._q_ref[i] 

        J_e_x = np.empty((3,3))
        # _ = q.rminus(q_ref, J_e_x)
        _ = q.lminus(q_ref, J_e_x)

        blk_size = self.pos_state_size
        H_err = (J_e_x.T * 2) @ self._Q[:blk_size, :blk_size] @ J_e_x
        H_xi = 2 * self._Q[blk_size:, blk_size:]
        
        return np.block([
            [ H_err, np.zeros((blk_size,blk_size)) ],
            [ np.zeros((blk_size,blk_size)), H_xi  ],
        ])

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.action_size,self.state_size))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        return 2 * self.R

# =================================================================================
# SE3 Cost
# =================================================================================

class SE3TrackingQuadraticGaussNewtonCost(BaseCost):

    """
        Stage Cost defined on Lie Algebra for SE(3) ErrorDynamics.
        - Implemented with Jax for autodiff.
        - 2nd order quadratic cost, penalizing both position deviation and velocity deviation.
        - Used for tracking the given reference trajectory
        - Used for exact dynamics, i.e. state is \Psi
        - Stage cost 
                l( (X_k, xi_k), u_k, k ) = ||Log(Xbar_k^{-1} X_k)||^2_Q1 
                                            + ||xi_k - xibar_k||^2_Q2
                                            + ||u||^2_R
        - Terminal cost
                l( (X_k, xi_k), u_k, k ) = ||Log(Xbar_N^{-1} X_N)||^2_P1 
                                            + ||xi_N - xibar_N||^2_P2
    """

    def __init__(self, Q, R, P, q_ref, xi_ref,
                 state_size=(6,6), action_size=6, **kwargs):
        """Constructor.

        Args:
            Q: State weighting matrix for the stage cost. Shape: [state_size, state_size].
                This matrix penalizes the state deviation (pos & vel) from the reference at each time step.
            R: Control weighting matrix for the stage cost. Shape: [action_size, action_size].
                This matrix penalizes the magnitude of control inputs at each time step.
            P: State weighting matrix for the terminal cost. Shape: [state_size, state_size].
                This matrix penalizes the state deviation (pos & vel) from the reference at the final time step.
            q_ref: configuration reference, list of SE(3) matrix,
                 (N, 4, 4)
            xi_ref: velocity reference along the trajectory, described in Lie Algebra,
                 (N, velocity_size,)
            state_size: Tuple of State variable dimension, 
                ( error state size, velocity state size ).
            action_size: Input variable dimension.
            **kwargs: Additional keyword-arguments for backup usage.
        """
        self._state_size = state_size[0] + state_size[1]
        self._error_state_size = state_size[0] 
        self._vel_state_size = state_size[1] 
        self._action_size = action_size

        # print("Cost: Start Reference Manifization")
        # self._q_ref = [ SE32manifSE3(q) for q in q_ref ]
        self._q_ref = parallel_SE32manifSE3(q_ref)
        # print("Cost: End Manifization")
        self._xi_ref = xi_ref

        self._Q = Q
        self._R = R
        self._P = P

        super(SE3TrackingQuadraticGaussNewtonCost, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size
    
    @property
    def error_state_size(self):
        """Error-state size."""
        return self._error_state_size

    @property
    def vel_state_size(self):
        """Velocity state size."""
        return self._vel_state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size
    
    @property
    def Q(self):
        """State cost coefficient matrix in the quadratic stage cost."""
        return self._Q
    
    @property
    def R(self):
        """Input cost coefficient matrix in the quadratic stage cost."""
        return self._R
    
    @property
    def P(self):
        """State cost coefficient matrix in the quadratic terminal cost."""
        return self._P
    
    def _err(self, x, i):
        """Return the error with the reference
        """
        q, xi = x 
        q = SE32manifSE3( q )

        q_ref = self._q_ref[i] 
        xi_ref = self._xi_ref[i]

        q_err = manifse32se3( q.lminus( q_ref ).coeffs() )
        # q_err = manifse32se3( q.rminus( q_ref ).coeffs() )

        vel_err = xi - xi_ref

        return q_err, vel_err
    
    def _l(self, x, u, i ):
        """Stage cost function.

        Args:
            x: Current state, tuple of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.

        Returns:
            Instantaneous cost (scalar).
        """
        q, xi = x 
        q = SE32manifSE3( q )
        u = u.reshape(self.action_size, 1)

        q_ref = self._q_ref[i] 
        xi_ref = self._xi_ref[i]

        # Compute the logarithmic map of the pose error.
        # q_err = q.rminus( q_ref ).coeffs()
        q_err = q.lminus( q_ref ).coeffs()
        q_err = manifse32se3( q_err ).reshape(6,1)
        q_cost = q_err.T @ self._Q[:self._error_state_size, :self._error_state_size] @ q_err

        # Compute velocity error.
        vel_err = xi - xi_ref
        vel_err = vel_err.reshape(6,1)
        v_cost = vel_err.T @ self._Q[self._error_state_size:, self._error_state_size:] @ vel_err

        # Compute control cost.
        u_cost = u.T @ self._R @ u

        return (q_cost + v_cost + u_cost).reshape(-1)[0]

    def _l_terminal(self, x, i):
        """Terminal cost function.

        Args:
            x: Current state, list of pose and twist, [q, xi]:
                q:  SE3 matrix, 4x4
                xi: twist velocity
            i: Current time step.

        Returns:
            Instantaneous cost (scalar).
        """
        q, xi = x 
        q = SE32manifSE3( q )

        q_ref = self._q_ref[i] 
        xi_ref = self._xi_ref[i]

        # Compute the logarithmic map of the pose error.
        # q_err = q.rminus( q_ref ).coeffs()
        q_err = q.lminus( q_ref ).coeffs()
        q_err = manifse32se3(q_err).reshape(6,1)
        q_cost = q_err.T @ self._Q[:self._error_state_size, :self._error_state_size] @ q_err

        # Compute velocity error.
        vel_err = xi - xi_ref
        vel_err = vel_err.reshape(6,1)
        v_cost = vel_err.T @ self._Q[self._error_state_size:, self._error_state_size:] @ vel_err

        return (q_cost + v_cost).reshape(-1)[0]

    
    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self._l_terminal(x,i)

        return self._l(x,u,i)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        q, xi = x 
        q = SE32manifSE3( q )

        q_ref = self._q_ref[i] 
        xi_ref = self._xi_ref[i]

        J_e_x = np.empty((6,6))
        # err = manifse32se3(q.rminus(q_ref, J_e_x)).reshape(6,1)
        err = manifse32se3(q.lminus(q_ref, J_e_x)).reshape(6,1)
        J_e_x = Jmnf2J(J_e_x)
        J_err = (J_e_x.T * 2) @ self._Q[:self._error_state_size, :self._error_state_size] @ err

        J_xi = 2 * self._Q[self._error_state_size:, self._error_state_size:] @ ( xi - xi_ref ).reshape(6,1)
        
        return np.vstack(
            (J_err, J_xi)
        ).reshape((self.state_size,))

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        return 2 * self._R @ u

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.
            Derived using Gauss-Newton.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        q, _ = x 
        q = SE32manifSE3( q )

        q_ref = self._q_ref[i] 

        J_e_x = np.empty((6,6))
        # _ = q.rminus(q_ref, J_e_x)
        _ = q.lminus(q_ref, J_e_x)
        J_e_x = Jmnf2J(J_e_x)
        H_err = (J_e_x.T * 2) @ self._Q[:self._error_state_size, :self._error_state_size] @ J_e_x

        H_xi = 2 * self._Q[self._error_state_size:, self._error_state_size:]
        
        return np.block([
            [ H_err, np.zeros((6,6)) ],
            [ np.zeros((6,6)), H_xi  ],
        ])

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        return np.zeros((self.action_size,self.state_size))

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state, list of pose and twist, [q, xi].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        return 2 * self._R

# =================================================================================
# Lagrangian Cost
# =================================================================================

class ALConstrainedCost(BaseCost):

    """
        Given cost and constraints, construct the Augmented Lagrangian LA:
            LA( (X_k, xi_k), u_k, k ) = l( x, u, k )
                                        + lambda^T * g(x, u)
                                        + 1/2 * g(x, u).T * Imu * g(x, u)
    """

    def __init__(self, cost, constraints, N,
                 state_size=(6,6), action_size=6, **kwargs):
        """Constructor.

        Args:
            cost: Instantied cost
            constraints: Instantiated constraints from traoptlibrary, 
                e.g. input constraints
            state_size: Tuple of State variable dimension, 
                ( error state size, velocity state size ).
            action_size: Input variable dimension.
            **kwargs: Additional keyword-arguments for backup usage.
        """
        self._state_size = state_size[0] + state_size[1]
        self._error_state_size = state_size[0] 
        self._vel_state_size = state_size[1] 
        self._action_size = action_size
        self._constr_size = constraints.constr_size

        self.constr = constraints
        self.cost = cost
        self.N = N
        
        self.lmbd = np.zeros(( N+1, self._constr_size ))
        self.mu = 0.
        self.Imu = np.zeros(( N+1, self._constr_size, self._constr_size))

        super(ALConstrainedCost, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size
    
    @property
    def error_state_size(self):
        """Error-state size."""
        return self._error_state_size

    @property
    def vel_state_size(self):
        """Velocity state size."""
        return self._vel_state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def constr_size(self):
        """Number of constraints at each stage."""
        return self._constr_size
        
    def l(self, x, u, i, terminal=False):

        if terminal:
            g_reshape = self.constr.g( x,u,i, terminal=True ).reshape( self.constr_size,1 )
            LA = self.cost.l( x, u, i, terminal=True ) \
                + self.lmbd[i].T @ self.constr.g( x, u, i, terminal=True ) \
                + 0.5 * (g_reshape.T @ self.Imu[i] @ g_reshape).reshape(-1)[0]
            return LA

        g_reshape = self.constr.g( x,u,i ).reshape( self.constr_size,1 )
        LA = self.cost.l( x,u,i ) + self.lmbd[i].T @ self.constr.g( x,u,i ) \
            + 0.5 * (g_reshape.T @ self.Imu[i] @ g_reshape).reshape(-1)[0]

        return LA

    def l_x(self, x, u, i, terminal=False):

        if terminal:
            g = self.constr.g(x,u,i, terminal=True )
            gx = self.constr.g_x( x,u,i, terminal=True )
            lx = self.cost.l_x( x,u,i, terminal=True) \
                + gx.T @ ( self.lmbd[i] + self.Imu[i] @ g )
            return lx
        
        g = self.constr.g(x,u,i)
        gx = self.constr.g_x( x,u,i )
        lx = self.cost.l_x( x,u,i ) \
            + gx.T @ ( self.lmbd[i] + self.Imu[i] @ g )
        return lx

    def l_u(self, x, u, i, terminal=False):

        if terminal:
            g = self.constr.g(x,u,i, terminal=True )
            gu = self.constr.g_u( x,u,i, terminal=True )
            lu = self.cost.l_u( x,u,i, terminal=True) \
                + gu.T @ ( self.lmbd[i] + self.Imu[i] @ g )
            return lu
        
        g = self.constr.g( x,u,i )
        gu = self.constr.g_u( x,u,i )
        lu = self.cost.l_u( x,u,i ) \
            + gu.T @ ( self.lmbd[i] + self.Imu[i] @ g )
        return lu

    def l_uu(self, x, u, i, terminal=False):

        if terminal:
            gu = self.constr.g_u( x,u,i, terminal=True)
            luu = self.cost.l_uu( x,u,i, terminal=True) \
                + gu.T @ self.Imu[i] @ gu
            return luu
        
        gu = self.constr.g_u( x,u,i )
        luu = self.cost.l_uu( x,u,i ) \
            + gu.T @ self.Imu[i] @ gu
        return luu
    
    def l_xx(self, x, u, i, terminal=False):

        if terminal:
            gx = self.constr.g_x( x,u,i, terminal=True)
            lxx = self.cost.l_xx( x,u,i, terminal=True) \
                + gx.T @ self.Imu[i] @ gx
            return lxx
        
        gx = self.constr.g_x( x,u,i )
        lxx = self.cost.l_xx( x,u,i ) \
            + gx.T @ self.Imu[i] @ gx
        return lxx

    def l_ux(self, x, u, i, terminal=False):

        if terminal:
            gx = self.constr.g_x( x,u,i, terminal=True)
            gu = self.constr.g_u( x,u,i, terminal=True)
            lux = self.cost.l_ux( x,u,i, terminal=True) \
                + gu.T @ self.Imu[i] @ gx
            return lux
        
        gx = self.constr.g_x( x,u,i )
        gu = self.constr.g_u( x,u,i )
        lux = self.cost.l_ux( x,u,i ) \
            + gu.T @ self.Imu[i] @ gx
        return lux

# =================================================================================
# ErrorState Cost
# =================================================================================

class ErrorStateSE3ApproxTrackingQuadraticAutodiffCost(BaseCost):

    """
        Instantaneous Stage Cost defined on Lie Algebra for SE(3) Error-state Dynamics.
        - Implemented with Jax for autodiff.
        - 2nd order quadratic cost, penalizing both position deviation and velocity deviation.
        - Used for tracking the given reference trajectory
        - Implemented for error-state \psi = Log(\bar{X}^{-1} X) 
    """

    def __init__(self, Q, R, P, xi_ref,
                 state_size=(6,6), action_size=6, **kwargs):
        """Constructs an AutoDiffCost.

        Args:
            Q: State weighting matrix for the stage cost. Shape: [state_size, state_size].
                This matrix penalizes the state deviation (pos & vel) from the reference at each time step.
            R: Control weighting matrix for the stage cost. Shape: [action_size, action_size].
                This matrix penalizes the magnitude of control inputs at each time step.
            P: State weighting matrix for the terminal cost. Shape: [state_size, state_size].
                This matrix penalizes the state deviation (pos & vel) from the reference at the final time step.
            xi_ref: List of velocity reference, described in Lie Algebra,
                 (N, velocity_size, 1)
            state_size: Tuple of State variable dimension, 
                ( error state size, velocity state size ).
            action_size: Input variable dimension.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._state_size = state_size[0] + state_size[1]
        self._error_state_size = state_size[0] 
        self._vel_state_size = state_size[1] 
        self._action_size = action_size

        self._xi_ref = xi_ref
        self._Q = jnp.array(Q)
        self._R = jnp.array(R)
        self._P = jnp.array(P)

        self._l = jit(self._l)

        self._l_x = jit(jacfwd(self._l))
        self._l_u = jit(jacfwd(self._l, argnums=1))

        self._l_xx = jit(hessian(self._l, argnums=0))
        self._l_ux = jit(jacfwd( jacfwd(self._l, argnums=1) ))
        self._l_uu = jit(hessian(self._l, argnums=1))

        # Terminal cost only depends on x, so we only need to evaluate the x
        # partial derivatives.

        self._l_terminal = jit(self._l_terminal)
        self._l_x_terminal = jit(jacfwd(self._l_terminal))
        self._l_xx_terminal = jit(hessian(self._l_terminal))

        super(ErrorStateSE3ApproxTrackingQuadraticAutodiffCost, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size
    
    @property
    def error_state_size(self):
        """Error-state size."""
        return self._error_state_size

    @property
    def vel_state_size(self):
        """Velocity state size."""
        return self._vel_state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size
    
    @property
    def Q(self):
        """State cost coefficient matrix in the quadratic stage cost."""
        return self._Q
    
    @property
    def R(self):
        """Input cost coefficient matrix in the quadratic stage cost."""
        return self._R
    
    @property
    def P(self):
        """State cost coefficient matrix in the quadratic terminal cost."""
        return self._P
    
    def xi_ref(self, i) :
        """Return the Lie Algebra velocity xi reference xi_ref at time index i."""
        return self._xi_ref[i]
    
    def _l(self, x, u, i ):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.

        Returns:
            Instantaneous cost (scalar).
        """
        x = x.reshape(self.state_size, 1)
        u = u.reshape(self.action_size, 1)

        Ct = jnp.block([
            [ jnp.identity( self.error_state_size ), jnp.zeros((self.vel_state_size,self.vel_state_size)) ],
            [ -adjoint( self.xi_ref(i) ), jnp.identity( self.vel_state_size ) ],
        ])
        dt = jnp.vstack(
            ( jnp.zeros((self.error_state_size,1)), self.xi_ref(i) )
        )
        yt = Ct @ x - dt
        
        return (yt.T @ self.Q @ yt + u.T @ self.R @ u).reshape(-1)[0]
    
    def _l_terminal(self, x, i):
        """Terminal cost function.

        Args:
            x: Current state [state_size].
            i: Current time step.

        Returns:
            Terminal cost (scalar).
        """
        x = x.reshape(self.state_size, 1)
        
        Ct = jnp.block([
            [ jnp.identity( self.error_state_size ), jnp.zeros((self.vel_state_size,self.vel_state_size)) ],
            [ -adjoint( self.xi_ref(i) ), jnp.identity( self.vel_state_size ) ],
        ])
        dt = jnp.vstack(
            ( jnp.zeros((self.error_state_size,1)), self.xi_ref(i) )
        )
        yt = Ct @ x - dt
        
        return (yt.T @ self.P @ yt).reshape(-1)[0]

    
    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self._l_terminal(x,i)

        return self._l(x,u,i)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return self._l_x_terminal(x,i).reshape(self.state_size,)

        return self._l_x(x,u,i).reshape(self.state_size,)

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        return self._l_u(x,u,i).reshape(self.action_size,)

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        if terminal:
            return self._l_xx_terminal(x,i).reshape(self.state_size,self.state_size)

        return self._l_xx(x,u,i).reshape(self.state_size,self.state_size)

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        return self._l_ux(x,u,i).reshape(self.action_size,self.state_size)

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        return self._l_uu(x,u,i).reshape(self.action_size,self.action_size)
    

class ErrorStateSE3ApproxGenerationQuadraticAutodiffCost(BaseCost):

    """
        Instantaneous Stage Cost defined on Lie Algebra for SE(3) Error-state Dynamics.
        - Implemented with Jax for autodiff.
        - 2nd order quadratic cost, penalizing .
        - Used for generation of a trajectory to go to the goal configuraiton, not tracking.
    """

    def __init__(self, Q, R, P, X_ref, X_goal, 
                 state_size=(6,6), action_size=6, **kwargs):
        """Constructs an AutoDiff Cost.

        Args:
            Q: State weighting matrix for the stage cost. 
                This matrix penalizes the state deviation (only pos) from the goal at each time step.
                ( error_state_size, error_state_size )
            R: Control weighting matrix for the stage cost. 
                This matrix penalizes the magnitude of control inputs at each time step.
                ( action_size, action_size )
            P: State weighting matrix for the terminal cost.
                This matrix penalizes the state deviation (only pos) from the goal at the final time step.
                 ( error_state_size, error_state_size )
            X_ref: Reference trajecotory.
                Note: this is described in 7-d quat-pos vector.
                (N, 4+3, 1).
            X_goal: Final goal SE(3) configuration for trajectory generation,
                (4, 4).
            state_size: Tuple of State variable dimension, 
                ( error_state_size, vel_state_size ).
            action_size: Input variable dimension.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._state_size = state_size[0] + state_size[1]
        self._error_state_size = state_size[0] 
        self._vel_state_size = state_size[1] 
        self._action_size = action_size

        self._X_ref = jnp.array(X_ref)
        self._X_goal = jnp.array(X_goal)
        print(f"The goal configuration is \n{self._X_goal}")

        # self._X_goal_inv = np.linalg.inv( X_goal )

        self._N = X_ref.shape[0] - 1

        def update_phigoal_onlyinv(Xref, Xgoal):
            # print(Xref.shape)
            return jnp.linalg.inv(quatpos2SE3(Xref)) @ Xgoal
        self._vec_update_phigoal_onlyinv = jax.jit(
            jax.vmap(update_phigoal_onlyinv, in_axes=[0,None])
        )

        def vec_update_phigoal(Xref, Xgoal):
            se3_batch = self._vec_update_phigoal_onlyinv(Xref, Xgoal)
            # se3_batch = [ jnp.linalg.inv(quatpos2SE3(mat)) @ Xgoal for mat in Xref ]
            phi_goal_new = [se3_vee(logm(np.array(mat))) for mat in se3_batch]
            return jnp.array(phi_goal_new)
        self._vec_update_phigoal = vec_update_phigoal

        self._phi_goal = self._vec_update_phigoal( self._X_ref, self._X_goal )

        self._Q = jnp.array(Q)
        self._R = jnp.array(R)
        self._P = jnp.array(P)

        # self._l = jit(self._l)

        self._l_x = jit(jacfwd(self._l))
        self._l_u = jit(jacfwd(self._l, argnums=1))

        self._l_xx = jit(hessian(self._l, argnums=0))
        self._l_ux = jit(jacfwd( jacfwd(self._l, argnums=1) ))
        self._l_uu = jit(hessian(self._l, argnums=1))

        # Terminal cost only depends on x, so we only need to evaluate the x
        # partial derivatives.

        # self._l_terminal = jit(self._l_terminal)
        self._l_x_terminal = jit(jacfwd(self._l_terminal))
        self._l_xx_terminal = jit(hessian(self._l_terminal))

        super(ErrorStateSE3ApproxGenerationQuadraticAutodiffCost, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size
    
    @property
    def error_state_size(self):
        """Error-state size."""
        return self._error_state_size

    @property
    def vel_state_size(self):
        """Velocity state size."""
        return self._vel_state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size
    
    @property
    def Q(self):
        """State cost coefficient matrix in the quadratic stage cost."""
        return self._Q
    
    @property
    def R(self):
        """Input cost coefficient matrix in the quadratic stage cost."""
        return self._R
    
    @property
    def P(self):
        """State cost coefficient matrix in the quadratic terminal cost."""
        return self._P
    
    @property
    def q_goal(self):
        """Final trajectory goal configuration."""
        return self._q_goal
    
    # @property
    # def q_goal_inv(self):
    #     """Inverse of Final trajectory goal configuration."""
    #     return self._q_goal_inv
    
    def X_ref(self, i) :
        """Return the quat-pos reference at time index i."""
        return self._X_ref[i]
    
    def ref_reinitialize( self, new_X_ref ) :
        """Re-initialize the error-state cost, with the new error-state rollout trajecotory."""
        self._X_ref = new_X_ref
        self._phi_goal = self._vec_update_phigoal( self._X_ref, self._X_goal )
        return self._phi_goal
    
    def _l(self, x, u, i ):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.

        Returns:
            Instantaneous cost (scalar).
        """
        x = x.reshape(self.state_size, 1)
        u = u.reshape(self.action_size, 1)

        x_err = x[:self.error_state_size]
        # yt = logm( self.q_goal_inv @ quatpos2SE3( self.X_ref(i) ) @ expm(x_err) )
        # yt =  se3_vee( logm(  self.q_goal_inv @ expm( se3_hat(x_err) )))

        yt = x_err - self._phi_goal[i].reshape((self.error_state_size,1))

        # print(f'yt: {yt}, u: {u}')
        cost_value = (yt.T @ self.Q @ yt + u.T @ self.R @ u)
        # print(f'Cost value: {cost_value}, type: {type(cost_value)}')

        return cost_value.reshape(-1)[0]
    
    def _l_terminal(self, x, i):
        """Terminal cost function.

        Args:
            x: Current state [state_size].
            i: Current time step.

        Returns:
            Terminal cost (scalar).
        """
        x = x.reshape(self.state_size, 1)
        
        x_err = x[:self.error_state_size]
        # yt = logm( self.q_goal_inv @ quatpos2SE3( self.X_ref(i) ) @ expm(x_err) )
        # yt =  se3_vee( self.q_goal_inv @ expm( se3_hat(x_err) )  )

        yt = x_err - self._phi_goal[i].reshape((self.error_state_size,1))
        
        return (yt.T @ self.P @ yt).reshape(-1)[0]

    
    def l(self, x, u, i, terminal=False):
        """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
        if terminal:
            return self._l_terminal(x,i)

        return self._l(x,u,i)

    def l_x(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
        if terminal:
            return self._l_x_terminal(x,i).reshape(self.state_size,)

        return self._l_x(x,u,i).reshape(self.state_size,)

    def l_u(self, x, u, i, terminal=False):
        """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros(self._action_size)

        return self._l_u(x,u,i).reshape(self.action_size,)

    def l_xx(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
        if terminal:
            return self._l_xx_terminal(x,i).reshape(self.state_size,self.state_size)

        return self._l_xx(x,u,i).reshape(self.state_size,self.state_size)

    def l_ux(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._state_size))

        return self._l_ux(x,u,i).reshape(self.action_size,self.state_size)

    def l_uu(self, x, u, i, terminal=False):
        """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
        if terminal:
            # Not a function of u, so the derivative is zero.
            return np.zeros((self._action_size, self._action_size))

        return self._l_uu(x,u,i).reshape(self.action_size,self.action_size)