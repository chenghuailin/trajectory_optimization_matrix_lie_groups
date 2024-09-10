import abc
import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd, hessian, jit
from traoptlibrary.traopt_utilis import adjoint, quatpos2SE3, se3_vee, se3_hat
from jax.scipy.linalg import expm
from scipy.linalg import logm
from functools import partial

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
    

class ErrorStateSE3TrackingQuadratic2ndOrderAutodiffCost(BaseCost):

    """
        Instantaneous Stage Cost defined on Lie Algebra for SE(3) Error-state Dynamics.
        - Implemented with Jax for autodiff.
        - 2nd order quadratic cost, penalizing both position deviation and velocity deviation.
        - Used for tracking the given reference trajectory
        - The same setup as Sangli's Paper
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

        super(ErrorStateSE3TrackingQuadratic2ndOrderAutodiffCost, self).__init__()

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
    

class ErrorStateSE3GenerationQuadratic1stOrderAutodiffCost(BaseCost):

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
        print(f"The goal configuration is {self._X_goal}")

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

        super(ErrorStateSE3GenerationQuadratic1stOrderAutodiffCost, self).__init__()

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