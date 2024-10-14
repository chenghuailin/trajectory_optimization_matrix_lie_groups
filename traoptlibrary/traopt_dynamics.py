import abc
import numpy as np
import jax.numpy as jnp
import jax
from jax import jacfwd, hessian, jit
from traoptlibrary.traopt_utilis import skew, adjoint, coadjoint, se3_hat, \
            SE32quatpos, SE32manifSE3, se32manifse3, Jmnf2J
from jax.scipy.linalg import expm
from scipy.linalg import logm
import scipy

class BaseDynamics():

    """Dynamics Model."""

    @property
    @abc.abstractmethod
    def state_size(self):
        """State size."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_size(self):
        """Action size."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        raise NotImplementedError

    @abc.abstractmethod
    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        raise NotImplementedError

    @abc.abstractmethod
    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Note:
            This is not necessary to implement if you're planning on skipping
            Hessian evaluation as the iLQR implementation does by default.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        raise NotImplementedError
    

class AutoDiffDynamics(BaseDynamics):

    """Auto-differentiated Dynamics Model implemented with Jax."""

    def __init__(self, f, state_size, action_size, hessians=False, **kwargs):
        """Constructs an AutoDiffDynamics model.

        Args:
            f: Discretized dynamics function (e.g. after RK4):
                Args:
                    x: Batch of state variables.
                    u: Batch of action variables.
                    i: Batch of time step variables.
                Returns:
                    f: Batch of next state variables.
            state_size: State variable dimension.
            action_size: Input variable dimension.
            hessians: Evaluate the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            **kwargs: Additional keyword-arguments to pass to any potential 
                function, e.g. in the preivious version `theano.function()`.
        """

        self._state_size = state_size
        self._action_size = action_size

        self._f = jit(f)
        self._f_x = jit(jacfwd(f))
        self._f_u = jit(jacfwd(f, argnums=1))

        self._has_hessians = hessians
        if hessians:
            self._f_xx = jit(hessian(f, argnums=0))
            self._f_ux = jit(jacfwd( jacfwd(f, argnums=1) ))
            self._f_uu = jit(hessian(f, argnums=1))

        super(AutoDiffDynamics, self).__init__()

    @property
    def state_size(self):
        """State size."""
        return self._state_size

    @property
    def action_size(self):
        """Action size."""
        return self._action_size

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return self._has_hessians

    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return self._f(x,u,i)

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        return self._f_x(x,u,i)

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        return self._f_u(x,u,i)

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_xx(x,u,i)

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_ux(x,u,i)

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_uu(x,u,i)
    

class SE3Dynamics(BaseDynamics):

    """Error-State SE(3) Dynamics Model"""

    def __init__(self, J, dt, integration_method="euler",
                    state_size=(6,6), action_size=6, 
                    hessians=False, debug = None, 
                    **kwargs):
        """Constructs an Dynamics model for SE(3).

        Args:
            J: Inertia matrix, diag(I_b, m * I_3), 
                m : body mass,
                I_b : moment of inertia in the body frame.
            dt: Sampling time.
            integration_method: integration method for dynamics,
                "euler": euler method,
                "rk4": Runga Kutta 4 method.
            state_size: Tuple of State variable dimension, 
                ( error state size, velocity state size ).
            action_size: Input variable dimension.
            hessians: Evaluate the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            **kwargs: Additional keyword-arguments to pass to any potential 
                function, e.g. in the preivious version `theano.function()`.
        """

        self._state_size = state_size[0] + state_size[1]
        self._error_state_size = state_size[0] 
        self._vel_state_size = state_size[1] 
        self._action_size = action_size

        self._Ib = J[0:3, 0:3] 
        self._m = J[4,4]
        self._J = J
        self._Jinv = np.linalg.inv(J)
        self._dt = dt

        self._Bt = np.vstack(
            (np.zeros((self.error_state_size, self.action_size)),self.Jinv )
        )

        self._integration_method = integration_method
        if integration_method == "euler":
            # self._f = jit(self.fd_euler)
            self._f = self.fd_euler
        elif integration_method == "rk4":
            # self._f = jit(self.fd_rk4)
            raise ValueError("RK4 not implemented yet.")
        else:
            raise ValueError("Invalid integration method. Choose 'euler' or 'rk4'.")
        
        self._has_hessians = hessians
        # if hessians:
        #     self._f_xx = jit(hessian(self._f, argnums=0))
        #     self._f_ux = jit(jacfwd( jacfwd(self._f, argnums=1) ))
        #     self._f_uu = jit(hessian(self._f, argnums=1))

        self._debug = debug
        
        super(SE3Dynamics, self).__init__()

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
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return self._has_hessians

    @property
    def Ib(self):
        """Moment of inertia in the body frame."""
        return self._Ib

    @property
    def m(self):
        """Mass of the system."""
        return self._m

    @property
    def J(self):
        """Inertia matrix of the system."""
        return self._J
    
    @property
    def Jinv(self):
        """Inverse of the inertia matrix."""
        return self._Jinv
    
    @property
    def dt(self):
        """Sampling time of the system dynamics."""
        return self._dt
    
    def fc(self, x, u, i):
        """ Continuous nonlinear dynamicsf.

        Args:
            x: Current state [state_size], stacked by
                error-state and velocity, both on Lie Algebra
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """

        q, xi = x 

        # TODO: decide if rewriting into manif or reordering xi is needed
        q_dot = q @ scipy.linalg.expm( se3_hat(xi))
        xi_dot =  self.Jinv @ ( coadjoint( xi ) @ self.J @ xi + u )
        
        return [q_dot, xi_dot]
    
    def fd_euler( self, x, u, i ):
        """ Descrtized dynamics with Eular method.

        Args:
            x: Current state [state_size], stacked by
                error-state and velocity, both on Lie Algebra
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """

        q, xi = x 
        xi = xi.reshape(self.vel_state_size, 1)
        u = u.reshape(self.action_size, 1)

        q_next = q @ scipy.linalg.expm( se3_hat(xi) * self.dt )
        xi_next = xi + self.Jinv @ ( coadjoint( xi ) @ self.J @ xi + u ) * self.dt  

        return [q_next, xi_next.reshape(self.vel_state_size,)]
    
    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return self._f(x,u,i)
    
    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """

        q, xi = x 
        omega = xi[:3]
        v = xi[3:]

        q = SE32manifSE3( q )
        xi = se32manifse3( xi )
        
        J_q_q = np.empty((6,6))
        J_q_xih = np.empty((6,6))
        _ = q.rplus(xi * self.dt, J_q_q, J_q_xih)

        J_q_q = Jmnf2J(J_q_q)
        J_q_xi = Jmnf2J(J_q_xih) * self.dt

        G = np.block([
            [skew( self.Ib @ omega ), self.m * skew( v )],
            [self.m * skew( v ), np.zeros((3,3))],        
        ])
        H = self.Jinv @ ( coadjoint( xi.coeffs() ) @ self.J + G )

        return np.block([
                    [J_q_q,             J_q_xi],
                    [np.zeros((6,6)),   np.identity(6) + H*self.dt],
                ])

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        return self._Bt * self.dt

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_xx(x,u,i)

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_ux(x,u,i)

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_uu(x,u,i)
    
    
class ErrorStateSE3ApproxLinearRolloutDynamics(BaseDynamics):

    """Error-State SE(3) Dynamics Model implemented with Jax"""

    def __init__(self, J, q_ref, xi_ref, dt, integration_method="euler",
                    state_size=(6,6), action_size=6, 
                    hessians=False, debug = None, 
                    autodiff_dyn=True, **kwargs):
        """Constructs an Dynamics model for SE(3).

        Args:
            J: Inertia matrix, diag(I_b, m * I_3), 
                m : body mass,
                I_b : moment of inertia in the body frame.
            q_ref: List of Lie Group reference, (N, 4, 4)
            xi_ref: List of velocity reference, described in Lie Algebra,
                 (N, velocity_size, 1)
            dt: Sampling time.
            integration_method: integration method for dynamics,
                "euler": euler method,
                "rk4": Runga Kutta 4 method.
            state_size: Tuple of State variable dimension, 
                ( error state size, velocity state size ).
            action_size: Input variable dimension.
            hessians: Evaluate the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            **kwargs: Additional keyword-arguments to pass to any potential 
                function, e.g. in the preivious version `theano.function()`.
        """

        self._state_size = state_size[0] + state_size[1]
        self._error_state_size = state_size[0] 
        self._vel_state_size = state_size[1] 
        self._action_size = action_size

        self._q_ref = jnp.array(q_ref)
        self._xi_ref = jnp.array(xi_ref)
        self._q_ref_inv = jnp.array([ np.linalg.inv(q) for q in q_ref])

        self._Ib = J[0:3, 0:3] 
        self._m = J[4,4]
        self._J = J
        self._Jinv = jnp.linalg.inv(J)

        self._Bt = jnp.vstack(
            (jnp.zeros((self.error_state_size, self.action_size)),self.Jinv )
        )

        if q_ref.shape[0] != xi_ref.shape[0]:
            raise ValueError("Group reference X and velocity reference \
                            should share the same time horizon")
        self._N = q_ref.shape[0] - 1
        self._dt = dt

        self._At_list = jnp.empty((self._N, self._state_size,self._state_size))
        self._Bt_list = jnp.empty((self._N, self._state_size,self._action_size))
        self._ht_list = jnp.empty((self._N, self._state_size,self._state_size))

        # TODO: Use jit for faster computation
        self._integration_method = integration_method
        if integration_method == "euler":
            self._f = jit(self.fd_euler)
        elif integration_method == "rk4":
            self._f = jit(self.fd_rk4)
        else:
            raise ValueError("Invalid integration method. Choose 'euler' or 'rk4'.")
        
        self._autodiff_dyn = autodiff_dyn

        self._f_x = jit(jacfwd(self._f))
        self._f_u = jit(jacfwd(self._f, argnums=1))

        self._has_hessians = hessians
        if hessians:
            self._f_xx = jit(hessian(self._f, argnums=0))
            self._f_ux = jit(jacfwd( jacfwd(self._f, argnums=1) ))
            self._f_uu = jit(hessian(self._f, argnums=1))

        self._debug = debug

        def update_Xref(q_ref, x):
            q_ref_new = q_ref @ expm( se3_hat(x[:self.error_state_size]) )
            return q_ref_new
        
        def update_xi_ref(xs):
            return xs[self.error_state_size:]
        
        # Use vmap to parallelize the update_ref function
        self._vec_update_qref = jax.jit(jax.vmap(update_Xref))
        self._vec_update_xi_ref = jax.jit(jax.vmap(update_xi_ref))

        super(ErrorStateSE3ApproxLinearRolloutDynamics, self).__init__()

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
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return self._has_hessians

    @property
    def Ib(self):
        """Moment of inertia in the body frame."""
        return self._Ib

    @property
    def m(self):
        """Mass of the system."""
        return self._m

    @property
    def J(self):
        """Inertia matrix of the system."""
        return self._J
    
    @property
    def Jinv(self):
        """Inverse of the inertia matrix."""
        return self._Jinv

    @property
    def N(self):
        """The horizon for dynamics to be valid, due to horizon of given reference."""
        return self._N
    
    @property
    def dt(self):
        """Sampling time of the system dynamics."""
        return self._dt
    
    @property
    def q_ref(self) :
        """Return the Lie group reference q_ref at time index i."""
        return self._q_ref
    
    @property
    def xi_ref(self) :
        """Return the velocity reference xi_ref."""
        return self._xi_ref

    def get_q_ref(self, i) :
        """Return the Lie group reference q_ref at time index i."""
        return self._q_ref[i]

    def get_xi_ref(self, i) :
        """Return the Lie Algebra velocity xi reference xi_ref at time index i."""
        return self._xi_ref[i]
    
    def ref_reinitialize_serial( self, xs ) :
        """Re-initialize the error-state dynamics, with the new error-state rollout trajecotory.
            In serial programming style, not recommended, only for comparison.
        """
        
        for i in range(self.N + 1):
            self._q_ref = self._q_ref.at[i].set( 
                self._q_ref[i] @ expm( se3_hat( xs[i, :6]) )
            )
            self._xi_ref = self._xi_ref.at[i].set(
                xs[i, self.error_state_size:].reshape(6,1)
            )
        
        return self._X_ref, self._xi_ref

    def ref_reinitialize(self, xs):
        """Re-initialize the error-state dynamics, 
        with the new error-state rollout trajectory in a parallel style."""
        
        self._q_ref= self._vec_update_qref(self._q_ref, xs)
        self._xi_ref = self._vec_update_xi_ref(xs)

        if self._autodiff_dyn:

            if self._integration_method == "euler":
                self._f = jit(self.fd_euler)
            elif self._integration_method == "rk4":
                self._f = jit(self.fd_rk4)
            else:
                raise ValueError("Invalid integration method. Choose 'euler' or 'rk4'.")
        
            self._f_x = jit(jacfwd(self._f))
            self._f_u = jit(jacfwd(self._f, argnums=1))

            if self._has_hessians:
                self._f_xx = jit(hessian(self._f, argnums=0))
                self._f_ux = jit(jacfwd( jacfwd(self._f, argnums=1) ))
                self._f_uu = jit(hessian(self._f, argnums=1))

        return self._q_ref, self._xi_ref
    
    def At(self, x, u, i):
        """ Return the Jacobian matrix A. 
        
        Args:
            x: Current state [state_size], stacked by
                error-state and velocity, both on Lie Algebra
            u: Current control [action_size].
            i: Current time step.

        Returns:
            At: [state_size, state_size]
        """

        x = x.reshape(self.state_size, 1)
        u = u.reshape(self.action_size, 1)

        xi = x[-self.vel_state_size:]
        omega = xi[:3]
        v = xi[-3:]

        G = jnp.block([
            [skew( self.Ib @ omega ), self.m * skew( v )],
            [self.m * skew( v ), jnp.zeros((3,3))],        
        ])
        Ht = self.Jinv @ ( coadjoint( xi ) @ self.J + G )

        At = jnp.block([
            [- adjoint( self.get_xi_ref(i) ), jnp.identity( self.error_state_size )],
            [jnp.zeros((self.vel_state_size, self.error_state_size)), Ht]
        ])

        return At
    
    def Bt(self, x, u, i):
        """ Return the Jacobian matrix A. 
        
        Args:
            x: Current state [state_size], stacked by
                error-state and velocity, both on Lie Algebra
            u: Current control [action_size].
            i: Current time step.

        Returns:
            At: [state_size, state_size]
        """
        return self._Bt

    def fc(self, x, u, i):
        """ Continuous linearized dynamicsf.

        Args:
            x: Current state [state_size], stacked by
                error-state and velocity, both on Lie Algebra
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """

        # psi = x[:self.error_state_size]

        x = x.reshape(self.state_size, 1)
        u = u.reshape(self.action_size, 1)

        xi = x[-self.vel_state_size:]
        omega = xi[:3]
        v = xi[-3:]

        G = jnp.block([
            [skew( self.Ib @ omega ), self.m * skew( v )],
            [self.m * skew( v ), jnp.zeros((3,3))],        
        ])
        Ht = self.Jinv @ ( coadjoint( xi ) @ self.J + G )
        bt = - self.Jinv @ G @ xi

        # print("\nG is shape of", G.shape, "with value \n", G)
        # print("\nHt is shape of", Ht.shape, "with value \n", Ht)
        # print("\nbt is shape of", bt.shape, "with value \n", bt)

        At = jnp.block([
            [- adjoint( self.get_xi_ref(i) ), jnp.identity( self.error_state_size )],
            [jnp.zeros((self.vel_state_size, self.error_state_size)), Ht]
        ])
        Bt = jnp.vstack((jnp.zeros((self.error_state_size, self.action_size)),
                self.Jinv ))
        ht = jnp.vstack( (-self.get_xi_ref(i).reshape(self.vel_state_size,1), bt ))

        # print("\nAt is shape of", At.shape, "with value \n", At)
        # print("\nBt is shape of", Bt.shape, "with value \n", Bt)
        # print("\nht is shape of", ht.shape, "with value \n", ht)

        # self._At = At
        # self._Bt = Bt

        xt_dot = At @ x + Bt @ u + ht

        if self._debug and self._debug.get('vel_zero'):
            xt_dot = xt_dot.at[-self.vel_state_size:].set(0)
        
        return xt_dot.reshape(self.state_size,)
        # return xt_dot
    
    def fd_euler( self, x, u, i ):
        """ Descrtized dynamics with Eular method.

        Args:
            x: Current state [state_size], stacked by
                error-state and velocity, both on Lie Algebra
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return x.reshape(self.state_size,) + self.fc( x,u,i ) * self.dt
        # return x + self.fc( x,u,i ) * self.dt
    
    def fd_rk4( self, x, u, i ):
        """ Descrtized dynamics with RK4 method.

        Args:
            x: Current state [state_size], stacked by
                error-state and velocity, both on Lie Algebra
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        s1 = self.fc( x, u, i)
        s2 = self.fc( x+ self.dt/2*s1, u, i )
        s3 = self.fc( x+ self.dt/2*s2, u, i )
        s4 = self.fc( x+ self.dt*s3, u, i )
        x_next = x + self.dt/6 * ( s1 + 2 * s2 + 2 * s3 + s4 )
    
        return x_next.reshape(self.state_size,)

    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return self._f(x,u,i)
    
    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """

        if self._debug and self._debug.get('derivative_compare'):
            analytical = self.At(x,u,i) * self.dt + jnp.identity(self.state_size)
            autodiff = self._f_x(x,u,i).reshape(self.state_size,self.state_size)
            return analytical, autodiff
        elif not self._autodiff_dyn:            
            return self.At(x,u,i) * self.dt + jnp.identity(self.state_size)
        else:
            return self._f_x(x,u,i).reshape(self.state_size,self.state_size)

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
    
        if self._debug and self._debug.get('derivative_compare'):
            analytical = self.Bt(x,u,i) * self.dt
            autodiff = self._f_u(x,u,i).reshape(self.state_size,self.action_size)
            return analytical, autodiff
        elif not self._autodiff_dyn:            
            return self.Bt(x,u,i) * self.dt
        else:
            return self._f_u(x,u,i).reshape(self.state_size,self.action_size)

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_xx(x,u,i)

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_ux(x,u,i)

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_uu(x,u,i)
    
    def _fc_vel(self, xi, u, i):
        """ Continuous nonlinear dynamics for velocity.

        Args:
            xi: velocity state [vel_state_size]
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next velocity state [vel_state_size].
        """
        
        xi = xi.reshape(self.vel_state_size, 1)
        u = u.reshape(self.action_size, 1)

        xi_dot =  self.Jinv @ ( coadjoint( xi ) @ self.J @ xi + u )

        if self._debug and self._debug.get('vel_zero'):
            xi_dot = np.zeros((self.action_size, 1))
        
        return xi_dot.reshape(self.vel_state_size,)
    
    def _fd_euler_fc_vel( self, x, u, i ):
        """ Discretized nonlinear velocity dynamics with Euler method.

        Args:
            x: Current velocity state [vel_state_size]
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next velocity state [vel_state_size].
        """
        return x.reshape(self.vel_state_size,) + self._fc_vel( x,u,i ) * self.dt
    
    def _fd_euler_fc_group( self, q, xi, u, i ):
        """ Discretized nonlinear group complete dynamics with euler method.

        Args:
            q: Current configuration SE(3), [4,4].
            xi: Current velocity state [vel_state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            q_next: Next configuration SE(3), [4,4].
            xi_next: Next velocity state [vel_state_size].
        """

        q_next = q @ expm( se3_hat( xi ) * self.dt )
        xi_next = self._fd_euler_fc_vel(xi, u, i)

        return q_next, xi_next 
    

class ErrorStateSE3ApproxNonlinearRolloutDynamics(BaseDynamics):

    """Error-State SE(3) Dynamics Model implemented with Jax"""

    def __init__(self, J, u0, q0, xi0, dt, 
                errstate_integration="euler", rollout_integration ="euler",
                state_size=(6,6), action_size=6, 
                hessians=False, debug = None, **kwargs):
        """Constructs an Dynamics model for SE(3).

        Args:
            J: Inertia matrix, diag(I_b, m * I_3), 
                m : body mass,
                I_b : moment of inertia in the body frame.
            u0: Initial input sequence, (N, actiion size)
            q0: Initial configuration, SE(3), (4,4).
            xi0: Initial velocity, (velocity_state_size,).
            dt: Sampling time.
            errstate_integration: integration method for dynamics,
                note this is for the error-state linear dynamics,
                "euler": euler method,
                "rk4": Runga Kutta 4 method.
            rollout_integration: integration method for rollout,
                "euler": euler method,
                "rk4": Runga Kutta 4 method.
            state_size: Tuple of State variable dimension, 
                ( error state size, velocity state size ).
            action_size: Input variable dimension.
            hessians: Evaluate the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            debug: Debug mode or not.
            **kwargs: Additional keyword-arguments to pass to any potential 
                function, e.g. in the preivious version `theano.function()`.
        """

        self._state_size = state_size[0] + state_size[1]
        self._error_state_size = state_size[0] 
        self._vel_state_size = state_size[1] 
        self._action_size = action_size

        self._Ib = J[0:3, 0:3] 
        self._m = J[4,4]
        self._J = J
        self._Jinv = jnp.linalg.inv(J)

        self._At = jnp.empty((self._state_size,self._state_size))
        self._Bt = jnp.empty((self._state_size,self._action_size))
        self._ht = jnp.empty((self._state_size, 1))
        
        self._N = u0.shape[0] - 1
        self._dt = dt

        self._debug = debug

        self.errstate_integration = errstate_integration
        if self.errstate_integration == "euler":
            self._f = jit(self._fd_euler_fc_errstate)
        elif self.errstate_integration == "rk4":
            self._f = jit(self._fd_rk4_fc_errstate)
        else:
            raise ValueError("Invalid integration method. Choose 'euler' or 'rk4'.")
       
        self.rollout_integration = rollout_integration
        if self.rollout_integration == "euler":
            self._f_rollout = jit(self._fd_euler_fc_group)
        elif self.rollout_integration == "rk4":
            self._f_rollout = jit(self._fd_rk4_fc_group)
        else:
            raise ValueError("Invalid integration method. Choose 'euler' or 'rk4'.")
        
        self._q_ref, self._xi_ref = self.rollout_nominal_with_input_list( q0, xi0, u0 )
        self._q_ref = jnp.array(self._q_ref)
        self._xi_ref = jnp.array(self._xi_ref)
                
        self._f_x = jit(jacfwd(self._f))
        self._f_u = jit(jacfwd(self._f, argnums=1))
 
        self._has_hessians = hessians
        if hessians:
            self._f_xx = jit(hessian(self._f, argnums=0))
            self._f_ux = jit(jacfwd( jacfwd(self._f, argnums=1) ))
            self._f_uu = jit(hessian(self._f, argnums=1))

        def err2config(q_ref, x):
            q_ref_new = SE32quatpos( 
                    q_ref @ expm( se3_hat(x[:6]) )
            )
            return q_ref_new
        self._vec_update_Xref = jax.jit(jax.vmap(err2config))

        super(ErrorStateSE3ApproxNonlinearRolloutDynamics, self).__init__()

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
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return self._has_hessians

    @property
    def Ib(self):
        """Moment of inertia in the body frame."""
        return self._Ib

    @property
    def m(self):
        """Mass of the system."""
        return self._m

    @property
    def J(self):
        """Inertia matrix of the system."""
        return self._J
    
    @property
    def Jinv(self):
        """Inverse of the inertia matrix."""
        return self._Jinv

    @property
    def N(self):
        """The horizon for dynamics to be valid, due to horizon of given reference."""
        return self._N
    
    @property
    def dt(self):
        """Sampling time of the system dynamics."""
        return self._dt
    
    @property
    def At(self):
        """Matrix At of the error-state linearization."""
        return self._At
    
    @property
    def Bt(self):
        """Matrix Bt of the error-state linearization."""
        return self._Bt
    
    @property
    def q_ref(self) :
        """Return the Lie group reference q_ref."""
        return self._q_ref
    
    @property
    def xi_ref(self) :
        """Return the velocity reference xi_ref."""
        return self._xi_ref

    def get_q_ref(self, i) :
        """Return the Lie group reference q_ref at time index i."""
        return self._q_ref[i]

    def get_xi_ref(self, i) :
        """Return the Lie Algebra velocity xi reference xi_ref at time index i."""
        return self._xi_ref[i]

    def rollout_nominal_with_input_list(self, q0, xi0, u):
        """Nonlinear rollout the SE3 trajectory with given input sequence."""
        
        N = u.shape[0]
        q_ref = np.empty((N+1, 4, 4))
        xi_ref = np.empty((N+1, self._vel_state_size,))

        q_ref[0] = q0
        xi_ref[0] = xi0
        
        for i in range(N):
            q_ref[i+1], xi_ref[i+1] = self.f_rollout( q_ref[i], xi_ref[i], u[i], i )
        
        return q_ref, xi_ref

    def ref_update(self, xs):
        """Update the reference configuration trajectory with the error-state.
            Might not be needed in nonlinear rollout"""
        self._q_ref= self._vec_update_Xref(self._q_ref, xs)
        return self._q_ref

    def _fc_errstate(self, x, u, i):
        """ Continuous linearized dynamicsf.

        Args:
            x: Current state [state_size], stacked by
                error-state and velocity, both on Lie Algebra
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """

        # psi = x[:self.error_state_size]

        x = x.reshape(self.state_size, 1)
        u = u.reshape(self.action_size, 1)

        xi = x[-self.vel_state_size:]
        omega = xi[:3]
        v = xi[-3:]

        G = jnp.block([
            [skew( self.Ib @ omega ), self.m * skew( v )],
            [self.m * skew( v ), jnp.zeros((3,3))],        
        ])
        Ht = self.Jinv @ ( coadjoint( xi ) @ self.J + G )
        bt = - self.Jinv @ G @ xi

        At = jnp.block([
            [- adjoint( self.get_xi_ref(i) ), jnp.identity( self.error_state_size )],
            [jnp.zeros((self.vel_state_size, self.error_state_size)), Ht]
        ])
        Bt = jnp.vstack((jnp.zeros((self.error_state_size, self.action_size)),
                self.Jinv ))
        ht = jnp.vstack( (-self.get_xi_ref(i).reshape(self.vel_state_size,1), bt ))

        xt_dot = At @ x + Bt @ u + ht

        self._At = At
        self._Bt = Bt
        self._ht = ht

        if self._debug and self._debug.get('vel_zero'):
            xt_dot = xt_dot.at[-self.vel_state_size:].set(0)
        
        return xt_dot.reshape(self.state_size,)
    
    def _fd_euler_fc_errstate( self, x, u, i ):
        """ Descrtized linearized error-state dynamics with Eular method.

        Args:
            x: Current state [state_size], stacked by
                error-state and velocity, both on Lie Algebra
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return x.reshape(self.state_size,) + self._fc_errstate( x,u,i ) * self.dt
        # return x + self.fc( x,u,i ) * self.dt
    
    def _fd_rk4_fc_errstate( self, x, u, i ):
        """ Descrtized linearized error-state dynamics with RK4 method.

        Args:
            x: Current state [state_size], stacked by
                error-state and velocity, both on Lie Algebra
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        s1 = self._fc_errstate( x, u, i)
        s2 = self._fc_errstate( x+ self.dt/2*s1, u, i )
        s3 = self._fc_errstate( x+ self.dt/2*s2, u, i )
        s4 = self._fc_errstate( x+ self.dt*s3, u, i )
        x_next = x + self.dt/6 * ( s1 + 2 * s2 + 2 * s3 + s4 )
    
        return x_next.reshape(self.state_size,)
    
    def _fc_vel(self, xi, u, i):
        """ Continuous nonlinear dynamics for velocity.

        Args:
            xi: velocity state [vel_state_size]
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next velocity state [vel_state_size].
        """
        
        xi = xi.reshape(self.vel_state_size, 1)
        u = u.reshape(self.action_size, 1)

        xi_dot =  self.Jinv @ ( coadjoint( xi ) @ self.J @ xi + u )

        if self._debug and self._debug.get('vel_zero'):
            xi_dot = np.zeros((self.action_size, 1))
        
        return xi_dot.reshape(self.vel_state_size,)
    
    def _fd_euler_fc_vel( self, x, u, i ):
        """ Discretized nonlinear velocity dynamics with Euler method.

        Args:
            x: Current velocity state [vel_state_size]
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next velocity state [vel_state_size].
        """
        return x.reshape(self.vel_state_size,) + self._fc_vel( x,u,i ) * self.dt
    
    def _fd_rk4_fc_vel( self, x, u, i ):
        """ Discretized nonlinear velocity dynamics with RK4 method.

        Args:
            x: Current velocity state [vel_state_size]
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next velocity state [vel_state_size].
        """
        s1 = self._fc_vel( x, u, i)
        s2 = self._fc_vel( x+ self.dt/2*s1, u, i )
        s3 = self._fc_vel( x+ self.dt/2*s2, u, i )
        s4 = self._fc_vel( x+ self.dt*s3, u, i )
        x_next = x + self.dt/6 * ( s1 + 2 * s2 + 2 * s3 + s4 )
        return x_next.reshape(self.vel_state_size,)
    
    def _fd_euler_fc_group( self, q, xi, u, i ):
        """ Discretized nonlinear group complete dynamics with euler method.

        Args:
            q: Current configuration SE(3), [4,4].
            xi: Current velocity state [vel_state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            q_next: Next configuration SE(3), [4,4].
            xi_next: Next velocity state [vel_state_size].
        """

        q_next = q @ expm( se3_hat( xi ) * self.dt )
        xi_next = self._fd_euler_fc_vel(xi, u, i)

        return q_next, xi_next 
    
    def _fd_rk4_fc_group( self, q, xi, u, i ):
        """ Discretized nonlinear velocity dynamics with RK4 method.

        Args:
            q: Current configuration SE(3), [4,4].
            xi: Current velocity state [vel_state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            q_next: Next configuration SE(3), [4,4].
            xi_next: Next velocity state [vel_state_size].
        """

        q_next = q @ expm( se3_hat( xi ) * self.dt )
        xi_next = self._fd_rk4_fc_vel(xi, u, i)
        return q_next, xi_next 
    
    def f(self, x, u, i):
        """Linearized error-state dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return self._f(x,u,i)
    
    def f_rollout( self, q, xi, u, i ):
        """Nonlinear dynamics for rollout.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        return self._f_rollout( q, xi, u, i )
    
    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        return self._f_x(x,u,i).reshape(self.state_size,self.state_size)
        # return self._f_x(x,u,i)

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        return self._f_u(x,u,i).reshape(self.state_size,self.action_size)

    def f_xx(self, x, u, i):
        """Second partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dx^2 [state_size, state_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_xx(x,u,i)

    def f_ux(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/dudx [state_size, action_size, state_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_ux(x,u,i)

    def f_uu(self, x, u, i):
        """Second partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            d^2f/du^2 [state_size, action_size, action_size].
        """
        if not self._has_hessians:
            raise NotImplementedError

        return self._f_uu(x,u,i)