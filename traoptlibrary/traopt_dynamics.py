import abc
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, hessian, jit
from traopt_utilis import skew, unskew, se3_hat

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
    

class ErrorStateSE3AutoDiffDynamics(BaseDynamics):

    """Error-State SE(3) Dynamics Model implemented with Jax for Derivatvies"""

    def __init__(self, J, X_ref, xi_ref, dt, integration_method="euler",
                    state_size=(6,6), action_size=6, hessians=False, **kwargs):
        """Constructs an Dynamics model for SE(3).

        Args:
            J: Inertia matrix, diag(I_b, m * I_3), 
                m : body mass,
                I_b : moment of inertia in the body frame.
            X_ref: List of Lie Group reference, (N, error_state_size, 1)
            xi_ref: List of velocity reference, described in Lie Algebra,
                 (N, velocity_size, 1)
            dt: Sampling time
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

        self._X_ref = X_ref
        self._xi_ref = xi_ref
        self._x_ref = np.concatenate(( X_ref, xi_ref ), axis = 1)

        self._Ib = J[0:3, 0:3] 
        self._m = J[4,4]
        self._J = J
        self._Jinv = np.linalg.inv(J)
        
        if X_ref.shape[0] != xi_ref.shape[0]:
            raise ValueError("Group reference X and velocity reference should share the same time horizon")
        self._horizon = X_ref.shape[0]
        self._dt = dt

        self.integration_method = integration_method
        if integration_method == "euler":
            self.f = self.fd_euler
        elif integration_method == "rk4":
            self.f = self.fd_rk4
        else:
            raise ValueError("Invalid integration method. Choose 'euler' or 'rk4'.")
        
        self._f = jit(self.f)
        self._f_x = jit(jacfwd(self.f))
        self._f_u = jit(jacfwd(self.f, argnums=1))

        self._has_hessians = hessians
        if hessians:
            self._f_xx = jit(hessian(self.f, argnums=0))
            self._f_ux = jit(jacfwd( jacfwd(self.f, argnums=1) ))
            self._f_uu = jit(hessian(self.f, argnums=1))

        super(ErrorStateSE3AutoDiffDynamics, self).__init__()

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
        return self._Ib

    @property
    def J(self):
        """Inertia matrix of the system."""
        return self._J
    
    @property
    def Jinv(self):
        """Inverse of the inertia matrix."""
        return self._Jinv

    @property
    def horizon(self):
        """The horizon for dynamics to be valid, due to horizon of given reference."""
        return self._horizon
    
    @property
    def dt(self):
        """Sampling time of the system dynamics."""
        return self._dt

    def xi_ref(self, i) :
        """Return the Lie Algebra velocity xi reference xi_ref at time index i."""
        return self._xi_ref[i]

    def X_ref(self, i) :
        """Return the Lie group reference X_ref at time index i."""
        return self._X_ref[i]

    def x_ref(self, i) :
        """Return the concatenated Lie group and Lie algebra reference X_ref at time index i."""
        return self._x_ref[i]

    def adjoint( self, xi ):
        """ Get the the adjoint matrix representation of Lie Algebra."""
        w = np.array([xi[0], xi[1], xi[2]])
        v = np.array([xi[3], xi[4], xi[5]])
        adx = np.block([
            [skew(w), np.zeros((3, 3))],
            [skew(v), skew(w)]
        ])
        return adx
    
    def coadjoint( self, xi ):
        """ Get the the coadjoint matrix representation of Lie Algebra."""
        return self.adjoint(xi).T

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
        xi = x[-self.vel_state_size:]
        omega = xi[:3]
        v = xi[-3:]

        # print("v is shape of \n", v.shape, "with value", v)

        G = np.block([
            [skew( self.Ib @ omega ), self.m * skew( v )],
            [self.m * skew( v ), np.zeros((3,3))],        
        ])
        Ht = - self.Jinv @ ( self.coadjoint( xi ) @ self.J + G )
        bt = - self.Jinv @ G @ xi

        # print("\nG is shape of", G.shape, "with value \n", G)
        # print("\nHt is shape of", Ht.shape, "with value \n", Ht)
        # print("\nbt is shape of", bt.shape, "with value \n", bt)

        At = np.block([
            [- self.adjoint( self.xi_ref(i) ), np.identity( self.error_state_size )],
            [np.zeros((self.vel_state_size, self.error_state_size)), Ht]
        ])
        Bt = np.vstack((np.zeros((self.error_state_size, self.action_size)),
                self.Jinv ))
        ht = np.vstack( (-self.xi_ref(i), bt ))

        # print("\nAt is shape of", At.shape, "with value \n", At)
        # print("\nBt is shape of", Bt.shape, "with value \n", Bt)
        # print("\nht is shape of", ht.shape, "with value \n", ht)

        xt_dot = At @ x + Bt @ u + ht
        
        return xt_dot
    
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
        return x + self.fc( x,u,i ) * self.dt
    
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
    
        return x_next

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
    
