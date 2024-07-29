"""Instantaneous Cost Function."""

import six
import abc
import numpy as np
import theano.tensor as T
from scipy.optimize import approx_fprime
from .autodiff import as_function, hessian_scalar, jacobian_scalar

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

    def __init__(self, l, l_terminal, x_inputs, u_inputs, i=None, **kwargs):
        """Constructs an AutoDiffCost.

        Args:
            l: Vector Theano tensor expression for instantaneous cost.
                This needs to be a function of x and u and must return a scalar.
            l_terminal: Vector Theano tensor expression for terminal cost.
                This needs to be a function of x only and must retunr a scalar.
            x_inputs: Theano state input variables [state_size].
            u_inputs: Theano action input variables [action_size].
            i: Theano tensor time step variable.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
        self._i = T.dscalar("i") if i is None else i
        self._x_inputs = x_inputs
        self._u_inputs = u_inputs

        non_t_inputs = np.hstack([x_inputs, u_inputs]).tolist()
        inputs = np.hstack([x_inputs, u_inputs, self._i]).tolist()
        terminal_inputs = np.hstack([x_inputs, self._i]).tolist()

        x_dim = len(x_inputs)
        u_dim = len(u_inputs)

        self._J = jacobian_scalar(l, non_t_inputs)
        self._Q = hessian_scalar(l, non_t_inputs)

        self._l = as_function(l, inputs, name="l", **kwargs)

        self._l_x = as_function(self._J[:x_dim], inputs, name="l_x", **kwargs)
        self._l_u = as_function(self._J[x_dim:], inputs, name="l_u", **kwargs)

        self._l_xx = as_function(self._Q[:x_dim, :x_dim],
                                 inputs,
                                 name="l_xx",
                                 **kwargs)
        self._l_ux = as_function(self._Q[x_dim:, :x_dim],
                                 inputs,
                                 name="l_ux",
                                 **kwargs)
        self._l_uu = as_function(self._Q[x_dim:, x_dim:],
                                 inputs,
                                 name="l_uu",
                                 **kwargs)

        # Terminal cost only depends on x, so we only need to evaluate the x
        # partial derivatives.
        self._J_terminal = jacobian_scalar(l_terminal, x_inputs)
        self._Q_terminal = hessian_scalar(l_terminal, x_inputs)

        self._l_terminal = as_function(l_terminal,
                                       terminal_inputs,
                                       name="l_term",
                                       **kwargs)
        self._l_x_terminal = as_function(self._J_terminal[:x_dim],
                                         terminal_inputs,
                                         name="l_term_x",
                                         **kwargs)
        self._l_xx_terminal = as_function(self._Q_terminal[:x_dim, :x_dim],
                                          terminal_inputs,
                                          name="l_term_xx",
                                          **kwargs)

        super(AutoDiffCost, self).__init__()

    @property
    def x(self):
        """The state variables."""
        return self._x_inputs

    @property
    def u(self):
        """The control variables."""
        return self._u_inputs

    @property
    def i(self):
        """The time step variable."""
        return self._i

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
            z = np.hstack([x, i])
            return np.asscalar(self._l_terminal(*z))

        z = np.hstack([x, u, i])
        return np.asscalar(self._l(*z))

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
            z = np.hstack([x, i])
            return np.array(self._l_x_terminal(*z))

        z = np.hstack([x, u, i])
        return np.array(self._l_x(*z))

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

        z = np.hstack([x, u, i])
        return np.array(self._l_u(*z))

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
            z = np.hstack([x, i])
            return np.array(self._l_xx_terminal(*z))

        z = np.hstack([x, u, i])
        return np.array(self._l_xx(*z))

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

        z = np.hstack([x, u, i])
        return np.array(self._l_ux(*z))

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

        z = np.hstack([x, u, i])
        return np.array(self._l_uu(*z))