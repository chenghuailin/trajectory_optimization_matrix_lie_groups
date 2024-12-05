import abc
import numpy as np


class BaseConstraint():

    """Base constraint."""

    # @abc.abstractmethod
    # def g_sum(self, xs, us, *args, **kwargs):
    #     """Evaluate the inequaltiy constraints.

    #     Args:
    #         xs: state [N, state_size]..
    #         us: control input [N, action_size].
    #         *args, **kwargs: Additional positional and key-word arguments.

    #     Returns:
    #         constr: evaluated constraints [num_constr,].
    #     """
    #     raise NotImplementedError
    
    @abc.abstractmethod
    def g(self, x, u, i, terminal=False, *args, **kwargs):
        """Evaluate the stage-wise inequaltiy constraints.

        Args:
            x: state [state_size, ].
            u: control input [action_size, ].
            *args, **kwargs: Additional positional and key-word arguments.

        Returns:
            constr: evaluated constraints [num_constr_per_stage,].
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def g_x(self, x, u, i, terminal=False, *args, **kwargs):
        """Partial derivative of stage-wise inequaltiy constraint w.r.t x.

        Args:
            x: state [state_size, ].
            u: control input [action_size, ].
            *args, **kwargs: Additional positional and key-word arguments.

        Returns:
            g_x: evaluated constraints [num_constr_per_stage, state_size].
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def g_u(self, x, u, i, terminal=False, *args, **kwargs):
        """Partial derivative of stage-wise inequaltiy constraint w.r.t u.

        Args:
            x: state [state_size, ].
            u: control input [action_size, ].
            *args, **kwargs: Additional positional and key-word arguments.

        Returns:
            g_u: evaluated constraints [num_constr_per_stage, action_size].
        """
        raise NotImplementedError
    

class InputConstraint(BaseConstraint):

    """Input constraints."""

    def __init__( self, input_lb, input_ub, 
                  state_size=(6,6), action_size=6 ):

        self._state_size = state_size[0] + state_size[1]
        self._error_state_size = state_size[0] 
        self._vel_state_size = state_size[1] 
        self._action_size = action_size

        self._lb = input_lb
        self._ub = input_ub

        self._constr_size =  2 * action_size

    @property
    def lb(self):
        return self._lb
    
    @property
    def ub(self):
        return self._ub

    @property
    def constr_size(self):
        return self._constr_size

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

    @abc.abstractmethod
    def g(self, x, u, i, terminal=False, *args, **kwargs):
        """Evaluate the stage-wise inequaltiy constraints.

        Args:
            x: state [state_size, ].
            u: control input [action_size, ].
            *args, **kwargs: Additional positional and key-word arguments.

        Returns:
            constr: evaluated constraints [num_constr_per_stage,].
        """
        if terminal:
            return np.zeros((self._constr_size,))

        return np.concatenate([
            self.lb - u, 
            u - self.ub
        ])
    
    @abc.abstractmethod
    def g_x(self, x, u, i, terminal=False, *args, **kwargs):
        """Partial derivative of stage-wise inequaltiy constraint w.r.t x.

        Args:
            x: state [state_size, ].
            u: control input [action_size, ].
            *args, **kwargs: Additional positional and key-word arguments.

        Returns:
            g_x: evaluated constraints [num_constr_per_stage, state_size].
        """
        # if terminal:
        #     return np.zeros([ self.constr_size, self.state_size ])

        return np.zeros([ self.constr_size, self.state_size ])
    
    @abc.abstractmethod
    def g_u(self, x, u, i, terminal=False, *args, **kwargs):
        """Partial derivative of stage-wise inequaltiy constraint w.r.t u.

        Args:
            x: state [state_size, ].
            u: control input [action_size, ].
            *args, **kwargs: Additional positional and key-word arguments.

        Returns:
            g_u: evaluated constraints [num_constr_per_stage, action_size].
        """
        if terminal:
            return np.zeros([ self.constr_size, self.action_size ])

        return np.vstack([ 
            -1*np.identity( self.action_size ), np.identity( self.action_size ) 
        ])
    
    