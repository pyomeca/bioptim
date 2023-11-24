from typing import Callable

from casadi import MX, SX, DM, vertcat, horzcat
import numpy as np

from ..misc.enums import Node, QuadratureRule, PhaseDynamics

class PenaltyHelpers:
    
    @staticmethod
    def t0(penalty, penalty_node_idx, get_t0: Callable):
        """
        Parameters
        ----------
        penalty: PenaltyFunctionAbstract
            The penalty function
        penalty_node_idx: int
            The index of the node in the penalty
        get_t0: Callable
            A function that returns the time of the node. It is expected to return stepwise time
        
        TODO COMPLETE
        """
        if penalty.transition or penalty.multinode_penalty:
            phases = penalty.nodes_phase
            nodes = penalty.multinode_idx
        else:
            phases = [penalty.phase]
            nodes = [penalty.node_idx[penalty_node_idx]]

        if len(phases) > 1:
            raise NotImplementedError("penalty cost over multiple phases is not implemented yet")
        
        return vertcat(*[_reshape_to_vector(get_t0(phases[0], n)[0]) for n in nodes])

    @staticmethod
    def phases_dt(penalty, get_all_dt: Callable):
        """
        Parameters
        ----------
        penalty: PenaltyFunctionAbstract
            The penalty function
        get_all_dt: Callable
            A function that returns the dt of the all phases

        TODO COMPLETE
        """

        return _reshape_to_vector(get_all_dt())

    @staticmethod
    def states(penalty, penalty_node_idx, get_state_decision: Callable):
        if isinstance(penalty.phase, list) and len(penalty.phase) > 1:
            raise NotImplementedError("penalty cost over multiple phases is not implemented yet")

        if penalty.transition or penalty.multinode_penalty:
            x = get_state_decision(penalty.phase, penalty.node_idx[penalty_node_idx])
            return _reshape_to_vector(x)
        
        elif penalty.derivative:
            x = get_state_decision(penalty.phase, penalty.node_idx[penalty_node_idx])
            x_next = get_state_decision(penalty.phase, penalty.node_idx[penalty_node_idx + 1])
            return x.reshape((-1, 1)), x_next[:, 0].reshape((-1, 1))

        else:
            x = get_state_decision(penalty.phase, penalty.node_idx[penalty_node_idx])

        if penalty.explicit_derivative:
            x = _reshape_to_vector(x)
            x = vertcat(x, get_state_decision(penalty.phase, penalty.node_idx[penalty_node_idx] + 1)[:, 0])
        
        return _reshape_to_vector(x)


    @staticmethod
    def controls(penalty, penalty_node_idx, get_control_decision: Callable):
        u = get_control_decision(penalty.phase, penalty.node_idx[penalty_node_idx])

        if penalty.explicit_derivative:
            u = _reshape_to_vector(u)
            u = vertcat(u, get_control_decision(penalty.phase, penalty.node_idx[penalty_node_idx] + 1)[:, 0])
        
        return _reshape_to_vector(u)

    @staticmethod
    def parameters(penalty, get_parameter: Callable):
        p = get_parameter()
        return _reshape_to_vector(p)

    @staticmethod
    def stochastic(penalty, penalty_node_idx, get_stochastic: Callable):
        s = get_stochastic(penalty.phase, penalty.node_idx[penalty_node_idx])
        return _reshape_to_vector(s)

    @staticmethod
    def weight(penalty):
        return penalty.weight

    @staticmethod
    def target(penalty, penalty_node_idx):
        if penalty.target is None:
            return np.ndarray([])
        
        target = penalty.target[0][..., penalty.node_idx.index(penalty_node_idx)]
        return _reshape_to_vector(target)


def _reshape_to_vector(m):
    if isinstance(m, (SX, MX, DM)):
        return m.reshape((-1, 1))
    elif isinstance(m, np.ndarray):
        return m.reshape((-1, 1), order="F")
    else:
        raise RuntimeError("Invalid type to reshape")
