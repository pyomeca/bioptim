from typing import Callable

from casadi import MX, SX, DM, vertcat
import numpy as np

from ..misc.enums import Node, QuadratureRule, PhaseDynamics

class PenaltyHelpers:
    @staticmethod
    def weight(penalty):
        return penalty.weight
    
    @staticmethod
    def dt(penalty):
        if penalty.transition or penalty.multinode_penalty:
            return 1  # That is so multinode penalties behave like Mayer functions
        else:
            return penalty.dt

    @staticmethod
    def states(penalty, penalty_node_idx, get_state_decision: Callable):
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
    def parameters(penalty, penalty_node_idx, get_parameter: Callable):
        p = get_parameter(penalty.phase, penalty.node_idx[penalty_node_idx])
        return _reshape_to_vector(p)

    @staticmethod
    def stochastic(penalty, penalty_node_idx, get_stochastic: Callable):
        s = get_stochastic(penalty.phase, penalty.node_idx[penalty_node_idx])
        return _reshape_to_vector(s)


def _reshape_to_vector(m):
    if isinstance(m, (SX, MX, DM)):
        return m.reshape((-1, 1))
    elif isinstance(m, np.ndarray):
        return m.reshape((-1, 1), order="F")
    else:
        raise RuntimeError("Invalid type to reshape")
