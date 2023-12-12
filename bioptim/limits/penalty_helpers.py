from typing import Callable

from casadi import MX, SX, DM, vertcat
import numpy as np

from ..misc.enums import QuadratureRule, PhaseDynamics, ControlType


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
            phases, nodes = _get_multinode_indices(penalty)
            phase = phases[0]
            node = nodes[0]
        else:
            phase = penalty.phase
            node = penalty.node_idx[penalty_node_idx]

        return get_t0(phase, node)[0, 0]

    @staticmethod
    def phases_dt(penalty, ocp, get_all_dt: Callable):
        """
        Parameters
        ----------
        penalty: PenaltyFunctionAbstract
            The penalty function
        get_all_dt: Callable
            A function that returns the dt of the all phases

        TODO COMPLETE
        """

        return _reshape_to_vector(_reshape_to_vector(get_all_dt(ocp.time_phase_mapping.to_first.map_idx)))

    @staticmethod
    def states(penalty, index, get_state_decision: Callable):
        if isinstance(penalty.phase, list) and len(penalty.phase) > 1:
            raise NotImplementedError("penalty cost over multiple phases is not implemented yet")
                
        index = 0 if index is None else penalty.node_idx[index]

        if penalty.integration_rule in (QuadratureRule.APPROXIMATE_TRAPEZOIDAL,) or penalty.integrate:
            return _reshape_to_vector(get_state_decision(penalty.phase, index, slice(0, -1)))
        
        elif penalty.derivative or penalty.explicit_derivative:
            x0 = _reshape_to_vector(get_state_decision(penalty.phase, index, slice(0, 0)))
            x1 = _reshape_to_vector(get_state_decision(penalty.phase, index, slice(-1, -1)))
            return vertcat(x1, x0) if penalty.derivative else vertcat(x0, x1)

        elif penalty.transition or penalty.multinode_penalty:
            x = []
            phases, nodes = _get_multinode_indices(penalty)
            for phase, node in zip(phases, nodes):
                x.append(_reshape_to_vector(get_state_decision(phase, node, slice(0, 0))))
            return _vertcat(x)
        
        else:
            return _reshape_to_vector(get_state_decision(penalty.phase, index, slice(0, 0)))



    @staticmethod
    def controls(penalty, index, get_control_decision: Callable):
        index = 0 if index is None else penalty.node_idx[index]

        if penalty.transition or penalty.multinode_penalty:
            u = []
            phases, nodes = _get_multinode_indices(penalty)
            for phase, node in zip(phases, nodes):
                u.append(_reshape_to_vector(get_control_decision(phase, node, slice(0, 0))))
            return _vertcat(u)

        elif penalty.integrate or penalty.derivative or penalty.explicit_derivative:
            return _reshape_to_vector(get_control_decision(penalty.phase, index, slice(0, -1)))
            
        else:
            return _reshape_to_vector(get_control_decision(penalty.phase, index, slice(0, 0)))

    @staticmethod
    def parameters(penalty, ocp, get_parameter: Callable):
        p = get_parameter()
        return _reshape_to_vector(p)

    @staticmethod
    def weight(penalty):
        return penalty.weight

    @staticmethod
    def target(penalty, penalty_node_idx):
        if penalty.target is None:
            return np.array([])
        
        if penalty.integration_rule in (QuadratureRule.APPROXIMATE_TRAPEZOIDAL, QuadratureRule.TRAPEZOIDAL):
            target0 = penalty.target[0][..., penalty_node_idx]
            target1 = penalty.target[1][..., penalty_node_idx]
            return np.vstack((target0, target1)).T

        return penalty.target[0][..., penalty_node_idx]


def _get_multinode_indices(penalty):
    if penalty.transition:
        if len(penalty.nodes_phase) != 2 or len(penalty.multinode_idx) != 2:
            raise RuntimeError("Transition must have exactly 2 nodes and 2 phases")
        phases = [penalty.nodes_phase[1], penalty.nodes_phase[0]]
        nodes = [penalty.multinode_idx[1], penalty.multinode_idx[0]]
    else:
        phases = penalty.nodes_phase
        nodes = penalty.multinode_idx
    return phases, nodes


def _reshape_to_vector(m):
    """
    Reshape a matrix to a vector (column major)
    """

    if isinstance(m, (SX, MX, DM)):
        return m.reshape((-1, 1))
    elif isinstance(m, np.ndarray):
        return m.reshape((-1, 1), order="F")
    else:
        raise RuntimeError("Invalid type to reshape")


def _vertcat(v):
    """
    Vertically concatenate a list of vectors
    """

    if not isinstance(v, list):
        raise ValueError("_vertcat must be called with a list of vectors")
    
    data_type = type(v[0])
    for tp in v:
        if not isinstance(tp, data_type):
            raise ValueError("All elements of the list must be of the same type")

    if isinstance(v[0], (SX, MX, DM)):
        return vertcat(*v)
    elif isinstance(v[0], np.ndarray):
        return np.vstack(v)
    else:
        raise RuntimeError("Invalid type to vertcat")
