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
    def states(penalty, ocp, penalty_node_idx, get_state_decision: Callable):
        if isinstance(penalty.phase, list) and len(penalty.phase) > 1:
            raise NotImplementedError("penalty cost over multiple phases is not implemented yet")

        if penalty.transition or penalty.multinode_penalty:
            x = []
            phases, nodes = _get_multinode_indices(penalty)
            for phase, node in zip(phases, nodes):
                tp = get_state_decision(phase, node)
                if penalty.transition:
                    tp = tp[:, 0:1]
                x.append(_reshape_to_vector(tp))
            return _vertcat(x)

        penalty_node_index = 0 if penalty_node_idx is None else penalty.node_idx[penalty_node_idx]
        x = get_state_decision(penalty.phase, penalty_node_index)

        need_end_point = penalty.integration_rule in (QuadratureRule.APPROXIMATE_TRAPEZOIDAL,)
        if need_end_point or penalty.derivative or penalty.explicit_derivative:
            x = _reshape_to_vector(x)
            x_next = get_state_decision(penalty.phase, penalty_node_index + 1)[:, 0]
            x = vertcat(x_next, x) if penalty.derivative else vertcat(x, x_next)  #TODO: change order ?

        return _reshape_to_vector(x)

    @staticmethod
    def controls(penalty, ocp, penalty_node_idx, get_control_decision: Callable):
        
        def _get_control_internal(_phase, _node):
            _nlp = ocp.nlp[_phase]

            _u = get_control_decision(_phase, _node)
            if _nlp.phase_dynamics == PhaseDynamics.ONE_PER_NODE and _node >= _nlp.n_controls_nodes:
                if isinstance(_u, (MX, SX, DM)):
                    return type(_u)()
                elif isinstance(_u, np.ndarray):
                    return np.ndarray((0, 1))
                else:
                    raise RuntimeError("Invalid type for control")
                
            return _u[:, 0]  # That is so Linear controls don't return two columns, it will be dealty with later

        if penalty.transition or penalty.multinode_penalty:
            u = []
            phases, nodes = _get_multinode_indices(penalty)
            for phase, node in zip(phases, nodes):
                u.append(_reshape_to_vector(_get_control_internal(phase, node)))
            return _vertcat(u)

        penalty_node_index = 0 if penalty_node_idx is None else penalty.node_idx[penalty_node_idx]
        u = _get_control_internal(penalty.phase, penalty_node_index)

        nlp = ocp.nlp[penalty.phase]
        is_linear = nlp.control_type == ControlType.LINEAR_CONTINUOUS
        if is_linear or penalty.integrate or penalty.derivative or penalty.explicit_derivative:
            u = _reshape_to_vector(u)
            
            next_node = penalty_node_index + 1  # (0 if penalty.derivative else 1)
            if penalty.derivative and nlp.phase_dynamics == PhaseDynamics.ONE_PER_NODE and next_node >= nlp.n_controls_nodes:
                next_node -= 1
            step = 0  # TODO: This should be 1 for integrate if TRAPEZOIDAL
            next_u = _get_control_internal(penalty.phase, next_node)
            if np.sum(next_u.shape) > 0:
                u = vertcat(next_u, u) if penalty.derivative else vertcat(u, next_u)

        return _reshape_to_vector(u)

    @staticmethod
    def parameters(penalty, ocp, get_parameter: Callable):
        p = get_parameter()
        return _reshape_to_vector(p)

    @staticmethod
    def stochastic_variables(penalty, ocp, penalty_node_idx, get_stochastic_decision: Callable):
        if penalty.transition or penalty.multinode_penalty:
            s = []
            phases, nodes = _get_multinode_indices(penalty)
            for phase, node in zip(phases, nodes):
                tp = get_stochastic_decision(phase, node)
                if penalty.transition:
                    tp = tp[:, 0:1]
                s.append(_reshape_to_vector(tp))
            return _vertcat(s)

        penalty_node_index = 0 if penalty_node_idx is None else penalty.node_idx[penalty_node_idx]
        s = get_stochastic_decision(penalty.phase, penalty_node_index)

        need_end_point = penalty.integration_rule in (QuadratureRule.APPROXIMATE_TRAPEZOIDAL,)
        if need_end_point or penalty.derivative or penalty.explicit_derivative:
            s = _reshape_to_vector(s)
            s_next = get_stochastic_decision(penalty.phase, penalty_node_index + 1)[:, 0]
            s = vertcat(s_next, s) if penalty.derivative else vertcat(s, s_next)

        return _reshape_to_vector(s)

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
