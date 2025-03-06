from typing import Callable, Protocol

import numpy as np
from casadi import MX, SX, DM, vertcat

from ..misc.enums import PhaseDynamics, ControlType


class PenaltyProtocol(Protocol):
    transition: bool  # If the penalty is a transition penalty
    multinode_penalty: bool  # If the penalty is a multinode penalty
    phase: int  # The phase of the penalty (only for non multinode or transition penalties)
    nodes_phase: list[int]  # The phases of the penalty (only for multinode penalties)
    node_idx: list[int]  # The node index of the penalty (only for non multinode or transition penalties)
    multinode_idx: list[int]  # The node index of the penalty (only for multinode penalties)
    subnodes_are_decision_states: list[bool]  # If the subnodes are decision states (e.g. collocation points)
    integrate: bool  # If the penalty is an integral penalty
    derivative: bool  # If the penalty is a derivative penalty
    explicit_derivative: bool  # If the penalty is an explicit derivative penalty
    phase_dynamics: list[PhaseDynamics]  # The dynamics of the penalty (only for multinode penalties)
    ns = list[int]  # The number of shooting points of problem (only for multinode penalties)
    control_types: ControlType  # The control type of the penalties


class PenaltyHelpers:
    @staticmethod
    def t0(penalty, index, get_t0: Callable):
        """
        This method returns the t0 of a penalty.
        """

        if penalty.multinode_penalty:
            phases, nodes, sub_nodes = _get_multinode_indices(penalty, False)
            phase, node, sub_node = phases[0], nodes[0], sub_nodes[0]
        else:
            phase, node = penalty.phase, penalty.node_idx[index]

        return get_t0(phase, node)

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
    def states(penalty, index, get_state_decision: Callable, is_constructing_penalty: bool = False):
        """
        get_state_decision: Callable[int, int, slice]
            A function that returns the state decision of a given phase, node and subnodes (or steps)
            If the subnode requests slice(0, None), it actually does not expect the very last node (equal to starting
            of the next node), if it needs so, it will actively asks for slice(-1, None) to get the last node.
            When slice(-1, None) is requested, if it is in constructing phase of the penalty, it expects cx_end.
            The slice(-1, None) will not be requested when the penalty is not being constructed (it will request
            slice(0, 1) of the following node instead)
        """
        if isinstance(penalty.phase, list) and len(penalty.phase) > 1:
            raise NotImplementedError("penalty cost over multiple phases is not implemented yet")

        node = penalty.node_idx[index]

        if penalty.multinode_penalty:
            x = []
            phases, nodes, sub_nodes = _get_multinode_indices(penalty, is_constructing_penalty)
            for phase, node, sub_node in zip(phases, nodes, sub_nodes):
                x.append(_reshape_to_vector(get_state_decision(phase, node, sub_node)))
            return _vertcat(x)

        else:
            subnodes = slice(0, None if node < penalty.ns[0] and penalty.subnodes_are_decision_states[0] else 1)
            x0 = _reshape_to_vector(get_state_decision(penalty.phase, node, subnodes))

            if is_constructing_penalty:
                if node < penalty.ns[0]:
                    x1 = _reshape_to_vector(get_state_decision(penalty.phase, node, slice(-1, None)))
                else:
                    x1 = type(x0).sym("dummy_x", 0, 1)
            else:
                x1 = _reshape_to_vector(get_state_decision(penalty.phase, node + 1, slice(0, 1)))
            return vertcat(x0, x1)

    @staticmethod
    def controls(penalty, index, get_control_decision: Callable, is_constructing_penalty: bool = False):
        node = penalty.node_idx[index]

        if penalty.multinode_penalty:
            u = []
            phases, nodes, sub_nodes = _get_multinode_indices(penalty, is_constructing_penalty)
            for phase, node, sub in zip(phases, nodes, sub_nodes):
                u.append(_reshape_to_vector(get_control_decision(phase, node, sub)))
            return _vertcat(u)

        if is_constructing_penalty:
            if penalty.control_types[0] in (ControlType.LINEAR_CONTINUOUS,):
                final_subnode = None if node < penalty.ns[0] else 1
                u = _reshape_to_vector(get_control_decision(penalty.phase, node, slice(0, final_subnode)))
            else:
                u = _reshape_to_vector(get_control_decision(penalty.phase, node, slice(0, 1)))  # cx_start
                if node < penalty.ns[0] - 1:
                    u1 = _reshape_to_vector(get_control_decision(penalty.phase, node, slice(-1, None)))
                    u = vertcat(u, u1)
                elif node < penalty.ns[0] and penalty.control_types[0] == ControlType.CONSTANT_WITH_LAST_NODE:
                    u1 = _reshape_to_vector(get_control_decision(penalty.phase, node, slice(-1, None)))
                    u = vertcat(u, u1)
                else:
                    pass

        else:
            u0 = _reshape_to_vector(get_control_decision(penalty.phase, node, slice(0, 1)))
            u1 = _reshape_to_vector(get_control_decision(penalty.phase, node + 1, slice(0, 1)))
            u = _vertcat([u0, u1])

        return u

    @staticmethod
    def parameters(penalty, index, get_parameter_decision: Callable):
        node = penalty.node_idx[index]
        p = get_parameter_decision(penalty.phase, node, None)
        return _reshape_to_vector(p)

    @staticmethod
    def numerical_timeseries(penalty, index, get_numerical_timeseries: Callable):
        node = penalty.node_idx[index]
        if penalty.multinode_penalty:
            # numerical timeseries are expected to be provided only at the shooting node.
            for i_phase in penalty.nodes_phase:
                d = get_numerical_timeseries(i_phase, node, 0)  # cx_start
                if d.shape[0] != 0:
                    raise NotImplementedError(
                        "Numerical data timeseries is not implemented for multinode penalties yet."
                    )
                    # Note to the developers: We do not think this will raise an error at runtime,
                    # but the results will be wrong is cx_start or cx_end are used in multiple occasions with different values.
        else:
            d = get_numerical_timeseries(penalty.phase, node, 0)  # cx_start

        if d.shape != (0, 0):
            d = _reshape_to_vector(d)

        return d

    @staticmethod
    def weight(penalty):
        return penalty.weight

    @staticmethod
    def target(penalty, penalty_node_idx):
        if penalty.target is None:
            return np.array([])

        if penalty.integrate:
            target0 = penalty.target[..., penalty_node_idx]
            target1 = penalty.target[..., penalty_node_idx + 1]
            return np.vstack((target0, target1)).T

        return penalty.target[..., penalty_node_idx]


def _get_multinode_indices(penalty, is_constructing_penalty: bool):
    if not penalty.multinode_penalty:
        raise RuntimeError("This function should only be called for multinode penalties")

    phases = penalty.nodes_phase
    nodes = penalty.multinode_idx
    sub_nodes = []
    for subnode in penalty.sub_nodes:
        if subnode == -1:
            sub_nodes.append(slice(-1, None))
        else:
            sub_nodes.append(slice(subnode, subnode+1))

    return phases, nodes, sub_nodes

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
