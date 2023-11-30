from typing import Callable

from casadi import MX, SX, DM, vertcat, horzcat
import numpy as np

from ..misc.enums import Node, QuadratureRule, PhaseDynamics, ControlType


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
            x = []
            phases, nodes = _get_multinode_indices(penalty)
            for phase, node in zip(phases, nodes):
                tp = get_state_decision(phase, node)
                if penalty.transition:
                    tp = tp[:, 0:1]
                x.append(_reshape_to_vector(tp))
            return _vertcat(x)
        

        x = get_state_decision(penalty.phase, penalty.node_idx[penalty_node_idx])
        
        if penalty.derivative or penalty.explicit_derivative:
            x = _reshape_to_vector(x)
            x_next = get_state_decision(penalty.phase, penalty.node_idx[penalty_node_idx] + 1)[:, 0]
            x = vertcat(x_next, x) if penalty.derivative else vertcat(x, x_next) 

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

        u = _get_control_internal(penalty.phase, penalty.node_idx[penalty_node_idx])

        is_linear = ocp.nlp[penalty.phase].control_type == ControlType.LINEAR_CONTINUOUS
        if is_linear or penalty.integrate or penalty.derivative or penalty.explicit_derivative:
            u = _reshape_to_vector(u)
            
            next_node = penalty.node_idx[penalty_node_idx] + (0 if penalty.derivative else 1)
            step = 0  # TODO: This should be 1 for integrate if TRAPEZOIDAL
            next_u = _get_control_internal(penalty.phase, next_node)
            if np.sum(next_u.shape) > 0:
                u = vertcat(next_u, u) if penalty.derivative else vertcat(u, next_u)

        return _reshape_to_vector(u)

    @staticmethod
    def parameters(penalty, get_parameter: Callable):
        p = get_parameter()
        return _reshape_to_vector(p)

    @staticmethod
    def stochastic(penalty, penalty_node_idx, get_stochastic: Callable):
        if penalty.transition or penalty.multinode_penalty:
            x = []
            phases, nodes = _get_multinode_indices(penalty)
            for phase, node in zip(phases, nodes):
                tp = get_stochastic(phase, node)
                if penalty.transition:
                    tp = tp[:, 0:1]
                x.append(_reshape_to_vector(tp))
            return _vertcat(x)
        
        s = get_stochastic(penalty.phase, penalty.node_idx[penalty_node_idx])
        return _reshape_to_vector(s)

    @staticmethod
    def weight(penalty):
        return penalty.weight

    @staticmethod
    def target(penalty, penalty_node_idx):
        if penalty.target is None:
            return np.array([])
        
        return penalty.target[0][..., penalty_node_idx]

        # if penalty.target is None:
        #     target = []
        # elif (
        #     penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
        #     or penalty.integration_rule == QuadratureRule.TRAPEZOIDAL
        # ):
        #     target0 = format_target(penalty, penalty.target[0], idx)
        #     target1 = format_target(penalty, penalty.target[1], idx)
        #     target = np.vstack((target0, target1)).T
        # else:
        #     target = format_target(penalty, penalty.target[0], idx)



    def _get_x_u_s_at_idx(ocp, nlp, penalty, _idx, is_unscaled):
        """ """

        if penalty.transition:
            ocp = ocp
            cx = ocp.cx

            all_nlp = ocp.nlp

            phase_node0 = penalty.nodes_phase[0]
            phase_node1 = penalty.nodes_phase[1]

            node_idx_0 = penalty.all_nodes_index[0]
            node_idx_1 = penalty.all_nodes_index[1]

            u0_mode = get_control_modificator(ocp, penalty, 0)
            u1_mode = get_control_modificator(ocp, penalty, 1)

            _x_0 = get_padded_array(
                nlp=all_nlp[phase_node0],
                attribute="X" if is_unscaled else "X_scaled",
                node_idx=node_idx_0,
                target_length=all_nlp[phase_node1].X_scaled[node_idx_1].shape[0],
                casadi_constructor=cx,
            )
            _x_1 = get_padded_array(
                nlp=all_nlp[phase_node1],
                attribute="X" if is_unscaled else "X_scaled",
                node_idx=node_idx_1,
                target_length=all_nlp[phase_node0].X_scaled[node_idx_0].shape[0],
                casadi_constructor=cx,
            )

            _s_0 = get_padded_array(
                nlp=all_nlp[phase_node0],
                attribute="S" if is_unscaled else "S_scaled",
                node_idx=node_idx_0,
                target_length=all_nlp[phase_node1].S[node_idx_1].shape[0],
                casadi_constructor=cx,
            )
            _s_1 = get_padded_array(
                nlp=all_nlp[phase_node1],
                attribute="S" if is_unscaled else "S_scaled",
                node_idx=node_idx_1,
                target_length=all_nlp[phase_node0].S[node_idx_0].shape[0],
                casadi_constructor=cx,
            )

            is_shared_dynamics_0, is_node0_within_control_limit, len_u_0 = get_node_control_info(
                all_nlp[phase_node0], node_idx_0, attribute="U" if is_unscaled else "U_scaled"
            )
            is_shared_dynamics_1, is_node1_within_control_limit, len_u_1 = get_node_control_info(
                all_nlp[phase_node1], node_idx_1, attribute="U" if is_unscaled else "U_scaled"
            )

            _u_0 = get_padded_control_array(
                all_nlp[phase_node0],
                node_idx_0,
                attribute="U" if is_unscaled else "U_scaled",
                u_mode=u0_mode,
                target_length=len_u_1,
                is_shared_dynamics_target=is_shared_dynamics_1,
                is_within_control_limit_target=is_node1_within_control_limit,
                casadi_constructor=cx,
            )

            _u_1 = get_padded_control_array(
                all_nlp[phase_node1],
                node_idx_1,
                attribute="U" if is_unscaled else "U_scaled",
                u_mode=u1_mode,
                target_length=len_u_0,
                is_shared_dynamics_target=is_shared_dynamics_0,
                is_within_control_limit_target=is_node0_within_control_limit,
                casadi_constructor=cx,
            )

            _x = vertcat(_x_1, _x_0)
            _u = vertcat(_u_1, _u_0)
            _s = vertcat(_s_1, _s_0)

        elif penalty.multinode_penalty:

            # Make an exception to the fact that U is not available for the last node
            _x = ocp.cx()
            _u = ocp.cx()
            _s = ocp.cx()
            for i in range(len(penalty.nodes_phase)):
                nlp_i = ocp.nlp[penalty.nodes_phase[i]]
                index_i = penalty.multinode_idx[i]
                ui_mode = get_control_modificator(ocp, _penalty=penalty, index=i)

                if is_unscaled:
                    _x_tp = nlp_i.cx()
                    if penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
                        _x_tp = vertcat(_x_tp, nlp_i.X[index_i][:, 0])
                    else:
                        for i in range(nlp_i.X[index_i].shape[1]):
                            _x_tp = vertcat(_x_tp, nlp_i.X[index_i][:, i])
                    _u_tp = (
                        nlp_i.U[index_i - ui_mode]
                        if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or index_i < len(nlp_i.U)
                        else []
                    )
                    _s_tp = nlp_i.S[index_i]
                else:
                    _x_tp = nlp_i.cx()
                    if penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
                        _x_tp = vertcat(_x_tp, nlp_i.X_scaled[index_i][:, 0])
                    else:
                        for i in range(nlp_i.X_scaled[index_i].shape[1]):
                            _x_tp = vertcat(_x_tp, nlp_i.X_scaled[index_i][:, i])
                    _u_tp = (
                        nlp_i.U_scaled[index_i - ui_mode]
                        if nlp_i.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or index_i < len(nlp_i.U_scaled)
                        else []
                    )
                    _s_tp = nlp_i.S_scaled[index_i]

                _x = vertcat(_x, _x_tp)
                _u = vertcat(_u, _u_tp)
                _s = vertcat(_s, _s_tp)

        elif penalty.integrate:
            if is_unscaled:
                _x = nlp.cx()
                for i in range(nlp.X[_idx].shape[1]):
                    _x = vertcat(_x, nlp.X[_idx][:, i])
                _u = (
                    nlp.U[_idx][:, 0]
                    if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or _idx + 1 < len(nlp.U)
                    else []
                )
                _s = nlp.S[_idx]
            else:
                _x = nlp.cx()
                for i in range(nlp.X_scaled[_idx].shape[1]):
                    _x = vertcat(_x, nlp.X_scaled[_idx][:, i])
                _u = (
                    nlp.U_scaled[_idx][:, 0]
                    if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or _idx + 1 < len(nlp.U_scaled)
                    else []
                )
                _s = nlp.S_scaled[_idx]
        
        else:
            if is_unscaled:
                _x = nlp.cx()
                if (
                    penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                    or penalty.integration_rule == QuadratureRule.TRAPEZOIDAL
                ):
                    _x = vertcat(_x, nlp.X[_idx][:, 0])
                else:
                    for i in range(nlp.X[_idx].shape[1]):
                        _x = vertcat(_x, nlp.X[_idx][:, i])

                # Watch out, this is ok for all of our current built-in functions, but it is not generally ok to do that
                if (
                    _idx == nlp.ns
                    and nlp.ode_solver.is_direct_collocation
                    and nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
                    and penalty.node[0] != Node.END
                    and penalty.integration_rule != QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                ):
                    for i in range(1, nlp.X[_idx - 1].shape[1]):
                        _x = vertcat(_x, nlp.X[_idx - 1][:, i])

                _u = (
                    nlp.U[_idx][:, 0]
                    if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or _idx < len(nlp.U)
                    else []
                )
                _s = nlp.S[_idx][:, 0]
            else:
                _x = nlp.cx()
                if (
                    penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                    or penalty.integration_rule == QuadratureRule.TRAPEZOIDAL
                ):
                    _x = vertcat(_x, nlp.X_scaled[_idx][:, 0])
                else:
                    for i in range(nlp.X_scaled[_idx].shape[1]):
                        _x = vertcat(_x, nlp.X_scaled[_idx][:, i])

                # Watch out, this is ok for all of our current built-in functions, but it is not generally ok to do that
                if (
                    _idx == nlp.ns
                    and nlp.ode_solver.is_direct_collocation
                    and nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
                    and penalty.node[0] != Node.END
                    and penalty.integration_rule != QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                ):
                    for i in range(1, nlp.X_scaled[_idx - 1].shape[1]):
                        _x = vertcat(_x, nlp.X_scaled[_idx - 1][:, i])

                if sum(penalty.weighted_function[_idx].size_in(3)) == 0:
                    _u = []
                elif nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE and _idx == len(nlp.U_scaled):
                    _u = nlp.U_scaled[_idx - 1][:, 0]
                elif _idx < len(nlp.U_scaled):
                    _u = nlp.U_scaled[_idx][:, 0]
                else:
                    _u = []
                _s = nlp.S_scaled[_idx][:, 0]

        if penalty.explicit_derivative:
            if _idx < nlp.ns:
                if is_unscaled:
                    x = nlp.X[_idx + 1][:, 0]
                    if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE and _idx  == len(nlp.U):
                        u = nlp.U[_idx][:, 0]
                    elif _idx + 1 < len(nlp.U):
                        u = nlp.U[_idx + 1][:, 0]
                    else:
                        u = []
                    s = nlp.S[_idx + 1][:, 0]
                else:
                    x = nlp.X_scaled[_idx + 1][:, 0]
                    if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE and _idx == len(nlp.U_scaled):
                        u = nlp.U_scaled[_idx][:, 0]
                    elif _idx + 1 < len(nlp.U_scaled):
                        u = nlp.U_scaled[_idx + 1][:, 0]
                    else:
                        u = []
                    s = nlp.S_scaled[_idx + 1][:, 0]

                _x = vertcat(_x, x)
                _u = vertcat(_u, u)
                _s = vertcat(_s, s)
            else:
                print("coucou")

        if penalty.derivative:
            if _idx < nlp.ns:
                if is_unscaled:
                    x = nlp.X[_idx + 1][:, 0]
                    if _idx + 1 == len(nlp.U):
                        u = nlp.U[_idx][:, 0]
                    elif _idx + 1 < len(nlp.U):
                        u = nlp.U[_idx + 1][:, 0]
                    else:
                        u = []
                    s = nlp.S[_idx + 1][:, 0]
                else:
                    x = nlp.X_scaled[_idx + 1][:, 0]
                    if _idx + 1 == len(nlp.U_scaled):
                        u = nlp.U_scaled[_idx][:, 0]
                    elif _idx + 1 < len(nlp.U_scaled):
                        u = nlp.U_scaled[_idx + 1][:, 0]
                    else:
                        u = []
                    s = nlp.S_scaled[_idx + 1][:, 0]

                _x = vertcat(_x, x)
                _u = vertcat(_u, u)
                _s = vertcat(_s, s)

        if penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
            if is_unscaled:
                x = nlp.X[_idx + 1][:, 0]
                s = nlp.S[_idx + 1][:, 0]
            else:
                x = nlp.X_scaled[_idx + 1][:, 0]
                s = nlp.S_scaled[_idx + 1][:, 0]
            _x = vertcat(_x, x)
            _s = vertcat(_s, s)
            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                if is_unscaled:
                    u = (
                        nlp.U[_idx + 1][:, 0]
                        if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or _idx + 1 < len(nlp.U)
                        else []
                    )
                else:
                    u = (
                        nlp.U_scaled[_idx + 1][:, 0]
                        if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or _idx + 1 < len(nlp.U_scaled)
                        else []
                    )
                _u = vertcat(_u, u)

        elif penalty.integration_rule == QuadratureRule.TRAPEZOIDAL:
            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                if is_unscaled:
                    u = (
                        nlp.U[_idx + 1][:, 0]
                        if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or _idx + 1 < len(nlp.U)
                        else []
                    )
                else:
                    u = (
                        nlp.U_scaled[_idx + 1][:, 0]
                        if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or _idx + 1 < len(nlp.U_scaled)
                        else []
                    )
                _u = vertcat(_u, u)
        return _x, _u, _s


def get_padded_array(
    nlp, attribute: str, node_idx: int, casadi_constructor: Callable, target_length: int = None
) -> SX | MX:
    """
    Get a padded array of the correct length

    Parameters
    ----------
    nlp: NonLinearProgram
        The current phase
    attribute: str
        The attribute to get the array from such as "X", "X_scaled", "U", "U_scaled", "S", "S_scaled"
    node_idx: int
        The node index in the current phase
    target_length: int
        The target length of the array, in some cases, one side can be longer than the other one
        (e.g. when using uneven transition phase with a different of states between the two phases)
    casadi_constructor: Callable
        The casadi constructor to use that either build SX or MX

    Returns
    -------
    SX | MX
        The padded array
    """
    padded_array = getattr(nlp, attribute)[node_idx][:, 0]
    len_x = padded_array.shape[0]

    if target_length is None:
        target_length = len_x

    if target_length > len_x:
        fake_padding = casadi_constructor(target_length - len_x, 1)
        padded_array = vertcat(padded_array, fake_padding)

    return padded_array


def get_control_modificator(ocp, _penalty, index: int):
    current_phase = ocp.nlp[_penalty.nodes_phase[index]]
    current_node = _penalty.nodes[index]
    phase_dynamics = current_phase.phase_dynamics
    number_of_shooting_points = current_phase.ns

    is_shared_dynamics = phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
    is_end_or_shooting_point = current_node == Node.END or current_node == number_of_shooting_points

    return 1 if is_shared_dynamics and is_end_or_shooting_point else 0


def get_node_control_info(nlp, node_idx, attribute: str):
    """This returns the information about the control at a given node to format controls properly"""
    is_shared_dynamics = nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
    is_within_control_limit = node_idx < len(nlp.U_scaled)
    len_u = getattr(nlp, attribute)[0].shape[0]

    return is_shared_dynamics, is_within_control_limit, len_u


def get_padded_control_array(
    nlp,
    node_idx: int,
    u_mode: int,
    attribute: str,
    target_length: int,
    is_within_control_limit_target: bool,
    is_shared_dynamics_target: bool,
    casadi_constructor: Callable,
):
    """
    Get a padded array of the correct length

    Parameters
    ----------
    nlp: NonLinearProgram
        The current phase
    node_idx: int
        The node index in the current phase
    u_mode: int
        The control mode see get_control_modificator
    attribute: str
        The attribute to get the array from such as "X", "X_scaled", "U", "U_scaled", "S", "S_scaled"
    target_length: int
        The target length of the array, in some cases, one side can be longer than the other one
        (e.g. when using uneven transition phase with a different of states between the two phases)
    is_within_control_limit_target: bool
        If the target node of a given phase is within the control limit
        (e.g. when using uneven transition phase with a different of states between the two phases)
    is_shared_dynamics_target: bool
        If the target node of a given phase is shared during the phase
        (e.g. when using uneven transition phase with a different of states between the two phases)
    casadi_constructor: Callable
        The casadi constructor to use that either build SX or MX

    Returns
    -------
    SX | MX
        The padded array
    """

    is_shared_dynamics, is_within_control_limit, len_u = get_node_control_info(nlp, node_idx, attribute=attribute)

    _u_sym = []

    if is_shared_dynamics or is_within_control_limit:
        should_apply_fake_padding_on_u_sym = target_length > len_u and (
            is_within_control_limit_target or is_shared_dynamics_target
        )
        _u_sym = getattr(nlp, attribute)[node_idx - u_mode]

        if should_apply_fake_padding_on_u_sym:
            fake_padding = casadi_constructor(target_length - len_u, 1)
            _u_sym = vertcat(_u_sym, fake_padding)

    return _u_sym

def format_target(penalty, target_in: np.ndarray, idx: int) -> np.ndarray:
    """
    Format the target of a penalty to a numpy array

    Parameters
    ----------
    penalty:
        The penalty with a target
    target_in: np.ndarray
        The target of the penalty
    idx: int
        The index of the node
    Returns
    -------
        np.ndarray
            The target of the penalty formatted to a numpy ndarray
    """
    if len(target_in.shape) not in [2, 3]:
        raise NotImplementedError("penalty target with dimension != 2 or 3 is not implemented yet")

    target_out = target_in[..., penalty.node_idx.index(idx)]

    return target_out


# TO KEEP!!!!


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
