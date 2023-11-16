from time import perf_counter
from typing import Callable
from sys import platform

from casadi import Importer
import numpy as np
from casadi import horzcat, vertcat, sum1, sum2, nlpsol, SX, MX, reshape

from ..gui.plot import OnlineCallback
from ..limits.path_conditions import Bounds
from ..misc.enums import InterpolationType, ControlType, Node, QuadratureRule, PhaseDynamics
from bioptim.optimization.solution.solution import Solution
from ..optimization.non_linear_program import NonLinearProgram


def generic_online_optim(interface, ocp, show_options: dict = None):
    """
    Declare the online callback to update the graphs while optimizing

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the current OptimalControlProgram
    show_options: dict
        The options to pass to PlotOcp
    """

    if platform != "linux":
        raise RuntimeError("Online graphics are only available on Linux")
    interface.options_common["iteration_callback"] = OnlineCallback(ocp, show_options=show_options)


def generic_solve(interface) -> dict:
    """
    Solve the prepared ocp

    Returns
    -------
    A reference to the solution
    """

    all_objectives = interface.dispatch_obj_func()
    all_g, all_g_bounds = interface.dispatch_bounds()

    if interface.opts.show_online_optim:
        interface.online_optim(interface.ocp, interface.opts.show_options)

    # Thread here on (f and all_g) instead of individually for each function?
    interface.sqp_nlp = {"x": interface.ocp.variables_vector, "f": sum1(all_objectives), "g": all_g}
    interface.c_compile = interface.opts.c_compile
    options = interface.opts.as_dict(interface)

    if interface.c_compile:
        if not interface.ocp_solver or interface.ocp.program_changed:
            nlpsol("nlpsol", interface.solver_name.lower(), interface.sqp_nlp, options).generate_dependencies("nlp.c")
            interface.ocp_solver = nlpsol("nlpsol", interface.solver_name, Importer("nlp.c", "shell"), options)
            interface.ocp.program_changed = False
    else:
        interface.ocp_solver = nlpsol("solver", interface.solver_name.lower(), interface.sqp_nlp, options)

    v_bounds = interface.ocp.bounds_vectors
    v_init = interface.ocp.init_vector
    interface.sqp_limits = {
        "lbx": v_bounds[0],
        "ubx": v_bounds[1],
        "lbg": all_g_bounds.min,
        "ubg": all_g_bounds.max,
        "x0": v_init,
    }

    if interface.lam_g is not None:
        interface.sqp_limits["lam_g0"] = interface.lam_g
    if interface.lam_x is not None:
        interface.sqp_limits["lam_x0"] = interface.lam_x

    # Solve the problem
    tic = perf_counter()
    interface.out = {"sol": interface.ocp_solver.call(interface.sqp_limits)}
    interface.out["sol"]["solver_time_to_optimize"] = interface.ocp_solver.stats()["t_wall_total"]
    interface.out["sol"]["real_time_to_optimize"] = perf_counter() - tic
    interface.out["sol"]["iter"] = interface.ocp_solver.stats()["iter_count"]
    interface.out["sol"]["inf_du"] = (
        interface.ocp_solver.stats()["iterations"]["inf_du"] if "iteration" in interface.ocp_solver.stats() else None
    )
    interface.out["sol"]["inf_pr"] = (
        interface.ocp_solver.stats()["iterations"]["inf_pr"] if "iteration" in interface.ocp_solver.stats() else None
    )
    # To match acados convention (0 = success, 1 = error)
    interface.out["sol"]["status"] = int(not interface.ocp_solver.stats()["success"])
    interface.out["sol"]["solver"] = interface.solver_name

    return interface.out


def generic_set_lagrange_multiplier(interface, sol: Solution):
    """
    Set the lagrange multiplier from a solution structure

    Parameters
    ----------
    sol: dict
        A solution structure where the lagrange multipliers are set
    """

    interface.lam_g = sol.lam_g
    interface.lam_x = sol.lam_x
    return sol


def generic_dispatch_bounds(interface):
    """
    Parse the bounds of the full ocp to a SQP-friendly one
    """

    all_g = interface.ocp.cx()
    all_g_bounds = Bounds("all_g", interpolation=InterpolationType.CONSTANT)

    all_g = vertcat(all_g, interface.get_all_penalties(interface.ocp, interface.ocp.g_internal))
    for g in interface.ocp.g_internal:
        all_g_bounds.concatenate(g.bounds)

    all_g = vertcat(all_g, interface.get_all_penalties(interface.ocp, interface.ocp.g_implicit))
    for g in interface.ocp.g_implicit:
        all_g_bounds.concatenate(g.bounds)

    all_g = vertcat(all_g, interface.get_all_penalties(interface.ocp, interface.ocp.g))
    for g in interface.ocp.g:
        all_g_bounds.concatenate(g.bounds)

    for nlp in interface.ocp.nlp:
        all_g = vertcat(all_g, interface.get_all_penalties(nlp, nlp.g_internal))
        for g in nlp.g_internal:
            for _ in g.node_idx:
                all_g_bounds.concatenate(g.bounds)

        all_g = vertcat(all_g, interface.get_all_penalties(nlp, nlp.g_implicit))
        for g in nlp.g_implicit:
            for _ in g.node_idx:
                all_g_bounds.concatenate(g.bounds)

        all_g = vertcat(all_g, interface.get_all_penalties(nlp, nlp.g))
        for g in nlp.g:
            for _ in g.node_idx:
                all_g_bounds.concatenate(g.bounds)

    if isinstance(all_g_bounds.min, (SX, MX)) or isinstance(all_g_bounds.max, (SX, MX)):
        raise RuntimeError(f"{interface.solver_name} doesn't support SX/MX types in constraints bounds")
    return all_g, all_g_bounds


def generic_dispatch_obj_func(interface):
    """
    Parse the objective functions of the full ocp to a SQP-friendly one

    Returns
    -------
    SX | MX
        The objective function
    """

    all_objectives = interface.ocp.cx()
    all_objectives = vertcat(all_objectives, interface.get_all_penalties(interface.ocp, interface.ocp.J_internal))
    all_objectives = vertcat(all_objectives, interface.get_all_penalties([], interface.ocp.J))

    for nlp in interface.ocp.nlp:
        all_objectives = vertcat(all_objectives, interface.get_all_penalties(nlp, nlp.J_internal))
        all_objectives = vertcat(all_objectives, interface.get_all_penalties(nlp, nlp.J))

    return all_objectives


def generic_get_all_penalties(interface, nlp: NonLinearProgram, penalties, is_unscaled=False):
    """
    Parse the penalties of the full ocp to a SQP-friendly one

    Parameters
    ----------
    interface:
        A reference to the current interface
    nlp: NonLinearProgram
        The nonlinear program to parse the penalties from
    penalties:
        The penalties to parse
    is_unscaled: bool
        If the penalty is unscaled or scaled
    Returns
    -------
    TODO
    """

    param = interface.ocp.cx(interface.ocp.parameters.cx)
    out = interface.ocp.cx()
    for penalty in penalties:
        if not penalty:
            continue

        if penalty.multi_thread:
            if penalty.target is not None and len(penalty.target[0].shape) != 2:
                raise NotImplementedError("multi_thread penalty with target shape != [n x m] is not implemented yet")
            target = penalty.target[0] if penalty.target is not None else []

            x = nlp.cx()
            u = nlp.cx()
            s = nlp.cx()
            for idx in penalty.node_idx:
                x_tp, u_tp, s_tp = get_x_u_s_at_idx(interface, nlp, penalty, idx, is_unscaled)
                x = horzcat(x, x_tp)
                u = horzcat(u, u_tp)
                s = horzcat(s, s_tp)

            # We can call penalty.weighted_function[0] since multi-thread declares all the node at [0]
            time = interface.ocp.node_time(phase_idx=nlp.phase_idx, node_idx=penalty.node_idx[-1])
            p = reshape(penalty.weighted_function[0](time, x, u, param, s, penalty.weight, target, penalty.dt), -1, 1)

        else:
            p = interface.ocp.cx()
            for idx in penalty.node_idx:
                if penalty.target is None:
                    target = []
                elif (
                    penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                    or penalty.integration_rule == QuadratureRule.TRAPEZOIDAL
                ):
                    target0 = format_target(penalty, penalty.target[0], idx)
                    target1 = format_target(penalty, penalty.target[1], idx)
                    target = np.vstack((target0, target1)).T
                else:
                    target = format_target(penalty, penalty.target[0], idx)

                if np.isnan(np.sum(target)):
                    continue

                if not nlp:
                    x = []
                    u = []
                    s = []
                else:
                    x, u, s = get_x_u_s_at_idx(interface, nlp, penalty, idx, is_unscaled)
                    time = interface.ocp.node_time(phase_idx=0 if nlp == [] else nlp.phase_idx, node_idx=idx)
                    p = vertcat(
                        p, penalty.weighted_function[idx](time, x, u, param, s, penalty.weight, target, penalty.dt)
                    )

        out = vertcat(out, sum2(p))
    return out


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


def get_control_modificator(ocp, _penalty, index: int):
    current_phase = ocp.nlp[_penalty.nodes_phase[index]]
    current_node = _penalty.nodes[index]
    phase_dynamics = current_phase.phase_dynamics
    number_of_shooting_points = current_phase.ns

    is_shared_dynamics = phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE
    is_end_or_shooting_point = current_node == Node.END or current_node == number_of_shooting_points

    return 1 if is_shared_dynamics and is_end_or_shooting_point else 0


def get_x_u_s_at_idx(interface, nlp, _penalty, _idx, is_unscaled):
    """ """

    if _penalty.transition:
        ocp = interface.ocp
        cx = interface.ocp.cx

        all_nlp = interface.ocp.nlp

        phase_node0 = _penalty.nodes_phase[0]
        phase_node1 = _penalty.nodes_phase[1]

        node_idx_0 = _penalty.all_nodes_index[0]
        node_idx_1 = _penalty.all_nodes_index[1]

        u0_mode = get_control_modificator(ocp, _penalty, 0)
        u1_mode = get_control_modificator(ocp, _penalty, 1)

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

    elif _penalty.multinode_penalty:
        ocp = interface.ocp

        # Make an exception to the fact that U is not available for the last node
        _x = ocp.cx()
        _u = ocp.cx()
        _s = ocp.cx()
        for i in range(len(_penalty.nodes_phase)):
            nlp_i = ocp.nlp[_penalty.nodes_phase[i]]
            index_i = _penalty.multinode_idx[i]
            ui_mode = get_control_modificator(ocp, _penalty=_penalty, index=i)

            if is_unscaled:
                _x_tp = nlp_i.cx()
                if _penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
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
                if _penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
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

    elif _penalty.integrate:
        if is_unscaled:
            _x = nlp.cx()
            for i in range(nlp.X[_idx].shape[1]):
                _x = vertcat(_x, nlp.X[_idx][:, i])
            _u = (
                nlp.U[_idx][:, 0]
                if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or _idx < len(nlp.U)
                else []
            )
            _s = nlp.S[_idx]
        else:
            _x = nlp.cx()
            for i in range(nlp.X_scaled[_idx].shape[1]):
                _x = vertcat(_x, nlp.X_scaled[_idx][:, i])
            _u = (
                nlp.U_scaled[_idx][:, 0]
                if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE or _idx < len(nlp.U_scaled)
                else []
            )
            _s = nlp.S_scaled[_idx]
    else:
        if is_unscaled:
            _x = nlp.cx()
            if (
                _penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                or _penalty.integration_rule == QuadratureRule.TRAPEZOIDAL
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
                and _penalty.node[0] != Node.END
                and _penalty.integration_rule != QuadratureRule.APPROXIMATE_TRAPEZOIDAL
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
                _penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                or _penalty.integration_rule == QuadratureRule.TRAPEZOIDAL
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
                and _penalty.node[0] != Node.END
                and _penalty.integration_rule != QuadratureRule.APPROXIMATE_TRAPEZOIDAL
            ):
                for i in range(1, nlp.X_scaled[_idx - 1].shape[1]):
                    _x = vertcat(_x, nlp.X_scaled[_idx - 1][:, i])

            if sum(_penalty.weighted_function[_idx].size_in(1)) == 0:
                _u = []
            elif nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE and _idx == len(nlp.U_scaled):
                _u = nlp.U_scaled[_idx - 1][:, 0]
            elif _idx < len(nlp.U_scaled):
                _u = nlp.U_scaled[_idx][:, 0]
            else:
                _u = []
            _s = nlp.S_scaled[_idx][:, 0]

    if _penalty.explicit_derivative:
        if _idx < nlp.ns:
            if is_unscaled:
                x = nlp.X[_idx + 1][:, 0]
                if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE and _idx + 1 == len(nlp.U):
                    u = nlp.U[_idx][:, 0]
                elif _idx + 1 < len(nlp.U):
                    u = nlp.U[_idx + 1][:, 0]
                else:
                    u = []
                s = nlp.S[_idx + 1][:, 0]
            else:
                x = nlp.X_scaled[_idx + 1][:, 0]
                if nlp.phase_dynamics == PhaseDynamics.SHARED_DURING_THE_PHASE and _idx + 1 == len(nlp.U_scaled):
                    u = nlp.U_scaled[_idx][:, 0]
                elif _idx + 1 < len(nlp.U_scaled):
                    u = nlp.U_scaled[_idx + 1][:, 0]
                else:
                    u = []
                s = nlp.S_scaled[_idx + 1][:, 0]

            _x = vertcat(_x, x)
            _u = vertcat(_u, u)
            _s = vertcat(_s, s)

    if _penalty.derivative:
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

    if _penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL:
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

    elif _penalty.integration_rule == QuadratureRule.TRAPEZOIDAL:
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
