from time import perf_counter

from casadi import Importer, Function
from casadi import horzcat, vertcat, sum1, sum2, nlpsol, SX, MX, DM, reshape
import numpy as np

from .solver_interface import SolverInterface
from ..gui.online_callback_multiprocess import OnlineCallbackMultiprocess
from ..gui.online_callback_multiprocess_server import OnlineCallbackMultiprocessServer
from ..gui.online_callback_server import OnlineCallbackServer
from ..limits.path_conditions import Bounds
from ..limits.penalty_helpers import PenaltyHelpers, Slicy
from ..misc.enums import InterpolationType, OnlineOptim
from ..optimization.non_linear_program import NonLinearProgram
from ..optimization.solution.solution import Solution


from ..misc.parameters_types import (
    AnyDictOptional,
    Bool,
    AnyDict,
    CX,
    DoubleNpArrayTuple,
    Int,
    Range,
)


def generic_online_optim(interface: SolverInterface, ocp, show_options: AnyDictOptional = None):
    """
    Declare the online callback to update the graphs while optimizing

    Parameters
    ----------
    interface: SolverInterface
        A reference to the current interface
    ocp: OptimalControlProgram
        A reference to the current OptimalControlProgram
    show_options: dict
        The options to pass to PlotOcp, if online_optim is OnlineOptim.SERVER or OnlineOptim.MULTIPROCESS_SERVER there are
        additional options:
            - host: The host to connect to (only for OnlineOptim.SERVER)
            - port: The port to connect to (only for OnlineOptim.SERVER)
    """
    if show_options is None:
        show_options = {}

    online_optim = interface.opts.online_optim.get_default()
    if online_optim == OnlineOptim.MULTIPROCESS:
        to_call = OnlineCallbackMultiprocess
    elif online_optim == OnlineOptim.SERVER:
        to_call = OnlineCallbackServer
    elif online_optim == OnlineOptim.MULTIPROCESS_SERVER:
        to_call = OnlineCallbackMultiprocessServer
    else:
        raise ValueError(f"online_optim {online_optim} is not implemented yet")

    interface.options_common["iteration_callback"] = to_call(ocp, **show_options)


def _vectors_are_equal(
    vector1: CX,
    vector2: CX,
    v: CX,
    v_init: DM,
    v_min: DM,
    v_max: DM,
) -> Bool:
    if vector1 is None or vector2 is None:
        return False
    if vector1.size1() != vector2.size1() or vector1.size2() != vector2.size2():
        return False

    # We test the equality at three points (min, max, init) hoping any differences will be caught
    func = Function("equality", [v], [vector1 - vector2])
    return np.sum(func(v_init)) == 0.0 and np.sum(func(v_min)) == 0.0 and np.sum(func(v_max)) == 0.0


def generic_solve(interface: SolverInterface, expand_during_shake_tree: Bool = False) -> AnyDict:
    """
    Solve the prepared ocp

    Parameters
    ----------
    interface: GenericInterface
        A reference to the current interface
    expand_during_shake_tree: bool
        If the tree should be expanded during the shake tree

    Returns
    -------
    A reference to the solution
    """

    v = interface.ocp.variables_vector
    v_bounds = interface.ocp.bounds_vectors
    v_init = interface.ocp.init_vector

    # Shake the tree if needed for objectives
    all_objectives_tp = interface.dispatch_obj_func()
    can_skip_shake = _vectors_are_equal(
        interface.pre_shake_tree_objectives, all_objectives_tp, v=v, v_init=v_init, v_min=v_bounds[0], v_max=v_bounds[1]
    )
    if can_skip_shake:
        all_objectives = interface.shaked_objectives
    else:
        all_objectives = _shake_penalties_tree(interface.ocp, all_objectives_tp, v, v_bounds, expand_during_shake_tree)
    interface.pre_shake_tree_objectives = all_objectives_tp
    interface.shaked_objectives = all_objectives

    # Shake the tree if needed for constraints
    all_g_tp, all_g_bounds = interface.dispatch_bounds()
    can_skip_shake = _vectors_are_equal(
        interface.pre_shake_tree_constraints, all_g_tp, v=v, v_init=v_init, v_min=v_bounds[0], v_max=v_bounds[1]
    )
    if can_skip_shake:
        all_g = interface.shaked_constraints
    else:
        all_g = _shake_penalties_tree(interface.ocp, all_g_tp, v, v_bounds, expand_during_shake_tree)
    interface.pre_shake_tree_constraints = all_g_tp
    interface.shaked_constraints = all_g

    # Set online_optim and show_online_optim options
    if interface.opts.show_online_optim is not None:
        if interface.opts.online_optim is not None:
            raise ValueError("show_online_optim and online_optim cannot be simultaneous set")
        interface.opts.online_optim = OnlineOptim.DEFAULT if interface.opts.show_online_optim else None

    if interface.opts.online_optim is not None:
        interface.online_optim(interface.ocp, interface.opts.show_options)

    # Thread here on (f and all_g) instead of individually for each function?
    interface.nlp = {"x": v, "f": sum1(all_objectives), "g": all_g}
    interface.c_compile = interface.opts.c_compile
    options = interface.opts.as_dict(interface)

    if interface.c_compile:
        if not interface.ocp_solver or interface.ocp.program_changed:
            nlpsol("nlpsol", interface.solver_name.lower(), interface.nlp, options).generate_dependencies("nlp.c")
            interface.ocp_solver = nlpsol("nlpsol", interface.solver_name, Importer("nlp.c", "shell"), options)
            interface.ocp.program_changed = False
    else:
        interface.ocp_solver = nlpsol("solver", interface.solver_name.lower(), interface.nlp, options)

    interface.limits = {
        "lbx": v_bounds[0],
        "ubx": v_bounds[1],
        "lbg": all_g_bounds.min,
        "ubg": all_g_bounds.max,
        "x0": v_init,
    }

    if interface.lam_g is not None:
        interface.limits["lam_g0"] = interface.lam_g
    if interface.lam_x is not None:
        interface.limits["lam_x0"] = interface.lam_x

    # Solve the problem
    tic = perf_counter()
    interface.out = {"sol": interface.ocp_solver.call(interface.limits)}
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

    # Make sure the graphs are showing the last iteration
    if "iteration_callback" in interface.options_common:
        to_eval = [
            interface.out["sol"]["x"],
            interface.out["sol"]["f"],
            interface.out["sol"]["g"],
            interface.out["sol"]["lam_x"],
            interface.out["sol"]["lam_g"],
            interface.out["sol"]["lam_p"],
        ]
        interface.options_common["iteration_callback"].eval(to_eval, enforce=True)
    return interface.out


def _shake_penalties_tree(ocp, penalties_cx: CX, v: CX, v_bounds: DoubleNpArrayTuple, expand: Bool):
    """
    Remove the dt in the objectives and constraints if they are constant

    Note: Shake tree is a metaphor for saying it makes every unnecessary variables disappear / fall

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the current OptimalControlProgram
    penalties_cx
        all the penalties of the ocp (objective, constraints)
    v
        full vector of variables of the ocp
    v_bounds
        all the bounds of the variables, use to detect constant variables if min == max
    expand : bool
        expand if possible the penalty but can failed but ignored if there is matrix inversion or newton descent for example

    Returns
    -------
    The penalties without extra variables

    """
    dt = []
    for i in range(ocp.n_phases):
        # If min == max, then it's a constant
        if v_bounds[0][i] == v_bounds[1][i]:
            dt.append(v_bounds[0][i])
        else:
            dt.append(v[i])

    # Shake the tree
    penalty = Function("penalty", [v], [penalties_cx])
    if expand:
        try:
            penalty = penalty.expand()
        except RuntimeError:
            # This happens mostly when, for instance, there is a Newton decent in the penalty
            pass
    return penalty(vertcat(*dt, v[len(dt) :]))


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


def generic_dispatch_bounds(interface, include_g: Bool, include_g_internal: Bool):
    """
    Parse the bounds of the full ocp to a SQP-friendly one

    Parameters
    ----------
    interface:
        A reference to the current interface
    include_g: bool
        If the g bounds should be included
    include_g_internal: bool
        If the g_internal bounds should be included
    """

    all_g = interface.ocp.cx()
    all_g_bounds = Bounds("all_g", interpolation=InterpolationType.CONSTANT)

    if include_g_internal:
        all_g = vertcat(all_g, interface.get_all_penalties(interface.ocp, interface.ocp.g_internal))
        for g in interface.ocp.g_internal:
            if g != []:
                all_g_bounds.concatenate(g.bounds)

    if include_g:
        all_g = vertcat(all_g, interface.get_all_penalties(interface.ocp, interface.ocp.g))
        for g in interface.ocp.g:
            if g != []:
                all_g_bounds.concatenate(g.bounds)

    for nlp in interface.ocp.nlp:
        if include_g_internal:
            all_g = vertcat(all_g, interface.get_all_penalties(nlp, nlp.g_internal))
            for g in nlp.g_internal:
                if g != []:
                    for _ in g.node_idx:
                        all_g_bounds.concatenate(g.bounds)

        if include_g:
            all_g = vertcat(all_g, interface.get_all_penalties(nlp, nlp.g))
            for g in nlp.g:
                if g != []:
                    for _ in g.node_idx:
                        all_g_bounds.concatenate(g.bounds)

    if isinstance(all_g_bounds.min, (SX, MX)) or isinstance(all_g_bounds.max, (SX, MX)):
        raise RuntimeError(f"{interface.solver_name} doesn't support SX/MX types in constraints bounds")
    return all_g, all_g_bounds


def generic_dispatch_obj_func(interface) -> CX:
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


def generic_get_all_penalties(interface, nlp: NonLinearProgram, penalties, scaled: Bool = True):
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
    scaled: bool
        If the penalty should be the scaled [True] or unscaled [False]
    Returns
    -------
    out: CX
        A vector of all penalty values
    """

    ocp = interface.ocp

    out = interface.ocp.cx()
    for penalty in penalties:
        if not penalty:
            continue

        phases_dt = PenaltyHelpers.phases_dt(penalty, interface.ocp, lambda _: interface.ocp.dt_parameter.cx)

        if penalty.multi_thread:
            if penalty.target is not None and len(penalty.target.shape) != 2:
                raise NotImplementedError("multi_thread penalty with target shape != [n x m] is not implemented yet")

            t0 = nlp.cx()
            x = nlp.cx()
            u = nlp.cx()
            a = nlp.cx()
            d = None
            weight = DM()
            target = DM()
            for idx in range(len(penalty.node_idx)):
                t0_tp, x_tp, u_tp, p, a_tp, d_tp, weight_tp, target_tp = _get_weighted_function_inputs(
                    penalty, idx, ocp, nlp, scaled
                )

                t0 = horzcat(t0, t0_tp)
                if idx != 0 and x_tp.shape[0] != x.shape[0]:
                    tp = ocp.cx.nan(x.shape[0], 1)
                    tp[: x_tp.shape[0], :] = x_tp
                    x_tp = tp
                x = horzcat(x, x_tp)
                if idx != 0 and u_tp.shape[0] != u.shape[0]:
                    tp = ocp.cx.nan(u.shape[0], 1)
                    tp[: u_tp.shape[0], :] = u_tp
                    u_tp = tp
                u = horzcat(u, u_tp)
                if idx != 0 and a_tp.shape[0] != a.shape[0]:
                    tp = ocp.cx.nan(a.shape[0], 1)
                    tp[: a_tp.shape[0], :] = a_tp
                    a_tp = tp
                a = horzcat(a, a_tp)
                d = horzcat(d, d_tp) if d is not None else d_tp
                weight = horzcat(weight, weight_tp)
                target = horzcat(target, target_tp)

            # We can call penalty.weighted_function[0] since multi-thread declares all the node at [0]
            tp = reshape(penalty.weighted_function[0](t0, phases_dt, x, u, p, a, d, weight, target), -1, 1)
        else:
            tp = interface.ocp.cx()
            for idx in range(len(penalty.node_idx)):
                if nlp:
                    nlp.states.node_index = penalty.node_idx[idx]
                    nlp.controls.node_index = penalty.node_idx[idx]
                    nlp.algebraic_states.node_index = penalty.node_idx[idx]
                t0, x, u, p, a, d, weight, target = _get_weighted_function_inputs(penalty, idx, ocp, nlp, scaled)

                node_idx = penalty.node_idx[idx]
                tp = vertcat(tp, penalty.weighted_function[node_idx](t0, phases_dt, x, u, p, a, d, weight, target))

        out = vertcat(out, sum2(tp))
    return out


def _get_weighted_function_inputs(penalty, penalty_idx: Int, ocp, nlp: NonLinearProgram, scaled: Bool):
    t0 = PenaltyHelpers.t0(penalty, penalty_idx, lambda p_idx, n_idx: ocp.node_time(phase_idx=p_idx, node_idx=n_idx))

    weight = PenaltyHelpers.weight(penalty, penalty_idx)
    target = PenaltyHelpers.target(penalty, penalty_idx)

    if nlp:
        x = PenaltyHelpers.states(
            penalty,
            penalty_idx,
            lambda p_idx, n_idx, sn_idx: _get_x(ocp, p_idx, n_idx, sn_idx, scaled, penalty),
        )
        u = PenaltyHelpers.controls(
            penalty,
            penalty_idx,
            lambda p_idx, n_idx, sn_idx: _get_u(ocp, p_idx, n_idx, sn_idx, scaled, penalty),
        )
        p = PenaltyHelpers.parameters(
            penalty, penalty_idx, lambda p_idx, n_idx, sn_idx: _get_p(ocp, p_idx, n_idx, sn_idx, scaled)
        )
        a = PenaltyHelpers.states(
            penalty,
            penalty_idx,
            lambda p_idx, n_idx, sn_idx: _get_a(ocp, p_idx, n_idx, sn_idx, scaled, penalty),
        )
        d = PenaltyHelpers.numerical_timeseries(
            penalty,
            penalty_idx,
            lambda p_idx, n_idx, sn_idx: get_numerical_timeseries(ocp, p_idx, n_idx, sn_idx),
        )
    else:
        x = []
        u = []
        p = PenaltyHelpers.parameters(
            penalty, penalty_idx, lambda p_idx, n_idx, sn_idx: _get_p(ocp, p_idx, n_idx, sn_idx, scaled)
        )
        a = []
        d = []

    return t0, x, u, p, a, d, weight, target


def _get_x(ocp, phase_idx: Int, node_idx: Int, subnodes_idx: Slicy, scaled: Bool, penalty):
    values = ocp.nlp[phase_idx].X_scaled if scaled else ocp.nlp[phase_idx].X
    x = PenaltyHelpers.get_states(ocp, penalty, phase_idx, node_idx, subnodes_idx, values)
    return x


def _get_u(ocp, phase_idx: Int, node_idx: Int, subnodes_idx: Slicy, scaled: Bool, penalty):
    values = ocp.nlp[phase_idx].U_scaled if scaled else ocp.nlp[phase_idx].U
    u = PenaltyHelpers.get_controls(ocp, penalty, phase_idx, node_idx, subnodes_idx, values)
    return u


def _get_p(ocp, phase_idx: Int, node_idx: Int, subnodes_idx: Slicy, scaled: Bool):
    return ocp.parameters.scaled.cx if scaled else ocp.parameters.scaled


def _get_a(ocp, phase_idx: Int, node_idx: Int, subnodes_idx: Slicy, scaled: Bool, penalty):
    values = ocp.nlp[phase_idx].A_scaled if scaled else ocp.nlp[phase_idx].A
    a = PenaltyHelpers.get_states(ocp, penalty, phase_idx, node_idx, subnodes_idx, values)
    return a


def get_numerical_timeseries(ocp, phase_idx: Int, node_idx: Int, subnodes_idx: Slicy):
    timeseries = ocp.nlp[phase_idx].numerical_data_timeseries
    if timeseries is None:
        return ocp.cx()
    else:
        values = None
        for i_d, (key, array) in enumerate(timeseries.items()):
            for i_element in range(array.shape[1]):
                if values is None:
                    values = array[:, i_element, node_idx]
                else:
                    values = vertcat(values, array[:, i_element, node_idx])
        return values
