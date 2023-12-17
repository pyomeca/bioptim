from time import perf_counter
from typing import Callable
from sys import platform

from casadi import Importer, Function
import numpy as np
from casadi import horzcat, vertcat, sum1, sum2, nlpsol, SX, MX, reshape

from ..gui.plot import OnlineCallback
from ..limits.path_conditions import Bounds
from ..limits.penalty_helpers import PenaltyHelpers
from ..misc.enums import InterpolationType, ControlType, Node, QuadratureRule, PhaseDynamics
from bioptim.optimization.solution.solution import Solution
from ..optimization.non_linear_program import NonLinearProgram
from ..dynamics.ode_solver import OdeSolver


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


def generic_solve(interface, expand_during_shake_tree=False) -> dict:
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

    all_objectives = interface.dispatch_obj_func()
    all_objectives = _shake_tree_for_penalties(interface.ocp, all_objectives, v, v_bounds, expand_during_shake_tree)

    all_g, all_g_bounds = interface.dispatch_bounds()
    all_g = _shake_tree_for_penalties(interface.ocp, all_g, v, v_bounds, expand_during_shake_tree)

    if interface.opts.show_online_optim:
        interface.online_optim(interface.ocp, interface.opts.show_options)

    # Thread here on (f and all_g) instead of individually for each function?
    interface.sqp_nlp = {"x": v, "f": sum1(all_objectives), "g": all_g}
    interface.c_compile = interface.opts.c_compile
    options = interface.opts.as_dict(interface)

    if interface.c_compile:
        if not interface.ocp_solver or interface.ocp.program_changed:
            nlpsol("nlpsol", interface.solver_name.lower(), interface.sqp_nlp, options).generate_dependencies("nlp.c")
            interface.ocp_solver = nlpsol("nlpsol", interface.solver_name, Importer("nlp.c", "shell"), options)
            interface.ocp.program_changed = False
    else:
        interface.ocp_solver = nlpsol("solver", interface.solver_name.lower(), interface.sqp_nlp, options)

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


def _shake_tree_for_penalties(ocp, penalties_cx, v, v_bounds, expand):
    """
    Remove the dt in the objectives and constraints if they are constant

    Parameters
    ----------
    ocp
    penalties_cx
    v
    v_bounds

    Returns
    -------

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
            # This happens mostly when there is a Newton decent in the penalty
            pass
    return penalty(vertcat(*dt, v[len(dt):]))


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


def generic_get_all_penalties(interface, nlp: NonLinearProgram, penalties, scaled=True):
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
    TODO
    """

    ocp = interface.ocp

    out = interface.ocp.cx()
    for penalty in penalties:
        if not penalty:
            continue

        phases_dt = PenaltyHelpers.phases_dt(penalty, interface.ocp, lambda _: interface.ocp.dt_parameter.cx)
        p = PenaltyHelpers.parameters(penalty, lambda: interface.ocp.parameters.cx)

        if penalty.multi_thread:
            if penalty.target is not None and len(penalty.target.shape) != 2:
                raise NotImplementedError("multi_thread penalty with target shape != [n x m] is not implemented yet")

            t0 = nlp.cx()
            x = nlp.cx()
            u = nlp.cx()
            s = nlp.cx()
            weight = np.ndarray((0,))
            target = nlp.cx()
            for idx in range(len(penalty.node_idx)):
                t0_tp, x_tp, u_tp, s_tp, weight_tp, target_tp = _get_weighted_function_inputs(penalty, idx, ocp, nlp, scaled)
                
                t0 = horzcat(t0, t0_tp)
                x = horzcat(x, x_tp)
                if idx != 0 and u_tp.shape[0] != u.shape[0]: 
                    tp = u.zeros(u.shape[0], 1)
                    tp[:u_tp.shape[0], :] = u_tp
                    u_tp = tp
                u = horzcat(u, u_tp)
                s = horzcat(s, s_tp)
                weight = np.concatenate((weight, [weight_tp]))
                target = horzcat(target, target_tp)

            # We can call penalty.weighted_function[0] since multi-thread declares all the node at [0]
            tp = reshape(penalty.weighted_function[0](t0, phases_dt, x, u, p, s, penalty.weight, target), -1, 1)

        else:
            tp = interface.ocp.cx()
            for idx in range(len(penalty.node_idx)):
                nlp.states.node_index = penalty.node_idx[idx]
                nlp.controls.node_index = penalty.node_idx[idx]
                nlp.parameters.node_index = penalty.node_idx[idx]
                nlp.stochastic_variables.node_index = penalty.node_idx[idx]

                t0, x, u, s, weight, target = _get_weighted_function_inputs(penalty, idx, ocp, nlp, scaled)

                node_idx = penalty.node_idx[idx]
                tp = vertcat(
                    tp, penalty.weighted_function[node_idx](t0, phases_dt, x, u, p, s, weight, target)
                )

        out = vertcat(out, sum2(tp))
    return out


def _get_weighted_function_inputs(penalty, penalty_idx, ocp, nlp, scaled):
    t0 = PenaltyHelpers.t0()

    weight = PenaltyHelpers.weight(penalty)
    target = PenaltyHelpers.target(penalty, penalty_idx)

    if nlp:
        x = PenaltyHelpers.states(penalty, penalty_idx, lambda p_idx, n_idx, sn_idx: _get_x(ocp, p_idx, n_idx, sn_idx, scaled))
        u = PenaltyHelpers.controls(penalty, penalty_idx, lambda p_idx, n_idx, sn_idx: _get_u(ocp, p_idx, n_idx, sn_idx, scaled))
        s = PenaltyHelpers.states(penalty, penalty_idx, lambda p_idx, n_idx, sn_idx: _get_s(ocp, p_idx, n_idx, sn_idx, scaled))
    else:
        x = []
        u = []
        s = []

    return t0, x, u, s, weight, target,


def _get_x(ocp, phase_idx, node_idx, subnodes_idx, scaled):
    values = ocp.nlp[phase_idx].X_scaled if scaled else ocp.nlp[phase_idx].X
    return values[node_idx][:, subnodes_idx] if node_idx < len(values) else ocp.cx()
    

def _get_u(ocp, phase_idx, node_idx, subnodes_idx, scaled):
    values = ocp.nlp[phase_idx].U_scaled if scaled else ocp.nlp[phase_idx].U
    return values[node_idx][:, subnodes_idx] if node_idx < len(values) else ocp.cx()


def _get_s(ocp, phase_idx, node_idx, subnodes_idx, scaled):
    values = ocp.nlp[phase_idx].S_scaled if scaled else ocp.nlp[phase_idx].S
    return values[node_idx][:, subnodes_idx] if node_idx < len(values) else ocp.cx()
