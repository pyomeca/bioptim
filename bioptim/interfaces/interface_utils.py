from time import perf_counter
from sys import platform

from casadi import Importer
import numpy as np
from casadi import horzcat, vertcat, sum1, sum2, nlpsol, SX, MX, reshape

from ..gui.plot import OnlineCallback
from ..limits.path_conditions import Bounds
from ..misc.enums import InterpolationType, ControlType, Node, IntegralApproximation
from ..optimization.solution import Solution
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

    if platform == "win32":
        raise RuntimeError("Online graphics are not available on Windows")
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

    interface.sqp_nlp = {"x": interface.ocp.v.vector, "f": sum1(all_objectives), "g": all_g}
    interface.c_compile = interface.opts.c_compile
    options = interface.opts.as_dict(interface)

    if interface.c_compile:
        if not interface.ocp_solver or interface.ocp.program_changed:
            nlpsol("nlpsol", interface.solver_name.lower(), interface.sqp_nlp, options).generate_dependencies("nlp.c")
            interface.ocp_solver = nlpsol("nlpsol", interface.solver_name, Importer("nlp.c", "shell"), options)
            interface.ocp.program_changed = False
    else:
        interface.ocp_solver = nlpsol("solver", interface.solver_name.lower(), interface.sqp_nlp, options)

    v_bounds = interface.ocp.v.bounds
    v_init = interface.ocp.v.init
    interface.sqp_limits = {
        "lbx": v_bounds.min,
        "ubx": v_bounds.max,
        "lbg": all_g_bounds.min,
        "ubg": all_g_bounds.max,
        "x0": v_init.init,
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
    all_g_bounds = Bounds(interpolation=InterpolationType.CONSTANT)

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
    nlp: NonLinearProgram
        The nonlinear program to parse the penalties from
    penalties:
        The penalties to parse
    Returns
    -------

    """

    def format_target(target_in: np.array) -> np.array:
        """
        Format the target of a penalty to a numpy array

        Parameters
        ----------
        target_in: np.array
            The target of the penalty
        Returns
        -------
            np.array
                The target of the penalty formatted to a numpy array
        """
        if len(target_in.shape) == 2:
            target_out = target_in[:, penalty.node_idx.index(idx)]
        elif len(target_in.shape) == 3:
            target_out = target_in[:, :, penalty.node_idx.index(idx)]
        else:
            raise NotImplementedError("penalty target with dimension != 2 or 3 is not implemented yet")
        return target_out

    def get_x_and_u_at_idx(_penalty, _idx, is_unscaled):


        """
        TODO: Change here
        """
        if _penalty.transition:
            ocp = interface.ocp
            if is_unscaled:
                x_pre = ocp.nlp[_penalty.phase_pre_idx].X[-1]
                x_post = ocp.nlp[_penalty.phase_post_idx].X[0][:, 0]
                u_pre = ocp.nlp[_penalty.phase_pre_idx].U[-1]
                u_post = ocp.nlp[_penalty.phase_post_idx].U[0]
            else:
                x_pre = ocp.nlp[_penalty.phase_pre_idx].X_scaled[-1]
                x_post = ocp.nlp[_penalty.phase_post_idx].X_scaled[0][:, 0]
                u_pre = ocp.nlp[_penalty.phase_pre_idx].U_scaled[-1]
                u_post = ocp.nlp[_penalty.phase_post_idx].U_scaled[0]

            _x = vertcat(x_pre, x_post)
            _u = vertcat(u_pre, u_post)
        elif _penalty.binode_constraint:
            ocp = interface.ocp
            # Make an exception to the fact that U is not available for the last node
            mod_u0 = 1 if _penalty.first_node == Node.END else 0
            mod_u1 = 1 if _penalty.second_node == Node.END else 0

            if is_unscaled:
                x_first = ocp.nlp[_penalty.phase_first_idx].X[_penalty.node_idx[0]]
                x_second = ocp.nlp[_penalty.phase_second_idx].X[_penalty.node_idx[1]][:, 0]
                u_first = ocp.nlp[_penalty.phase_first_idx].U[_penalty.node_idx[0] - mod_u0]
                u_second = ocp.nlp[_penalty.phase_second_idx].U[_penalty.node_idx[1] - mod_u1]
            else:
                x_first = ocp.nlp[_penalty.phase_first_idx].X_scaled[_penalty.node_idx[0]]
                x_second = ocp.nlp[_penalty.phase_second_idx].X_scaled[_penalty.node_idx[1]][:, 0]
                u_first = ocp.nlp[_penalty.phase_first_idx].U_scaled[_penalty.node_idx[0] - mod_u0]
                u_second = ocp.nlp[_penalty.phase_second_idx].U_scaled[_penalty.node_idx[1] - mod_u1]

            _x = vertcat(x_first, x_second)
            _u = vertcat(u_first, u_second)
        elif _penalty.integrate:
            if is_unscaled:
                _x = nlp.X[_idx]
                _u = nlp.U[_idx][:, 0] if _idx < len(nlp.U) else []
            else:
                _x = nlp.X_scaled[_idx]
                _u = nlp.U_scaled[_idx][:, 0] if _idx < len(nlp.U_scaled) else []
        else:
            if is_unscaled:
                _x = nlp.X[_idx][:, 0]
                _u = nlp.U[_idx][:, 0] if _idx < len(nlp.U) else []
            else:
                _x = nlp.X_scaled[_idx][:, 0]
                _u = nlp.U_scaled[_idx][:, 0] if _idx < len(nlp.U_scaled) else []

        if _penalty.derivative or _penalty.explicit_derivative:
            if is_unscaled:
                x = nlp.X[_idx + 1][:, 0]
                u = nlp.U[_idx + 1][:, 0] if _idx + 1 < len(nlp.U) else []
            else:
                x = nlp.X_scaled[_idx + 1][:, 0]
                u = nlp.U_scaled[_idx + 1][:, 0] if _idx + 1 < len(nlp.U_scaled) else []

            _x = horzcat(_x, x)
            _u = horzcat(_u, u)

        if _penalty.integration_rule == IntegralApproximation.TRAPEZOIDAL:
            if is_unscaled:
                x = nlp.X[_idx + 1][:, 0]
            else:
                x = nlp.X_scaled[_idx + 1][:, 0]
            _x = horzcat(_x, x)
            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                if is_unscaled:
                    u = nlp.U[_idx + 1][:, 0] if _idx + 1 < len(nlp.U) else []
                else:
                    u = nlp.U_scaled[_idx + 1][:, 0] if _idx + 1 < len(nlp.U_scaled) else []
                _u = horzcat(_u, u)

        if _penalty.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL:
            if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                if is_unscaled:
                    u = nlp.U[_idx + 1][:, 0] if _idx + 1 < len(nlp.U) else []
                else:
                    u = nlp.U_scaled[_idx + 1][:, 0] if _idx + 1 < len(nlp.U_scaled) else []
                _u = horzcat(_u, u)
        return _x, _u

    param = interface.ocp.cx(interface.ocp.v.parameters_in_list.cx_start)
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
            for idx in penalty.node_idx:
                x_tp, u_tp = get_x_and_u_at_idx(penalty, idx, is_unscaled)
                x = horzcat(x, x_tp)
                u = horzcat(u, u_tp)
            if (
                penalty.derivative or penalty.explicit_derivative or penalty.node[0] == Node.ALL
            ) and nlp.control_type == ControlType.CONSTANT:
                u = horzcat(u, u[:, -1])

            p = reshape(penalty.weighted_function(x, u, param, penalty.weight, target, penalty.dt), -1, 1)

        else:
            p = interface.ocp.cx()
            for idx in penalty.node_idx:
                if penalty.target is None:
                    target = []
                elif (
                    penalty.integration_rule == IntegralApproximation.TRAPEZOIDAL
                    or penalty.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
                ):
                    target0 = format_target(penalty.target[0])
                    target1 = format_target(penalty.target[1])
                    target = np.vstack((target0, target1)).T
                else:
                    target = format_target(penalty.target[0])

                if np.isnan(np.sum(target)):
                    continue

                if not nlp:
                    x = []
                    u = []
                else:
                    x, u = get_x_and_u_at_idx(penalty, idx, is_unscaled)
                p = vertcat(p, penalty.weighted_function(x, u, param, penalty.weight, target, penalty.dt))
        out = vertcat(out, sum2(p))
    return out
