from time import perf_counter
from sys import platform

from casadi import Importer
import numpy as np
from casadi import horzcat, vertcat, sum1, sum2, nlpsol, SX, MX, reshape

from ..gui.plot import OnlineCallback
from ..limits.path_conditions import Bounds
from ..misc.enums import InterpolationType, ControlType, Node, QuadratureRule
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
        """ """
        def get_control_modificator(index):
            return (
                1
                if ocp.assume_phase_dynamics
                and (
                    _penalty.nodes[index] == Node.END
                    or _penalty.nodes[index] == ocp.nlp[_penalty.nodes_phase[index]].ns
                )
                else 0
            )

        if _penalty.transition:
            ocp = interface.ocp

            u0_mode = get_control_modificator(0)
            u1_mode = get_control_modificator(1)

            if is_unscaled:
                if interface.ocp.nlp[_penalty.nodes_phase[1]].X[_penalty.all_nodes_index[1]][:, 0].shape[0] > interface.ocp.nlp[_penalty.nodes_phase[0]].X[_penalty.all_nodes_index[0]][:, 0].shape[0]:
                    fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[1]].X[_penalty.all_nodes_index[1]][:, 0].shape[0] - interface.ocp.nlp[_penalty.nodes_phase[0]].X[_penalty.all_nodes_index[0]][:, 0].shape[0], 1)
                    _x_0 = vertcat(
                        interface.ocp.nlp[_penalty.nodes_phase[0]].X[_penalty.all_nodes_index[0]][:, 0],
                        fake,
                    )
                else:
                    _x_0 = interface.ocp.nlp[_penalty.nodes_phase[0]].X[_penalty.all_nodes_index[0]][:, 0]
                if interface.ocp.nlp[_penalty.nodes_phase[0]].X[_penalty.all_nodes_index[0]][:, 0].shape[0] > interface.ocp.nlp[_penalty.nodes_phase[1]].X[_penalty.all_nodes_index[1]][:, 0].shape[0]:
                    fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[0]].X[
                                                            _penalty.all_nodes_index[0]][:, 0].shape[0] -
                                                interface.ocp.nlp[_penalty.nodes_phase[1]].X[
                                                    _penalty.all_nodes_index[1]][:, 0].shape[0], 1)
                    _x_1 = vertcat(
                        interface.ocp.nlp[_penalty.nodes_phase[1]].X[_penalty.all_nodes_index[1]][:, 0],
                        fake,
                    )
                else:
                    _x_1 = interface.ocp.nlp[_penalty.nodes_phase[1]].X[_penalty.all_nodes_index[1]][:, 0]
                _x = vertcat(_x_1, _x_0)

                if ocp.assume_phase_dynamics or _penalty.all_nodes_index[0] < len(interface.ocp.nlp[_penalty.nodes_phase[0]].U):
                    if interface.ocp.nlp[_penalty.nodes_phase[1]].U[_penalty.all_nodes_index[1] - u1_mode].shape[0] > interface.ocp.nlp[_penalty.nodes_phase[0]].U[_penalty.all_nodes_index[0] - u0_mode].shape[0]:
                        fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[1]].U[_penalty.all_nodes_index[1] - u1_mode].shape[0] - interface.ocp.nlp[_penalty.nodes_phase[0]].U[_penalty.all_nodes_index[0] - u0_mode].shape[0], 1)
                        _u_0 = vertcat(
                            interface.ocp.nlp[_penalty.nodes_phase[0]].U[_penalty.all_nodes_index[0] - u0_mode],
                            fake,
                        )
                    else:
                        _u_0 = interface.ocp.nlp[_penalty.nodes_phase[0]].U[_penalty.all_nodes_index[0] - u0_mode]
                else:
                    _u_0 = []
                if ocp.assume_phase_dynamics or _penalty.all_nodes_index[1] < len(interface.ocp.nlp[_penalty.nodes_phase[1]].U):
                    if interface.ocp.nlp[_penalty.nodes_phase[0]].U[_penalty.all_nodes_index[0] - u0_mode].shape[0] > interface.ocp.nlp[_penalty.nodes_phase[1]].U[_penalty.all_nodes_index[1] - u1_mode].shape[0]:
                        fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[0]].U[_penalty.all_nodes_index[0] - u0_mode].shape[0] - interface.ocp.nlp[_penalty.nodes_phase[1]].U[_penalty.all_nodes_index[1] - u1_mode].shape[0], 1)
                        _u_1 = vertcat(
                            interface.ocp.nlp[_penalty.nodes_phase[1]].U[_penalty.all_nodes_index[1] - u1_mode],
                            fake,
                        )
                    else:
                        _u_1 = interface.ocp.nlp[_penalty.nodes_phase[1]].U[_penalty.all_nodes_index[1] - u1_mode]
                else:
                    _u_1 = []
                _u = vertcat(_u_1, _u_0)

                if interface.ocp.nlp[_penalty.nodes_phase[1]].S[_penalty.all_nodes_index[1]][:, 0].shape[0] > interface.ocp.nlp[_penalty.nodes_phase[0]].S[_penalty.all_nodes_index[0]][:, 0].shape[0]:
                    fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[1]].S[_penalty.all_nodes_index[1]][:, 0].shape[0] - interface.ocp.nlp[_penalty.nodes_phase[0]].S[_penalty.all_nodes_index[0]][:, 0].shape[0], 1)
                    _s_0 = vertcat(
                        interface.ocp.nlp[_penalty.nodes_phase[0]].S[_penalty.all_nodes_index[0]][:, 0],
                        fake,
                    )
                else:
                    _s_0 = interface.ocp.nlp[_penalty.nodes_phase[0]].S[_penalty.all_nodes_index[0]][:, 0]
                if interface.ocp.nlp[_penalty.nodes_phase[0]].S[_penalty.all_nodes_index[0]][:, 0].shape[0] > interface.ocp.nlp[_penalty.nodes_phase[1]].S[_penalty.all_nodes_index[1]][:, 0].shape[0]:
                    fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[0]].S[
                                                            _penalty.all_nodes_index[0]][:, 0].shape[0] -
                                                interface.ocp.nlp[_penalty.nodes_phase[1]].S[
                                                    _penalty.all_nodes_index[1]][:, 0].shape[0], 1)
                    _s_1 = vertcat(
                        interface.ocp.nlp[_penalty.nodes_phase[1]].S[_penalty.all_nodes_index[1]][:, 0],
                        fake,
                    )
                else:
                    _s_1 = interface.ocp.nlp[_penalty.nodes_phase[1]].S[_penalty.all_nodes_index[1]][:, 0]
                _s = vertcat(_s_1, _s_0)

            else:
                if interface.ocp.nlp[_penalty.nodes_phase[1]].X_scaled[_penalty.all_nodes_index[1]][:, 0].shape[0] > \
                        interface.ocp.nlp[_penalty.nodes_phase[0]].X_scaled[_penalty.all_nodes_index[0]][:, 0].shape[0]:
                    fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[1]].X_scaled[
                                                            _penalty.all_nodes_index[1]][:, 0].shape[0] -
                                                interface.ocp.nlp[_penalty.nodes_phase[0]].X_scaled[
                                                    _penalty.all_nodes_index[0]][:, 0].shape[0], 1)
                    _x_0 = vertcat(
                        interface.ocp.nlp[_penalty.nodes_phase[0]].X_scaled[_penalty.all_nodes_index[0]][:, 0],
                        fake,
                    )
                else:
                    _x_0 = interface.ocp.nlp[_penalty.nodes_phase[0]].X_scaled[_penalty.all_nodes_index[0]][:, 0]
                if interface.ocp.nlp[_penalty.nodes_phase[0]].X_scaled[_penalty.all_nodes_index[0]][:, 0].shape[0] > \
                        interface.ocp.nlp[_penalty.nodes_phase[1]].X_scaled[_penalty.all_nodes_index[1]][:, 0].shape[0]:
                    fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[0]].X_scaled[
                                                            _penalty.all_nodes_index[0]][:, 0].shape[0] -
                                                interface.ocp.nlp[_penalty.nodes_phase[1]].X_scaled[
                                                    _penalty.all_nodes_index[1]][:, 0].shape[0], 1)
                    _x_1 = vertcat(
                        interface.ocp.nlp[_penalty.nodes_phase[1]].X_scaled[_penalty.all_nodes_index[1]][:, 0],
                        fake,
                    )
                else:
                    _x_1 = interface.ocp.nlp[_penalty.nodes_phase[1]].X_scaled[_penalty.all_nodes_index[1]][:, 0]
                _x = vertcat(_x_1, _x_0)

                if ocp.assume_phase_dynamics or _penalty.all_nodes_index[0] < len(
                        interface.ocp.nlp[_penalty.nodes_phase[0]].U_scaled):
                    if interface.ocp.nlp[_penalty.nodes_phase[1]].U_scaled[_penalty.all_nodes_index[1] - u1_mode].shape[0] > \
                            interface.ocp.nlp[_penalty.nodes_phase[0]].U_scaled[_penalty.all_nodes_index[0] - u0_mode].shape[0]:
                        fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[1]].U_scaled[
                            _penalty.all_nodes_index[1] - u1_mode].shape[0] - interface.ocp.nlp[_penalty.nodes_phase[0]].U_scaled[
                                                        _penalty.all_nodes_index[0] - u0_mode].shape[0], 1)
                        _u_0 = vertcat(
                            interface.ocp.nlp[_penalty.nodes_phase[0]].U_scaled[_penalty.all_nodes_index[0] - u0_mode],
                            fake,
                        )
                    else:
                        _u_0 = interface.ocp.nlp[_penalty.nodes_phase[0]].U_scaled[_penalty.all_nodes_index[0] - u0_mode]
                else:
                    _u_0 = []
                if ocp.assume_phase_dynamics or _penalty.all_nodes_index[1] < len(
                        interface.ocp.nlp[_penalty.nodes_phase[1]].U_scaled):
                    if interface.ocp.nlp[_penalty.nodes_phase[0]].U_scaled[_penalty.all_nodes_index[0] - u0_mode].shape[0] > \
                            interface.ocp.nlp[_penalty.nodes_phase[1]].U_scaled[_penalty.all_nodes_index[1] - u1_mode].shape[0]:
                        fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[0]].U_scaled[
                            _penalty.all_nodes_index[0] - u0_mode].shape[0] - interface.ocp.nlp[_penalty.nodes_phase[1]].U_scaled[
                                                        _penalty.all_nodes_index[1] - u1_mode].shape[0], 1)
                        _u_1 = vertcat(
                            interface.ocp.nlp[_penalty.nodes_phase[1]].U_scaled[_penalty.all_nodes_index[1] - u1_mode],
                            fake,
                        )
                    else:
                        _u_1 = interface.ocp.nlp[_penalty.nodes_phase[1]].U_scaled[_penalty.all_nodes_index[1] - u1_mode]
                else:
                    _u_1 = []
                _u = vertcat(_u_1, _u_0)

                if interface.ocp.nlp[_penalty.nodes_phase[1]].S_scaled[_penalty.all_nodes_index[1]][:, 0].shape[0] > \
                        interface.ocp.nlp[_penalty.nodes_phase[0]].S_scaled[_penalty.all_nodes_index[0]][:, 0].shape[0]:
                    fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[1]].S_scaled[
                                                            _penalty.all_nodes_index[1]][:, 0].shape[0] -
                                                interface.ocp.nlp[_penalty.nodes_phase[0]].S_scaled[
                                                    _penalty.all_nodes_index[0]][:, 0].shape[0], 1)
                    _s_0 = vertcat(
                        interface.ocp.nlp[_penalty.nodes_phase[0]].S_scaled[_penalty.all_nodes_index[0]][:, 0],
                        fake,
                    )
                else:
                    _s_0 = interface.ocp.nlp[_penalty.nodes_phase[0]].S_scaled[_penalty.all_nodes_index[0]][:, 0]
                if interface.ocp.nlp[_penalty.nodes_phase[0]].S_scaled[_penalty.all_nodes_index[0]][:, 0].shape[0] > \
                        interface.ocp.nlp[_penalty.nodes_phase[1]].S_scaled[_penalty.all_nodes_index[1]][:, 0].shape[0]:
                    fake = interface.ocp.cx(interface.ocp.nlp[_penalty.nodes_phase[0]].S_scaled[
                                                            _penalty.all_nodes_index[0]][:, 0].shape[0] -
                                                interface.ocp.nlp[_penalty.nodes_phase[1]].S_scaled[
                                                    _penalty.all_nodes_index[1]][:, 0].shape[0], 1)
                    _s_1 = vertcat(
                        interface.ocp.nlp[_penalty.nodes_phase[1]].S_scaled[_penalty.all_nodes_index[1]][:, 0],
                        fake,
                    )
                else:
                    _s_1 = interface.ocp.nlp[_penalty.nodes_phase[1]].S_scaled[_penalty.all_nodes_index[1]][:, 0]
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
                ui_mode = get_control_modificator(i)

                if is_unscaled:
                    _x_tp = nlp_i.cx()
                    for i in range(nlp_i.X[index_i].shape[1]):
                        _x_tp = vertcat(_x_tp, nlp_i.X[index_i][:, i])
                    _u_tp = nlp_i.U[index_i - ui_mode] if ocp.assume_phase_dynamics or index_i < len(nlp_i.U) else []
                    _s_tp = nlp_i.S[index_i]
                else:
                    _x_tp = nlp_i.cx()
                    for i in range(nlp_i.X_scaled[index_i].shape[1]):
                        _x_tp = vertcat(_x_tp, nlp_i.X_scaled[index_i][:, i])
                    _u_tp = (
                        nlp_i.U_scaled[index_i - ui_mode]
                        if ocp.assume_phase_dynamics or index_i < len(nlp_i.U_scaled)
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
                _u = nlp.U[_idx][:, 0] if nlp.assume_phase_dynamics or _idx < len(nlp.U) else []
                _s = nlp.S[_idx]
            else:
                _x = nlp.cx()
                for i in range(nlp.X_scaled[_idx].shape[1]):
                    _x = vertcat(_x, nlp.X_scaled[_idx][:, i])
                _u = nlp.U_scaled[_idx][:, 0] if nlp.assume_phase_dynamics or _idx < len(nlp.U_scaled) else []
                _s = nlp.S_scaled[_idx]
        else:  # do we really need this if ?
            if is_unscaled:
                _x = nlp.cx()
                for i in range(nlp.X[_idx].shape[1]):
                    _x = vertcat(_x, nlp.X[_idx][:, i])
                _u = nlp.U[_idx][:, 0] if nlp.assume_phase_dynamics or _idx < len(nlp.U) else []
                _s = nlp.S[_idx][:, 0]
            else:
                _x = nlp.cx()
                for i in range(nlp.X_scaled[_idx].shape[1]):
                    _x = vertcat(_x, nlp.X_scaled[_idx][:, i])
                if sum(_penalty.weighted_function[_idx].size_in(1)) == 0:
                    _u = []
                elif nlp.assume_phase_dynamics and _idx == len(nlp.U_scaled):
                    _u = nlp.U_scaled[_idx - 1][:, 0]
                elif _idx < len(nlp.U_scaled):
                    _u = nlp.U_scaled[_idx][:, 0]
                else:
                    _u = []
                _s = nlp.S_scaled[_idx][:, 0]

        if _penalty.derivative or _penalty.explicit_derivative:
            if _idx < nlp.ns:
                if is_unscaled:
                    x = nlp.X[_idx + 1][:, 0]
                    if nlp.assume_phase_dynamics and _idx + 1 == len(nlp.U):
                        u = nlp.U[_idx][:, 0]
                    elif _idx + 1 < len(nlp.U):
                        u = nlp.U[_idx + 1][:, 0]
                    else:
                        u = []
                    s = nlp.S[_idx + 1][:, 0]
                else:
                    x = nlp.X_scaled[_idx + 1][:, 0]
                    if nlp.assume_phase_dynamics and _idx + 1 == len(nlp.U_scaled):
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
            if nlp.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
                if is_unscaled:
                    u = nlp.U[_idx + 1][:, 0] if nlp.assume_phase_dynamics or _idx + 1 < len(nlp.U) else []
                else:
                    u = nlp.U_scaled[_idx + 1][:, 0] if nlp.assume_phase_dynamics or _idx + 1 < len(nlp.U_scaled) else []
                _u = vertcat(_u, u)

        if _penalty.integration_rule == QuadratureRule.TRAPEZOIDAL:
            if nlp.control_type in (ControlType.LINEAR_CONTINUOUS, ControlType.CONSTANT_WITH_LAST_NODE):
                if is_unscaled:
                    u = nlp.U[_idx + 1][:, 0] if nlp.assume_phase_dynamics or _idx + 1 < len(nlp.U) else []
                else:
                    u = nlp.U_scaled[_idx + 1][:, 0] if nlp.assume_phase_dynamics or _idx + 1 < len(nlp.U_scaled) else []
                _u = vertcat(_u, u)
        return _x, _u, _s

    param = interface.ocp.cx(interface.ocp.parameters.cx)
    out = interface.ocp.cx()
    for penalty in penalties:
        if penalty.multinode_penalty:
            phase_idx = penalty.nodes_phase[0]
        else:
            phase_idx = penalty.phase
        if interface.ocp.nlp[phase_idx].motor_noise is not None:
            motor_noise = interface.ocp.nlp[phase_idx].motor_noise
            sensory_noise = interface.ocp.nlp[phase_idx].sensory_noise
        else:
            motor_noise = interface.ocp.nlp[phase_idx].cx()
            sensory_noise = interface.ocp.nlp[phase_idx].cx()

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
                x_tp, u_tp, s_tp = get_x_and_u_at_idx(penalty, idx, is_unscaled)
                x = vertcat(x, x_tp)
                u = vertcat(u, u_tp)
                s = vertcat(s, s_tp)
            if (
                penalty.derivative or penalty.explicit_derivative or penalty.node[0] == Node.ALL
            ) and nlp.control_type == ControlType.CONSTANT:
                u = vertcat(u, u[:, -1])

            # We can call penalty.weighted_function[0] since multi-thread declares all the node at [0]
            p = reshape(
                penalty.weighted_function[0](
                    x, u, param, s, motor_noise, sensory_noise, penalty.weight, target, penalty.dt
                ),
                -1,
                1,
            )

        else:
            p = interface.ocp.cx()
            for idx in penalty.node_idx:
                if penalty.target is None:
                    target = []
                elif (
                    penalty.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                    or penalty.integration_rule == QuadratureRule.TRAPEZOIDAL
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
                    s = []
                else:
                    x, u, s = get_x_and_u_at_idx(penalty, idx, is_unscaled)
                    p = vertcat(
                        p,
                        penalty.weighted_function[idx](
                            x, u, param, s, motor_noise, sensory_noise, penalty.weight, target, penalty.dt
                        ),
                    )

        out = vertcat(out, sum2(p))
    return out
