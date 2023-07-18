import numpy as np
from casadi import MX, SX, Function, horzcat, vertcat, jacobian, vcat, hessian
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from ..misc.enums import (
    ControlType,
)
from ..misc.enums import QuadratureRule


def check_conditioning(ocp):
    """
    Visualisation of jacobian and hessian contraints and hessian objective for each phase at initial time
    """

    def get_u(nlp, u: MX | SX, dt: MX | SX):
        """
        Get the control at a given time

        Parameters
        ----------
        nlp: NonlinearProgram
            The nonlinear program
        u: MX | SX
            The control matrix
        dt: MX | SX
            The time a which control should be computed

        Returns
        -------
        The control at a given time
        """

        if nlp.control_type in (ControlType.CONSTANT, ControlType.NONE):
            return u
        elif nlp.control_type == ControlType.LINEAR_CONTINUOUS:
            return u[:, 0] + (u[:, 1] - u[:, 0]) * dt
        else:
            raise RuntimeError(f"{nlp.control_type} ControlType not implemented yet")

    def jacobian_hessian_constraints():
        """
        Returns
        -------
        A list with jacobian matrix of constraints evaluates at initial time for each phase
        A list with the rank of each jacobian matrix
        A list with the different type of constraints
        A list with norms of hessian matrix of constraints at initial time for each phase
        """

        jacobian_list = []
        jacobian_rank = []
        tick_labels_list = []
        hessian_norm_list = []

        # JACOBIAN
        for nlp in ocp.nlp:
            list_constraints = []
            for constraints in nlp.g:
                node_index = constraints.node_idx[0]  # TODO deal with assume_phase_dynamics=False
                nlp.states.node_index = node_index
                nlp.states_dot.node_index = node_index
                nlp.controls.node_index = node_index
                nlp.stochastic_variables.node_index = node_index

                for axis in range(
                    0,
                    constraints.function[node_index](
                        nlp.states.cx_start, nlp.controls.cx_start, nlp.parameters.cx, nlp.stochastic_variables.cx_start
                    ).shape[0],
                ):
                    # depends if there are parameters
                    if nlp.parameters.shape == 0:
                        vertcat_obj = vertcat(*nlp.X_scaled, *nlp.U_scaled, nlp.parameters.cx, *nlp.S)
                    else:
                        vertcat_obj = vertcat(*nlp.X_scaled, *nlp.U_scaled, *[nlp.parameters.cx, *nlp.S])

                    list_constraints.append(
                        jacobian(
                            constraints.function[constraints.node_idx[0]](
                                nlp.states.cx_start,
                                nlp.controls.cx_start,
                                nlp.parameters.cx,
                                nlp.stochastic_variables.cx_start,
                            )[axis],
                            vertcat_obj,
                        )
                    )

            jacobian_cas = vcat(list_constraints).T

            # depends if there are parameters
            if nlp.parameters.shape == 0:
                vertcat_obj = vertcat(*nlp.X_scaled, *nlp.U_scaled, nlp.parameters.cx, *nlp.S)
            else:
                vertcat_obj = vertcat(*nlp.X_scaled, *nlp.U_scaled, *[nlp.parameters.cx], *nlp.S)

            jac_func = Function(
                "jacobian",
                [vertcat_obj],
                [jacobian_cas],
            )

            nb_x_init = sum([nlp.x_init[key].shape[0] for key in nlp.x_init.keys()])
            nb_u_init = sum([nlp.u_init[key].shape[0] for key in nlp.u_init.keys()])
            nb_s_init = sum([nlp.s_init[key].shape[0] for key in nlp.s_init.keys()])

            # evaluate jac_func at X_init, U_init, considering the parameters
            x_init = np.zeros((len(nlp.X), nb_x_init))
            u_init = np.zeros((len(nlp.U), nb_u_init))
            param_init = np.array([ocp.parameter_init[key].shape[0] for key in ocp.parameter_init.keys()])
            s_init = np.zeros((len(nlp.S), nb_s_init))

            for key in nlp.states.keys():
                nlp.x_init[key].check_and_adjust_dimensions(len(nlp.states[key]), nlp.ns + 1)
                for node_index in range(nlp.ns + 1):
                    nlp.states.node_index = node_index
                    x_init[node_index, nlp.states[key].index] = np.array(nlp.x_init[key].init.evaluate_at(node_index))
            for key in nlp.controls.keys():
                nlp.u_init[key].check_and_adjust_dimensions(len(nlp.controls[key]), nlp.ns)
                for node_index in range(nlp.ns):
                    nlp.controls.node_index = node_index
                    u_init[node_index, nlp.controls[key].index] = np.array(nlp.u_init[key].init.evaluate_at(node_index))
            for key in nlp.stochastic_variables.keys():
                nlp.s_init[key].check_and_adjust_dimensions(len(nlp.stochastic_variables[key]), nlp.ns)
                for node_index in range(nlp.ns):
                    nlp.stochastic_variables.node_index = node_index
                    s_init[node_index, nlp.stochastic_variables[key].index] = np.array(
                        nlp.s_init[key].init.evaluate_at(node_index)
                    )

            x_init = x_init.reshape((x_init.size, 1))
            u_init = u_init.reshape((u_init.size, 1))
            param_init = param_init.reshape((param_init.size, 1))
            s_init = s_init.reshape((s_init.size, 1))

            jacobian_matrix = np.array(jac_func(np.vstack((x_init, u_init, param_init, s_init))))

            jacobian_list.append(jacobian_matrix)

            # calculate jacobian rank
            if jacobian_matrix.size > 0:
                rank = np.linalg.matrix_rank(jacobian_matrix)
                jacobian_rank.append(rank)
            else:
                jacobian_rank.append("No constraints")

            # HESSIAN
            tick_labels = []
            list_hessian = []
            list_norm = []
            for constraints in nlp.g:
                node_index = constraints.node_idx[0]  # TODO deal with assume_phase_dynamics=False
                nlp.states.node_index = node_index
                nlp.states_dot.node_index = node_index
                nlp.controls.node_index = node_index
                nlp.stochastic_variables.node_index = node_index

                for axis in range(
                    0,
                    constraints.function[node_index](
                        nlp.states.cx_start, nlp.controls.cx_start, nlp.parameters.cx, nlp.stochastic_variables.cx_start
                    ).shape[0],
                ):
                    # find all equality constraints
                    if constraints.bounds.min[axis][0] == constraints.bounds.max[axis][0]:
                        # parameters
                        if nlp.parameters.shape == 0:
                            vertcat_obj = vertcat(*nlp.X_scaled, *nlp.U_scaled, nlp.parameters.cx, *nlp.S)
                        else:
                            vertcat_obj = vertcat(*nlp.X_scaled, *nlp.U_scaled, *[nlp.parameters.cx, *nlp.S])

                        hessian_cas = hessian(
                            constraints.function[node_index](
                                nlp.states.cx_start,
                                nlp.controls.cx_start,
                                nlp.parameters.cx,
                                nlp.stochastic_variables.cx_start,
                            )[axis],
                            vertcat_obj,
                        )[0]

                        tick_labels.append(constraints.name)

                        hes_func = Function(
                            "hessian",
                            [vertcat_obj],
                            [hessian_cas],
                        )

                        hessian_matrix = np.array(hes_func(np.vstack((x_init, u_init, param_init, s_init))))

                        # append hessian list
                        list_hessian.append(hessian_matrix)

            tick_labels_list.append(tick_labels)

            # calculate norm
            for hessian_idx in list_hessian:
                norm = np.linalg.norm(hessian_idx)
                list_norm.append(norm)
            array_norm = np.array(list_norm).reshape(len(list_hessian), 1)
            hessian_norm_list.append(array_norm)

        return jacobian_list, jacobian_rank, tick_labels_list, hessian_norm_list

    def check_constraints_plot():
        """
        Visualisation of jacobian matrix and hessian norm matrix
        """
        jacobian_list, jacobian_rank, tick_labels_list, hessian_norm_list = jacobian_hessian_constraints()

        max_norm = []
        min_norm = []
        if hessian_norm_list[0].size != 0:
            for hessian_norm in hessian_norm_list:
                max_norm.append(np.ndarray.max(hessian_norm))
                min_norm.append(np.ndarray.min(hessian_norm))
            min_norm = min(min_norm)
            max_norm = max(max_norm)
        else:
            max_norm = 0
            min_norm = 0

        max_jac = []
        min_jac = []
        if jacobian_list[0].size != 0:
            for jacobian_obj in jacobian_list:
                max_jac.append(np.ndarray.max(jacobian_obj))
                min_jac.append(np.ndarray.min(jacobian_obj))
            max_jac = max(max_jac)
            min_jac = min(min_jac)
        else:
            max_jac = 0
            min_jac = 0

        # PLOT GENERAL
        fig, axis = plt.subplots(1, 2 * ocp.n_phases)
        for ax in range(ocp.n_phases):
            # Jacobian plot
            jacobian_list[ax][(jacobian_list[ax] == 0)] = np.nan
            current_cmap = mcm.get_cmap("seismic")
            current_cmap.set_bad(color="k")
            norm = mcolors.TwoSlopeNorm(vmin=min_jac - 0.01, vmax=max_jac + 0.01, vcenter=0)
            im = axis[ax].imshow(jacobian_list[ax], aspect="auto", cmap=current_cmap, norm=norm)
            axis[ax].set_title("Jacobian constraints \n Phase " + str(ax), fontweight="bold", fontsize=8)
            axis[ax].text(
                0,
                jacobian_list[ax].shape[0] * 1.08,
                "Matrix rank = "
                + str(jacobian_rank[ax])
                + "\n Number of constraints = "
                + str(jacobian_list[ax].shape[1]),
                horizontalalignment="center",
                fontweight="bold",
                fontsize=8,
            )
            axis[ax].text(
                0,
                jacobian_list[ax].shape[0] * 1.1,
                "The rank should be equal to the number of constraints",
                horizontalalignment="center",
                fontsize=6,
            )
            cbar_ax = fig.add_axes([0.02, 0.4, 0.015, 0.3])
            fig.colorbar(im, cax=cbar_ax)

            # Hessian constraints plot
            hessian_norm_list[ax][~(hessian_norm_list[ax] != 0).astype(bool)] = np.nan
            current_cmap2 = mcm.get_cmap("seismic")
            current_cmap2.set_bad(color="k")
            norm2 = mcolors.TwoSlopeNorm(vmin=min_norm - 0.01, vmax=max_norm + 0.01, vcenter=0)
            yticks = []
            for i in range(len(hessian_norm_list[ax])):
                yticks.append(i)

            im2 = axis[ax + ocp.n_phases].imshow(hessian_norm_list[ax], aspect="auto", cmap=current_cmap2, norm=norm2)
            axis[ax + ocp.n_phases].set_title(
                "Hessian constraint norms \n Phase " + str(ax), fontweight="bold", fontsize=8
            )
            axis[ax + ocp.n_phases].set_xticks([0])
            axis[ax + ocp.n_phases].set_xticklabels(["Norms should be close to 0"], fontsize=8)
            axis[ax + ocp.n_phases].set_yticks(yticks)
            axis[ax + ocp.n_phases].set_yticklabels(tick_labels_list[ax], fontsize=6, rotation=90)
            cbar_ax2 = fig.add_axes([0.95, 0.4, 0.015, 0.3])
            fig.colorbar(im2, cax=cbar_ax2)
        fig.legend(["Black = 0"], loc="upper left")
        plt.suptitle("Check conditioning for constraints ", color="b", fontsize=15, fontweight="bold")

    def hessian_objective():
        """

        Returns
        -------
        A list with the hessian of objectives evaluate at initial time for each phase
        A list with the condition numbers of each phases
        A list that indicates if the objective is convexe or not
        """

        hessian_obj_list = []
        for phase, nlp in enumerate(ocp.nlp):
            for obj in nlp.J:
                objective = 0

                node_index = obj.node_idx[0]  # TODO deal with assume_phase_dynamics=False
                nlp.states.node_index = node_index
                nlp.states_dot.node_index = node_index
                nlp.controls.node_index = node_index
                nlp.stochastic_variables.node_index = node_index

                # Test every possibility
                if obj.multinode_penalty or obj.transition:
                    phase = ocp.nlp[phase - 1]
                    nlp_post = nlp
                    states_pre = phase.states.cx_end
                    states_post = nlp_post.states.cx_start
                    controls_pre = phase.controls.cx_end
                    controls_post = nlp_post.controls.cx_start
                    stochastic_pre = phase.stochastic_cx_end
                    stochastic_post = nlp_post.stochastic_cx_start
                    state_cx = vertcat(states_pre, states_post)
                    control_cx = vertcat(controls_pre, controls_post)
                    stochastic_cx = vertcat(stochastic_pre, stochastic_post)

                else:
                    if obj.integrate:
                        state_cx = horzcat(*([nlp.states.cx_start] + nlp.states.cx_intermediates_list))
                    else:
                        state_cx = nlp.states.cx_start
                    control_cx = nlp.controls.cx_start
                    stochastic_cx = nlp.stochastic_variables.cx_start
                    if obj.explicit_derivative:
                        if obj.derivative:
                            raise RuntimeError("derivative and explicit_derivative cannot be simultaneously true")
                        state_cx = horzcat(state_cx, nlp.states.cx_end)
                        control_cx = horzcat(control_cx, nlp.controls.cx_end)
                        stochastic_cx = horzcat(stochastic_cx, nlp.stochastic_cx_end)

                if obj.derivative:
                    state_cx = horzcat(nlp.states.cx_end, nlp.states.cx_start)
                    control_cx = horzcat(nlp.controls.cx_end, nlp.controls.cx_start)
                    stochastic_cx = horzcat(nlp.stochastic.cx_end, nlp.stochastic.cx_start)

                dt_cx = nlp.cx.sym("dt", 1, 1)
                is_trapezoidal = (
                    obj.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                    or obj.integration_rule == QuadratureRule.TRAPEZOIDAL
                )

                if is_trapezoidal:
                    state_cx = (
                        horzcat(nlp.states.cx_start, nlp.states.cx_end)
                        if obj.integration_rule == QuadratureRule.APPROXIMATE_TRAPEZOIDAL
                        else nlp.states.cx_start
                    )
                    control_cx = (
                        horzcat(nlp.controls.cx_start)
                        if nlp.control_type == ControlType.CONSTANT
                        else horzcat(nlp.controls.cx_start, nlp.controls.cx_end)
                    )
                    stochastic_cx = (
                        horzcat(nlp.stochastic_variables.cx_start)
                        if nlp.control_type == ControlType.CONSTANT
                        else horzcat(nlp.stochastic_variables.cx_start, nlp.stochastic_variables.cx_end)
                    )

                if obj.target is None:
                    p = obj.weighted_function[node_index](
                        state_cx,
                        control_cx,
                        nlp.parameters.cx,
                        stochastic_cx,
                        obj.weight,
                        [],
                        obj.dt,
                    )
                else:
                    p = obj.weighted_function[node_index](
                        state_cx,
                        control_cx,
                        nlp.parameters.cx,
                        stochastic_cx,
                        obj.weight,
                        obj.target,
                        obj.dt,
                    )

                for i in range(p.shape[0]):
                    objective += p[i] ** 2

            # create function to build the hessian
            # parameters
            if nlp.parameters.shape == 0:
                vertcat_obj = vertcat(*nlp.X_scaled, *nlp.U_scaled, nlp.parameters.cx, *nlp.S)
            else:
                vertcat_obj = vertcat(*nlp.X_scaled, *nlp.U_scaled, *[nlp.parameters.cx], *nlp.S)

            hessian_cas = hessian(objective, vertcat_obj)[0]

            hes_func = Function(
                "hessian",
                [vertcat_obj],
                [hessian_cas],
            )

            nb_x_init = sum([nlp.x_init[key].shape[0] for key in nlp.x_init.keys()])
            nb_u_init = sum([nlp.u_init[key].shape[0] for key in nlp.u_init.keys()])
            nb_s_init = sum([nlp.s_init[key].shape[0] for key in nlp.s_init.keys()])

            # evaluate jac_func at X_init, U_init, considering the parameters
            x_init = np.zeros((len(nlp.X), nb_x_init))
            u_init = np.zeros((len(nlp.U), nb_u_init))
            param_init = np.array([nlp.x_init[key].shape[0] for key in ocp.parameter_init.keys()])
            s_init = np.zeros((len(nlp.S), nb_s_init))

            for key in nlp.states.keys():
                nlp.x_init[key].check_and_adjust_dimensions(len(nlp.states[key]), nlp.ns + 1)
                for node_index in range(nlp.ns + 1):
                    nlp.states.node_index = node_index
                    x_init[node_index, nlp.states[key].index] = np.array(nlp.x_init[key].init.evaluate_at(node_index))
            for key in nlp.controls.keys():
                nlp.u_init[key].check_and_adjust_dimensions(len(nlp.controls[key]), nlp.ns)
                for node_index in range(nlp.ns):
                    nlp.controls.node_index = node_index
                    u_init[node_index, nlp.controls[key].index] = np.array(nlp.u_init[key].init.evaluate_at(node_index))
            for key in nlp.stochastic_variables.keys():
                nlp.s_init[key].check_and_adjust_dimensions(len(nlp.stochastic_variables[key]), nlp.ns)
                for node_index in range(nlp.ns):
                    nlp.stochastic_variables.node_index = node_index
                    s_init[node_index, nlp.stochastic_variables[key].index] = np.array(
                        nlp.s_init[key].init.evaluate_at(node_index)
                    )

            x_init = x_init.reshape((x_init.size, 1))
            u_init = u_init.reshape((u_init.size, 1))
            param_init = param_init.reshape((param_init.size, 1))
            s_init = s_init.reshape((s_init.size, 1))

            hessian_obj_matrix = np.array(hes_func(np.vstack((x_init, u_init, param_init, s_init))))
            hessian_obj_list.append(hessian_obj_matrix)

        # Convexity checking (positive semi-definite hessian)
        # On R (convexe), the objective is convexe if and only if the hessian is positive semi definite (psd)
        # And, as the hessian is symetric (Schwarz), the hessian is psd if and only if the eigenvalues are positive
        convexity = []
        condition_number = []
        for matrix in range(len(hessian_obj_list)):
            eigen_values = np.linalg.eigvals(hessian_obj_list[matrix])
            ev_max = min(eigen_values)
            ev_min = max(eigen_values)
            if ev_min == 0:
                condition_number.append(" /!\ Ev_min is 0")
            if ev_min != 0:
                condition_number.append(np.abs(ev_max) / np.abs(ev_min))
            convexity.append("Possible")
            for ev in range(eigen_values.size):
                if eigen_values[ev] < 0:
                    convexity[matrix] = "False"
                    break

        return hessian_obj_list, condition_number, convexity

    def check_objective_plot():
        """
        Visualisation of hessian objective matrix
        """

        hessian_obj_list, condition_number, convexity = hessian_objective()

        max_hes = []
        min_hes = []
        for hessian_matrix_obj in hessian_obj_list:
            max_hes.append(np.ndarray.max(hessian_matrix_obj))
            min_hes.append(np.ndarray.min(hessian_matrix_obj))
        min_hes = min(min_hes)
        max_hes = max(max_hes)

        # PLOT GENERAL
        fig_obj, axis_obj = plt.subplots(1, ocp.n_phases)
        for ax in range(ocp.n_phases):
            hessian_obj_list[ax][~(hessian_obj_list[ax] != 0).astype(bool)] = np.nan
            current_cmap3 = mcm.get_cmap("seismic")
            current_cmap3.set_bad(color="k")
            norm = mcolors.TwoSlopeNorm(vmin=min_hes - 0.01, vmax=max_hes + 0.01, vcenter=0)
            if ocp.n_phases == 1:
                im3 = axis_obj.imshow(hessian_obj_list[ax], cmap=current_cmap3, norm=norm)
                axis_obj.set_title("Hessian objective \n Phase " + str(ax), fontweight="bold", fontsize=8)
                axis_obj.text(
                    hessian_obj_list[ax].shape[0] / 2,
                    hessian_obj_list[ax].shape[0] * 1.1,
                    "Convexity = " + convexity[ax],
                    horizontalalignment="center",
                    fontweight="bold",
                    fontsize=8,
                )
                axis_obj.text(
                    hessian_obj_list[ax].shape[0] / 2,
                    hessian_obj_list[ax].shape[0] * 1.2,
                    "|λmax|/|λmin| = Condition number = " + condition_number[ax],
                    horizontalalignment="center",
                    fontweight="bold",
                    fontsize=8,
                )
            else:
                im3 = axis_obj[ax].imshow(hessian_obj_list[ax], cmap=current_cmap3, norm=norm)
                axis_obj[ax].set_title("Hessian objective \n Phase " + str(ax), fontweight="bold", fontsize=8)
                axis_obj[ax].text(
                    hessian_obj_list[ax].shape[0] / 2,
                    hessian_obj_list[ax].shape[0] * 1.1,
                    "Convexity = " + convexity[ax],
                    horizontalalignment="center",
                    fontweight="bold",
                    fontsize=8,
                )
                axis_obj[ax].text(
                    hessian_obj_list[ax].shape[0] / 2,
                    hessian_obj_list[ax].shape[0] * 1.2,
                    "|λmax|/|λmin| = Condition number = " + condition_number[ax],
                    horizontalalignment="center",
                    fontweight="bold",
                    fontsize=8,
                )
            cbar_ax3 = fig_obj.add_axes([0.02, 0.4, 0.015, 0.3])
            fig_obj.colorbar(im3, cax=cbar_ax3)
        fig_obj.legend(["Black = 0"], loc="upper left")
        fig_obj.text(
            0.5,
            0.1,
            "Every hessian should be convexe \n Condition numbers should be close to 0",
            horizontalalignment="center",
            fontsize=12,
            fontweight="bold",
        )
        plt.suptitle("Check conditioning for objectives", color="b", fontsize=15, fontweight="bold")

    check_constraints_plot()
    check_objective_plot()

    plt.show()
