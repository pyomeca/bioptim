import numpy as np
from casadi import MX, SX, Function, horzcat, vertcat, jacobian, vcat, hessian
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from ..misc.enums import (
    ControlType,
)
from ..misc.enums import IntegralApproximation


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
        for phase in ocp.nlp:
            list_constraints = []
            for constraints in phase.g:
                node_index = constraints.node_idx[0]  # TODO deal with assume_phase_dynamics=False
                phase.states.node_index = node_index
                phase.states_dot.node_index = node_index
                phase.controls.node_index = node_index

                for axis in range(
                    0,
                    constraints.function[node_index](
                        phase.states.cx_start, phase.controls.cx_start, phase.parameters.cx_start
                    ).shape[0],
                ):
                    # depends if there are parameters
                    if phase.parameters.shape == 0:
                        vertcat_obj = vertcat(*phase.X_scaled, *phase.U_scaled, phase.parameters.cx_start)
                    else:
                        vertcat_obj = vertcat(*phase.X_scaled, *phase.U_scaled, *[phase.parameters.cx_start])

                    list_constraints.append(
                        jacobian(
                            constraints.function[constraints.node_idx[0]](
                                phase.states.cx_start,
                                phase.controls.cx_start,
                                phase.parameters.cx_start,
                            )[axis],
                            vertcat_obj,
                        )
                    )

            jacobian_cas = vcat(list_constraints).T

            # depends if there are parameters
            if phase.parameters.shape == 0:
                vertcat_obj = vertcat(*phase.X_scaled, *phase.U_scaled, phase.parameters.cx_start)
            else:
                vertcat_obj = vertcat(*phase.X_scaled, *phase.U_scaled, *[phase.parameters.cx_start])

            jac_func = Function(
                "jacobian",
                [vertcat_obj],
                [jacobian_cas],
            )

            # evaluate jac_func at X_init, U_init, considering the parameters
            x_init = np.zeros((len(phase.X), phase.x_init.shape[0]))
            U_init = np.zeros((len(phase.U), phase.u_init.shape[0]))
            Param_init = np.array(phase.parameters.initial_guess.init)

            for n_shooting in range(0, phase.ns + 1):
                x_init[n_shooting, :] = np.array(phase.x_init.init.evaluate_at(n_shooting))
            for n_shooting in range(0, phase.ns):
                U_init[n_shooting, :] = np.array(phase.u_init.init.evaluate_at(n_shooting))

            x_init = x_init.reshape((x_init.size, 1))
            U_init = U_init.reshape((U_init.size, 1))
            Param_init = Param_init.reshape((np.array(phase.parameters.initial_guess.init).size, 1))

            jacobian_matrix = np.array(jac_func(np.vstack((x_init, U_init, Param_init))))

            jacobian_list.append(jacobian_matrix)

            # calculate jacobian rank
            if (jacobian_matrix.size == 0) == False:
                rank = np.linalg.matrix_rank(jacobian_matrix)
                jacobian_rank.append(rank)
            else:
                jacobian_rank.append("No constraints")

            # HESSIAN
            tick_labels = []
            list_hessian = []
            list_norm = []
            for constraints in phase.g:
                node_index = constraints.node_idx[0]  # TODO deal with assume_phase_dynamics=False
                phase.states.node_index = node_index
                phase.states_dot.node_index = node_index
                phase.controls.node_index = node_index

                for axis in range(
                    0,
                    constraints.function[node_index](
                        phase.states.cx_start, phase.controls.cx_start, phase.parameters.cx_start
                    ).shape[0],
                ):
                    # find all equality constraints
                    if (constraints.bounds.min[axis][0] == constraints.bounds.max[axis][0]) == True:
                        # parameters
                        if (phase.parameters.shape == 0) == True:
                            vertcat_obj = vertcat(*phase.X_scaled, *phase.U_scaled, phase.parameters.cx_start)
                        else:
                            vertcat_obj = vertcat(*phase.X_scaled, *phase.U_scaled, *[phase.parameters.cx_start])

                        hessian_cas = hessian(
                            constraints.function[node_index](
                                phase.states.cx_start,
                                phase.controls.cx_start,
                                phase.parameters.cx_start,
                            )[axis],
                            vertcat_obj,
                        )[0]

                        tick_labels.append(constraints.name)

                        hes_func = Function(
                            "hessian",
                            [vertcat_obj],
                            [hessian_cas],
                        )

                        # evaluate hes_func en X_init, U_init, with parameters
                        x_init = np.zeros((len(phase.X), phase.x_init.shape[0]))
                        U_init = np.zeros((len(phase.U), phase.u_init.shape[0]))
                        Param_init = np.array(phase.parameters.initial_guess.init)

                        for n_shooting in range(0, phase.ns + 1):
                            x_init[n_shooting, :] = np.array(phase.x_init.init.evaluate_at(n_shooting))
                        for n_shooting in range(0, phase.ns):
                            U_init[n_shooting, :] = np.array(phase.u_init.init.evaluate_at(n_shooting))

                        x_init = x_init.reshape((x_init.size, 1))
                        U_init = U_init.reshape((U_init.size, 1))
                        Param_init = Param_init.reshape((np.array(phase.parameters.initial_guess.init).size, 1))

                        hessian_matrix = np.array(hes_func(np.vstack((x_init, U_init, Param_init))))

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
        for ax in range(0, ocp.n_phases):
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
            for i in range(0, len(hessian_norm_list[ax])):
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
        for phase, nlp_phase in enumerate(ocp.nlp):
            for obj in nlp_phase.J:
                objective = 0

                node_index = obj.node_idx[0]  # TODO deal with assume_phase_dynamics=False
                nlp_phase.states.node_index = node_index
                nlp_phase.states_dot.node_index = node_index
                nlp_phase.controls.node_index = node_index

                # Test every possibility
                if obj.binode_constraint or obj.transition:
                    nlp = ocp.nlp[phase - 1]
                    nlp_post = nlp_phase
                    states_pre = nlp.states.cx_end
                    states_post = nlp_post.states.cx_start
                    controls_pre = nlp.controls.cx_end
                    controls_post = nlp_post.controls.cx_start
                    state_cx = vertcat(states_pre, states_post)
                    control_cx = vertcat(controls_pre, controls_post)

                else:
                    if obj.integrate:
                        state_cx = horzcat(*([nlp_phase.states.cx_start] + nlp_phase.states.cx_intermediates_list))
                        control_cx = nlp_phase.controls.cx_start
                    else:
                        state_cx = nlp_phase.states.cx_start
                        control_cx = nlp_phase.controls.cx_start
                    if obj.explicit_derivative:
                        if obj.derivative:
                            raise RuntimeError("derivative and explicit_derivative cannot be simultaneously true")
                        state_cx = horzcat(state_cx, nlp_phase.states.cx_end)
                        control_cx = horzcat(control_cx, nlp_phase.controls.cx_end)

                if obj.derivative:
                    state_cx = horzcat(nlp_phase.states.cx_end, nlp_phase.states.cx_start)
                    control_cx = horzcat(nlp_phase.controls.cx_end, nlp_phase.controls.cx_start)

                dt_cx = nlp_phase.cx.sym("dt", 1, 1)
                is_trapezoidal = (
                    obj.integration_rule == IntegralApproximation.TRAPEZOIDAL
                    or obj.integration_rule == IntegralApproximation.TRUE_TRAPEZOIDAL
                )

                if is_trapezoidal:
                    state_cx = (
                        horzcat(nlp_phase.states.cx_start, nlp_phase.states.cx_end)
                        if obj.integration_rule == IntegralApproximation.TRAPEZOIDAL
                        else nlp_phase.states.cx_start
                    )
                    control_cx = (
                        horzcat(nlp_phase.controls.cx_start)
                        if nlp_phase.control_type == ControlType.CONSTANT
                        else horzcat(nlp_phase.controls.cx_start, nlp_phase.controls.cx_end)
                    )
                    control_cx_end = get_u(nlp_phase, control_cx, dt_cx)

                if obj.target is None:
                    p = obj.weighted_function[node_index](
                        state_cx,
                        control_cx,
                        nlp_phase.parameters.cx_start,
                        obj.weight,
                        [],
                        obj.dt,
                    )
                else:
                    p = obj.weighted_function[node_index](
                        state_cx,
                        control_cx,
                        nlp_phase.parameters.cx_start,
                        obj.weight,
                        obj.target,
                        obj.dt,
                    )

                for i in range(0, p.shape[0]):
                    objective += p[i] ** 2

            # create function to build the hessian
            # parameters
            if nlp_phase.parameters.shape == 0:
                vertcat_obj = vertcat(*nlp_phase.X_scaled, *nlp_phase.U_scaled, nlp_phase.parameters.cx_start)
            else:
                vertcat_obj = vertcat(*nlp_phase.X_scaled, *nlp_phase.U_scaled, *[nlp_phase.parameters.cx_start])

            hessian_cas = hessian(objective, vertcat_obj)[0]

            hes_func = Function(
                "hessian",
                [vertcat_obj],
                [hessian_cas],
            )

            # evaluate hes_func at X_init, U_init, with parameters
            x_init = np.zeros((len(nlp_phase.X), nlp_phase.x_init.shape[0]))
            U_init = np.zeros((len(nlp_phase.U), nlp_phase.u_init.shape[0]))
            Param_init = np.array(nlp_phase.parameters.initial_guess.init)

            for n_shooting in range(0, nlp_phase.ns + 1):
                x_init[n_shooting, :] = np.array(nlp_phase.x_init.init.evaluate_at(n_shooting))
            for n_shooting in range(0, nlp_phase.ns):
                U_init[n_shooting, :] = np.array(nlp_phase.u_init.init.evaluate_at(n_shooting))

            x_init = x_init.reshape((x_init.size, 1))
            U_init = U_init.reshape((U_init.size, 1))
            Param_init = Param_init.reshape((np.array(nlp_phase.parameters.initial_guess.init).size, 1))

            hessian_obj_matrix = np.array(hes_func(np.vstack((x_init, U_init, Param_init))))
            hessian_obj_list.append(hessian_obj_matrix)

        # Convexity checking (positive semi-definite hessian)
        # On R (convexe), the objective is convexe if and only if the hessian is positive semi definite (psd)
        # And, as the hessian is symetric (Schwarz), the hessian is psd if and only if the eigenvalues are positive
        convexity = []
        condition_number = []
        for matrix in range(0, len(hessian_obj_list)):
            eigen_values = np.linalg.eigvals(hessian_obj_list[matrix])
            ev_max = min(eigen_values)
            ev_min = max(eigen_values)
            if ev_min == 0:
                condition_number.append(" /!\ Ev_min is 0")
            if ev_min != 0:
                condition_number.append(np.abs(ev_max) / np.abs(ev_min))
            convexity.append("Possible")
            for ev in range(0, eigen_values.size):
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
        for ax in range(0, ocp.n_phases):
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
