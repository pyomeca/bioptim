import numpy as np
from casadi import MX, SX, Function, horzcat, vertcat, jacobian, vcat, hessian
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from ..misc.enums import (
    ControlType,
)
from ..misc.enums import QuadratureRule
from ..dynamics.ode_solver import OdeSolver
from ..limits.penalty_helpers import PenaltyHelpers


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
    hessian_norm_list = []

    # JACOBIAN
    phases_dt = ocp.dt_parameter.cx
    for nlp in ocp.nlp:
        list_constraints = []

        for constraints in nlp.g:
            node_index = constraints.node_idx[0]  # TODO deal with phase_dynamics=PHASE_DYNAMICS.ONE_PER_NOE
            nlp.states.node_index = node_index
            nlp.states_dot.node_index = node_index
            nlp.controls.node_index = node_index
            nlp.algebraic_states.node_index = node_index

            if constraints.multinode_penalty:
                n_phases = ocp.n_phases
                for phase_idx in constraints.nodes_phase:
                    controllers.append(constraints.get_penalty_controller(ocp, ocp.nlp[phase_idx % n_phases]))
            else:
                controllers = [constraints.get_penalty_controller(ocp, nlp)]

            for axis in range(constraints.function[node_index].size_out("val")[0]):
                # depends if there are parameters
                if nlp.parameters.shape == 0:
                    vertcat_obj = vertcat([], *nlp.X_scaled, *nlp.U_scaled, nlp.parameters.cx, *nlp.A_scaled)
                else:
                    vertcat_obj = vertcat([], *nlp.X_scaled, *nlp.U_scaled, *[nlp.parameters.cx, *nlp.A_scaled])

                for controller in controllers:
                    controller.node_index = constraints.node_idx[0]

                _, t0, x, u, p, a = constraints.get_variable_inputs(controllers)

                list_constraints.append(
                    jacobian(
                        constraints.function[constraints.node_idx[0]](t0, phases_dt, x, u, p, a)[axis],
                        vertcat_obj,
                    )
                )

        jacobian_cas = vcat(list_constraints).T

        # depends if there are parameters
        if nlp.parameters.shape == 0:
            vertcat_obj = vertcat([], *nlp.X_scaled, *nlp.U_scaled, nlp.parameters.cx, *nlp.A)
        else:
            vertcat_obj = vertcat([], *nlp.X_scaled, *nlp.U_scaled, *[nlp.parameters.cx], *nlp.A)

        jac_func = Function(
            "jacobian",
            [vertcat_obj],
            [jacobian_cas],
        )

        nb_x_init = sum([nlp.x_init[key].shape[0] for key in nlp.x_init.keys()])
        nb_u_init = sum([nlp.u_init[key].shape[0] for key in nlp.u_init.keys()])
        nb_a_init = sum([nlp.a_init[key].shape[0] for key in nlp.a_init.keys()])

        # evaluate jac_func at X_init, U_init, considering the parameters
        time_init = np.array([], dtype=np.float64)
        x_init = np.zeros((len(nlp.X), nb_x_init))
        u_init = np.zeros((len(nlp.U), nb_u_init))
        param_init = np.array([ocp.parameter_init[key].shape[0] for key in ocp.parameter_init.keys()])
        a_init = np.zeros((len(nlp.A), nb_a_init))

        for key in nlp.states.keys():
            nlp.x_init[key].check_and_adjust_dimensions(len(nlp.states[key]), nlp.ns + 1)
            for node_index in range(nlp.ns + 1):
                nlp.states.node_index = node_index
                x_init[node_index, nlp.states[key].index] = np.array(nlp.x_init[key].init.evaluate_at(node_index))
        for key in nlp.controls.keys():
            if not key in nlp.controls:
                continue
            nlp.u_init[key].check_and_adjust_dimensions(len(nlp.controls[key]), nlp.ns)
            for node_index in range(nlp.ns):
                nlp.controls.node_index = node_index
                u_init[node_index, nlp.controls[key].index] = np.array(nlp.u_init[key].init.evaluate_at(node_index))
        for key in nlp.algebraic_states.keys():
            nlp.a_init[key].check_and_adjust_dimensions(len(nlp.algebraic_states[key]), nlp.ns)
            for node_index in range(nlp.ns):
                nlp.algebraic_states.node_index = node_index
                a_init[node_index, nlp.algebraic_states[key].index] = np.array(
                    nlp.a_init[key].init.evaluate_at(node_index)
                )

        time_init = time_init.reshape((time_init.size, 1))
        x_init = x_init.reshape((x_init.size, 1))
        u_init = u_init.reshape((u_init.size, 1))
        param_init = param_init.reshape((param_init.size, 1))
        a_init = a_init.reshape((a_init.size, 1))

        vector_init = np.vstack((time_init, x_init, u_init, param_init, a_init))
        jacobian_matrix = np.array(jac_func(vector_init))
        jacobian_list.append(jacobian_matrix)

        # calculate jacobian rank
        if jacobian_matrix.size > 0:
            rank = np.linalg.matrix_rank(jacobian_matrix)
            jacobian_rank.append(rank)
        else:
            jacobian_rank.append("No constraints")

        # HESSIAN
        list_hessian = []
        list_norm = []
        for constraints in nlp.g:
            node_index = constraints.node_idx[0]  # TODO deal with phase_dynamics=PhaseDynamics.ONE_PER_NODE
            nlp.states.node_index = node_index
            nlp.states_dot.node_index = node_index
            nlp.controls.node_index = node_index
            nlp.algebraic_states.node_index = node_index

            if constraints.multinode_penalty:
                n_phases = ocp.n_phases
                for phase_idx in constraints.nodes_phase:
                    controllers.append(constraints.get_penalty_controller(ocp, ocp.nlp[phase_idx % n_phases]))
            else:
                controllers = [constraints.get_penalty_controller(ocp, nlp)]

            for axis in range(constraints.function[node_index].size_out("val")[0]):
                # find all equality constraints
                if constraints.bounds.min[axis][0] == constraints.bounds.max[axis][0]:
                    vertcat_obj = vertcat([], *nlp.X_scaled, *nlp.U_scaled)  # time, states, controls
                    if nlp.parameters.shape == 0:
                        vertcat_obj = vertcat(vertcat_obj, nlp.parameters.cx)
                    else:
                        vertcat_obj = vertcat(vertcat_obj, *[nlp.parameters.cx])
                    vertcat_obj = vertcat(vertcat_obj, *nlp.A_scaled)

                    for controller in controllers:
                        controller.node_index = constraints.node_idx[0]
                    _, t0, x, u, p, a = constraints.get_variable_inputs(controllers)

                    hessian_cas = hessian(
                        constraints.function[node_index](t0, phases_dt, x, u, p, a)[axis], vertcat_obj
                    )[0]

                    hes_func = Function(
                        "hessian",
                        [vertcat_obj],
                        [hessian_cas],
                    )

                    vector_init = np.vstack((time_init, x_init, u_init, param_init, a_init))
                    hessian_matrix = np.array(hes_func(vector_init))
                    list_hessian.append(hessian_matrix)

        # calculate norm
        for hessian_idx in list_hessian:
            norm = np.linalg.norm(hessian_idx)
            list_norm.append(norm)
        array_norm = np.array(list_norm).reshape(len(list_hessian), 1)
        hessian_norm_list.append(array_norm)

    return jacobian_list, jacobian_rank, hessian_norm_list



def hessian_objective(v, ocp):
    """

    Returns
    -------
    A list with the hessian of objectives evaluate at initial time for each phase
    A list with the condition numbers of each phases
    A list that indicates if the objective is convexe or not
    """

    hessian_obj_list = []
    phases_dt = ocp.dt_parameter.cx
    for phase, nlp in enumerate(ocp.nlp):
        for obj in nlp.J:
            objective = 0

            node_index = obj.node_idx[0]  # TODO deal with phase_dynamics=PhaseDynamics.ONE_PER_NODE
            nlp.states.node_index = node_index
            nlp.states_dot.node_index = node_index
            nlp.controls.node_index = node_index
            nlp.algebraic_states.node_index = node_index

            if obj.multinode_penalty:
                n_phases = ocp.n_phases
                for phase_idx in obj.nodes_phase:
                    controllers.append(obj.get_penalty_controller(ocp, ocp.nlp[phase_idx % n_phases]))
            else:
                controllers = [obj.get_penalty_controller(ocp, nlp)]

            for controller in controllers:
                controller.node_index = obj.node_idx[0]
            _, t0, x, u, p, a = obj.get_variable_inputs(controllers)

            target = PenaltyHelpers.target(obj, obj.node_idx.index(node_index))

            penalty = obj.weighted_function[node_index](t0, phases_dt, x, u, p, a, obj.weight, target)

            for i in range(penalty.shape[0]):
                objective += penalty[i] ** 2

        # create function to build the hessian
        vertcat_obj = vertcat([], *nlp.X_scaled, *nlp.U_scaled)  # time, states, controls
        if nlp.parameters.shape == 0:
            vertcat_obj = vertcat(vertcat_obj, nlp.parameters.cx)
        else:
            vertcat_obj = vertcat(vertcat_obj, *[nlp.parameters.cx])
        if vertcat(*nlp.A_scaled).shape[0] > 0:
            vertcat_obj = vertcat(vertcat_obj, *nlp.A_scaled)

        hessian_cas = hessian(objective, vertcat_obj)[0]

        hes_func = Function(
            "hessian",
            [vertcat_obj],
            [hessian_cas],
        )

        nb_x_init = sum([nlp.x_init[key].shape[0] for key in nlp.x_init.keys()])
        nb_u_init = sum([nlp.u_init[key].shape[0] for key in nlp.u_init.keys()])
        nb_a_init = sum([nlp.a_init[key].shape[0] for key in nlp.a_init.keys()])

        # evaluate jac_func at X_init, U_init, considering the parameters
        time_init = np.array([], dtype=np.float64)
        x_init = np.zeros((len(nlp.X), nb_x_init))
        u_init = np.zeros((len(nlp.U), nb_u_init))
        param_init = np.array([nlp.x_init[key].shape[0] for key in ocp.parameter_init.keys()])
        a_init = np.zeros((len(nlp.A), nb_a_init))

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
        for key in nlp.algebraic_states.keys():
            nlp.a_init[key].check_and_adjust_dimensions(len(nlp.algebraic_states[key]), nlp.ns)
            for node_index in range(nlp.ns):
                nlp.algebraic_states.node_index = node_index
                a_init[node_index, nlp.algebraic_states[key].index] = np.array(
                    nlp.a_init[key].init.evaluate_at(node_index)
                )

        time_init = time_init.reshape((time_init.size, 1))
        x_init = x_init.reshape((x_init.size, 1))
        u_init = u_init.reshape((u_init.size, 1))
        param_init = param_init.reshape((param_init.size, 1))
        a_init = a_init.reshape((a_init.size, 1))
        vector_init = np.vstack((time_init, x_init, u_init, param_init, a_init))

        hessian_obj_matrix = np.array(hes_func(vector_init))
        hessian_obj_list.append(hessian_obj_matrix)

    # Convexity checking (positive semi-definite hessian)
    # On R (convex), the objective is convex if and only if the hessian is positive semi definite (psd)
    # And, as the hessian is symmetric (Schwarz), the hessian is psd if and only if the eigenvalues are positive
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
        convexity.append("positive semi-definite" if np.all(eigen_values > 0) else "not positive semi-definite")

    return hessian_obj_list, condition_number, convexity


def create_conditioning_plots(ocp):
    
    cmap = mcm.get_cmap("seismic")
    # cmap.set_bad(color="k")
    
    # PLOT CONSTRAINTS
    fig_constraints, axis_constraints = plt.subplots(1, 2 * ocp.n_phases, num="Check conditioning for constraints")
    tick_labels_list = []
    for i_phase, nlp in enumerate(ocp.nlp):
        nb_constraints = len(nlp.g) + len(nlp.g_internal)
        # Jacobian plot
        fake_jacobian = np.zeros((nb_constraints, nlp.nx))
        im = axis_constraints[i_phase].imshow(fake_jacobian, aspect="auto", cmap=cmap)
        axis_constraints[i_phase].set_title("Jacobian constraints \n Phase " + str(i_phase) + "\nMatrix rank = NA \n Number of constraints = NA", fontweight="bold", fontsize=12)
        # colorbar
        cbar_ax = fig_constraints.add_axes([0.02, 0.4, 0.015, 0.3])
        fig_constraints.colorbar(im, cax=cbar_ax)

        # Hessian constraints plot
        fake_hessian = np.zeros((nlp.g.shape[0], nlp.nx))
        yticks = []
        tick_labels = []
        for i_g, g in enumerate(nlp.g):
            yticks.append(i_g)
            tick_labels.append(g.name)
        tick_labels_list.append(tick_labels)

        im2 = axis_constraints[i_phase + ocp.n_phases].imshow(fake_hessian, aspect="auto", cmap=cmap)
        axis_constraints[i_phase + ocp.n_phases].set_title(
            "Hessian constraint norms (Norms should be close to 0) \n Phase " + str(i_phase), fontweight="bold", fontsize=12
        )
        axis_constraints[i_phase + ocp.n_phases].set_yticks(yticks)
        axis_constraints[i_phase + ocp.n_phases].set_yticklabels(tick_labels, fontsize=6, rotation=90)
        # colobar
        cbar_ax2 = fig_constraints.add_axes([0.95, 0.4, 0.015, 0.3])
        fig_constraints.colorbar(im2, cax=cbar_ax2)

    fig_constraints.legend(["Black = 0"], loc="upper left")
    plt.suptitle("The rank should be equal to the number of constraints", color="b", fontsize=15, fontweight="bold")
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


    # PLOT OBJECTIVES
    fig_obj, axis_obj = plt.subplots(1, ocp.n_phases, num="Check conditioning for objectives")
    if ocp.n_phases == 1:
        axis_obj = [axis_obj]
    for i_phase, nlp in enumerate(ocp.nlp):

        # Hessian objective plot
        fake_hessian_obj = np.zeros((nlp.J.shape[0], nlp.nx))
        im3 = axis_obj[i_phase].imshow(fake_hessian_obj, cmap=cmap)
        axis_obj[i_phase].set_title("Hessian objective \n Phase " + str(i_phase) + "\nConvexity = NA \n|λmax|/|λmin| = Condition number = NA", fontweight="bold", fontsize=12)
        # colobar
        cbar_ax3 = fig_obj.add_axes([0.02, 0.4, 0.015, 0.3])
        fig_obj.colorbar(im3, cax=cbar_ax3)

    fig_obj.legend(["Black = 0"], loc="upper left")
    plt.suptitle("Every hessian should be convexe \n Condition numbers should be close to 0", color="b", fontsize=15,
                 fontweight="bold")
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    ocp.conditioning_plots = {
        "axis_constraints": axis_constraints,
        "axis_obj": axis_obj,
    }

    
def update_constraints_plot(v, ocp):

    # --------------------------------------------------------------------------------------------
    jacobian_list, jacobian_rank, hessian_norm_list = jacobian_hessian_constraints()
    axis_constraints = ocp.conditioning_plots["axis_constraints"]
    cmap = mcm.get_cmap("seismic")

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

    for i_phase, nlp in enumerate(ocp.nlp):
        # Jacobian plot
        jacobian_list[i_phase][(jacobian_list[i_phase] == 0)] = np.nan
        norm = mcolors.TwoSlopeNorm(vmin=min_jac - 0.01, vmax=max_jac + 0.01, vcenter=0)
        axis_constraints[i_phase].set_data(jacobian_list[i_phase], aspect="auto", cmap=cmap, norm=norm)
        axis_constraints[i_phase].set_title("Jacobian constraints \n Phase " + str(i_phase) + "\nMatrix rank = " + str(
            jacobian_rank[i_phase]) + "\n Number of constraints = " + str(jacobian_list[i_phase].shape[1]),
                                            fontweight="bold",
                                            fontsize=12)

        # Hessian constraints plot
        hessian_norm_list[i_phase][~(hessian_norm_list[i_phase] != 0).astype(bool)] = np.nan
        norm_squared = mcolors.TwoSlopeNorm(vmin=min_norm - 0.01, vmax=max_norm + 0.01, vcenter=0)
        axis_constraints[i_phase + ocp.n_phases].set_data(hessian_norm_list[i_phase], aspect="auto", cmap=cmap,
                                                              norm=norm_squared)

    
def update_objective_plot(v, ocp):

    hessian_obj_list, condition_number, convexity = hessian_objective()
    axis_obj = ocp.conditioning_plots["axis_obj"]
    cmap = mcm.get_cmap("seismic")

    max_hes = []
    min_hes = []
    for hessian_matrix_obj in hessian_obj_list:
        max_hes.append(np.ndarray.max(hessian_matrix_obj))
        min_hes.append(np.ndarray.min(hessian_matrix_obj))
    min_hes = min(min_hes)
    max_hes = max(max_hes)

    for i_phase, nlp in enumerate(ocp.nlp):
        # Hessian objective plot
        norm = mcolors.TwoSlopeNorm(vmin=min_hes - 0.01, vmax=max_hes + 0.01, vcenter=0)
        axis_obj[i_phase].set_data(hessian_obj_list[i_phase], cmap=cmap, norm=norm)


def check_conditioning(ocp):
    """
    Visualisation of jacobian and hessian contraints and hessian objective for each phase at initial time
    """

    create_conditioning_plots(ocp)
    update_constraints_plot(v, ocp)
    update_objective_plot(v, ocp)

    plt.show()
