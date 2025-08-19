import numpy as np
from casadi import Function, jacobian, hessian, sum1
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from ..interfaces.ipopt_interface import IpoptInterface


def jacobian_hessian_constraints(variables_vector, all_g):
    """
    Returns
    -------
    A list with jacobian matrix of constraints evaluates at initial time for each phase
    A list with the rank of each jacobian matrix
    A list with the different type of constraints
    A list with norms of hessian matrix of constraints at initial time for each phase
    """

    # JACOBIAN
    constraints_jac_func = Function(
        "constraints_jacobian",
        [variables_vector],
        [jacobian(all_g, variables_vector)],
    )

    # HESSIAN
    constraints_hess_func = []
    for i in range(all_g.shape[0]):
        constraints_hess_func.append(
            Function(
                "constraints_hessian",
                [variables_vector],
                [hessian(all_g[i], variables_vector)[0]],
            )
        )

    return constraints_jac_func, constraints_hess_func


def evaluate_jacobian_hessian_constraints(v, ocp):

    # JACOBIAN
    constraints_jac_func = ocp.conditioning_plots["constraints_jac_func"]
    evaluated_constraints_jacobian = constraints_jac_func(v)
    jacobian_matrix = np.array(evaluated_constraints_jacobian)

    # Jacobian rank
    if jacobian_matrix.size > 0:
        jacobian_rank = np.linalg.matrix_rank(jacobian_matrix)
    else:
        jacobian_rank = "No constraints"

    # HESSIAN
    constraints_hess_func = ocp.conditioning_plots["constraints_hess_func"]
    hess_min_mean_max = np.zeros((len(constraints_hess_func), 3))
    for i in range(len(constraints_hess_func)):
        evaluated_constraints_hessian = constraints_hess_func[i](v)
        hess_min_mean_max[i, 0] = np.min(evaluated_constraints_hessian)
        hess_min_mean_max[i, 1] = np.mean(evaluated_constraints_hessian)
        hess_min_mean_max[i, 2] = np.max(evaluated_constraints_hessian)

    return jacobian_matrix, jacobian_rank, hess_min_mean_max


def hessian_objective(variables_vector, all_objectives):
    """

    Returns
    -------
    A list with the hessian of objectives evaluate at initial time for each phase
    A list with the condition numbers of each phases
    A list that indicates if the objective is convexe or not
    """

    objectives_hess_func = Function(
        "hessian",
        [variables_vector],
        [hessian(sum1(all_objectives), variables_vector)[0]],
    )

    return objectives_hess_func


def evaluate_hessian_objective(v, ocp):
    """

    Returns
    -------
    A list with the hessian of objectives evaluate at initial time for each phase
    A list with the condition numbers of each phases
    A list that indicates if the objective is convexe or not
    """

    objectives_hess_func = ocp.conditioning_plots["objectives_hess_func"]
    evaluated_objectives_hessian = objectives_hess_func(v)
    hessian_matrix = np.array(evaluated_objectives_hessian)

    # Convexity checking (positive semi-definite hessian)
    # On R (convex), the objective is convex if and only if the hessian is positive semi definite (psd)
    # And, as the hessian is symmetric (Schwarz), the hessian is psd if and only if the eigenvalues are positive
    eigen_values = np.linalg.eigvals(hessian_matrix)
    ev_max = np.max(eigen_values)
    ev_min = np.min(eigen_values)
    if ev_min == 0:
        condition_number = "/!\\ min eigen value is 0"
    if ev_min != 0:
        condition_number = np.abs(ev_max) / np.abs(ev_min)
    convexity = (
        "positive semi-definite"
        if np.all(eigen_values > 0)
        else f"not positive semi-definite (min: {ev_min}, max: {ev_max})"
    )

    return hessian_matrix, condition_number, convexity


def create_conditioning_plots(ocp):

    cmap = mcm.get_cmap("seismic")
    cmap.set_bad(color="k")
    interface = IpoptInterface(ocp)
    variables_vector = ocp.variables_vector
    all_g, _ = interface.dispatch_bounds(include_g=True, include_g_internal=False)
    all_objectives = interface.dispatch_obj_func()
    nb_variables = variables_vector.shape[0]
    nb_constraints = all_g.shape[0]

    constraints_jac_func, constraints_hess_func = jacobian_hessian_constraints(variables_vector, all_g)
    objectives_hess_func = hessian_objective(variables_vector, all_objectives)

    # PLOT CONSTRAINTS
    fig_constraints, axis_constraints = plt.subplots(1, 2, num="Check conditioning for constraints")

    # Jacobian plot
    fake_jacobian = np.zeros((nb_constraints, nb_variables))
    im_constraints_jacobian = axis_constraints[0].imshow(fake_jacobian, aspect="auto", cmap=cmap)
    axis_constraints[0].set_title(
        "Jacobian constraints \nMatrix rank = NA \n Number of constraints = NA", fontweight="bold", fontsize=12
    )
    # colorbar
    cbar_ax = fig_constraints.add_axes([0.02, 0.4, 0.015, 0.3])
    fig_constraints.colorbar(im_constraints_jacobian, cax=cbar_ax)

    # Hessian constraints plot
    fake_hessian = np.zeros((nb_constraints, 3))
    im_constraints_hessian = axis_constraints[1].imshow(fake_hessian, aspect="auto", cmap=cmap)
    axis_constraints[1].set_title(
        "Hessian constraint norms (Norms should be close to 0)", fontweight="bold", fontsize=12
    )
    axis_constraints[1].set_xticks([0, 1, 2])
    axis_constraints[1].set_xticklabels(["Min", "Mean", "Max"])
    # colobar
    cbar_ax2 = fig_constraints.add_axes([0.95, 0.4, 0.015, 0.3])
    fig_constraints.colorbar(im_constraints_hessian, cax=cbar_ax2)

    fig_constraints.legend(["Black = 0"], loc="upper left")
    plt.suptitle("The rank should be equal to the number of constraints", color="b", fontsize=15, fontweight="bold")
    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    except:
        pass

    # PLOT OBJECTIVES
    fig_obj, axis_obj = plt.subplots(1, 1, num="Check conditioning for objectives")

    # Hessian objective plot
    fake_hessian_obj = np.zeros((nb_variables, nb_variables))
    im_objectives_hessian = axis_obj.imshow(fake_hessian_obj, cmap=cmap)
    axis_obj.set_title("Convexity = NA \n|λmax|/|λmin| = Condition number = NA", fontweight="bold", fontsize=12)
    axis_obj.set_xlabel("Hessian objectives")
    # colobar
    cbar_ax3 = fig_obj.add_axes([0.02, 0.4, 0.015, 0.3])
    fig_obj.colorbar(im_objectives_hessian, cax=cbar_ax3)

    fig_obj.legend(["Black = 0"], loc="upper left")
    plt.suptitle(
        "Every hessian should be convexe (positive) and Condition numbers should be close to 0",
        color="b",
        fontsize=15,
        fontweight="bold",
    )
    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    except:
        pass

    ocp.conditioning_plots = {
        "axis_constraints": axis_constraints,
        "axis_obj": axis_obj,
        "im_constraints_jacobian": im_constraints_jacobian,
        "im_constraints_hessian": im_constraints_hessian,
        "im_objectives_hessian": im_objectives_hessian,
        "constraints_jac_func": constraints_jac_func,
        "constraints_hess_func": constraints_hess_func,
        "objectives_hess_func": objectives_hess_func,
    }


def update_constraints_plot(v, ocp):

    jacobian_matrix, jacobian_rank, hess_min_mean_max = evaluate_jacobian_hessian_constraints(v, ocp)
    axis_constraints = ocp.conditioning_plots["axis_constraints"]
    im_constraints_jacobian = ocp.conditioning_plots["im_constraints_jacobian"]
    im_constraints_hessian = ocp.conditioning_plots["im_constraints_hessian"]

    # Jacobian plot
    jacobian_matrix[jacobian_matrix == 0] = np.nan
    jac_min = np.min(jacobian_matrix) if jacobian_matrix.shape[0] != 0 else 0
    jac_max = np.max(jacobian_matrix) if jacobian_matrix.shape[0] != 0 else 0
    norm = mcolors.TwoSlopeNorm(vmin=jac_min - 0.01, vmax=jac_max + 0.01, vcenter=0)
    im_constraints_jacobian.set_data(jacobian_matrix)
    im_constraints_jacobian.set_norm(norm)
    axis_constraints[0].set_title(
        f"Jacobian constraints \nMatrix rank = {str(jacobian_rank)}\n Number of constraints = {str(jacobian_matrix.shape[0])}",
        fontweight="bold",
        fontsize=12,
    )

    # Hessian constraints plot
    hess_min = np.min(hess_min_mean_max) if hess_min_mean_max.shape[0] != 0 else 0
    hess_max = np.max(hess_min_mean_max) if hess_min_mean_max.shape[0] != 0 else 0
    norm = mcolors.TwoSlopeNorm(vmin=hess_min - 0.01, vmax=hess_max + 0.01, vcenter=0)
    im_constraints_hessian.set_data(hess_min_mean_max)
    im_constraints_hessian.set_norm(norm)


def update_objective_plot(v, ocp):

    hessian_matrix, condition_number, convexity = evaluate_hessian_objective(v, ocp)
    axis_obj = ocp.conditioning_plots["axis_obj"]
    im_objectives_hessian = ocp.conditioning_plots["im_objectives_hessian"]
    cmap = mcm.get_cmap("seismic")

    # Hessian objective plot
    hess_min = np.min(hessian_matrix) if hessian_matrix.shape[0] != 0 else 0
    hess_max = np.max(hessian_matrix) if hessian_matrix.shape[0] != 0 else 0
    norm = mcolors.TwoSlopeNorm(vmin=hess_min - 0.01, vmax=hess_max + 0.01, vcenter=0)
    im_objectives_hessian.set_data(hessian_matrix)
    im_objectives_hessian.set_norm(norm)
    axis_obj.set_title(
        f"Hessian objective \nConvexity = {convexity} \n|λmax|/|λmin| = Condition number = {condition_number}",
        fontweight="bold",
        fontsize=12,
    )


def update_conditioning_plots(v, ocp):
    update_constraints_plot(v, ocp)
    update_objective_plot(v, ocp)
    plt.draw()


def check_conditioning(ocp):
    """
    Visualisation of jacobian and hessian contraints and hessian objective for each phase at initial time
    """

    create_conditioning_plots(ocp)
    v_init = ocp.init_vector
    update_constraints_plot(v_init, ocp)
    update_objective_plot(v_init, ocp)

    plt.show()
