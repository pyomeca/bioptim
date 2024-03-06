import numpy as np
from casadi import MX, SX, Function, horzcat, vertcat, jacobian, vcat, hessian
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from ..misc.enums import ControlType
from ..misc.enums import QuadratureRule
from ..dynamics.ode_solver import OdeSolver
from ..limits.penalty_helpers import PenaltyHelpers
from ..interfaces.ipopt_interface import IpoptInterface


def jacobian_hessian_constraints(v, ocp):
    """
    Returns
    -------
    A list with jacobian matrix of constraints evaluates at initial time for each phase
    A list with the rank of each jacobian matrix
    A list with the different type of constraints
    A list with norms of hessian matrix of constraints at initial time for each phase
    """

    interface = IpoptInterface(ocp)
    variables_vector = ocp.variables_vector
    all_g, _ = interface.dispatch_bounds()

    # JACOBIAN
    constraints_jac_func = Function(
        "constraints_jacobian",
        [variables_vector],
        [jacobian(all_g, variables_vector)],
    )
    evaluated_constraints_jacobian = constraints_jac_func(v)
    jacobian_matrix = np.array(evaluated_constraints_jacobian)

    # Jacobian rank
    if jacobian_matrix.size > 0:
        jacobian_rank = np.linalg.matrix_rank(jacobian_matrix)
    else:
        jacobian_rank = "No constraints"


    # HESSIAN
    constraints_hess_func = Function(
        "constraints_hessian",
        [variables_vector],
        [hessian(all_g, variables_vector)],
    )
    evaluated_constraints_hessian = constraints_hess_func(v)
    hessian_matrix = np.array(evaluated_constraints_hessian)

    # Hessian norm
    hessian_norm = np.linalg.norm(hessian_matrix)

    return jacobian_matrix, jacobian_rank, hessian_matrix, hessian_norm



def hessian_objective(v, ocp):
    """

    Returns
    -------
    A list with the hessian of objectives evaluate at initial time for each phase
    A list with the condition numbers of each phases
    A list that indicates if the objective is convexe or not
    """

    interface = IpoptInterface(ocp)
    variables_vector = ocp.variables_vector
    all_objectives = interface.dispatch_obj_func()

    objectives_hess_func = Function(
        "hessian",
        [variables_vector],
        [hessian(all_objectives, variables_vector)],
    )
    evaluated_objectives_hessian = objectives_hess_func(v)
    hessian_matrix = np.array(evaluated_objectives_hessian)

    # Convexity checking (positive semi-definite hessian)
    # On R (convex), the objective is convex if and only if the hessian is positive semi definite (psd)
    # And, as the hessian is symmetric (Schwarz), the hessian is psd if and only if the eigenvalues are positive
    eigen_values = np.linalg.eigvals(hessian_matrix)
    ev_max = min(eigen_values)
    ev_min = max(eigen_values)
    if ev_min == 0:
        condition_number = " /!\ Ev_min is 0"
    if ev_min != 0:
        condition_number = np.abs(ev_max) / np.abs(ev_min)
    convexity = "positive semi-definite" if np.all(eigen_values > 0) else "not positive semi-definite"

    return hessian_matrix, condition_number, convexity


def create_conditioning_plots(ocp):
    
    cmap = mcm.get_cmap("seismic")
    # cmap.set_bad(color="k")
    interface = IpoptInterface(ocp)
    variables_vector = ocp.variables_vector
    all_g, _ = interface.dispatch_bounds()
    all_objectives = interface.dispatch_obj_func()
    nb_variables = variables_vector.shape[0]
    nb_constraints = all_g.shape[0]
    nb_obj = all_objectives.shape[0]
    
    # PLOT CONSTRAINTS
    fig_constraints, axis_constraints = plt.subplots(1, 2, num="Check conditioning for constraints")

    # Jacobian plot
    fake_jacobian = np.zeros((nb_constraints, nb_variables))
    im = axis_constraints[0].imshow(fake_jacobian, aspect="auto", cmap=cmap)
    axis_constraints[0].set_title("Jacobian constraints \nMatrix rank = NA \n Number of constraints = NA", fontweight="bold", fontsize=12)
    # colorbar
    cbar_ax = fig_constraints.add_axes([0.02, 0.4, 0.015, 0.3])
    fig_constraints.colorbar(im, cax=cbar_ax)

    # Hessian constraints plot
    fake_hessian = np.zeros((nb_constraints, nb_variables))

    im2 = axis_constraints[1].imshow(fake_hessian, aspect="auto", cmap=cmap)
    axis_constraints[1].set_title(
        "Hessian constraint norms (Norms should be close to 0)", fontweight="bold", fontsize=12
    )
    # colobar
    cbar_ax2 = fig_constraints.add_axes([0.95, 0.4, 0.015, 0.3])
    fig_constraints.colorbar(im2, cax=cbar_ax2)

    fig_constraints.legend(["Black = 0"], loc="upper left")
    plt.suptitle("The rank should be equal to the number of constraints", color="b", fontsize=15, fontweight="bold")
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


    # PLOT OBJECTIVES
    fig_obj, axis_obj = plt.subplots(1, 1, num="Check conditioning for objectives")

    # Hessian objective plot
    fake_hessian_obj = np.zeros((nb_obj, nb_variables))
    im3 = axis_obj.imshow(fake_hessian_obj, cmap=cmap)
    axis_obj.set_title("Hessian objective \nConvexity = NA \n|λmax|/|λmin| = Condition number = NA", fontweight="bold", fontsize=12)
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

    jacobian_matrix, jacobian_rank, hessian_matrix, hessian_norm = jacobian_hessian_constraints(v, ocp)
    axis_constraints = ocp.conditioning_plots["axis_constraints"]
    cmap = mcm.get_cmap("seismic")

    # Jacobian plot
    jacobian_matrix[jacobian_matrix == 0] = np.nan
    norm = mcolors.TwoSlopeNorm(vmin=np.min(jacobian_matrix) - 0.01, vmax=np.max(jacobian_matrix) + 0.01, vcenter=0)
    axis_constraints[0].set_data(jacobian_matrix, aspect="auto", cmap=cmap, norm=norm)
    axis_constraints[0].set_title("Jacobian constraints \nMatrix rank = " + str(
        jacobian_rank) + "\n Number of constraints = " + str(jacobian_matrix.shape[1]),
                                        fontweight="bold",
                                        fontsize=12)

    # Hessian constraints plot
    hessian_norm[hessian_norm == 0] = np.nan
    norm = mcolors.TwoSlopeNorm(vmin=np.min(hessian_norm) - 0.01, vmax=np.max(hessian_matrix) + 0.01, vcenter=0)
    axis_constraints[1].set_data(hessian_norm, aspect="auto", cmap=cmap, norm=norm)

    
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
    v_init = ocp.init_vector
    update_constraints_plot(v_init, ocp)
    update_objective_plot(v_init, ocp)

    plt.show()
