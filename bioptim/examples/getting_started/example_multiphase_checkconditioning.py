"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement and
a the at different marker at the end of each phase. Moreover a constraint on the rotation is imposed on the cube.
Finally, an objective for the transition continuity on the control is added. Please note that the "last" control
of the previous phase is the last shooting node (and not the node arrival).
It is designed to show how one can define a multiphase optimal control program
"""

import casadi as cas
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    PenaltyNode,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Node,
    Solver,
    CostType,
    PlotType,
)

def minimize_difference(all_pn: PenaltyNode):
    return all_pn[0].nlp.controls.cx_end - all_pn[1].nlp.controls.cx

def prepare_ocp(
    biorbd_model_path: str = "models/cube.bioMod", ode_solver: OdeSolver = OdeSolver.RK4(), long_optim: bool = False
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use
    long_optim: bool
        If the solver should solve the precise optimization (500 shooting points) or the approximate (50 points)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Problem parameters
    if long_optim:
        n_shooting = (100, 300, 100)
    else:
        n_shooting = (20, 30, 20)
    final_time = (2, 5, 4)
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)
    objective_functions.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=100,
        phase=1,
        quadratic=True,
    )

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=0)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m1", phase=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2", phase=2)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    for bounds in x_bounds:
        for i in [1, 3, 4, 5]:
            bounds[i, [0, -1]] = 0
    x_bounds[0][2, 0] = 0.0
    x_bounds[2][2, [0, -1]] = [0.0, 1.57]

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """

    ocp = prepare_ocp(long_optim=False)
    ocp.add_plot_penalty(CostType.ALL)

    """"##########################################
    ####verif rang jacobienne instant init####
    ##########################################"""


    #calculate all jacobian constraints and all hessian matrix norms
    jacobian_list = []
    jacobian_rank = []
    tick_labels_list = []
    hessian_norm_list = []
    for phase in range (0,len(ocp.nlp)):
        jacobian_cas = cas.MX()
        list_constraints = []
        for i in range (0,len(ocp.nlp[phase].g)):
            for axis in range (0, ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx, ocp.nlp[phase].parameters.cx).shape[0]):

                #depends if there are parameters
                if (ocp.nlp[phase].parameters.shape == 0) == True :
                    list_constraints.append(cas.jacobian(
                        ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx, ocp.nlp[phase].parameters.cx)[axis],
                        cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, ocp.nlp[phase].parameters.cx)))
                else:
                    list_constraints.append(cas.jacobian(
                            ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx,
                            ocp.nlp[phase].parameters.cx)[axis],cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, *[ocp.nlp[phase].parameters.cx])))


        jacobian_cas = cas.vcat(list_constraints).T

        #depends if there are parameters
        if (ocp.nlp[phase].parameters.shape == 0) == True:
            jac_func = cas.Function("jacobian", [cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, ocp.nlp[phase].parameters.cx)], [jacobian_cas])
        else:
            jac_func = cas.Function("jacobian",[cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, *[ocp.nlp[phase].parameters.cx])],[jacobian_cas])

        #evaluate jac_func at X_init, U_init, considering the parameters

        X_init = np.zeros((len(ocp.nlp[phase].X), ocp.nlp[phase].x_init.shape[0]))
        U_init = np.zeros((len(ocp.nlp[phase].U), ocp.nlp[phase].u_init.shape[0]))
        Param_init = np.array(ocp.nlp[phase].parameters.initial_guess.init)

        for n_shooting in range (0,ocp.nlp[phase].ns+1):
            X_init[n_shooting, :] = np.array(ocp.nlp[phase].x_init.init.evaluate_at(n_shooting))
        for n_shooting in range (0,ocp.nlp[phase].ns):
            U_init[n_shooting, :] = np.array(ocp.nlp[phase].u_init.init.evaluate_at(n_shooting))

        X_init = X_init.reshape((X_init.size, 1))
        U_init = U_init.reshape((U_init.size, 1))
        Param_init = Param_init.reshape((np.array(ocp.nlp[phase].parameters.initial_guess.init).size, 1))

        jacobian_matrix = np.array(jac_func(np.vstack((X_init, U_init, Param_init))))

        jacobian_matrix[15][2] = 20 ############################################################################################TEST

        jacobian_list.append(jacobian_matrix)

        #verification jacobian rank
        rank = np.linalg.matrix_rank(jacobian_matrix)
        jacobian_rank.append(rank)
        # if rank == len(ocp.nlp[phase].g): ############a corriger
        #     print('Phase ' + str(phase) + ' : constraints ok')
        # if rank != len(ocp.nlp[phase].g): ############a corriger
        #     print('Phase ' + str(phase) + ' : constraints not ok')


        """"###############################################################
        ##verif contrainte egalité linéaire si norme hessian proche de 0##
        #################################################################"""
        tick_labels = []
        list_hessian = []
        list_norm = []
        for i in range (0,len(ocp.nlp[phase].g)):
            for axis in range (0, ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx, ocp.nlp[phase].parameters.cx).shape[0]):

                #find all equality constraints
                if (ocp.nlp[phase].g[i].bounds.min[axis][0] == ocp.nlp[phase].g[i].bounds.max[axis][0]) == True:

                    #parameters
                    if (ocp.nlp[phase].parameters.shape == 0) == True :
                        hessienne_cas = cas.hessian(
                            ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx, ocp.nlp[phase].parameters.cx)[axis],
                            cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, ocp.nlp[phase].parameters.cx))[0]
                    else:
                        hessienne_cas = cas.hessian(
                                ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx,
                                ocp.nlp[phase].parameters.cx)[axis], cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, *[ocp.nlp[phase].parameters.cx]))[0]
                    tick_labels.append(ocp.nlp[phase].g[i].name)

                    #parameters
                    if (ocp.nlp[phase].parameters.shape == 0) == True:
                        hes_func = cas.Function("hessian", [
                            cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, ocp.nlp[phase].parameters.cx)],
                                                [hessienne_cas])
                    else:
                        hes_func = cas.Function("hessian", [
                            cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, *[ocp.nlp[phase].parameters.cx])],
                                                [hessienne_cas])

                    # evaluate hes_func en X_init, U_init, with parameters

                    X_init = np.zeros((len(ocp.nlp[phase].X), ocp.nlp[phase].x_init.shape[0]))
                    U_init = np.zeros((len(ocp.nlp[phase].U), ocp.nlp[phase].u_init.shape[0]))
                    Param_init = np.array(ocp.nlp[phase].parameters.initial_guess.init)

                    for n_shooting in range(0, ocp.nlp[phase].ns + 1):
                        X_init[n_shooting, :] = np.array(ocp.nlp[phase].x_init.init.evaluate_at(n_shooting))
                    for n_shooting in range(0, ocp.nlp[phase].ns):
                        U_init[n_shooting, :] = np.array(ocp.nlp[phase].u_init.init.evaluate_at(n_shooting))

                    X_init = X_init.reshape((X_init.size, 1))
                    U_init = U_init.reshape((U_init.size, 1))
                    Param_init = Param_init.reshape((np.array(ocp.nlp[phase].parameters.initial_guess.init).size, 1))

                    hessian = np.array(hes_func(np.vstack((X_init, U_init, Param_init))))

                    #on met la hessienne de cote
                    list_hessian.append(hessian)

                else:
                    do = 'nothing'

        tick_labels_list.append(tick_labels)

        # calcul norm
        for nb_hessian in range(0, len(list_hessian)):
            norm = 0
            for row in range(list_hessian[nb_hessian].shape[0]):
                for column in range(list_hessian[nb_hessian].shape[1]):
                    norm += list_hessian[nb_hessian][row, column] ** 2

            list_norm.append(norm)
        array_norm = np.array(list_norm).reshape(len(list_hessian), 1)

        array_norm[0][0] = 65 ##################################################################################################TEST

        hessian_norm_list.append(array_norm)

    hessian_norm_list[0][2][0] = 80 #################################################################################################
    jacobian_list[0][50][3] = 50#####################################################################################################
    jacobian_list[0][78][2] = 0.2



    max_norm = []
    min_norm = []
    for i in range (0,len(hessian_norm_list)):
        max_norm.append(np.ndarray.max(hessian_norm_list[i]))
        min_norm.append(np.ndarray.min(hessian_norm_list[i]))
    max_jac = []
    min_jac = []
    for i in range(0, len(jacobian_list)):
        max_jac.append(np.ndarray.max(jacobian_list[i]))
        min_jac.append(np.ndarray.min(jacobian_list[i]))

    min_norm = min(min_norm)
    max_norm = max(max_norm)
    max_jac = max(max_jac)
    min_jac = min(min_jac)

    #PLOT GENERAL
    fig, axis = plt.subplots(1, 2*len(ocp.nlp))
    for ax in range (0, len(ocp.nlp)):
        # Jacobian plot
        jacobian_list[ax][~(jacobian_list[ax] != 0).astype(bool)] = np.nan
        cmap = plt.cm.seismic
        current_cmap = mcm.get_cmap('seismic')
        current_cmap.set_bad(color='k')
        norm = mcolors.TwoSlopeNorm(vmin=min_jac-0.1, vmax=max_jac, vcenter=0)
        im = axis[ax].imshow(jacobian_list[ax], aspect='auto', cmap=current_cmap, norm=norm)#, vmin=min_jac, vmax=max_jac,
        axis[ax].set_title('Jacobian constraints \n Phase ' + str(ax), fontweight='bold', fontsize=8)
        axis[ax].text(0, jacobian_list[ax].shape[0]*1.08, 'Matrix rank = ' + str(jacobian_rank[ax]) + '\n Number of constraints = ' + str(len(hessian_norm_list[ax])), horizontalalignment='center', fontweight='bold', fontsize=8)
        axis[ax].text(0, jacobian_list[ax].shape[0] * 1.1, 'The rank should be equal to the number of constraints',horizontalalignment='center', fontsize=6)

        #Hessian constraints plot
        hessian_norm_list[ax][~(hessian_norm_list[ax] != 0).astype(bool)] = np.nan
        cmap2 = plt.cm.seismic
        current_cmap2 = mcm.get_cmap('seismic')
        current_cmap2.set_bad(color='k')
        norm2 = mcolors.TwoSlopeNorm(vmin=min_norm - 0.1, vmax=max_norm, vcenter=0)
        liste_ytick = []
        for i in range (0,len(hessian_norm_list[ax])):
            liste_ytick.append(i)

        im2 = axis[ax+len(ocp.nlp)].imshow(hessian_norm_list[ax], aspect='auto', cmap=current_cmap2, norm=norm2)#, vmin=min_norm, vmax=max_norm)
        axis[ax + len(ocp.nlp)].set_title('Hessian constraint norms \n Phase ' + str(ax), fontweight='bold', fontsize=8)
        axis[ax + len(ocp.nlp)].set_xticks([0])
        axis[ax + len(ocp.nlp)].set_xticklabels(['Norms should be close to 0'], fontsize=8)
        axis[ax + len(ocp.nlp)].set_yticks(liste_ytick)
        axis[ax + len(ocp.nlp)].set_yticklabels(tick_labels_list[ax], fontsize=6, rotation=90)

    cbar_ax = fig.add_axes([0.02, 0.4, 0.015, 0.3])
    cbar_ax2 = fig.add_axes([0.95, 0.4, 0.015, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    fig.colorbar(im2, cax=cbar_ax2)
    plt.suptitle('Check conditioning for constraints ', color='b', fontsize=15, fontweight='bold')
    plt.show()


    ################################################
    """"#########################################
    ####verif objectif convexe avec hessienne####
    ##########################################"""
    # for phase in range(0, len(ocp.nlp)):
    #     hessienne_obj = 0
    #     ocp.nlp[0].J[0].function_non_threaded(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx, ocp.nlp[phase].parameters.cx)

    # plt.subplot(1, 5, 5)
    # plt.imshow(np.diag(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])))
    # plt.xlabel('H ⪰ 0 : oui/non')
    # plt.title('Hessian objective', pad=15)
    # plt.text(5, 13, 'Hessienne des objectifs', horizontalalignment='center', fontweight='bold')
    # plt.text(5, 14, 'Si ⪰ 0, probleme convexe', horizontalalignment='center')
    # '\n λmax/λmin = , le conditionement doit etre proche de 0'

    ################################################
    ################################################

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
