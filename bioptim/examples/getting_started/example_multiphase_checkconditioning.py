"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement and
a the at different marker at the end of each phase. Moreover a constraint on the rotation is imposed on the cube.
Finally, an objective for the transition continuity on the control is added. Please note that the "last" control
of the previous phase is the last shooting node (and not the node arrival).
It is designed to show how one can define a multiphase optimal control program
"""

import casadi as cas
import matplotlib.pyplot as plt
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

    for phase in range (0,len(ocp.nlp)):
        jacobienne_cas = cas.MX()
        liste_contrainte = []
        for i in range (0,len(ocp.nlp[phase].g)):
            for axe in range (0, ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx, ocp.nlp[phase].parameters.cx).shape[0]):

                #gerer les parametres
                if (ocp.nlp[phase].parameters.shape == 0) == True :
                    liste_contrainte.append(cas.jacobian(
                        ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx, ocp.nlp[phase].parameters.cx)[axe],
                        cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, ocp.nlp[phase].parameters.cx)))
                else:
                    liste_contrainte.append(cas.jacobian(
                            ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx,
                            ocp.nlp[phase].parameters.cx)[axe],cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, *[ocp.nlp[phase].parameters.cx])))


        jacobienne_cas = cas.vcat(liste_contrainte).T

        # gerer les parametres
        if (ocp.nlp[phase].parameters.shape == 0) == True:
            jac_func = cas.Function("jacobienne", [cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, ocp.nlp[phase].parameters.cx)], [jacobienne_cas])
        else:
            jac_func = cas.Function("jacobienne",[cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, *[ocp.nlp[phase].parameters.cx])],[jacobienne_cas])

        #evaluation jac_func en X_init, U_init, avec param

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

        jacobienne = np.array(jac_func(np.vstack((X_init, U_init, Param_init))))

        # plt.matshow(jacobienne)
        # plt.show()

        #verification rang de la jacobienne
        rang = np.linalg.matrix_rank(jacobienne)

        if rang == len(ocp.nlp[phase].g):
            print('Phase ' + str(phase) + ' : contraintes ok')
        if rang != len(ocp.nlp[phase].g):
            print('Phase ' + str(phase) + ' : contraintes mal définies')


    ################################################
    """"#####################################################
    ####verif contrainte egalité linéaire si norme hessian proche de 0####
    #########################################################"""

    for phase in range (0,len(ocp.nlp)):
        liste_hessienne = []
        liste_norme_hessienne = []
        for i in range (0,len(ocp.nlp[phase].g)):
            for axe in range (0, ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx, ocp.nlp[phase].parameters.cx).shape[0]):

                #determiner si c'est une contraintes d'egalité
                if (ocp.nlp[phase].g[i].bounds.min[axe][0] == ocp.nlp[phase].g[i].bounds.max[axe][0]) == True:

                    #gerer les parametres
                    if (ocp.nlp[phase].parameters.shape == 0) == True :
                        hessienne_cas = cas.hessian(
                            ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx, ocp.nlp[phase].parameters.cx)[axe],
                            cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, ocp.nlp[phase].parameters.cx))[0]
                    else:
                        hessienne_cas = cas.hessian(
                                ocp.nlp[phase].g[i].function(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx,
                                ocp.nlp[phase].parameters.cx)[axe], cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, *[ocp.nlp[phase].parameters.cx]))[0]


                    # gerer les parametres
                    if (ocp.nlp[phase].parameters.shape == 0) == True:
                        hes_func = cas.Function("hessienne", [
                            cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, ocp.nlp[phase].parameters.cx)],
                                                [hessienne_cas])
                    else:
                        hes_func = cas.Function("hessienne", [
                            cas.vertcat(*ocp.nlp[phase].X, *ocp.nlp[phase].U, *[ocp.nlp[phase].parameters.cx])],
                                                [hessienne_cas])

                    # evaluation hes_func en X_init, U_init, avec param

                    X_init = np.zeros((len(ocp.nlp[phase].X), ocp.nlp[phase].x_init.shape[0]))
                    U_init = np.zeros((len(ocp.nlp[phase].U), ocp.nlp[phase].u_init.shape[0]))
                    Param_init = np.array(ocp.nlp[phase].parameters.initial_guess.init)

                    for n_shooting in range(0, ocp.nlp[phase].ns + 1):
                        X_init[n_shooting, :] = np.array(ocp.nlp[phase].x_init.init.evaluate_at(n_shooting))
                    for n_shooting in range(0, ocp.nlp[phase].ns):
                        U_init[n_shooting, :] = np.array(ocp.nlp[phase].u_init.init.evaluate_at(n_shooting))

                    X_init = X_init.reshape((X_init.size, 1))
                    U_init = U_init.reshape((U_init.size, 1))
                    Param_init = Param_init.reshape(
                        (np.array(ocp.nlp[phase].parameters.initial_guess.init).size, 1))

                    hessienne = np.array(hes_func(np.vstack((X_init, U_init, Param_init))))

                    #on met la hessienne de cote
                    liste_hessienne.append(hessienne)

                else:
                    do = 'nothing'

        # calcul norm
        for nb_hessienne in range(0, len(liste_hessienne)):
            norm = 0
            for row in range(liste_hessienne[nb_hessienne].shape[0]):
                for column in range(liste_hessienne[nb_hessienne].shape[1]):
                    norm += liste_hessienne[nb_hessienne][row, column] ** 2

            liste_norme_hessienne.append(norm)
        #liste_norme_hessienne[3] = 10
        # array_norm = np.array(liste_norme_hessienne).reshape(len(liste_hessienne), 1)
        # plt.matshow(array_norm)
        # plt.show()

    ################################################
    """"#########################################
    ####verif objectif convexe avec hessienne####
    ##########################################"""
    for phase in range(0, len(ocp.nlp)):
        hessienne_obj = 0
        ocp.nlp[0].J[0].function_non_threaded(ocp.nlp[phase].states.cx, ocp.nlp[phase].controls.cx, ocp.nlp[phase].parameters.cx)




    ################################################
    ################################################

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=True))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
