"""
# TODO: Remove all the examples/muscle_driven_with_contact and make sure everything is properly tested
All the examples in muscle_driven_with_contact are merely to show some dynamics and prepare some OCP for the tests.
It is not really relevant and will be removed when unitary tests for the dynamics will be implemented
"""


import importlib.util
from pathlib import Path

import numpy as np
import biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
)

# Load track_segment_on_rt
spec = importlib.util.spec_from_file_location(
    "data_to_track", str(Path(__file__).parent) + "/contact_forces_inequality_constraint_muscle.py"
)
data_to_track = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_to_track)


def prepare_ocp(
    biorbd_model_path, phase_time, n_shooting, muscle_activations_ref, contact_forces_ref, ode_solver=OdeSolver.RK4()
):
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -500, 500, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MUSCLES_CONTROL, target=muscle_activations_ref)
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTACT_FORCES, target=contact_forces_ref)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=0.001)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_ALL_CONTROLS, weight=0.001)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)

    # Path constraint
    n_q = biorbd_model.nbQ()
    n_qdot = n_q
    pose_at_first_node = [0, 0, -0.75, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * biorbd_model.nbGeneralizedTorque() + [activation_min] * biorbd_model.nbMuscleTotal(),
        [tau_max] * biorbd_model.nbGeneralizedTorque() + [activation_max] * biorbd_model.nbMuscleTotal(),
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [activation_init] * biorbd_model.nbMuscleTotal())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        phase_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
    )


def main():
    # Define the problem
    model_path = "2segments_4dof_2contacts_1muscle.bioMod"
    final_time = 0.7
    ns = 20

    # Generate data using another optimization that will be feedback in as tracking data
    ocp_to_track = data_to_track.prepare_ocp(
        biorbd_model_path=model_path,
        phase_time=final_time,
        n_shooting=ns,
        min_bound=50,
        max_bound=np.inf,
    )
    sol = ocp_to_track.solve()
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]
    x = np.concatenate((q, qdot))
    u = np.concatenate((tau, mus))
    contact_forces_ref = np.array(ocp_to_track.nlp[0].contact_forces_func(x[:, :-1], u[:, :-1], []))
    muscle_activations_ref = mus

    # Track these data
    ocp = prepare_ocp(
        biorbd_model_path=model_path,
        phase_time=final_time,
        n_shooting=ns,
        muscle_activations_ref=muscle_activations_ref[:, :-1],
        contact_forces_ref=contact_forces_ref,
    )

    def add_custom_plots(ocp, nb_phases):

        def casadi_func(J_MX):
            func = Function("val", [J_MX], [J_MX])
            return func

        def casadi_concat(MX_array, New_MX):
            func = Function("val", [MX_array, New_MX], [cas.horzcat(MX_array, New_MX)])
            return func

        def casadi_func_objectives(casadi_func_eval, nlp, i_objectives):
            # J_values = np.array([])
            for i_subobjective in range(len(nlp.J[i_objectives])):
                # MX_SYM = cas.MX.sym("res", 1, 1)
                # MX_SYM_2 = cas.MX.sym("res_2", 1, i_subobjective+1)
                # casadi_func_concat = casadi_concat(MX_SYM, MX_SYM_2)

                # J_values = casadi_func_concat(J_values, casadi_func_eval(nlp.J[i_objectives][i_subobjective]["val"]))
                if i_subobjective == 0:
                    J_values = casadi_func_eval(nlp.J[i_objectives][0]["val"])
                else:
                    J_values = vertcat(J_values, casadi_func_eval(nlp.J[i_objectives][i_subobjective]["val"]))

            return J_values

        MX_SYM = cas.MX.sym("res", 1, 1)
        casadi_func_eval = casadi_func(MX_SYM)

        for i_phase in range(nb_phases):
            # Plot Objectives
            for i_objectives in range(len(ocp.nlp[i_phase].J)):
                # casadi_objectives = casadi_func_objectives(ocp, i_phase, i_objectives)
                ocp.add_plot('OBJECTIVE_'+ocp.nlp[i_phase].J[i_objectives][0]['objective'].name, casadi_func_objectives(casadi_func_eval, ocp.nlp[i_phase], i_objectives), node_index=ocp.nlp[i_phase].J[i_objectives][0]['node_index'], phase=i_phase, plot_type=PlotType.INTEGRATED, casadi_func_eval=casadi_func_eval, i_objectives=i_objectives)
    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
