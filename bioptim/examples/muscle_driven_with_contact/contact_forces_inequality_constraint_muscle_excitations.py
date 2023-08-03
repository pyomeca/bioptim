"""
# TODO: Remove all the examples/muscle_driven_with_contact and make sure everything is properly tested
All the examples in muscle_driven_with_contact are merely to show some dynamics and prepare some OCP for the tests.
It is not really relevant and will be removed when unitary tests for the dynamics will be implemented
"""

import platform

from matplotlib import pyplot as plt
import numpy as np
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BiMappingList,
    SelectionMapping,
    Dependency,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    Solver,
)


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, ode_solver=OdeSolver.RK4(), expand_dynamics=True):
    bio_model = BiorbdModel(biorbd_model_path)
    torque_min, torque_max, torque_init = -500.0, 500.0, 0.0
    activation_min, activation_max, activation_init = 0.0, 1.0, 0.5
    dof_mapping = BiMappingList()

    # adds a bimapping to bimappinglist
    # dof_mapping.add("tau", [None, None, None, 0], [3])
    # easier way is to use SelectionMapping which is a subclass of biMapping
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", bimapping=None, to_second=[None, None, None, 0], to_first=[3])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.MUSCLE_DRIVEN,
        with_excitations=True,
        with_residual_torque=True,
        with_contact=True,
        expand=expand_dynamics,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=1,
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
        contact_index=2,
    )

    # Path constraint
    n_q = bio_model.nb_q
    n_qdot = n_q
    n_mus = bio_model.nb_muscles
    pose_at_first_node = [0, 0, -0.75, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = pose_at_first_node
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = np.zeros((n_qdot,))
    x_bounds["muscles"] = [[activation_min] * n_mus, [activation_max] * n_mus]
    x_bounds["muscles"][:, 0] = np.zeros((n_mus,))

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = pose_at_first_node
    x_init["qdot"] = np.zeros((n_qdot,))
    x_init["muscles"] = np.zeros((n_mus,))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [torque_min] * len(dof_mapping["tau"].to_first), [torque_max] * len(dof_mapping["tau"].to_first)
    u_bounds["muscles"] = [activation_min] * bio_model.nb_muscles, [activation_max] * bio_model.nb_muscles

    u_init = InitialGuessList()
    u_init["tau"] = [torque_init] * len(dof_mapping["tau"].to_first)
    u_init["muscles"] = [activation_init] * bio_model.nb_muscles
    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        phase_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=dof_mapping,
        ode_solver=ode_solver,
        assume_phase_dynamics=True,
    )


def main():
    biorbd_model_path = "models/2segments_4dof_2contacts_1muscle.bioMod"
    t = 0.3
    ns = 10
    ocp = prepare_ocp(biorbd_model_path=biorbd_model_path, phase_time=t, n_shooting=ns, min_bound=50)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    nlp = ocp.nlp[0]
    nlp.model = BiorbdModel(biorbd_model_path)

    q = sol.states["q"]
    qdot = sol.states["qdot"]
    activations = sol.states["muscles"]
    tau = sol.controls["tau"]
    excitations = sol.controls["muscles"]

    x = np.concatenate((q, qdot, activations))
    u = np.concatenate((tau, excitations))
    contact_forces = np.array(nlp.contact_forces_func(x[:, :-1], u[:, :-1], []))

    names_contact_forces = ocp.nlp[0].model.contact_names
    for i, elt in enumerate(contact_forces):
        plt.plot(np.linspace(0, t, ns + 1)[:-1], elt, ".-", label=f"{names_contact_forces[i]}")
    plt.legend()
    plt.grid()
    plt.title("Contact forces")
    plt.show()

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
