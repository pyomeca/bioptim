"""
# TODO: Remove all the examples/muscle_driven_with_contact and make sure everything is properly tested
All the examples in muscle_driven_with_contact are merely to show some dynamics and prepare some OCP for the tests.
It is not really relevant and will be removed when unitary tests for the dynamics will be implemented
"""

from matplotlib import pyplot as plt
import numpy as np
import biorbd
from bioptim import (
    Node,
    OptimalControlProgram,
    ConstraintList,
    ConstraintFcn,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BiMapping,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
)


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, ode_solver=OdeSolver.RK4()):
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -500, 500, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5
    tau_mapping = BiMapping([None, None, None, 0], [3])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=np.inf,
        node=Node.ALL,
        contact_force_idx=1,
    )
    constraints.add(
        ConstraintFcn.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=np.inf,
        node=Node.ALL,
        contact_force_idx=2,
    )

    # Path constraint
    n_q = biorbd_model.nbQ()
    n_qdot = n_q
    n_mus = biorbd_model.nbMuscleTotal()
    pose_at_first_node = [0, 0, -0.75, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0].concatenate(Bounds([activation_min] * n_mus, [activation_max] * n_mus))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot + [0.5] * n_mus

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * n_qdot + [0.5] * n_mus)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add(
        [torque_min] * tau_mapping.to_first.len + [activation_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * tau_mapping.to_first.len + [activation_max] * biorbd_model.nbMuscleTotal(),
    )

    u_init = InitialGuessList()
    u_init.add([torque_init] * tau_mapping.to_first.len + [activation_init] * biorbd_model.nbMuscleTotal())
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
        constraints=constraints,
        tau_mapping=tau_mapping,
        ode_solver=ode_solver,
    )


def main():
    biorbd_model_path = "2segments_4dof_2contacts_1muscle.bioMod"
    t = 0.3
    ns = 10
    ocp = prepare_ocp(biorbd_model_path=biorbd_model_path, phase_time=t, n_shooting=ns, min_bound=50)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    nlp = ocp.nlp[0]
    nlp.model = biorbd.Model(biorbd_model_path)

    q = sol.states["q"]
    qdot = sol.states["qdot"]
    activations = sol.states["muscles"]
    tau = sol.controls["tau"]
    excitations = sol.controls["muscles"]

    x = np.concatenate((q, qdot, activations))
    u = np.concatenate((tau, excitations))
    contact_forces = np.array(nlp.contact_forces_func(x[:, :-1], u[:, :-1], []))

    names_contact_forces = ocp.nlp[0].model.contactNames()
    for i, elt in enumerate(contact_forces):
        plt.plot(np.linspace(0, t, ns + 1)[:-1], elt, ".-", label=f"{names_contact_forces[i].to_string()}")
    plt.legend()
    plt.grid()
    plt.title("Contact forces")
    plt.show()

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
