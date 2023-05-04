"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end. While doing so, a force pushes the box upward.
The solver must minimize the force needed to lift the box while reaching the marker in time.
It is designed to show how to use external forces. An example of external forces that depends on the state (for
example a spring) can be found at 'examples/torque_driven_ocp/spring_load.py'

Please note that the point of application of the external forces are defined in the bioMod file by the
externalforceindex tag in segment and is acting at the center of mass of this particular segment. Please note that
this segment MUST have at least one degree of freedom defined (translations and/or rotations). Otherwise, the
external_force is silently ignored. Bioptim expects external_forces to be a list (one element for each phase) of
np.ndarray of shape (6, i, n), where the 6 components are [Mx, My, Mz, Fx, Fy, Fz], for the ith force platform
(defined by the externalforceindex) for each node n
"""
import platform

import numpy as np
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str = "models/cube_with_forces.bioMod", ode_solver: OdeSolver = OdeSolver.RK4()
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Problem parameters
    n_shooting = 30
    final_time = 2

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)

    # Dynamics
    dynamics = DynamicsList()
    expand = False if isinstance(ode_solver, OdeSolver.IRK) else True
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, expand=expand)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")

    # External forces. external_forces is of len 1 because there is only one phase.
    # The array inside it is 6x2x30 since there is [Mx, My, Mz, Fx, Fy, Fz] for the two externalforceindex for each node
    external_forces = [[np.array([[0, 0, 0, 0, 0, -2], [0, 0, 0, 0, 0, 5]]).T for _ in range(n_shooting)]]
    external_forces[0][4] = np.array([[0, 0, 0, 0, 0, -22], [0, 0, 0, 0, 0, 52]]).T

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][3:6, [0, -1]] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (bio_model.nb_q + bio_model.nb_qdot))

    # Define control path constraint
    u_bounds = BoundsList()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds.add([tau_min] * bio_model.nb_tau, [tau_max] * bio_model.nb_tau)

    u_init = InitialGuessList()
    u_init.add([tau_init] * bio_model.nb_tau)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        external_forces=external_forces,
        ode_solver=ode_solver,
        assume_phase_dynamics=False,
    )


def main():
    """
    Solve an ocp with external forces and animates the solution
    """

    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == 'Linux'))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
