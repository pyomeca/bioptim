"""
This trivial example has two rodes and must superimpose a marker on one rod at the beginning and another marker on the
same rod at the end, while keeping the degrees of freedom opposed. It does this by imposing the symmetry as a
proportional constraint (equals to -1).

The proportional constraint simply creates a constraint such that: state[i] = coef * state[j], where the coef is the
proportional constraint.

The difference between symmetry_by_mapping and symmetry_by_constraint is that one (mapping) removes the degree of
freedom from the solver, while the other (constraints) imposes a proportional constraint (equals to -1) so they
are opposed.
Please note that even though removing a degree of freedom seems a good idea, it is unclear if it is actually faster when
solving with IPOPT.
"""

import biorbd
from bioptim import (
    Node,
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
)


def prepare_ocp(
    biorbd_model_path: str = "cubeSym.bioMod", ode_solver: OdeSolver = OdeSolver.RK4()
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        Path to the bioMod
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    n_shooting = 30
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)
    constraints.add(ConstraintFcn.PROPORTIONAL_STATE, node=Node.ALL, first_dof=2, second_dof=3, coef=-1)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][4:8, [0, -1]] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model.nbQ(), [tau_max] * biorbd_model.nbQ())

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model.nbQ())

    # ------------- #

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
    Solves an ocp where the symmetry is enforced by constraints, and animates it
    """

    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
