"""
This trivial example has two rodes and must superimpose a marker on one rod at the beginning and another marker on the
same rod at the end, while keeping the degrees of freedom opposed. It does this by imposing the symmetry as a
mapping, that is by completely removing the degree of freedom from the solver variables but interpreting the numbers
properly when computing the dynamics

A BiMapping is used. The way to understand the mapping is that if one is provided with two vectors, what
would be the correspondence between those vector. For instance, BiMapping([None, 0, 1, 2, -2], [0, 1, 2])
would mean that the first vector (v1) has 3 components and to create it from the second vector (v2), you would do:
v1 = [v2[0], v2[1], v2[2]]. Conversely, the second v2 has 5 components and is created from the vector v1 using:
v2 = [0, v1[0], v1[1], v1[2], -v1[2]]. For the dynamics, it is assumed that v1 is what is to be sent to the dynamic
functions (the full vector with all the degrees of freedom), while v2 is the one sent to the solver (the one with less
degrees of freedom).

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
    BiMappingList,
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
    dof_mappings = BiMappingList()
    dof_mappings.add("q", [0, 1, 2, -2], [0, 1, 2])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="m1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="m2")

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model, dof_mappings[0]))
    x_bounds[0][3:6, [0, -1]] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * len(dof_mappings["q"].to_first) * 2)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * len(dof_mappings["q"].to_first), [tau_max] * len(dof_mappings["q"].to_first))

    u_init = InitialGuessList()
    u_init.add([tau_init] * len(dof_mappings["q"].to_first))

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
        variable_mappings=dof_mappings,
    )


def main():
    """
    Solves an ocp where the symmetry must be respected, and animates it
    """

    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
