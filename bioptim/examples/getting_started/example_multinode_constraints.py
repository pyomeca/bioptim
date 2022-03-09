"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement and
a the at different marker at the end of each phase. Moreover a constraint on the rotation is imposed on the cube.
Extra constraints are defined between specific nodes of phases.
It is designed to show how one can define a multinode constraints and objectives in a multiphase optimal control program
"""


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
    MultinodeConstraintList,
    MultinodeConstraintFcn,
)


def minimize_difference(all_pn: PenaltyNode):
    return all_pn[0].nlp.controls.cx_end - all_pn[1].nlp.controls.cx


def prepare_ocp(
    biorbd_model_path: str = "models/cube.bioMod", ode_solver: OdeSolver = OdeSolver.RK4()
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Problem parameters
    n_shooting = (100, 300, 100)
    final_time = (2, 5, 4)
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=2)

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

    # Constraints
    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    multinode_constraints.add(
        MultinodeConstraintFcn.EQUALITY,
        phase_first_idx=0,
        phase_second_idx=2,
        first_node=Node.START,
        second_node=Node.START,
    )
    # Objectives with the weight as an argument
    multinode_constraints.add(
        MultinodeConstraintFcn.EQUALITY,
        phase_first_idx=0,
        phase_second_idx=2,
        first_node=2,
        second_node=Node.MID,
        weight=2,
    )
    # Objectives with the weight as an argument
    multinode_constraints.add(
        MultinodeConstraintFcn.EQUALITY,
        phase_first_idx=0,
        phase_second_idx=1,
        first_node=Node.MID,
        second_node=Node.END,
        weight=0.1,
    )

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
        multinode_constraints=multinode_constraints,
        ode_solver=ode_solver,
    )


def main():
    """
    Defines a multiphase ocp and animate the results
    """

    ocp = prepare_ocp(long_optim=True)

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    sol.print()
    sol.graphs()
    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
