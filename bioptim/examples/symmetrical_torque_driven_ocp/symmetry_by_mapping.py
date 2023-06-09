"""
This trivial example has two rodes and must superimpose a marker on one rod at the beginning and another marker on the
same rod at the end, while keeping the degrees of freedom opposed. It does this by imposing the symmetry as a
mapping, that is by completely removing the degree of freedom from the solver variables but interpreting the numbers
properly when computing the dynamics

A BiMapping is used to link the degrees of freedom of the model to the optimization variables, that is converting a
vector of dimension M to dimension N, and vice versa. The way to understand the mapping is that if one is provided with
two vectors, what would be the correspondence between those vectors.
For instance, BiMapping([0, 1, None, 2, 2], [0, 1, 3], [4]) would mean that the first vector (v1) has 3 components and
to create it from a vector v2 of dimension 5 (using the [0, 1, 2] mapping), you would do: v1 = [v2[0], v2[1], v2[3]],
while ignoring the v2[2] and v2[4]. Conversely, the second v2 has 5 components and is created from the vector v1 of
dimension 3 using the [0, 1, None, 2, 2]/[4] mapping: v2 = [v1[0], v1[1], 0, v1[2], -v1[2]].
While used in dynamics, it is assumed that v1 is what is to be sent to biorbd (the full vector with all
the degrees of freedom), while v2 is the one sent to the solver (the one with less degrees of freedom).

The difference between symmetry_by_mapping and symmetry_by_constraint is that one (mapping) removes the degree of
freedom from the solver, while the other (constraints) imposes a proportional constraint (equals to -1) so they
are opposed.
Please note that even though removing a degree of freedom seems a good idea, it is unclear if it is actually faster when
solving with IPOPT.

Please note that while BiMapping is used in that context for reducing dof, it is only one of many more
applications one can do with the Mappings
"""

import platform

from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BiMappingList,
    SelectionMapping,
    Dependency,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str = "models/cubeSym.bioMod",
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        Path to the bioMod
    ode_solver: OdeSolverBase
        The ode solver to use
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    # Problem parameters
    n_shooting = 30
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0
    dof_mappings = BiMappingList()

    # adds a bimapping to bimappinglist
    dof_mappings.add("q", to_second=[0, 1, None, 2, 2], to_first=[0, 1, 3], oppose_to_second=4)
    # easier way is to use SelectionMapping which is a subclass of biMapping
    bimap = SelectionMapping(
        nb_elements=bio_model.nb_dof,
        independent_indices=(0, 1, 3),
        dependencies=(Dependency(dependent_index=4, reference_index=3, factor=-1),),
    )
    dof_mappings.add("q", bimapping=bimap)
    dof_mappings.add("qdot", bimapping=bimap)
    dof_mappings.add("tau", bimapping=bimap)
    # For convenience, if only q is defined, qdot and tau are automatically defined too
    # While computing the derivatives, the states is 6 dimensions (3 for q and 3 for qdot) and controls is 3 dimensions
    # However, the forward dynamics ([q, qdot, tau] => qddot) needs 5 dimensions vectors (due to the chosen model)
    # 'to_second' is used to convert these 3 dimensions vectors (q, qdot and tau) to their corresponding 5 dimensions
    #       As discussed in the docstring at the beginning of the file, the first two dofs are conserved, the 3rd
    #       value is a numerical zero and the final two are equal but opposed.
    # The dynamics is computed (qddot) and returns a 5 dimensions vector
    # 'to_first' convert back this 5 dimensions qddot to a 3 dimensions needed by Ipopt
    #       the first two dofs are conserved and the 4th (index 3) is put at the last position (3rd component). The
    #       other dofs are ignored

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

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q", mapping=dof_mappings))
    x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot", mapping=dof_mappings))
    x_bounds["qdot"][:, [0, -1]] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("tau", min_bound=[tau_min] * len(dof_mappings["q"].to_first), max_bound=[tau_max] * len(dof_mappings["q"].to_first))

    # ------------- #

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        variable_mappings=dof_mappings,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Solves an ocp where the symmetry must be respected, and animates it
    """

    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
