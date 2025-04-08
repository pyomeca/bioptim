"""
This example mimics by essence what a jumper does which is maximizing the predicted height of the
center of mass at the peak of an aerial phase. It does so with a very simple two segments model though.
It is a clone of 'torque_driven_ocp/maximize_predicted_height_CoM.py' using
the option MINIMIZE_PREDICTED_COM_HEIGHT. It is different in the sense that the contact forces on ground have
to be downward (meaning that the object is limited to push on the ground, as one would expect when jumping, for
instance). Moreover, the lateral forces must respect some NON_SLIPPING constraint (that is the ground reaction
forces have to remain inside of the cone of friction).

It is designed to show how to use min_bound and max_bound values so they define inequality constraints instead
of equality constraints, which can be used with any ConstraintFcn
"""

import platform

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
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    PhaseDynamics,
    ContactType,
)


def prepare_ocp(
    biorbd_model_path: str,
    phase_time: float,
    n_shooting: int,
    min_bound: float,
    max_bound: float,
    mu: float,
    ode_solver: OdeSolverBase = OdeSolver.IRK(),
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
):
    """
    Prepare the actual control program to be solved

    Parameters
    ----------
    biorbd_model_path
        The path to the dynamic biorbd model
    phase_time
        The time of the phase
    n_shooting
        The number of discretization points of the phase
    min_bound
        The minimal bound of the inequality constraint
    max_bound
        The maximal bound of the inequalit constraint
    mu
        The coefficient of friction to use in the simulation
    ode_solver
        The integrator solver to use
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The OCP
    """

    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)
    tau_min, tau_max = -500, 500
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", to_second=[None, None, None, 0], to_first=[3])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        DynamicsFcn.TORQUE_DRIVEN,
        contact_type=[ContactType.RIGID_EXPLICIT],
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=1,
    )
    constraints.add(
        ConstraintFcn.TRACK_EXPLICIT_RIGID_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=2,
    )
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL_SHOOTING,
        normal_component_idx=(1, 2),
        tangential_component_idx=0,
        static_friction_coefficient=mu,
    )

    # Path constraint
    pose_at_first_node = [0, 0, -0.75, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = pose_at_first_node
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init["q"] = pose_at_first_node
    # No need to initialize qdot as it is 0

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * len(dof_mapping["tau"].to_first), [tau_max] * len(dof_mapping["tau"].to_first)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        phase_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        variable_mappings=dof_mapping,
        ode_solver=ode_solver,
    )


def main():
    model_path = "../torque_driven_ocp/models/2segments_4dof_2contacts.bioMod"
    t = 0.3
    ns = 10
    mu = 0.2
    ocp = prepare_ocp(
        biorbd_model_path=model_path,
        phase_time=t,
        n_shooting=ns,
        min_bound=50,
        max_bound=np.inf,
        mu=mu,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
