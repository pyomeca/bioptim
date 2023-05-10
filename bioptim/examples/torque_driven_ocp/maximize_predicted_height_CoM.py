"""
This example mimics by essence what a jumper does which is maximizing the predicted height of the
center of mass at the peak of an aerial phase. It does so with a very simple two segments model though.
It is designed to give a sense of the goal of the different MINIMIZE_COM functions and the use of
weight=-1 to maximize instead of minimizing.
"""

import platform

import numpy as np
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BiMappingList,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    OdeSolverBase,
    Axis,
    ConstraintList,
    ConstraintFcn,
    Node,
    Solver,
    RigidBodyDynamics,
)


def prepare_ocp(
    biorbd_model_path: str,
    phase_time: float,
    n_shooting: int,
    use_actuators: bool = False,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    objective_name: str = "MINIMIZE_PREDICTED_COM_HEIGHT",
    com_constraints: bool = False,
    rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
    assume_phase_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod file
    phase_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    use_actuators: bool
        If torque or torque activation should be used for the dynamics
    ode_solver: OdeSolver
        The ode solver to use
    objective_name: str
        The objective function to run ('MINIMIZE_PREDICTED_COM_HEIGHT',
        'MINIMIZE_COM_POSITION' or 'MINIMIZE_COM_VELOCITY')
    com_constraints: bool
        If a constraint on the COM should be applied
    rigidbody_dynamics: RigidBodyDynamics
        which transcription of rigidbody dynamics is chosen
    assume_phase_dynamics: bool
        If the dynamics equation within a phase is unique or changes at each node. True is much faster, but lacks the
        capability to have changing dynamics within a phase. A good example of when False should be used is when
        different external forces are applied at each node

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    bio_model = BiorbdModel(biorbd_model_path)

    if use_actuators:
        tau_min, tau_max, tau_init = -1, 1, 0
    else:
        tau_min, tau_max, tau_init = -500, 500, 0

    dof_mapping = BiMappingList()
    dof_mapping.add("tau", [None, None, None, 0], [3])

    # Add objective functions
    objective_functions = ObjectiveList()
    if objective_name == "MINIMIZE_PREDICTED_COM_HEIGHT":
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1)
    elif objective_name == "MINIMIZE_COM_POSITION":
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_POSITION, node=Node.ALL, axes=Axis.Z, weight=-1)
    elif objective_name == "MINIMIZE_COM_VELOCITY":
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_COM_VELOCITY, node=Node.ALL, axes=Axis.Z, weight=-1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1 / 100)

    # Dynamics
    dynamics = DynamicsList()
    if use_actuators:
        dynamics.add(DynamicsFcn.TORQUE_ACTIVATIONS_DRIVEN, with_contact=True)
    else:
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, rigidbody_dynamics=rigidbody_dynamics)

    # Constraints
    constraints = ConstraintList()
    if com_constraints:
        constraints.add(
            ConstraintFcn.TRACK_COM_VELOCITY,
            node=Node.ALL,
            min_bound=np.array([-100, -100, -100]),
            max_bound=np.array([100, 100, 100]),
        )
        constraints.add(
            ConstraintFcn.TRACK_COM_POSITION,
            node=Node.ALL,
            min_bound=np.array([-1, -1, -1]),
            max_bound=np.array([1, 1, 1]),
        )

    # Path constraint
    n_q = bio_model.nb_q
    n_qdot = n_q
    pose_at_first_node = [0, 0, -0.5, 0.5]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    if rigidbody_dynamics == RigidBodyDynamics.DAE_FORWARD_DYNAMICS:
        nu_sup = bio_model.nb_qddot
    elif rigidbody_dynamics == RigidBodyDynamics.DAE_INVERSE_DYNAMICS:
        nu_sup = bio_model.nb_qddot + bio_model.nb_contacts
    else:
        nu_sup = 0

    u_bounds = BoundsList()

    u_bounds.add(
        [tau_min] * (len(dof_mapping["tau"].to_first) + nu_sup), [tau_max] * (len(dof_mapping["tau"].to_first) + nu_sup)
    )

    u_init = InitialGuessList()
    u_init.add([tau_init] * (len(dof_mapping["tau"].to_first) + nu_sup))

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        phase_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints=constraints,
        variable_mappings=dof_mapping,
        ode_solver=ode_solver,
        assume_phase_dynamics=assume_phase_dynamics,
    )


def main():
    """
    Prepares and solves a maximal velocity at center of mass program and animates it
    """

    model_path = "models/2segments_4dof_2contacts.bioMod"
    t = 0.5
    ns = 20
    ocp = prepare_ocp(
        biorbd_model_path=model_path,
        phase_time=t,
        n_shooting=ns,
        use_actuators=False,
        objective_name="MINIMIZE_COM_VELOCITY",
        com_constraints=True,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == "Linux"))

    # --- Show results --- #
    sol.animate(n_frames=40)


if __name__ == "__main__":
    main()
