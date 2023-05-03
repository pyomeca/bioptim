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
    Solver,
)


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound, mu, ode_solver=OdeSolver.RK4()):
    # --- Options --- #
    # BioModel path
    bio_model = BiorbdModel(biorbd_model_path)
    tau_min, tau_max, tau_init = -500, 500, 0
    dof_mapping = BiMappingList()
    dof_mapping.add("tau", [None, None, None, 0], [3])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL_SHOOTING,
        contact_index=1,
    )
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
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
    n_q = bio_model.nb_q
    n_qdot = n_q
    pose_at_first_node = [0, 0, -0.75, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=bio_model.bounds_from_ranges(["q", "qdot"]))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * len(dof_mapping["tau"].to_first), [tau_max] * len(dof_mapping["tau"].to_first))

    u_init = InitialGuessList()
    u_init.add([tau_init] * len(dof_mapping["tau"].to_first))

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
        constraints,
        variable_mappings=dof_mapping,
        ode_solver=ode_solver,
        assume_phase_dynamics=True,
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
    sol = ocp.solve(Solver.IPOPT(show_online_optim=platform.system() == 'Linux'))

    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
