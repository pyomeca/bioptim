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
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
)


def prepare_ocp(biorbd_model_path, phase_time, n_shooting, min_bound, max_bound, mu, ode_solver=OdeSolver.RK4):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -500, 500, 0
    tau_mapping = BiMapping([None, None, None, 0], [3])

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT, weight=-1)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN_WITH_CONTACT)

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_force_idx=1,
    )
    constraints.add(
        ConstraintFcn.CONTACT_FORCE,
        min_bound=min_bound,
        max_bound=max_bound,
        node=Node.ALL,
        contact_force_idx=2,
    )
    constraints.add(
        ConstraintFcn.NON_SLIPPING,
        node=Node.ALL,
        normal_component_idx=(1, 2),
        tangential_component_idx=0,
        static_friction_coefficient=mu,
    )

    # Path constraint
    n_q = biorbd_model.nbQ()
    n_qdot = n_q
    pose_at_first_node = [0, 0, -0.75, 0.75]

    # Initialize x_bounds
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0][:, 0] = pose_at_first_node + [0] * n_qdot

    # Initial guess
    x_init = InitialGuessList()
    x_init.add(pose_at_first_node + [0] * n_qdot)

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * tau_mapping.to_first.len, [tau_max] * tau_mapping.to_first.len)

    u_init = InitialGuessList()
    u_init.add([tau_init] * tau_mapping.to_first.len)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        phase_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        tau_mapping=tau_mapping,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    model_path = "../torque_driven_ocp/2segments_4dof_2contacts.bioMod"
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
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    sol.animate()
