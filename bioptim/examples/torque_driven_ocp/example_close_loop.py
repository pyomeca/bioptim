"""
This example shown hot to use the objective MINIMIZE_JCS.
The third segments must stay aligned with the vertical.
Note that there are other ways to do this, here it is used to examplify how to use the function MINIMIZE_JCS.
"""
import biorbd as biorbd
import bioviz
import numpy as np
from Close_loop_try.myIntegrators import RK8
# from bioptim import (
#     BiorbdModel,
#     OptimalControlProgram,
#     DynamicsList,
#     DynamicsFcn,
#     ObjectiveList,
#     BoundsList,
#     InitialGuessList,
#     ObjectiveFcn,
#     ConstraintList,
#     ConstraintFcn,
#     Node,
# )

def forward_dynamics(t, x):
    q = x[0:nb_dof]
    dq = x[nb_dof:]

    ddq = m.ForwardDynamicsConstraintsDirect(q, dq, tau).to_array()

    return np.concatenate((dq, ddq))


def contact_forces_from_constrained_forward_dynamics(m, q, qdot, tau):
    return m.calcLoopConstraintForces(q, qdot, tau)[0].to_array()

# loading the model
model_name = "models/quadruple_pendulum_loopconstraint.bioMod"
m = biorbd.Model(model_name)

nb_dof = m.nbQ()

# torques
tau = np.ones(nb_dof) * 0

# states
x0 = np.zeros((nb_dof * 2,))
x0[0] = 0
x0[1] = np.pi / 2
x0[2] = 0
x0[3:] = 0

# final time (s)
t_fin = 1
# Integrator fixed-step
npt = 500 * t_fin
t = np.linspace(0, 1, int(npt)).T * t_fin
msize = 0.5

res_rk8 = RK8(t, forward_dynamics, x0)


# a posteriori computation of loop constraint forces
for i, ti in enumerate(t):
    q = res_rk8[0:nb_dof, i]
    dq = res_rk8[nb_dof:, i]
    f = contact_forces_from_constrained_forward_dynamics(m, q, dq, tau)
    print(f)

# Animate the model
biorbd_viz1 = bioviz.Viz(model_path="models/quadruple_pendulum_loopconstraint.bioMod", show_contacts=False,)
biorbd_viz1.load_movement(res_rk8[0:4, :])
biorbd_viz1.exec()