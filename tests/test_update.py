from pathlib import Path

import numpy as np
import biorbd

from biorbd_optim import (
    OptimalControlProgram,
    DynamicsTypeList,
    DynamicsType,
    BoundsOption,
    InitialConditionsOption,

)


def test_penalty_minimize_time():
    PROJECT_FOLDER = Path(__file__).parent / ".."
    biorbd_model = biorbd.Model(str(PROJECT_FOLDER) + "/examples/align/cube_and_line.bioMod")
    nq = biorbd_model.nbQ()

    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.TORQUE_DRIVEN)
    ocp = OptimalControlProgram(biorbd_model, dynamics, 10, 1.0)

    X_bounds = BoundsOption([-np.ones((nq*2, 1)), np.ones((nq*2, 1))])
    U_bounds = BoundsOption([-2.*np.ones((nq, 1)), 2.*np.ones((nq, 1))])
    ocp.update_bounds(X_bounds, U_bounds)
    X_init = InitialConditionsOption(0.5*np.ones((nq*2, 1)))
    U_init = InitialConditionsOption(-0.5*np.ones((nq, 1)))
    ocp.update_initial_guess(X_init, U_init)
    ocp.V_bounds
    ocp.V_init

    X_bounds = BoundsOption([-np.ones((nq*2, 1)), np.ones((nq*2, 1))])
    U_bounds = BoundsOption([-2.*np.ones((nq, 1)), 2.*np.ones((nq, 1))])
    ocp.update_bounds(X_bounds, U_bounds)
    X_init = InitialConditionsOption(0.5*np.ones((nq*2, 1)))
    U_init = InitialConditionsOption(-0.5*np.ones((nq, 1)))
    ocp.update_initial_guess(X_init, U_init)
    ocp.V_bounds
    ocp.V_init


    ocp.update_initial_guess(X_bounds, U_bounds)
    ocp.update_bounds(X_init, U_init)
