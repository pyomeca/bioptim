import os
import numpy as np
import pytest
from bioptim import (
    Bounds,
    BoundsList,
    InterpolationType,
    BiorbdModel,
    Objective,
    ObjectiveFcn,
    Dynamics,
    DynamicsFcn,
    InitialGuess,
    OptimalControlProgram,
)



def test_accessors_on_bounds_option():
    x_min = [-100] * 6
    x_max = [100] * 6
    x_bounds = Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT)
    x_bounds[:3] = 0
    x_bounds.min[3:] = -10
    x_bounds.max[1:3] = 10

    # Check accessor and min/max values to be equal
    np.testing.assert_almost_equal(x_bounds[:].min, x_bounds.min[:])
    np.testing.assert_almost_equal(x_bounds[:].max, x_bounds.max[:])

    # Check min and max have the right value
    np.testing.assert_almost_equal(x_bounds.min[:], np.array([[0], [0], [0], [-10], [-10], [-10]]))
    np.testing.assert_almost_equal(x_bounds.max[:], np.array([[0], [10], [10], [100], [100], [100]]))


def test_accessors_on_bounds_option_multidimensional():
    x_min = [[-100, -50, 0] for i in range(6)]
    x_max = [[100, 150, 200] for i in range(6)]
    x_bounds = Bounds(x_min, x_max, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds[:3, 0] = 0
    x_bounds.min[1:5, 1:] = -10
    x_bounds.max[1:5, 1:] = 10

    # Check accessor and min/max values to be equal
    np.testing.assert_almost_equal(x_bounds[:].min, x_bounds.min[:])
    np.testing.assert_almost_equal(x_bounds[:].max, x_bounds.max[:])

    # Check min and max have the right value
    np.testing.assert_almost_equal(
        x_bounds.min[:],
        np.array([[0, -50, 0], [0, -10, -10], [0, -10, -10], [-100, -10, -10], [-100, -10, -10], [-100, -50, 0]]),
    )
    np.testing.assert_almost_equal(
        x_bounds.max[:],
        np.array([[0, 150, 200], [0, 10, 10], [0, 10, 10], [100, 10, 10], [100, 10, 10], [100, 150, 200]]),
    )

def test_bounds_error_messages():
    """
    This tests that the error messages are properly raised. The OCP is adapted from the getting_started/pendulum.py example.
    """
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)
    biorbd_model_path = bioptim_folder + "/models/pendulum.bioMod"
    bio_model = BiorbdModel(biorbd_model_path)
    n_q = bio_model.nb_q

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Define states path constraint
    x_bounds = Bounds([-100] * n_q * 2, [100] * n_q * 2)
    x_bounds[1, :] = 0  # Prevent the model from actively rotate

    # Define control path constraint
    u_bounds = Bounds([-100] * n_q, [100] * n_q)
    u_bounds[1, :] = 0  # Prevent the model from actively rotate

    x_init = InitialGuess([0] * n_q * 2)
    u_init = InitialGuess([0] * n_q)

    # check the error messages
    with pytest.raises(
        RuntimeError,
        match="If you do not want to provide an x_bounds, you should declare x_bounds=None instead of an empty BoundsList",
    ):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting=5,
            phase_time=1,
            x_init=x_init,
            u_init=u_init,
            x_bounds=BoundsList(),
            u_bounds=u_bounds,
            objective_functions=objective_functions,
        )

    with pytest.raises(
            RuntimeError,
            match="If you do not want to provide a u_bounds, you should declare u_bounds=None instead of an empty BoundsList",
    ):
        OptimalControlProgram(
            bio_model,
            dynamics,
            n_shooting=5,
            phase_time=1,
            x_init=x_init,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=BoundsList(),
            objective_functions=objective_functions,
        )
