import os

from bioptim.gui.online_callback_server import _serialize_xydata, _deserialize_xydata
from bioptim.gui.plot import PlotOcp
from bioptim.optimization.optimization_vector import OptimizationVectorHelper
from casadi import DM, Function
import numpy as np


def test_serialize_deserialize():
    # Prepare a set of data to serialize and deserialize
    from bioptim.examples.getting_started import pendulum as ocp_module

    bioptim_folder = os.path.dirname(ocp_module.__file__)

    ocp = ocp_module.prepare_ocp(
        biorbd_model_path=bioptim_folder + "/models/pendulum.bioMod",
        final_time=1,
        n_shooting=40,
    )

    dummy_phase_times = OptimizationVectorHelper.extract_step_times(ocp, DM(np.ones(ocp.n_phases)))
    plotter = PlotOcp(ocp, dummy_phase_times=dummy_phase_times, show_bounds=True, only_initialize_variables=True)

    np.random.seed(42)
    xdata, ydata = plotter.parse_data(**{"x": np.random.rand(ocp.variables_vector.shape[0])[:, None]})

    # Serialize and deserialize the data
    serialized_data = _serialize_xydata(xdata, ydata)
    deserialized_xdata, deserialized_ydata = _deserialize_xydata(serialized_data)

    # Compare the outputs
    for x_phase, deserialized_x_phase in zip(xdata, deserialized_xdata):
        for x_node, deserialized_x_node in zip(x_phase, deserialized_x_phase):
            assert np.allclose(x_node, DM(deserialized_x_node))

    for y_variable, deserialized_y_variable in zip(ydata, deserialized_ydata):
        if isinstance(y_variable, np.ndarray):
            assert np.allclose(y_variable, deserialized_y_variable[0], equal_nan=True)
        else:
            for y_phase, deserialized_y_phase in zip(y_variable, deserialized_y_variable):
                assert np.allclose(y_phase, deserialized_y_phase)
